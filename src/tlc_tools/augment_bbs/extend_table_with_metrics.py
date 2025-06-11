from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, cast, Literal

import cv2
import numpy as np
import pacmap
import tlc
import torch
import torchvision.transforms as transforms
from PIL import ImageStat
from torch.utils.data import DataLoader
from tqdm import tqdm

from tlc_tools.common import InstanceConfig

from .instance_crop_dataset import InstanceCropDataset
from .label_utils import create_label_mappings

CLASSIFIER_EMBEDDING = "classif_Embedding"
CLASSIFIER_LABEL = "classif_Label"
CLASSIFIER_CONFIDENCE = "classif_Confidence"


# ========================
# SCHEMA FACTORY FUNCTIONS
# ========================


def create_embedding_schema(
    instance_type: Literal["bounding_boxes", "segmentations"], num_components: int
) -> tlc.Schema:
    """Create embedding schema appropriate for instance type."""
    return tlc.Schema(
        value=tlc.Float32Value(),
        size0=tlc.DimensionNumericValue(num_components, num_components),
        size1=tlc.DimensionNumericValue(0, 1000) if instance_type == "segmentations" else None,
    )


def create_label_schema(
    instance_type: Literal["bounding_boxes", "segmentations"],
    label_schema_template: tlc.Schema | None = None,
    background_label: int | None = None,
) -> tlc.Schema:
    """Create label schema appropriate for instance type."""
    if label_schema_template:
        # Use existing label schema as template
        label_schema = deepcopy(label_schema_template)
        if background_label is not None:
            assert hasattr(label_schema.value, "map") and label_schema.value.map is not None
            label_schema.value.map[background_label] = tlc.MapElement("background")
    else:
        # Label-free mode - create integer schema for predicted labels
        label_schema = tlc.Schema(value=tlc.Int32Value(), writable=False)

    label_schema.writable = False
    label_schema.size0 = tlc.DimensionNumericValue(0, 1000) if instance_type == "segmentations" else None

    return label_schema


def create_confidence_schema(instance_type: Literal["bounding_boxes", "segmentations"]) -> tlc.Schema:
    """Create confidence schema appropriate for instance type."""
    return tlc.Schema(
        value=tlc.Float32Value(),
        writable=False,
        size0=tlc.DimensionNumericValue(0, 1000) if instance_type == "segmentations" else None,
    )


def create_metrics_schema(instance_type: Literal["bounding_boxes", "segmentations"]) -> tlc.Schema:
    """Create image metrics schema appropriate for instance type."""
    return tlc.Schema(
        value=tlc.schema.Float32Value(),
        writable=False,
        size0=tlc.DimensionNumericValue(0, 1000) if instance_type == "segmentations" else None,
    )


def add_embedding_schemas_to_instance_properties(
    instance_properties_schema: tlc.Schema,
    instance_type: Literal["bounding_boxes", "segmentations"],
    num_components: int,
    use_pretrained: bool,
    label_column_path: str | None,
    allow_label_free: bool,
    background_label: int | None,
    new_table_schema: tlc.Schema,
) -> None:
    """Add embedding-related schemas to instance properties schema."""
    # Create and add embedding schema
    embedding_schema = create_embedding_schema(instance_type, num_components)
    if CLASSIFIER_EMBEDDING not in instance_properties_schema.values:
        instance_properties_schema.add_sub_schema(CLASSIFIER_EMBEDDING, embedding_schema)

    # Only add label and confidence schemas for non-pretrained models
    if not use_pretrained:
        # Create label schema
        label_schema_template = None
        if label_column_path and not allow_label_free:
            # Use existing label schema as template
            label_parts = label_column_path.split(".")
            temp_schema = new_table_schema
            for part in label_parts:
                temp_schema = temp_schema.values[part]
            label_schema_template = temp_schema

        label_schema = create_label_schema(instance_type, label_schema_template, background_label)
        confidence_schema = create_confidence_schema(instance_type)

        if CLASSIFIER_LABEL not in instance_properties_schema.values:
            instance_properties_schema.add_sub_schema(CLASSIFIER_LABEL, label_schema)
        if CLASSIFIER_CONFIDENCE not in instance_properties_schema.values:
            instance_properties_schema.add_sub_schema(CLASSIFIER_CONFIDENCE, confidence_schema)


def add_metrics_schemas_to_instance_properties(
    instance_properties_schema: tlc.Schema,
    instance_type: Literal["bounding_boxes", "segmentations"],
) -> None:
    """Add image metrics schemas to instance properties schema."""
    metrics_schema = create_metrics_schema(instance_type)

    for metric_name in ["brightness", "contrast", "sharpness"]:
        if metric_name not in instance_properties_schema.values:
            instance_properties_schema.add_sub_schema(metric_name, metrics_schema)


def custom_collate_fn(batch):
    """Custom collate function to handle PIL images alongside tensors.

    :param batch: List of (pil_image, tensor, label) tuples
    :return: (pil_images_list, tensor_batch, label_batch)
    """
    pil_images = [item[0] for item in batch]  # Keep PIL images in a list
    tensors = torch.stack([item[1] for item in batch])  # Stack tensors normally
    labels = torch.tensor([item[2] for item in batch])  # Convert labels to tensor

    return pil_images, tensors, labels


def calculate_bb_metrics(image, bb, bb_schema):
    """Calculate metrics for a single bounding box crop"""
    # Get the crop using BBCropInterface
    crop = tlc.BBCropInterface.crop(
        image, bb, bb_schema, x_max_offset=0, y_max_offset=0, x_scale_range=(1.0, 1.0), y_scale_range=(1.0, 1.0)
    )

    # Calculate metrics
    gray_crop = crop.convert("L")
    brightness = ImageStat.Stat(crop).mean[0]
    contrast = ImageStat.Stat(gray_crop).stddev[0]
    pixels = np.array(crop)
    sharpness = np.var(cv2.Laplacian(pixels, cv2.CV_64F))

    return {"brightness": float(brightness), "contrast": float(contrast), "sharpness": float(sharpness)}


def extend_table_with_metrics(
    input_table: tlc.Table,
    output_table_name: str,
    add_embeddings: bool = False,
    add_image_metrics: bool = False,
    model_checkpoint: str | None = None,
    model_name: str = "efficientnet_b0",
    batch_size: int = 64,
    num_components: int = 3,
    pacmap_reducer: pacmap.PaCMAP | None = None,
    fit_embeddings: np.ndarray | None = None,
    n_neighbors: int = 10,
    device: torch.device | None = None,
    reduce_last_dims: int = 0,
    max_memory_gb: int = 8,
    num_workers: int = 0,  # New parameter for DataLoader
    instance_config: InstanceConfig | None = None,  # New parameter
) -> tuple[str, pacmap.PaCMAP | None, np.ndarray | None]:
    """Extend table with embeddings and/or image metrics in a single pass.

    :param input_table: Input table to extend.
    :param output_table_name: Name of the output table.
    :param add_embeddings: Whether to add embeddings.
    :param add_image_metrics: Whether to add image metrics.
    :param model_checkpoint: Path to the model checkpoint to load.
    :param model_name: Name of the model to use.
    :param batch_size: Batch size for processing.
    :param num_components: Number of components for PaCMAP.
    :param pacmap_reducer: PaCMAP reducer to use.
    :param fit_embeddings: Fit embeddings to use.
    :param n_neighbors: Number of neighbors for PaCMAP.
    :param device: Device to use.
    :param reduce_last_dims: Number of dimensions to reduce from the end (0 means no reduction).
    :param max_memory_gb: Maximum memory to use in GB.
    :param num_workers: Number of workers for DataLoader.
    :param label_column_path: Path to the label column in the table (deprecated, use instance_config).
    :param instance_config: Instance configuration object with column/type/label info.

    :return: Tuple of output table URL, PaCMAP reducer, and fit embeddings.
    """
    if not (add_embeddings or add_image_metrics):
        raise ValueError("Must specify at least one type of metrics to add")

    # Resolve instance configuration - backward compatibility with label_column_path
    if instance_config is None:
        # Use factory method for new API
        instance_config = InstanceConfig.resolve(
            input_table=input_table,
            allow_label_free=False,  # Default to requiring labels for backward compatibility
        )
    else:
        # Ensure config is validated for this table (cached, so safe to call multiple times)
        instance_config._ensure_validated_for_table(input_table)

    instance_column = instance_config.instance_column
    instance_type = instance_config.instance_type
    label_column_path = instance_config.label_column_path
    instance_properties_column = instance_config.instance_properties_column

    # Check if we have labels when needed for embeddings
    # if add_embeddings and not instance_config.allow_label_free and instance_config.label_column_path is None:
    #     raise ValueError("Model checkpoint required for embeddings, and labels required for training model")

    # if add_embeddings and model_checkpoint is None and not instance_config.allow_label_free:
    #     raise ValueError("Model checkpoint required for embeddings (or use allow_label_free=True for pretrained model)")

    # Create dataset using the refactored BBCropDataset
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset - it handles the instance extraction logic
    dataset = InstanceCropDataset(
        input_table,
        transform=image_transform,
        instance_config=instance_config,
    )

    total_instances = len(dataset)
    print(f"Total instances to process: {total_instances}")

    # Create DataLoader with multi-worker support and custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep deterministic order
        num_workers=num_workers,
        pin_memory=bool(device and device.type == "cuda"),
        collate_fn=custom_collate_fn,  # Use custom collate for PIL images
    )

    # Collect embeddings and metrics if needed
    labels: list[int] = []
    confidences_list: list[float] = []
    image_metrics_list: list[dict[str, float]] = []
    if add_embeddings:
        # Load model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Device: ", device)

        # Determine if we're using a checkpoint or pretrained model
        use_pretrained = model_checkpoint is None and instance_config.allow_label_free

        if use_pretrained:
            print("Using pretrained model (label-free mode)")
            # For pretrained model, we don't have custom classes
            num_classes = 1000  # Standard ImageNet classes
            label_map = {}
            label_2_contiguous_idx: dict[int, int] = {}
            contiguous_2_label: dict[int, int] = {}
            background_label = None
            add_background = False
        else:
            # Load checkpoint first to get number of classes
            checkpoint = torch.load(cast(str, model_checkpoint), map_location=device, weights_only=True)
            num_classes = checkpoint["classifier.bias"].shape[0]

            # Get label map and determine if background was used - use instance_config
            if instance_config.label_column_path:
                label_map = input_table.get_simple_value_map(instance_config.label_column_path)
                if not label_map:
                    raise ValueError(f"Label map not found in table at path: {instance_config.label_column_path}")
            else:
                # Label-free mode - create dummy mappings
                label_map = {}

            label_2_contiguous_idx, contiguous_2_label, background_label, add_background = create_label_mappings(
                label_map, include_background=num_classes > len(label_map) if label_map else False
            )

        print(f"Label map: {label_map}")
        print(f"Using {num_classes} classes with background={add_background}")
        print(f"Label to contiguous mapping: {label_2_contiguous_idx}")
        print(f"Contiguous to label mapping: {contiguous_2_label}")

        import timm

        model = timm.create_model(model_name, pretrained=use_pretrained, num_classes=num_classes)
        if not use_pretrained:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        # Constants for chunking
        CHUNK_SIZE = 16384
        chunk_dir = os.path.join(os.path.expanduser("~"), ".tlc_temp_embeddings")
        os.makedirs(chunk_dir, exist_ok=True)

        # Process batches and save chunks
        current_chunk = []
        chunk_count = 0
        print("Processing batches and saving chunks...")

        # Get first batch to determine embedding shape (peek only)
        peek_dataloader = DataLoader(
            dataset,
            batch_size=1,  # Just one sample to peek
            shuffle=False,
            num_workers=0,  # No multiprocessing for peek
            collate_fn=custom_collate_fn,  # Use custom collate for PIL images
        )
        peek_batch = next(iter(peek_dataloader))
        _, peek_tensors, _ = peek_batch  # Unpack: pil_images, tensors, labels
        peek_images = peek_tensors.to(device)

        with torch.no_grad():
            peek_embedding = model.forward_features(peek_images).cpu().numpy()
            print(f"Shape of embedding: {peek_embedding.shape[1:]}")
            if reduce_last_dims > 0:
                peek_embedding = peek_embedding.mean(axis=tuple(range(-reduce_last_dims, 0)))
            if len(peek_embedding.shape) > 2:
                peek_embedding = peek_embedding.reshape(len(peek_embedding), -1)
            print(f"Shape of flattened embedding: {peek_embedding.shape}")

    # Process all batches (for embeddings and/or metrics)
    if add_embeddings or add_image_metrics:
        dataloader_iter = iter(dataloader)
        desc = "Running model inference" if add_embeddings else "Computing image metrics"
        for batch_data in tqdm(dataloader_iter, desc=desc, total=len(dataloader)):
            batch_pil_images, batch_tensors, batch_labels = batch_data  # Unpack the custom collate output

            # Collect image metrics if needed (using original PIL images)
            if add_image_metrics:
                for pil_image in batch_pil_images:
                    # Calculate metrics directly on original PIL image
                    gray_crop = pil_image.convert("L")
                    brightness = ImageStat.Stat(pil_image).mean[0]
                    contrast = ImageStat.Stat(gray_crop).stddev[0]
                    pixels = np.array(pil_image)
                    sharpness = np.var(cv2.Laplacian(pixels, cv2.CV_64F))

                    image_metrics_list.append(
                        {"brightness": float(brightness), "contrast": float(contrast), "sharpness": float(sharpness)}
                    )

            # Process embeddings if needed
            if add_embeddings:
                batch_images = batch_tensors.to(device)  # Use transformed tensors for model

                with torch.no_grad():
                    if not use_pretrained:
                        # Get predictions only for custom trained models
                        output = model(batch_images)
                        probabilities = torch.softmax(output, dim=1)
                        predicted_contiguous_labels = torch.argmax(output, dim=1)
                        confidences = torch.max(probabilities, dim=1)[0]

                        # Map contiguous labels back to original label space (if we have labels)
                        if not instance_config.allow_label_free or label_map:
                            predicted_original_labels = [
                                int(contiguous_2_label[idx.item()]) for idx in predicted_contiguous_labels
                            ]
                            labels.extend(predicted_original_labels)
                            confidences_list.extend(confidences.cpu().numpy())
                        else:
                            # Label-free mode - store predictions as-is
                            labels.extend(predicted_contiguous_labels.cpu().numpy().tolist())
                            confidences_list.extend(confidences.cpu().numpy())

                    # Get embeddings (only operation needed for pretrained models)
                    batch_embeddings = model.forward_features(batch_images).cpu().numpy().astype(np.float32)

                # Reduce dimensions if specified
                if reduce_last_dims > 0:
                    axes_to_reduce = tuple(range(-reduce_last_dims, 0))
                    batch_embeddings = batch_embeddings.mean(axis=axes_to_reduce)

                # Flatten if needed
                if len(batch_embeddings.shape) > 2:
                    batch_embeddings = batch_embeddings.reshape(len(batch_embeddings), -1)

                current_chunk.extend(batch_embeddings)

                # Save chunk if it reaches CHUNK_SIZE
                if len(current_chunk) >= CHUNK_SIZE:
                    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_count}.npy")
                    np.save(chunk_path, np.array(current_chunk[:CHUNK_SIZE]))
                    current_chunk = current_chunk[CHUNK_SIZE:]
                    chunk_count += 1

        # Save final chunk if any
        if current_chunk:
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_count}.npy")
            np.save(chunk_path, np.array(current_chunk))
            chunk_count += 1

        total_embeddings = total_instances
        embedding_dim = batch_embeddings.shape[1] if len(batch_embeddings) > 0 else 1280

        # Calculate memory requirements and sampling
        bytes_per_embedding = embedding_dim * 4  # float32 = 4 bytes
        max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        max_samples = int(max_memory_bytes / bytes_per_embedding)

        # Only sample and create reduced embeddings if we need to fit a new reducer
        if pacmap_reducer is None:
            assert fit_embeddings is None
            if total_embeddings > max_samples:
                print(f"Reducing embeddings from {total_embeddings} to {max_samples} samples")
                indices = np.random.choice(total_embeddings, size=max_samples, replace=False)
                indices.sort()  # Sort for more efficient reading

                # Read selected indices and create reduced embeddings
                embeddings_reduced = np.empty((max_samples, embedding_dim), dtype=np.float32)
                current_pos = 0

                print("Reading selected indices...")
                for chunk_idx in tqdm(range(chunk_count)):
                    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx}.npy")
                    chunk_data = np.load(chunk_path)
                    chunk_start = chunk_idx * CHUNK_SIZE

                    # Find which indices fall into this chunk
                    chunk_indices = [
                        i - chunk_start for i in indices if chunk_start <= i < chunk_start + len(chunk_data)
                    ]

                    if chunk_indices:
                        embeddings_reduced[current_pos : current_pos + len(chunk_indices)] = chunk_data[chunk_indices]
                        current_pos += len(chunk_indices)
            else:
                # When no sampling is needed, just read all chunks sequentially
                embeddings_reduced = np.empty((total_embeddings, embedding_dim), dtype=np.float32)
                current_pos = 0

                print("Reading all embeddings...")
                for chunk_idx in tqdm(range(chunk_count)):
                    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx}.npy")
                    chunk_data = np.load(chunk_path)
                    chunk_size = len(chunk_data)
                    embeddings_reduced[current_pos : current_pos + chunk_size] = chunk_data
                    current_pos += chunk_size
                max_samples = total_embeddings

            # Create and fit new PaCMAP reducer
            print(f"Applying PaCMAP with {num_components} components and {n_neighbors} neighbors")
            pacmap_reducer = pacmap.PaCMAP(n_components=num_components, n_neighbors=n_neighbors)
            print("Fitting PaCMAP on reduced embeddings")
            pacmap_reducer.fit_transform(embeddings_reduced)
            fit_embeddings = embeddings_reduced

        # Pre-allocate final embeddings array
        embeddings_nd = np.empty((total_embeddings, num_components), dtype=np.float32)

        # Process all embeddings in chunks
        print("Transforming all embeddings...")
        current_pos = 0
        for chunk_idx in tqdm(range(chunk_count)):
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx}.npy")
            chunk_data = np.load(chunk_path)
            chunk_embeddings = pacmap_reducer.transform(chunk_data, fit_embeddings)
            chunk_size = len(chunk_data)
            embeddings_nd[current_pos : current_pos + chunk_size] = chunk_embeddings
            current_pos += chunk_size

        # Clean up temporary files
        print("Cleaning up temporary files...")
        for chunk_idx in range(chunk_count):
            os.unlink(os.path.join(chunk_dir, f"chunk_{chunk_idx}.npy"))
        os.rmdir(chunk_dir)

        print("Done with embeddings")

    if add_image_metrics and not add_embeddings:
        print("Done with image metrics")

    # Create schema for new table
    new_table_schema = deepcopy(input_table.rows_schema)

    # Get the target schema for instance properties based on instance config
    instance_properties_schema = new_table_schema.values[instance_column].values[instance_properties_column]

    if add_embeddings:
        add_embedding_schemas_to_instance_properties(
            instance_properties_schema=instance_properties_schema,
            instance_type=instance_type,
            num_components=num_components,
            use_pretrained=use_pretrained,
            label_column_path=label_column_path,
            allow_label_free=instance_config.allow_label_free,
            background_label=background_label,
            new_table_schema=new_table_schema,
        )

    # Add image metrics schema if needed
    if add_image_metrics:
        add_metrics_schemas_to_instance_properties(
            instance_properties_schema=instance_properties_schema,
            instance_type=instance_type,
        )

    # Create TableWriter
    table_writer = tlc.TableWriter(
        root_url=input_table.root,
        project_name=input_table.project_name,
        dataset_name=input_table.dataset_name,
        table_name=output_table_name,
        description="Extended table with per instance embeddings and/or image metrics",
        column_schemas=new_table_schema.values,
        input_tables=[input_table.url],
    )

    # Get the hidden columns in the table (columns which are not part of the sample view of the table, e.g. "weight")
    hidden_column_names = [child.name for child in input_table.row_schema.sample_type_object.hidden_children]
    hidden_columns = {key: [row[key] for row in input_table.table_rows] for key in hidden_column_names}

    print(f"Processing with: embeddings={add_embeddings}, image_metrics={add_image_metrics}")

    # Get mapping from dataset index to (row_index, instance_index) using dataset's own logic
    dataset_index_to_row_instance = dataset.get_row_instance_mapping()

    # Create reverse lookup: (row_index, instance_index) -> dataset_index for O(1) lookups
    row_instance_to_dataset_index = {
        (row_idx, instance_idx): dataset_idx
        for dataset_idx, (row_idx, instance_idx) in enumerate(dataset_index_to_row_instance)
    }

    # Process each row and map embeddings back
    embedding_idx = 0
    for row_index, row in enumerate(tqdm(input_table.table_rows, desc="Processing rows")):
        new_row = row.copy()

        # Initialize embedding/label/confidence/metrics lists for this row's instances
        if instance_type == "segmentations":
            # For segmentations, we need to initialize lists in instance_properties
            initialize_segmentation_lists(
                new_row, instance_column, instance_properties_column, add_embeddings, add_image_metrics, use_pretrained
            )
            # Note: For bounding boxes, we add directly to each bb_list item (no initialization needed)

        # Get instances for this row
        instances = get_instances_for_row(new_row, instance_column, instance_type, instance_properties_column)

        # Process each instance in this row
        process_instances_for_row(
            row_index,
            new_row,
            instance_type,
            instance_column,
            instance_properties_column,
            row_instance_to_dataset_index,
            add_embeddings,
            add_image_metrics,
            embeddings_nd,
            labels,
            confidences_list,
            image_metrics_list,
            use_pretrained,
        )

        # Add the hidden columns to the new row
        for key in hidden_column_names:
            new_row[key] = hidden_columns[key][row_index]

        if instance_type == "segmentations":
            seg_sample_type = new_table_schema[instance_column].sample_type_object
            new_row[instance_column] = seg_sample_type.sample_from_row(new_row[instance_column])

        table_writer.add_row(new_row)

    # Finalize table
    table = table_writer.finalize()
    return table.url, pacmap_reducer, fit_embeddings


# ========================
# INSTANCE DATA ASSIGNMENT HELPERS
# ========================


def initialize_segmentation_lists(
    new_row: dict,
    instance_column: str,
    instance_properties_column: str,
    add_embeddings: bool,
    add_image_metrics: bool,
    use_pretrained: bool,
) -> None:
    """Initialize empty lists for segmentation instance properties."""
    if add_embeddings:
        if CLASSIFIER_EMBEDDING not in new_row[instance_column][instance_properties_column]:
            new_row[instance_column][instance_properties_column][CLASSIFIER_EMBEDDING] = []

        # Only initialize label and confidence lists for non-pretrained models
        if not use_pretrained:
            if CLASSIFIER_LABEL not in new_row[instance_column][instance_properties_column]:
                new_row[instance_column][instance_properties_column][CLASSIFIER_LABEL] = []
            if CLASSIFIER_CONFIDENCE not in new_row[instance_column][instance_properties_column]:
                new_row[instance_column][instance_properties_column][CLASSIFIER_CONFIDENCE] = []

    if add_image_metrics:
        for metric_name in ["brightness", "contrast", "sharpness"]:
            if metric_name not in new_row[instance_column][instance_properties_column]:
                new_row[instance_column][instance_properties_column][metric_name] = []


def assign_data_to_bbox_instance(
    instance: dict,
    embedding_data: dict[str, Any] | None = None,
    metrics_data: dict[str, float] | None = None,
) -> None:
    """Assign embeddings and metrics to a single bounding box instance."""
    if embedding_data:
        if "embedding" in embedding_data:
            instance[CLASSIFIER_EMBEDDING] = embedding_data["embedding"]
        if "label" in embedding_data:
            instance[CLASSIFIER_LABEL] = embedding_data["label"]
        if "confidence" in embedding_data:
            instance[CLASSIFIER_CONFIDENCE] = embedding_data["confidence"]

    if metrics_data:
        for metric_name, value in metrics_data.items():
            instance[metric_name] = value


def assign_data_to_segmentation_lists(
    new_row: dict,
    instance_column: str,
    instance_properties_column: str,
    embedding_data: dict[str, Any] | None = None,
    metrics_data: dict[str, float] | None = None,
) -> None:
    """Append embeddings and metrics to segmentation instance property lists."""
    instance_props = new_row[instance_column][instance_properties_column]

    if embedding_data:
        if "embedding" in embedding_data:
            instance_props[CLASSIFIER_EMBEDDING].append(embedding_data["embedding"])
        if "label" in embedding_data:
            instance_props[CLASSIFIER_LABEL].append(embedding_data["label"])
        if "confidence" in embedding_data:
            instance_props[CLASSIFIER_CONFIDENCE].append(embedding_data["confidence"])

    if metrics_data:
        for metric_name, value in metrics_data.items():
            instance_props[metric_name].append(value)


def create_embedding_data(
    embedding_idx: int,
    embeddings_nd: np.ndarray | None,
    labels: list[int] | None,
    confidences_list: list[float] | None,
    use_pretrained: bool,
) -> dict[str, Any] | None:
    """Create embedding data dict for assignment."""
    if embeddings_nd is None:
        return None

    data = {"embedding": embeddings_nd[embedding_idx].tolist()}

    if not use_pretrained and labels and confidences_list:
        data["label"] = int(labels[embedding_idx])
        data["confidence"] = float(confidences_list[embedding_idx])

    return data


def create_metrics_data(
    embedding_idx: int,
    image_metrics_list: list[dict[str, float]],
) -> dict[str, float] | None:
    """Create metrics data dict for assignment."""
    if not image_metrics_list or embedding_idx >= len(image_metrics_list):
        return None
    return image_metrics_list[embedding_idx]


def get_instances_for_row(
    new_row: dict,
    instance_column: str,
    instance_type: Literal["bounding_boxes", "segmentations"],
    instance_properties_column: str,
) -> list[Any]:
    """Get the list of instances for a row based on instance type."""
    if instance_type == "bounding_boxes":
        return cast(list[Any], new_row[instance_column][instance_properties_column])
    elif instance_type == "segmentations":
        return cast(list[Any], new_row[instance_column]["rles"])
    else:
        raise ValueError(f"Invalid instance type: {instance_type}")


def process_instances_for_row(
    row_index: int,
    new_row: dict,
    instance_type: Literal["bounding_boxes", "segmentations"],
    instance_column: str,
    instance_properties_column: str,
    row_instance_to_dataset_index: dict[tuple[int, int], int],
    add_embeddings: bool,
    add_image_metrics: bool,
    embeddings_nd: np.ndarray | None = None,
    labels: list[int] | None = None,
    confidences_list: list[float] | None = None,
    image_metrics_list: list[dict[str, float]] | None = None,
    use_pretrained: bool = False,
) -> None:
    """Process all instances for a single row and assign data."""
    instances = get_instances_for_row(new_row, instance_column, instance_type, instance_properties_column)

    # Process each instance in this row
    for instance_index in range(len(instances)):
        # Fast O(1) lookup to find the corresponding dataset index
        dataset_idx = row_instance_to_dataset_index.get((row_index, instance_index))

        if dataset_idx is None:
            # This instance wasn't processed by the dataset (shouldn't happen in normal cases)
            continue

        # Create data objects
        embedding_data = None
        metrics_data = None

        if add_embeddings:
            embedding_data = create_embedding_data(dataset_idx, embeddings_nd, labels, confidences_list, use_pretrained)

        if add_image_metrics:
            metrics_data = create_metrics_data(dataset_idx, image_metrics_list or [])

        # Assign data based on instance type
        if instance_type == "bounding_boxes":
            assign_data_to_bbox_instance(instances[instance_index], embedding_data, metrics_data)
        elif instance_type == "segmentations":
            assign_data_to_segmentation_lists(
                new_row, instance_column, instance_properties_column, embedding_data, metrics_data
            )
