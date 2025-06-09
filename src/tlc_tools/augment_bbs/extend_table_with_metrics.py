from __future__ import annotations

import os
from copy import deepcopy
from typing import cast

import cv2
import numpy as np
import pacmap
import tlc
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageStat
from torch.utils.data import DataLoader
from tqdm import tqdm

from .instance_crop_dataset import InstanceCropDataset
from .label_utils import create_label_mappings
from tlc_tools.common import InstanceConfig


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
    label_column_path: str = "bbs.bb_list.label",  # Keep for backward compatibility
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
        from tlc_tools.common import resolve_instance_config

        # Use legacy label_column_path for backward compatibility
        instance_config = resolve_instance_config(
            input_table=input_table,
            label_column_path=label_column_path if label_column_path != "bbs.bb_list.label" else None,
            allow_label_free=False,  # Default to requiring labels for backward compatibility
        )
        print("Warning: Using legacy label_column_path parameter. Consider using instance_config parameter.")

    # Check if we have labels when needed for embeddings
    if add_embeddings and not instance_config.allow_label_free and instance_config.label_column_path is None:
        raise ValueError("Model checkpoint required for embeddings, and labels required for training model")

    if add_embeddings and model_checkpoint is None and not instance_config.allow_label_free:
        raise ValueError("Model checkpoint required for embeddings")

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

    # Create DataLoader with multi-worker support
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep deterministic order
        num_workers=num_workers,
        pin_memory=True if device and device.type == "cuda" else False,
    )

    # Collect embeddings if needed
    labels: list[int] = []
    confidences_list: list[float] = []
    if add_embeddings:
        # Load model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Device: ", device)

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

        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
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
        )
        peek_batch = next(iter(peek_dataloader))
        peek_images = peek_batch[0].to(device)

        with torch.no_grad():
            peek_embedding = model.forward_features(peek_images).cpu().numpy()
            print(f"Shape of embedding: {peek_embedding.shape[1:]}")

        # Process all batches with the main dataloader
        dataloader_iter = iter(dataloader)
        for batch_data in tqdm(dataloader_iter, desc="Running model inference", total=len(dataloader)):
            batch_images = batch_data[0].to(device)  # Images are first element

            with torch.no_grad():
                # Get predictions
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

                # Get embeddings
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

    # Create schema for new table
    new_table_schema = deepcopy(input_table.rows_schema)

    # Get the target schema for instance properties based on instance config
    if instance_config.instance_column == "bbs":
        instance_properties_schema = new_table_schema.values["bbs"].values["bb_list"]
    elif instance_config.instance_column == "segmentations":
        instance_properties_schema = new_table_schema.values["segmentations"].values["instance_properties"]
    else:
        # For other instance types, try to find the schema
        instance_properties_schema = new_table_schema.values[instance_config.instance_column]

    if add_embeddings:
        # Create schema for embedding
        embedding_schema = tlc.Schema(
            value=tlc.Float32Value(),
            size0=tlc.DimensionNumericValue(num_components, num_components),
            size1=tlc.DimensionNumericValue(0, 1000),
        )

        # Create label and confidence schemas
        if instance_config.label_column_path and not instance_config.allow_label_free:
            # Use existing label schema as template
            label_parts = instance_config.label_column_path.split(".")
            temp_schema = new_table_schema
            for part in label_parts:
                temp_schema = temp_schema.values[part]
            label_schema = deepcopy(temp_schema)

            if background_label is not None:
                assert hasattr(label_schema.value, "map") and label_schema.value.map is not None
                label_schema.value.map[background_label] = tlc.MapElement("background")
        else:
            # Label-free mode - create integer schema for predicted labels
            label_schema = tlc.Schema(value=tlc.Int32Value(), writable=False)

        label_schema.writable = False
        label_schema.size0 = tlc.DimensionNumericValue(0, 1000)

        confidence_schema = tlc.Schema(value=tlc.Float32Value(), writable=False)
        confidence_schema.size0 = tlc.DimensionNumericValue(0, 1000)

        # Add schemas to instance properties
        if "classif_Embedding" not in instance_properties_schema.values:
            instance_properties_schema.add_sub_schema("classif_Embedding", embedding_schema)
        if "classif_Label" not in instance_properties_schema.values:
            instance_properties_schema.add_sub_schema("classif_Label", label_schema)
        if "classif_Confidence" not in instance_properties_schema.values:
            instance_properties_schema.add_sub_schema("classif_Confidence", confidence_schema)

    # Add image metrics schema if needed
    if add_image_metrics:
        # Add new metrics schema if they don't exist
        if "brightness" not in instance_properties_schema.values:
            instance_properties_schema.add_sub_value("brightness", tlc.schema.Float32Value(), writable=False)
        if "contrast" not in instance_properties_schema.values:
            instance_properties_schema.add_sub_value("contrast", tlc.schema.Float32Value(), writable=False)
        if "sharpness" not in instance_properties_schema.values:
            instance_properties_schema.add_sub_value("sharpness", tlc.schema.Float32Value(), writable=False)

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

    # Build mapping from dataset index to (row_index, instance_index)
    # This allows us to map embeddings back to the correct instances
    dataset_index_to_row_instance = []
    for row_index, row in enumerate(input_table.table_rows):
        # Get instances for this row using the same logic as the dataset
        if instance_config.instance_column == "bbs":
            instances = row["bbs"]["bb_list"]
        elif instance_config.instance_column == "segmentations":
            instances = row["segmentations"]["rles"]  # or another identifier
        else:
            # For other instance types, get the instances from the configured column
            instance_data = row[instance_config.instance_column]
            instances = instance_data if isinstance(instance_data, list) else [instance_data]

        for instance_index in range(len(instances)):
            dataset_index_to_row_instance.append((row_index, instance_index))

    # Process each row and map embeddings back
    embedding_idx = 0
    for row_index, row in enumerate(tqdm(input_table.table_rows, desc="Processing rows")):
        new_row = row.copy()

        if add_image_metrics:
            image = Image.open(tlc.Url(row["image"]).to_absolute().to_str())

        # Initialize embedding/label/confidence lists for this row's instances
        if add_embeddings:
            if instance_config.instance_column == "bbs":
                # For BBs, add directly to bb_list items
                pass  # We'll handle this in the instance loop
            elif instance_config.instance_column == "segmentations":
                # Initialize lists for segmentation instance properties
                if "classif_Embedding" not in new_row["segmentations"]["instance_properties"]:
                    new_row["segmentations"]["instance_properties"]["classif_Embedding"] = []
                if "classif_Label" not in new_row["segmentations"]["instance_properties"]:
                    new_row["segmentations"]["instance_properties"]["classif_Label"] = []
                if "classif_Confidence" not in new_row["segmentations"]["instance_properties"]:
                    new_row["segmentations"]["instance_properties"]["classif_Confidence"] = []

        # Get instances for this row
        if instance_config.instance_column == "bbs":
            instances = new_row["bbs"]["bb_list"]
        elif instance_config.instance_column == "segmentations":
            instances = new_row["segmentations"]["rles"]
        else:
            # For other instance types
            instance_data = new_row[instance_config.instance_column]
            instances = instance_data if isinstance(instance_data, list) else [instance_data]

        # Process each instance in this row
        for instance_index in range(len(instances)):
            # Find the corresponding dataset index for this (row_index, instance_index)
            while embedding_idx < len(dataset_index_to_row_instance) and dataset_index_to_row_instance[
                embedding_idx
            ] != (row_index, instance_index):
                embedding_idx += 1

            if embedding_idx >= len(dataset_index_to_row_instance):
                break

            if add_embeddings:
                if instance_config.instance_column == "bbs":
                    # Add directly to the bb item
                    instances[instance_index]["classif_Embedding"] = embeddings_nd[embedding_idx].tolist()
                    instances[instance_index]["classif_Label"] = int(labels[embedding_idx])
                    instances[instance_index]["classif_Confidence"] = float(confidences_list[embedding_idx])
                elif instance_config.instance_column == "segmentations":
                    # Add to instance properties lists
                    new_row["segmentations"]["instance_properties"]["classif_Embedding"].append(
                        embeddings_nd[embedding_idx].tolist()
                    )
                    new_row["segmentations"]["instance_properties"]["classif_Label"].append(int(labels[embedding_idx]))
                    new_row["segmentations"]["instance_properties"]["classif_Confidence"].append(
                        float(confidences_list[embedding_idx])
                    )

                embedding_idx += 1

            # TODO: Add image metrics support for instances
            # if add_image_metrics:
            #     metrics = calculate_bb_metrics(image, instance, schema)
            #     # Add metrics to instance
        # Add the hidden columns to the new row
        for key in hidden_column_names:
            new_row[key] = hidden_columns[key][row_index]

        seg_sample_type = new_table_schema["segmentations"].sample_type_object
        new_row["segmentations"] = seg_sample_type.sample_from_row(new_row["segmentations"])
        table_writer.add_row(new_row)

    # Finalize table
    table_writer.finalize()
    return table_writer.url, pacmap_reducer, fit_embeddings
