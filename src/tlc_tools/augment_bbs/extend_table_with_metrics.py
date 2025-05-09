from __future__ import annotations

import os
from copy import deepcopy
from io import BytesIO
from typing import cast

import cv2
import numpy as np
import pacmap
import timm
import tlc
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageStat
from tqdm import tqdm


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


def single_sample_bb_crop_iterator(sample, bb_schema, image_transform):
    """Process a single sample for bounding box crops."""
    image_filename = sample["image"]
    image_bytes = tlc.Url(image_filename).read()
    image = Image.open(BytesIO(image_bytes))
    w, h = image.size

    for bb in sample["bbs"]["bb_list"]:
        bb_crop = tlc.BBCropInterface.crop(image, bb, bb_schema, h, w)
        yield image_transform(bb_crop)


def bb_crop_iterator(input_table, bb_schema, image_transform):
    """Iterate over all bounding box crops in the table."""
    for sample in input_table:
        yield from single_sample_bb_crop_iterator(sample, bb_schema, image_transform)


def batched_bb_crop_iterator(input_table, bb_schema, image_transform, batch_size, device):
    """Create batches of bounding box crops."""
    batch = []
    for bb_crop in bb_crop_iterator(input_table, bb_schema, image_transform):
        batch.append(bb_crop)
        if len(batch) == batch_size:
            yield torch.stack(batch).to(device)
            batch = []
    if batch:
        yield torch.stack(batch).to(device)


def extend_table_with_metrics(
    input_table: tlc.Table,
    output_table_name: str,
    add_embeddings: bool = False,
    add_image_metrics: bool = False,
    model_checkpoint: str | None = None,
    model_name: str = "efficientnet_b0",
    batch_size: int = 32,
    num_components: int = 3,
    pacmap_reducer: pacmap.PaCMAP | None = None,
    fit_embeddings: np.ndarray | None = None,
    n_neighbors: int = 10,
    device: torch.device | None = None,
    reduce_last_dims: int = 0,
    max_memory_gb: int = 64,
):
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
    """
    if not (add_embeddings or add_image_metrics):
        raise ValueError("Must specify at least one type of metrics to add")

    if add_embeddings and model_checkpoint is None:
        raise ValueError("Model checkpoint required for embeddings")

    # Get total BB count for progress bar
    total_bb_count = sum(len(row["bbs"]["bb_list"]) for row in input_table)

    # Get BB schema for cropping
    bb_schema = input_table.rows_schema.values["bbs"].values["bb_list"]

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

        # Get label map and determine if background was used
        label_map = input_table.get_simple_value_map("bbs.bb_list.label")
        add_background = num_classes > len(label_map)
        background_label = max(label_map.keys()) + 1 if add_background else None

        # Create mapping from contiguous idx back to original labels
        label_2_contiguous_idx = {label: idx for idx, label in enumerate(label_map.keys())}
        if add_background:
            label_2_contiguous_idx[background_label] = len(label_2_contiguous_idx)
        contiguous_2_label = {idx: label for label, idx in label_2_contiguous_idx.items()}

        print(f"Label map: {label_map}")
        print(f"Using {num_classes} classes with background={add_background}")
        print(f"Label to contiguous mapping: {label_2_contiguous_idx}")
        print(f"Contiguous to label mapping: {contiguous_2_label}")

        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        # Setup image transformation
        image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Constants for chunking
        CHUNK_SIZE = 16384
        chunk_dir = os.path.join(os.path.expanduser("~"), ".tlc_temp_embeddings")
        os.makedirs(chunk_dir, exist_ok=True)

        # Process batches and save chunks
        current_chunk = []
        chunk_count = 0
        print("Processing batches and saving chunks...")

        # print shape of embedding only, removing batch dimension
        first_batch = next(batched_bb_crop_iterator(input_table, bb_schema, image_transform, batch_size, device))
        with torch.no_grad():
            first_embedding = model.forward_features(first_batch).cpu().numpy()
            print(f"Shape of embedding: {first_embedding.shape[1:]}")

        for batch in tqdm(
            batched_bb_crop_iterator(input_table, bb_schema, image_transform, batch_size, device),
            desc="Running model inference",
            total=total_bb_count // batch_size,
        ):
            with torch.no_grad():
                # Get predictions
                output = model(batch)
                probabilities = torch.softmax(output, dim=1)
                predicted_contiguous_labels = torch.argmax(output, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]

                # Map contiguous labels back to original label space
                predicted_original_labels = [contiguous_2_label[idx.item()] for idx in predicted_contiguous_labels]
                labels.extend(predicted_original_labels)
                confidences_list.extend(confidences.cpu().numpy())

                # Get embeddings
                batch_embeddings = model.forward_features(batch).cpu().numpy().astype(np.float32)

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

        total_embeddings = total_bb_count
        embedding_dim = batch_embeddings.shape[1]

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
    bb_list_schema = new_table_schema.values["bbs"].values["bb_list"]

    if add_embeddings:
        # Create schema for embedding
        embedding_schema = tlc.Schema(
            value=tlc.Float32Value(),
            size0=tlc.DimensionNumericValue(num_components, num_components),
        )

        # Create label and confidence schemas
        label_schema = deepcopy(bb_list_schema.values["label"])
        assert hasattr(label_schema.value, "map") and label_schema.value.map is not None
        label_schema.value.map[background_label] = tlc.MapElement("background")
        label_schema.writable = False
        confidence_schema = tlc.Schema(value=tlc.Float32Value(), writable=False)

        # Add schemas to bb_list
        if "classif_Embedding" not in bb_list_schema.values:
            bb_list_schema.add_sub_schema("classif_Embedding", embedding_schema)
        if "classif_Label" not in bb_list_schema.values:
            bb_list_schema.add_sub_schema("classif_Label", label_schema)
        if "classif_Confidence" not in bb_list_schema.values:
            bb_list_schema.add_sub_schema("classif_Confidence", confidence_schema)

    # Add image metrics schema if needed
    if add_image_metrics:
        # Add new metrics schema if they don't exist
        if "brightness" not in bb_list_schema.values:
            bb_list_schema.add_sub_value("brightness", tlc.schema.Float32Value(), writable=False)
        if "contrast" not in bb_list_schema.values:
            bb_list_schema.add_sub_value("contrast", tlc.schema.Float32Value(), writable=False)
        if "sharpness" not in bb_list_schema.values:
            bb_list_schema.add_sub_value("sharpness", tlc.schema.Float32Value(), writable=False)

    # Create TableWriter
    table_writer = tlc.TableWriter(
        project_name=input_table.project_name,
        dataset_name=input_table.dataset_name,
        table_name=output_table_name,
        description="Extended table with per Bounding Box embeddings and/or image metrics",
        column_schemas=new_table_schema.values,
        input_tables=[input_table.url],
    )

    # Get the hidden columns in the table (columns which are not part of the sample view of the table, e.g. "weight")
    hidden_column_names = [child.name for child in input_table.row_schema.sample_type_object.hidden_children]
    hidden_columns = {key: [row[key] for row in input_table.table_rows] for key in hidden_column_names}

    print(f"Processing with: embeddings={add_embeddings}, image_metrics={add_image_metrics}")
    # Process each row
    embedding_idx = 0
    for row_index, row in enumerate(tqdm(input_table, desc="Processing rows")):
        new_row = deepcopy(row)

        if add_image_metrics:
            image = Image.open(row["image"])

        for bb in new_row["bbs"]["bb_list"]:
            if add_embeddings:
                bb["classif_Embedding"] = embeddings_nd[embedding_idx].tolist()
                bb["classif_Label"] = int(labels[embedding_idx])
                bb["classif_Confidence"] = float(confidences_list[embedding_idx])
                embedding_idx += 1

            if add_image_metrics:
                metrics = calculate_bb_metrics(image, bb, bb_schema)
                bb["brightness"] = metrics["brightness"]
                bb["contrast"] = metrics["contrast"]
                bb["sharpness"] = metrics["sharpness"]

        # Add the hidden columns to the new row
        for key in hidden_column_names:
            new_row[key] = hidden_columns[key][row_index]

        table_writer.add_row(new_row)

    # Finalize table
    table_writer.finalize()
    return table_writer.url, pacmap_reducer, fit_embeddings
