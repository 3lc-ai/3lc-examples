"""Functions for adding embeddings to bounding boxes in TLC tables."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pacmap
import timm
import tlc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

from tlc_tools.common import infer_torch_device
from tlc_tools.datasets import BBCropDataset
from tlc_tools.split import split_table


def _convert_to_rgb(img):
    """Convert image to RGB format."""
    return img.convert("RGB")


def fine_tune_bb_classifier(
    table: tlc.Table,
    *,
    model_name: str = "efficientnet_b0",
    save_path: Union[str, Path],
    epochs: int = 10,
    batch_size: int = 32,
    train_val_split: float = 0.8,
    images_per_epoch: int = 1000,
    include_background: bool = False,
    x_max_offset: float = 0.1,
    y_max_offset: float = 0.1,
    x_scale_range: tuple[float, float] = (0.9, 1.1),
    y_scale_range: tuple[float, float] = (0.9, 1.1),
    learning_rate: float = 1e-4,
    lr_decay: float = 0.9516,
    num_workers: int = 8,
    verbose: bool = True,
) -> Path:
    """Fine-tune a classifier on bounding box crops."""
    device = infer_torch_device()

    # Split table into train/val
    splits = split_table(table, {"train": train_val_split, "val": 1 - train_val_split})
    train_table = splits["train"]
    val_table = splits["val"]

    # Setup transforms
    common_transforms = transforms.Compose(
        [
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_transforms = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            *common_transforms.transforms,
        ]
    )

    # Create datasets
    train_dataset = BBCropDataset(
        train_table,
        transform=train_transforms,
        add_background=include_background,
        is_train=True,
        x_max_offset=x_max_offset,
        y_max_offset=y_max_offset,
        x_scale_range=x_scale_range,
        y_scale_range=y_scale_range,
    )

    val_dataset = BBCropDataset(
        val_table,
        transform=common_transforms,
        add_background=False,
        is_train=False,
    )

    # Setup sampling
    num_bbs_per_image = [len(row["bbs"]["bb_list"]) for row in train_table.table_rows]
    sampler = WeightedRandomSampler(weights=num_bbs_per_image, num_samples=images_per_epoch)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    # Create model
    num_classes = len(table.get_simple_value_map("bbs.bb_list.label"))
    if include_background:
        num_classes += 1

    # with torch._dynamo.disable():  # Add this context manager
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # Track metrics for progress reporting
    prev_train_loss = None
    prev_train_acc = None
    prev_val_loss = None
    prev_val_acc = None

    # Training loop with epoch-level progress bar
    epoch_bar = tqdm.tqdm(range(epochs), desc="Fine-tuning classifier", disable=not verbose, leave=True)

    for epoch in epoch_bar:
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        # Training phase with hidden progress bar
        for inputs, labels in tqdm.tqdm(train_dataloader, desc="Training", leave=False, disable=not verbose):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(val_dataloader, desc="Validating", leave=False, disable=not verbose):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Calculate changes from previous epoch
        train_loss_delta = f" ({train_loss - prev_train_loss:+.4f})" if prev_train_loss is not None else ""
        train_acc_delta = f" ({(train_acc - prev_train_acc)*100:+.2f}%)" if prev_train_acc is not None else ""
        val_loss_delta = f" ({val_loss - prev_val_loss:+.4f})" if prev_val_loss is not None else ""
        val_acc_delta = f" ({(val_acc - prev_val_acc)*100:+.2f}%)" if prev_val_acc is not None else ""

        # Update previous values
        prev_train_loss = train_loss
        prev_train_acc = train_acc
        prev_val_loss = val_loss
        prev_val_acc = val_acc

        # Update progress bar with final epoch metrics
        if epoch == epochs - 1 and verbose:
            print(f"\n  Epoch {epoch+1}/{epochs}:")
            print(f"    Train: loss={train_loss:.4f}{train_loss_delta}, " f"acc={train_acc*100:.2f}%{train_acc_delta}")
            print(f"    Val:   loss={val_loss:.4f}{val_loss_delta}, " f"acc={val_acc*100:.2f}%{val_acc_delta}")

        # Update learning rate
        scheduler.step()

    # Save model
    torch.save(model.state_dict(), save_path)
    return save_path


def collect_bb_embeddings(
    table: tlc.Table,
    model_checkpoint: Union[str, Path],
    *,
    model_name: str = "efficientnet_b0",
    batch_size: int = 32,
    save_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect embeddings and predicted labels for all bounding boxes.

    Args:
        table: Input table containing images and bounding boxes
        model_checkpoint: Path to trained model checkpoint
        model_name: Name of timm model used for training
        batch_size: Batch size for inference
        save_dir: Optional directory to save embeddings and labels
        verbose: Whether to print progress messages

    Returns:
        tuple of (embeddings, labels) arrays
    """
    device = infer_torch_device()

    # Create model and load checkpoint
    num_classes = len(table.get_simple_value_map("bbs.bb_list.label"))
    model: torch.nn.Module = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True))
    model = model.to(device)
    model.eval()

    # Setup image transform
    image_transform = transforms.Compose(
        [
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def single_sample_bb_crop_iterator(sample):
        """Iterator over every transformed BB crop in a single sample."""
        image_filename = sample["image"]
        image_bytes = tlc.Url(image_filename).to_absolute(table.url).read()
        image = Image.open(BytesIO(image_bytes))
        w, h = image.size

        bb_schema = table.rows_schema.values["bbs"].values["bb_list"]
        for bb in sample["bbs"]["bb_list"]:
            bb_crop = tlc.BBCropInterface.crop(image, bb, bb_schema, h, w)
            yield image_transform(bb_crop)

    def bb_crop_iterator():
        """Iterator over every transformed BB crop in the dataset."""
        for sample in table:
            yield from single_sample_bb_crop_iterator(sample)

    def batched_bb_crop_iterator():
        """Batched iterator over every transformed BB crop in the dataset."""
        batch = []
        for bb_crop in bb_crop_iterator():
            batch.append(bb_crop)
            if len(batch) == batch_size:
                yield torch.stack(batch).to(device)
                batch = []
        if batch:
            yield torch.stack(batch).to(device)

    # Count total number of bounding boxes for progress bar
    total_bb_count = sum(len(row["bbs"]["bb_list"]) for row in table)

    # Collect embeddings and predictions
    all_labels: list[int] = []

    with torch.no_grad():
        # Add a model hook which saves activations
        all_embeddings = []

        def hook_fn(module, input, output):
            all_embeddings.append(output.cpu().numpy())

        hook_handle = model.global_pool.register_forward_hook(hook_fn)

        # Run inference on all crops
        for batch in tqdm.tqdm(
            batched_bb_crop_iterator(), desc="Collecting embeddings", total=total_bb_count // batch_size
        ):
            outputs = model(batch)
            predicted_labels = torch.argmax(outputs, dim=1)
            all_labels.extend(predicted_labels.cpu().numpy())

        hook_handle.remove()

    # Stack all embeddings and labels
    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)

    # Save if requested
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "embeddings.npy", embeddings)
        np.save(save_dir / "labels.npy", labels)

    if verbose:
        print(f"Collected embeddings shape: {embeddings.shape}")

    return embeddings, labels


def reduce_embeddings(
    embeddings: np.ndarray,
    *,
    n_components: int = 3,
    save_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Reduce dimensionality of embeddings using PaCMAP.

    Returns reduced embeddings array.
    """
    # Initialize PaCMAP with desired number of components
    reducer = pacmap.PaCMAP(n_components=n_components, random_state=42)

    # Fit and transform the embeddings
    reduced_embeddings = reducer.fit_transform(embeddings)
    assert isinstance(reduced_embeddings, np.ndarray)

    # Save if requested
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "reduced_embeddings.npy", reduced_embeddings)

    return reduced_embeddings


def add_embeddings_to_table(
    table: tlc.Table,
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    embedding_column_name: str = "embedding",
    predicted_label_column_name: str = "predicted_label",
    description: str = "Added embeddings from fine-tuned bb-classifier",
    verbose: bool = True,
) -> tlc.Table:
    """Create new table with embeddings added to bounding boxes.

    Args:
        table: Input table containing bounding boxes
        embeddings: Array of embeddings for each bounding box
        labels: Array of predicted labels for each bounding box
        embedding_column_name: Name of column to store embeddings
        predicted_label_column_name: Name of column to store predicted labels
        description: Description for the new table
        verbose: Whether to print progress messages

    Returns:
        New table with embeddings added to bounding boxes
    """
    # Create schema for embedding
    embedding_schema = tlc.Schema(
        value=tlc.Float32Value(),
        size0=tlc.DimensionNumericValue(embeddings.shape[1], embeddings.shape[1]),
    )

    # Create a schema for the new table
    new_table_schema = deepcopy(table.rows_schema)
    label_schema = deepcopy(new_table_schema.values["bbs"].values["bb_list"].values["label"])
    label_schema.writable = False
    new_table_schema.values["bbs"].values["bb_list"].add_sub_schema(embedding_column_name, embedding_schema)
    new_table_schema.values["bbs"].values["bb_list"].add_sub_schema(predicted_label_column_name, label_schema)

    # Create a TableWriter for the new table
    table_writer = tlc.TableWriter(
        project_name=table.project_name,
        dataset_name=table.dataset_name,
        table_name=f"{table.name}_with_embeddings",
        description=description,
        column_schemas=new_table_schema.values,
        input_tables=[table.url],
    )

    # Get the hidden columns in the table
    hidden_column_names = [child.name for child in table.row_schema.sample_type_object.hidden_children]
    hidden_columns = {key: [row[key] for row in table.table_rows] for key in hidden_column_names}

    # Iterate over the rows and add the embeddings
    embedding_idx = 0
    for row_index, row in enumerate(table):
        new_row = deepcopy(row)
        for bb in new_row["bbs"]["bb_list"]:
            bb[embedding_column_name] = embeddings[embedding_idx].tolist()
            bb[predicted_label_column_name] = int(labels[embedding_idx])
            embedding_idx += 1

        # Add the hidden columns to the new row
        for key in hidden_column_names:
            new_row[key] = hidden_columns[key][row_index]

        table_writer.add_row(new_row)

    # Create and return the new table
    output_table = table_writer.finalize()

    return output_table


def add_bb_embeddings(
    table: tlc.Table,
    *,
    # Model parameters
    model_name: str = "efficientnet_b0",
    save_path: str | Path = "./bb_embeddings",
    # Training parameters
    epochs: int = 10,
    batch_size: int = 32,
    train_val_split: float = 0.8,
    num_workers: int = 8,
    # Output parameters
    embedding_dim: int = 3,
    embedding_column_name: str = "embedding",
    predicted_label_column_name: str = "predicted_label",
    description: str | None = None,
    verbose: bool = True,
) -> tlc.Table:
    """Fine-tune a classifier on bounding box crops and add embeddings to table.

    This function:
    1. Fine-tunes a classifier on crops of the bounding boxes
    2. Uses the fine-tuned model to generate embeddings for each box
    3. Reduces embedding dimensionality using PaCMAP
    4. Creates a new table with embeddings added as a column

    Args:
        table: Input table containing images and bounding boxes
        model_name: Name of timm model to use
        save_path: Directory path where artifacts will be saved
    """
    # Ensure save_path is a directory
    save_path = Path(save_path)
    if save_path.exists() and not save_path.is_dir():
        raise ValueError(f"save_path must be a directory: {save_path}")

    # Create directory
    save_path.mkdir(parents=True, exist_ok=True)

    # Define artifact paths
    model_path = save_path / f"{model_name}_bb_classifier.pth"
    # embeddings_path = save_path / "embeddings.npy"
    # labels_path = save_path / "labels.npy"

    if verbose:
        print(f"Saving artifacts to: {save_path}")

    # Pass specific paths to component functions
    model_path = fine_tune_bb_classifier(
        table,
        model_name=model_name,
        save_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        train_val_split=train_val_split,
        num_workers=num_workers,
    )

    # Step 2: Collect embeddings
    if verbose:
        print("Step 2/4: Collecting embeddings")

    embeddings, labels = collect_bb_embeddings(
        table,
        model_path,
        model_name=model_name,
        batch_size=batch_size,
        save_dir=save_path,
        verbose=verbose,
    )

    # Step 3: Reduce dimensionality
    if verbose:
        print(f"Step 3/4: Reducing embeddings to {embedding_dim} dimensions")

    reduced_embeddings = reduce_embeddings(
        embeddings,
        n_components=embedding_dim,
        save_dir=save_path,
        verbose=verbose,
    )

    # Step 4: Create output table
    if verbose:
        print("Step 4/4: Creating output table")

    output_table = add_embeddings_to_table(
        table,
        reduced_embeddings,
        labels,
        embedding_column_name=embedding_column_name,
        predicted_label_column_name=predicted_label_column_name,
        description=description or f"Added embeddings from fine-tuned bb-classifier {model_name}",
        verbose=verbose,
    )

    if verbose:
        print(f"\nDone! Output table: {output_table.name}")
        print("=======================================")

    return output_table
