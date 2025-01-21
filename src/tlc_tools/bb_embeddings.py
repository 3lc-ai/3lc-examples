"""Functions for adding embeddings to bounding boxes in TLC tables."""

import os
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


def convert_to_rgb(img):
    """Convert image to RGB format."""
    return img.convert("RGB")


def fine_tune_bb_classifier(
    table: tlc.Table,
    *,
    model_name: str = "efficientnet_b0",
    checkpoint_dir: Optional[Union[str, Path]] = None,
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
) -> Path:
    """Fine-tune a classifier on bounding box crops from the table.

    Args:
        table: Input TLC table containing images and bounding boxes
        model_name: Name of timm model to use as backbone
        checkpoint_dir: Directory to save model checkpoint
        epochs: Number of epochs to train
        batch_size: Training batch size
        train_val_split: Fraction of data to use for training
        images_per_epoch: Number of images to sample per epoch
        include_background: Whether to include background class
        x_max_offset: Maximum horizontal jitter during training
        y_max_offset: Maximum vertical jitter during training
        x_scale_range: Range of horizontal scaling during training
        y_scale_range: Range of vertical scaling during training
        learning_rate: Initial learning rate
        lr_decay: Learning rate decay factor per epoch
        num_workers: Number of worker processes for data loading

    Returns:
        Path to saved model checkpoint
    """
    device = infer_torch_device()

    # Split table into train/val
    splits = split_table(table, {"train": train_val_split, "val": 1 - train_val_split})
    train_table = splits["train"]
    val_table = splits["val"]

    # Setup transforms
    common_transforms = transforms.Compose(
        [
            transforms.Lambda(convert_to_rgb),
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

    # Training loop
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]"):
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

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update learning rate
        scheduler.step()

        # Log metrics
        if False:
            tlc.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

    # Save model
    checkpoint_dir = Path(checkpoint_dir or ".")
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_bb_classifier.pth"
    torch.save(model.state_dict(), checkpoint_path)

    return checkpoint_path


def collect_bb_embeddings(
    table: tlc.Table,
    model_checkpoint: Union[str, Path],
    *,
    model_name: str = "efficientnet_b0",
    batch_size: int = 32,
    save_dir: Optional[Union[str, Path]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect embeddings and predicted labels for all bounding boxes.

    Args:
        table: Input table containing images and bounding boxes
        model_checkpoint: Path to trained model checkpoint
        model_name: Name of timm model used for training
        batch_size: Batch size for inference
        save_dir: Optional directory to save embeddings and labels

    Returns:
        tuple of (embeddings, labels) arrays
    """
    device = infer_torch_device()

    # Create model and load checkpoint
    num_classes = len(table.get_simple_value_map("bbs.bb_list.label"))
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_checkpoint))
    model = model.to(device)
    model.eval()

    # Setup image transform
    image_transform = transforms.Compose(
        [
            transforms.Lambda(convert_to_rgb),
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

    return embeddings, labels


def reduce_embeddings(
    embeddings: np.ndarray,
    *,
    n_components: int = 3,
    save_dir: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Reduce dimensionality of embeddings using PaCMAP.

    Returns reduced embeddings array.
    """
    # Initialize PaCMAP with desired number of components
    reducer = pacmap.PaCMAP(n_components=n_components, random_state=42)

    # Fit and transform the embeddings
    reduced_embeddings = reducer.fit_transform(embeddings)

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
) -> tlc.Table:
    """Create new table with embeddings added to bounding boxes.

    Args:
        table: Input table containing bounding boxes
        embeddings: Array of embeddings for each bounding box
        labels: Array of predicted labels for each bounding box
        embedding_column_name: Name of column to store embeddings
        predicted_label_column_name: Name of column to store predicted labels
        description: Description for the new table

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
    return table_writer.finalize()


def add_bb_embeddings(
    table: tlc.Table,
    *,
    # Model parameters
    model_name: str = "efficientnet_b0",
    checkpoint_dir: Optional[Union[str, Path]] = None,
    # Training parameters
    epochs: int = 10,
    batch_size: int = 32,
    train_val_split: float = 0.8,
    # Output parameters
    embedding_dim: int = 3,
    embedding_column_name: str = "embedding",
    predicted_label_column_name: str = "predicted_label",
    description: Optional[str] = None,
) -> tlc.Table:
    """Fine-tune a classifier on bounding box crops and add the resulting embeddings
    to the input table.

    This function:
    1. Fine-tunes a classifier on crops of the bounding boxes
    2. Uses the fine-tuned model to generate embeddings for each box
    3. Reduces embedding dimensionality using PaCMAP
    4. Creates a new table with embeddings added as a column

    Args:
        table: Input TLC table containing images and bounding boxes
        model_name: Name of timm model to use as backbone
        checkpoint_dir: Directory to save intermediate artifacts (model checkpoints etc)
        epochs: Number of epochs to train classifier
        batch_size: Batch size for training and inference
        train_val_split: Fraction of data to use for training
        embedding_dim: Number of dimensions to reduce embeddings to
        embedding_column_name: Name of column to store embeddings
        predicted_label_column_name: Name of column to store predicted labels
        description: Description for output table

    Returns:
        New TLC table with embeddings added to bounding boxes
    """
    # Use default checkpoint dir if none provided
    if checkpoint_dir is None:
        checkpoint_dir = Path(os.getenv("TRANSIENT_DATA_PATH", "./data"))
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Fine-tune classifier
    model_path = fine_tune_bb_classifier(
        table,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        train_val_split=train_val_split,
    )

    # Collect embeddings
    embeddings, labels = collect_bb_embeddings(
        table,
        model_path,
        model_name=model_name,
        batch_size=batch_size,
        save_dir=checkpoint_dir,
    )

    # Reduce dimensionality
    reduced_embeddings = reduce_embeddings(
        embeddings,
        n_components=embedding_dim,
        save_dir=checkpoint_dir,
    )

    # Create output table
    return add_embeddings_to_table(
        table,
        reduced_embeddings,
        labels,
        embedding_column_name=embedding_column_name,
        predicted_label_column_name=predicted_label_column_name,
        description=description,
    )
