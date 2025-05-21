from __future__ import annotations

import os

import numpy as np
import tlc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler

from tlc_tools.common import infer_torch_device

from .bb_crop_dataset import BBCropDataset
from .label_utils import create_label_mappings, get_label_name


def convert_to_rgb(img):
    """Convert image to RGB."""
    return img.convert("RGB")


def train_model(
    train_table_url: str,
    val_table_url: str,
    model_name: str = "efficientnet_b0",
    model_checkpoint: str = "./bb_classifier.pth",
    epochs: int = 20,
    batch_size: int = 32,
    include_background: bool = False,
    x_max_offset: float = 0.03,
    y_max_offset: float = 0.03,
    x_scale_range: tuple[float, float] = (0.95, 1.05),
    y_scale_range: tuple[float, float] = (0.95, 1.05),
    num_workers: int = 8,
    label_column_path: str = "bbs.bb_list.label",
) -> tuple[nn.Module, str]:
    """Train a model on bounding box crops from the given tables.

    :param train_table_url: URL of the table to train on.
    :param val_table_url: URL of the table to validate on.
    :param model_name: Name of the model to train.
    :param model_checkpoint: Path to the model checkpoint to load.
    :param epochs: Number of epochs to train.
    :param batch_size: Batch size for training.
    :param include_background: Whether to include the background class in the training.
    :param x_max_offset: Maximum offset in the x direction.
    :param y_max_offset: Maximum offset in the y direction.
    :param x_scale_range: Range of x scale factors.
    :param y_scale_range: Range of y scale factors.
    :param num_workers: Number of workers for data loading.
    :param label_column_path: Path to the label column in the table.
    """

    device = infer_torch_device()
    print(f"Using device: {device}")

    # Load tables
    train_table = tlc.Table.from_url(train_table_url)
    val_table = tlc.Table.from_url(val_table_url)

    # Get schema and number of classes
    label_map = train_table.get_simple_value_map(label_column_path)
    if not label_map:
        raise ValueError(f"Label map not found in table at path: {label_column_path}")
    print(f"Label map: {label_map}")

    # Create label mappings for training and validation
    label_2_contiguous_idx, contiguous_2_label, background_label, add_background = create_label_mappings(
        label_map, include_background=include_background
    )
    num_classes = len(label_2_contiguous_idx)
    background_freq = 1 / num_classes if add_background else 0

    print(f"Training with {num_classes} classes")
    print(f"Label to contiguous mapping: {label_2_contiguous_idx}")
    print(f"Contiguous to label mapping: {contiguous_2_label}")
    print(f"Using background: {add_background} (background_label={background_label})")

    # Setup transforms and datasets
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Lambda(convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Lambda(convert_to_rgb),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = BBCropDataset(
        train_table,
        transform=train_transforms,
        add_background=add_background,
        background_freq=background_freq,
        x_max_offset=x_max_offset,
        y_max_offset=y_max_offset,
        x_scale_range=x_scale_range,
        y_scale_range=y_scale_range,
    )

    val_dataset = BBCropDataset(
        val_table,
        transform=val_transforms,
        add_background=False,
    )

    # Calculate class frequencies across all bounding boxes
    class_counts: dict[int, int] = {}
    total_bbs = 0  # Add counter for total bounding boxes
    for row in train_table.table_rows:
        for bb in row["bbs"]["bb_list"]:
            label = bb["label"]
            class_counts[label] = class_counts.get(label, 0) + 1
            total_bbs += 1

    # Find the count of the most frequent class
    max_count = max(class_counts.values())

    # Calculate weights as (most_frequent_count / class_count)
    class_weights = {label: (max_count / count) for label, count in class_counts.items()}

    # Pre-allocate numpy array for weights
    bb_weights = np.zeros(total_bbs, dtype=np.float32)
    idx = 0
    print(f"Training on {len(train_table)} images with {total_bbs} bounding boxes")

    # Fill weights array
    for row in train_table.table_rows:
        for bb in row["bbs"]["bb_list"]:
            bb_weights[idx] = class_weights[bb["label"]]
            idx += 1

    # take the sqrt of the weights with numpy, found this to be better than just using the weights
    bb_weights = np.sqrt(bb_weights)

    # Print Number of weights
    print(f"Number of weights: {len(bb_weights)}")

    sampler = WeightedRandomSampler(weights=bb_weights, num_samples=min(len(bb_weights), len(train_table)))  # type: ignore[arg-type]

    # print all unique weights
    print(f"Unique weights: {list(set(bb_weights))}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    import timm

    # Create model and training components
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    best_mean_val_acc = 0.0

    # Create run and samplers
    checkpoint_base = os.path.basename(model_checkpoint)
    run = tlc.init(
        project_name=train_table.project_name,
        run_name="Train Bounding Box Classifier",
        description=f"Training BB Embeddings Model - Checkpoint: {checkpoint_base} ",
    )
    # Training loop
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_class_correct = torch.zeros(num_classes, device=device)
        train_class_total = torch.zeros(num_classes, device=device)

        pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Train]")
        for inputs, labels in pbar:
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

            # Update per-class accuracies
            for label in range(num_classes):
                mask = labels == label
                if mask.sum().item() > 0:  # Only update if we have samples for this class
                    train_class_correct[label] += (preds[mask] == labels[mask]).sum()
                    train_class_total[label] += mask.sum()

            pbar.set_postfix({"loss": f"{train_loss / train_total:.2f}", "acc": f"{train_correct / train_total:.2f}"})

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Calculate mean class accuracy for training
        train_class_acc = torch.zeros(num_classes, device=device)
        valid_classes = train_class_total > 0
        train_class_acc[valid_classes] = train_class_correct[valid_classes] / train_class_total[valid_classes]
        train_mean_class_acc = train_class_acc[valid_classes].mean().item()

        # Validation Phase
        isValRun = False
        if epoch % 1 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            val_class_correct = torch.zeros(num_classes, device=device)
            val_class_total = torch.zeros(num_classes, device=device)

            with torch.no_grad():
                pbar = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch + 1} [Val]")
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = outputs.max(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

                    # Update per-class accuracies
                    for label in range(num_classes):
                        mask = labels == label
                        if mask.sum().item() > 0:
                            val_class_correct[label] += (preds[mask] == labels[mask]).sum()
                            val_class_total[label] += mask.sum()

                    pbar.set_postfix({"loss": f"{val_loss / val_total:.4f}", "acc": f"{val_correct / val_total:.4f}"})

            val_loss /= val_total
            val_acc = val_correct / val_total

            # Calculate mean class accuracy for validation
            val_class_acc = torch.zeros(num_classes, device=device)
            valid_classes = val_class_total > 0
            val_class_acc[valid_classes] = val_class_correct[valid_classes] / val_class_total[valid_classes]

            # Only consider accuracies that are > 0 and
            valid_acc_mask = val_class_acc > 0
            val_mean_class_acc = val_class_acc[valid_acc_mask].mean().item() if valid_acc_mask.any() else 0
            val_min_class_acc = val_class_acc[valid_acc_mask].min().item() if valid_acc_mask.any() else 0
            val_max_class_acc = val_class_acc[valid_acc_mask].max().item() if valid_acc_mask.any() else 0

            # Find class names for min and max accuracies
            min_class_idx = int(val_class_acc[valid_acc_mask].argmin().item()) if valid_acc_mask.any() else -1
            max_class_idx = int(val_class_acc[valid_acc_mask].argmax().item()) if valid_acc_mask.any() else -1

            # Get actual indices (not mask indices)
            min_class_actual_idx = torch.nonzero(valid_acc_mask)[min_class_idx].item() if min_class_idx != -1 else -1
            max_class_actual_idx = torch.nonzero(valid_acc_mask)[max_class_idx].item() if max_class_idx != -1 else -1

            # Get class names from label map
            min_class_name = (
                get_label_name(contiguous_2_label[int(min_class_actual_idx)], label_map, background_label)
                if min_class_idx != -1
                else "N/A"
            )
            max_class_name = (
                get_label_name(contiguous_2_label[int(max_class_actual_idx)], label_map, background_label)
                if max_class_idx != -1
                else "N/A"
            )

            isValRun = True

            if val_mean_class_acc > best_mean_val_acc:
                best_mean_val_acc = val_mean_class_acc
                print(f"  New best validation mean class accuracy: {val_mean_class_acc:.4f}")
                # Save model checkpoint with original filename
                best_checkpoint_path = model_checkpoint
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"  Saved model checkpoint to {best_checkpoint_path}")

        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Train Mean Class Acc: {train_mean_class_acc:.4f}"
        )

        if isValRun:
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Mean Class Acc: {val_mean_class_acc:.4f}")
            print(
                f"  Min Class: {min_class_name} ({val_min_class_acc:.4f}), "
                f"Max Class: {max_class_name} ({val_max_class_acc:.4f})"
            )
            # Log per-class accuracies with label names
            for i in range(num_classes):
                if val_class_total[i] > 0:
                    # Get original label from contiguous index
                    original_label = contiguous_2_label[i]
                    # Get label name from the label map or use "background" for background class
                    label_name = get_label_name(original_label, label_map, background_label)
                    # Log with just the label name
                    tlc.log({f"val_{label_name}_acc": val_class_acc[i].item()})

            tlc.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_mean_class_acc": train_mean_class_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_mean_class_acc": val_mean_class_acc,
                    "val_min_class_acc": val_min_class_acc,
                    "val_max_class_acc": val_max_class_acc,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
        else:
            tlc.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_mean_class_acc": train_mean_class_acc,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

    run.set_status_completed()
    return model, best_checkpoint_path
