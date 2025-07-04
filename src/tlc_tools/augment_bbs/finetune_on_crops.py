from __future__ import annotations

import logging
import os

import numpy as np
import timm
import tlc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler

from tlc_tools.augment_bbs.instance_config import InstanceConfig
from tlc_tools.common import infer_torch_device

from .instance_crop_dataset import InstanceCropDataset
from .label_utils import create_label_mappings, get_label_name

logger = logging.getLogger(__name__)


def convert_to_rgb(img):
    """Convert image to RGB."""
    return img.convert("RGB")


def train_model(
    train_table_url: str,
    val_table_url: str,
    model_name: str = "efficientnet_b0",
    model_checkpoint: str = "./models/instance_classifier.pth",
    epochs: int = 20,
    batch_size: int = 32,
    include_background: bool = False,
    x_max_offset: float = 0.03,
    y_max_offset: float = 0.03,
    x_scale_range: tuple[float, float] = (0.95, 1.05),
    y_scale_range: tuple[float, float] = (0.95, 1.05),
    num_workers: int = 4,
    instance_config: InstanceConfig | None = None,
) -> tuple[nn.Module, str]:
    """Train a model on instance crops from the given tables.

    :param train_table_url: URL of the table to train on.
    :param val_table_url: URL of the table to validate on.
    :param model_name: Name of the model to train.
    :param model_checkpoint: Path to the model checkpoint to save.
    :param epochs: Number of epochs to train.
    :param batch_size: Batch size for training.
    :param include_background: Whether to include the background class in the training.
    :param x_max_offset: Maximum offset in the x direction.
    :param y_max_offset: Maximum offset in the y direction.
    :param x_scale_range: Range of x scale factors.
    :param y_scale_range: Range of y scale factors.
    :param num_workers: Number of workers for data loading.
    :param instance_config: Instance configuration object with column/type/label info.
    """

    device = infer_torch_device()
    logger.info(f"Using device: {device}")

    # Load tables
    train_table = tlc.Table.from_url(train_table_url)
    val_table = tlc.Table.from_url(val_table_url)

    # Resolve instance configuration for training table
    if instance_config is None:
        instance_config = InstanceConfig.resolve(
            input_table=train_table,
            allow_label_free=False,  # Training always requires labels
        )
        instance_config._ensure_validated_for_table(val_table)

    # Training cannot work without labels
    if instance_config.label_column_path is None:
        raise ValueError("Training requires labels, but no label column was found")

    logger.info("Instance configuration for training:")
    logger.info(f"  Instance column: {instance_config.instance_column}")
    logger.info(f"  Instance type: {instance_config.instance_type}")
    logger.info(f"  Label column path: {instance_config.label_column_path}")

    # Get schema and number of classes
    label_map = train_table.get_simple_value_map(instance_config.label_column_path)
    if not label_map:
        raise ValueError(f"Label map not found in table at path: {instance_config.label_column_path}")
    logger.info(f"Label map: {label_map}")

    # Create label mappings for training and validation
    label_2_contiguous_idx, contiguous_2_label, background_label, add_background = create_label_mappings(
        label_map, include_background=include_background
    )
    num_classes = len(label_2_contiguous_idx)
    background_freq = 1 / num_classes if add_background else 0

    logger.info(f"Training with {num_classes} classes")
    logger.info(f"Label to contiguous mapping: {label_2_contiguous_idx}")
    logger.info(f"Contiguous to label mapping: {contiguous_2_label}")
    logger.info(f"Using background: {add_background} (background_label={background_label})")

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

    train_dataset = InstanceCropDataset(
        train_table,
        transform=train_transforms,
        instance_config=instance_config,
        add_background=add_background,
        background_freq=background_freq,
        x_max_offset=x_max_offset,
        y_max_offset=y_max_offset,
        x_scale_range=x_scale_range,
        y_scale_range=y_scale_range,
        include_background_in_labels=add_background,  # Both datasets use same label mapping
    )

    val_dataset = InstanceCropDataset(
        val_table,
        transform=val_transforms,
        instance_config=instance_config,  # Use the SAME instance_config as train
        add_background=False,  # No background sampling for validation
        include_background_in_labels=add_background,  # But same label mapping as train
    )

    # Calculate class frequencies across all instances
    total_instances, class_counts = count_instances(train_table, instance_config)

    # Find the count of the most frequent class
    max_count = max(class_counts.values())

    # Calculate weights as (most_frequent_count / class_count)
    class_weights = {label: (max_count / count) for label, count in class_counts.items()}

    # Pre-allocate numpy array for weights
    logger.info(f"Training on {len(train_table)} images with {total_instances} instances")
    instance_weights = compute_instance_weights(train_table, total_instances, class_weights, instance_config)

    # take the sqrt of the weights with numpy, found this to be better than just using the weights
    instance_weights = np.sqrt(instance_weights)

    # Print Number of weights
    logger.info(f"Number of weights: {len(instance_weights)}")

    sampler = WeightedRandomSampler(weights=instance_weights, num_samples=min(len(instance_weights), len(train_table)))  # type: ignore[arg-type]

    # print all unique weights
    logger.info(f"Unique weights: {list(set(instance_weights))}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=InstanceCropDataset.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=InstanceCropDataset.collate_fn,
    )

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
        run_name="Train Instance Classifier",
        description=f"Training Instance Classifier - Checkpoint: {checkpoint_base} ",
    )
    # Training loop
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_class_correct = torch.zeros(num_classes, device=device)
        train_class_total = torch.zeros(num_classes, device=device)

        pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Train]")
        for _, inputs, labels in pbar:
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
                for _, inputs, labels in pbar:
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
                logger.info(f"  New best validation mean class accuracy: {val_mean_class_acc:.4f}")
                # Save model checkpoint with original filename
                best_checkpoint_path = model_checkpoint
                torch.save(model.state_dict(), best_checkpoint_path)
                logger.info(f"  Saved model checkpoint to {best_checkpoint_path}")

        scheduler.step()

        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Train Mean Class Acc: {train_mean_class_acc:.4f}"
        )

        if isValRun:
            logger.info(
                f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Mean Class Acc: {val_mean_class_acc:.4f}"
            )
            logger.info(
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


def compute_instance_weights(train_table, total_instances, class_weights, instance_config: InstanceConfig):
    """Compute instance weights for training based on class distribution."""
    instance_weights = np.zeros(total_instances, dtype=np.float32)
    idx = 0

    # Fill weights array using unified instance processing
    for row in train_table.table_rows:
        # Get instances for this row using the same logic as the dataset
        if instance_config.instance_type == "bounding_boxes":
            instances = row[instance_config.instance_column][instance_config.instance_properties_column]
            for instance in instances:
                # Extract label from instance
                label = instance["label"]  # Labels are directly in bb instances
                instance_weights[idx] = class_weights[label]
                idx += 1
        elif instance_config.instance_type == "segmentations":
            # For segmentations, labels are in instance_properties lists
            instance_properties = row[instance_config.instance_column][instance_config.instance_properties_column]
            for label in instance_properties["label"]:
                instance_weights[idx] = class_weights[label]
                idx += 1
        else:
            raise ValueError(f"Unsupported instance type: {instance_config.instance_type}")

    return instance_weights


def count_instances(train_table, instance_config: InstanceConfig):
    """Count total instances and class distribution for weight calculation."""
    total_instances = 0
    class_counts: dict[int, int] = {}

    for row in train_table.table_rows:
        # Use unified instance processing
        if instance_config.instance_type == "bounding_boxes":
            instances = row[instance_config.instance_column][instance_config.instance_properties_column]
            for instance in instances:
                label = instance["label"]  # Labels are directly in bb instances
                class_counts[label] = class_counts.get(label, 0) + 1
                total_instances += 1
        elif instance_config.instance_type == "segmentations":
            # For segmentations, labels are in instance_properties lists
            instance_properties = row[instance_config.instance_column][instance_config.instance_properties_column]
            for label in instance_properties["label"]:
                class_counts[label] = class_counts.get(label, 0) + 1
                total_instances += 1
        else:
            raise ValueError(f"Unsupported instance type: {instance_config.instance_type}")

    return total_instances, class_counts
