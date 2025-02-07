import timm
import tlc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler

from tlc_tools.common import infer_torch_device
from tlc_tools.datasets import BBCropDataset


def train_model(
    train_table_url: str | tlc.Url,
    val_table_url: str | tlc.Url,
    model_name: str = "efficientnet_b0",
    model_checkpoint: str = "./bb_classifier.pth",
    epochs: int = 20,
    batch_size: int = 32,
    include_background: bool = False,
    x_max_offset: float = 0.03,
    y_max_offset: float = 0.03,
    x_scale_range: tuple[float, float] = (0.95, 1.05),
    y_scale_range: tuple[float, float] = (0.95, 1.05),
):
    """Train a model on bounding box crops from the given tables."""

    device = infer_torch_device()
    print(f"Using device: {device}")

    # Load tables
    train_table = tlc.Table.from_url(train_table_url)
    val_table = tlc.Table.from_url(val_table_url)

    # Get schema and number of classes
    label_map = train_table.get_simple_value_map("bbs.bb_list.label")
    max_label = max(int(x) for x in label_map)  # Look at the index keys, not the label values
    num_classes = max_label + 1 if not include_background else max_label + 2
    print(f"Training with {num_classes} classes")

    # Setup transforms and datasets
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

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
        transform=val_transforms,
        add_background=False,
        is_train=False,
    )

    # Calculate class frequencies across all bounding boxes
    class_counts: dict[float, int] = {}
    for row in train_table.table_rows:
        for bb in row["bbs"]["bb_list"]:
            label = bb["label"]
            class_counts[label] = class_counts.get(label, 0) + 1

    # Find the count of the most frequent class
    max_count = max(class_counts.values())

    # Calculate weights as sqrt of (most_frequent_count / class_count)
    class_weights = {label: (max_count / count) ** 0.5 for label, count in class_counts.items()}

    # For each image, use the weight of its rarest class
    num_bbs_per_image = []
    isFirst = True
    for row in train_table:
        if not row["bbs"]["bb_list"]:  # Handle empty images
            num_bbs_per_image.append(1.0)
            if isFirst:
                print(f"Warning: Empty image found at index {row.index}")
                isFirst = False
            continue
        if isFirst:
            print(f"Training on {len(train_table)} images")
            isFirst = False

        # Find the rarest class in this image
        rarest_weight = max(class_weights[bb["label"]] for bb in row["bbs"]["bb_list"])
        num_bbs_per_image.append(rarest_weight)

    sampler = WeightedRandomSampler(weights=num_bbs_per_image, num_samples=len(train_table))

    # print all unique weights
    print(f"Unique weights: {list(set(num_bbs_per_image))}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True
    )

    # Create model and training components
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    best_val_acc = 0.0

    # Create run and samplers
    run = tlc.init(
        project_name=train_table.project_name,
        run_name="Train Bounding Box Classifier",
        description="Training BB Embeddings Model",
    )
    # Training loop
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
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

            pbar.set_postfix({"loss": f"{train_loss / train_total:.2f}", "acc": f"{train_correct / train_total:.2f}"})

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation Phase
        isValRun = False
        if epoch % 1 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
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

                    pbar.set_postfix({"loss": f"{val_loss / val_total:.4f}", "acc": f"{val_correct / val_total:.4f}"})

            val_loss /= val_total
            val_acc = val_correct / val_total
            isValRun = True

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  New best validation accuracy: {val_acc:.4f}")
                torch.save(model.state_dict(), model_checkpoint)
                print(f"  Saved model checkpoint to {model_checkpoint}")

        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if isValRun:
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            tlc.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
        else:
            tlc.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

    run.set_status_completed()
    return model
