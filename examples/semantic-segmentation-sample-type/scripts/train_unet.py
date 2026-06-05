# Copyright 2026 3LC Inc. All rights reserved.

"""Train a small UNet on the Oxford-IIIT Pets semseg tables and collect per-sample metrics.

The GT tables have three classes (background, pet, border), but border is an
ignore region, not a prediction target: the model outputs only background/pet,
border pixels are excluded from the loss (``ignore_index``), and IoU is computed
only over pixels where GT != border.

Collects, per sample and split:
- predicted_segmentation: model prediction at original image size, stored as
  semseg-as-RLE via the "semantic_segmentation" sample type (background/pet only)
- iou: mean IoU over the real classes, with border pixels masked out
"""

from __future__ import annotations

import argparse

import numpy as np
import tlc
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from semseg_sample_type import SemanticSegmentation, SemanticSegmentationSampleType
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_NAME = "oxford-pets-semseg-poc"
DATASET_NAME = "oxford-pets"
NUM_CLASSES = 2  # background, pet — border is ignore-only, never predicted
IGNORE_CLASS_ID = 2  # "border" in the GT tables
IGNORE_INDEX = 255
PREDICTED_CLASSES = {
    0: tlc.schemas.MapElement("background", display_color="#00000000"),
    1: tlc.schemas.MapElement("pet"),
}


# MODEL #####################################################################


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    """A small 3-level UNet."""

    def __init__(self, num_classes: int, base: int = 16) -> None:
        super().__init__()
        self.enc1 = DoubleConv(3, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# DATA ######################################################################


class PetsSegDataset(Dataset):
    """Wraps a tlc Table's sample view as a torch Dataset of resized (image, label_map) pairs."""

    def __init__(self, table: tlc.Table, image_size: int) -> None:
        self.table = table
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.table[idx]
        image = row["image"].convert("RGB").resize((self.image_size, self.image_size))
        image_tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0

        seg: SemanticSegmentation = row["segmentation"]
        label_map = seg.label_map.copy()
        label_map[label_map == IGNORE_CLASS_ID] = IGNORE_INDEX
        label_tensor = (
            F.interpolate(
                torch.from_numpy(label_map).long()[None, None].float(),
                size=(self.image_size, self.image_size),
                mode="nearest",
            )
            .long()
            .squeeze()
        )
        return image_tensor, label_tensor


# METRICS ###################################################################


def mean_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_class_id: int) -> float:
    """Mean IoU over the real classes, evaluated only where GT is not the ignore class."""
    valid = gt != ignore_class_id
    ious = []
    for class_id in range(num_classes):
        pred_mask = (pred == class_id) & valid
        gt_mask = (gt == class_id) & valid
        union = (pred_mask | gt_mask).sum()
        if union == 0:
            continue
        ious.append((pred_mask & gt_mask).sum() / union)
    return float(np.mean(ious)) if ious else 1.0


def collect_metrics(
    run: tlc.Run,
    table: tlc.Table,
    model: nn.Module,
    device: torch.device,
    image_size: int,
    epoch: int,
) -> float:
    """Predict on every sample at original size, write predicted RLEs + IoU to the run."""
    sample_type = SemanticSegmentationSampleType()
    predictions: list[dict] = []
    ious: list[float] = []

    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(table)), desc="collect", leave=False):
            row = table[idx]
            image = row["image"].convert("RGB")
            seg: SemanticSegmentation = row["segmentation"]
            width, height = image.size

            image_tensor = (
                torch.from_numpy(np.asarray(image.resize((image_size, image_size)))).permute(2, 0, 1).float() / 255.0
            )
            logits = model(image_tensor[None].to(device))
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            pred_map = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

            predicted = SemanticSegmentation(image_width=width, image_height=height, label_map=pred_map)
            predictions.append(sample_type.to_row(predicted))
            ious.append(mean_iou(pred_map, seg.label_map, NUM_CLASSES, IGNORE_CLASS_ID))

    run.add_metrics(
        {
            "predicted_segmentation": predictions,
            "iou": ious,
            "epoch": [epoch] * len(ious),
        },
        schema={"predicted_segmentation": SemanticSegmentationSampleType.schema(PREDICTED_CLASSES)},
        foreign_table_url=table.url,
    )
    return float(np.mean(ious))


# TRAINING ##################################################################


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_table = tlc.Table.from_names(table_name="train", dataset_name=DATASET_NAME, project_name=PROJECT_NAME)
    val_table = tlc.Table.from_names(table_name="val", dataset_name=DATASET_NAME, project_name=PROJECT_NAME)

    run = tlc.init(
        PROJECT_NAME,
        description="POC: TinyUNet on Oxford-IIIT Pets with semseg-as-RLE predictions",
        parameters=vars(args),
    )

    train_loader = DataLoader(
        PetsSegDataset(train_table, args.image_size), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(PetsSegDataset(val_table, args.image_size), batch_size=args.batch_size)

    model = TinyUNet(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.shape[0]
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_loss += criterion(model(images), labels).item() * images.shape[0]
        val_loss /= len(val_loader.dataset)

        tlc.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    train_miou = collect_metrics(run, train_table, model, device, args.image_size, epoch=args.epochs - 1)
    val_miou = collect_metrics(run, val_table, model, device, args.image_size, epoch=args.epochs - 1)
    tlc.log({"train_miou": train_miou, "val_miou": val_miou})
    print(f"train mIoU: {train_miou:.4f}, val mIoU: {val_miou:.4f}")

    run.set_status_completed()
    print(f"Run: {run.url}")


if __name__ == "__main__":
    main()
