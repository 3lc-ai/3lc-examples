# Copyright 2026 3LC Inc. All rights reserved.

"""Train a small UNet on the Oxford-IIIT Pets semseg tables and collect per-sample metrics.

The GT tables have three classes (background, pet, border), but border is an
ignore region, not a prediction target: the model outputs only background/pet,
border pixels are excluded from the loss (``ignore_index``), and IoU is computed
only over pixels where GT != border.

Collects, per sample and split, every ``--collect-frequency`` epochs (the final
epoch is always collected):
- predicted_segmentation: model prediction at original image size, stored as
  semseg-as-RLE via the "semantic_segmentation" sample type (background/pet only)
- iou / pet_iou: derived from the core confusion-matrix helper, border masked out
- confusion_matrix: the per-image C×C matrix (flattened), carried so the headline
  mIoU can be summed cumulatively dashboard-side rather than averaged per-image
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import tlc
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.functional as TF  # noqa: N812
from PIL import Image
from tlc.constants import SAMPLE_WEIGHT
from tlc.sample_types import (
    SemanticSegmentation,
    SemanticSegmentationSampleType,
    semseg_classes,
    void_id,
)
from tlc.helpers.semantic_segmentation_metrics import semantic_segmentation_metrics
from tlc.schemas import SemanticSegmentationRLESchema
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_NAME = "oxford-pets-semseg-poc"
DATASET_NAME = "oxford-pets"

# GT tables carry three classes; "border" is the void/ignore class. Mirrors the ingest
# class map so the ignore id is read back via void_id rather than hardcoded — eventually
# this would be read straight off the loaded GT table's segmentation schema.
GT_CLASSES = semseg_classes({0: "background", 1: "pet", 2: "border"}, background=0, void=2)
# The model predicts only background/pet — border is never a target, so it has no void class.
PREDICTED_CLASSES = semseg_classes({0: "background", 1: "pet"}, background=0)

NUM_CLASSES = 2  # background, pet — border is ignore-only, never predicted
FOREGROUND_CLASS_ID = 1  # "pet" — the class that actually matters for this dataset
IGNORE_CLASS_ID = void_id(GT_CLASSES)  # "border" in the GT tables
IGNORE_INDEX = 255


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
    """Wraps a tlc Table's sample view as a torch Dataset of resized (image, label_map) pairs.

    When ``augment`` is set (training only), applies random horizontal flip and a
    small rotation/scale jointly to image and label, plus image-only color jitter.
    Geometric transforms use bilinear for the image and nearest for the label;
    pixels rotated/scaled in from outside the frame become ``IGNORE_INDEX`` in the
    label (excluded from the loss) rather than spurious background.

    Rows with sample ``weight == 0`` are dropped, so samples excluded during
    Dashboard curation leave the training set without any code changes.
    """

    def __init__(self, table: tlc.Table, image_size: int, *, augment: bool = False) -> None:
        self.table = table
        self.image_size = image_size
        self.augment = augment
        # Honor curation: keep only rows with non-zero sample weight (default weight is 1.0).
        self.indices = [
            i for i, row in enumerate(table.table_rows) if float(row.get(SAMPLE_WEIGHT, 1.0)) > 0.0
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.table[self.indices[idx]]
        image = row["image"].convert("RGB").resize((self.image_size, self.image_size))

        seg: SemanticSegmentation = row["segmentation"]
        label_map = seg.mask.copy()
        label_map[label_map == IGNORE_CLASS_ID] = IGNORE_INDEX
        label = Image.fromarray(label_map.astype(np.uint8), mode="L").resize(
            (self.image_size, self.image_size), Image.NEAREST
        )

        if self.augment:
            image, label = self._augment(image, label)

        image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        label_tensor = torch.from_numpy(np.asarray(label).copy()).long()
        return image_tensor, label_tensor

    def _augment(self, image: Image.Image, label: Image.Image) -> tuple[Image.Image, Image.Image]:
        # torch RNG is seeded per-worker by the DataLoader, so this is reproducible and decorrelated.
        if torch.rand(1).item() < 0.5:
            image, label = TF.hflip(image), TF.hflip(label)

        angle = torch.empty(1).uniform_(-15, 15).item()
        scale = torch.empty(1).uniform_(0.9, 1.1).item()
        image = TF.affine(
            image, angle=angle, translate=(0, 0), scale=scale, shear=0,
            interpolation=TF.InterpolationMode.BILINEAR, fill=0,
        )
        label = TF.affine(
            label, angle=angle, translate=(0, 0), scale=scale, shear=0,
            interpolation=TF.InterpolationMode.NEAREST, fill=IGNORE_INDEX,
        )

        # Color jitter on the image only.
        image = TF.adjust_brightness(image, torch.empty(1).uniform_(0.8, 1.2).item())
        image = TF.adjust_contrast(image, torch.empty(1).uniform_(0.8, 1.2).item())
        image = TF.adjust_saturation(image, torch.empty(1).uniform_(0.8, 1.2).item())
        return image, label


# METRICS ###################################################################

# The per-class / mean IoU that used to live here (~40 lines, void-masked) graduated to
# core as ``tlc.helpers.semantic_segmentation_metrics`` (SPEC §4.3): every readout now
# derives from one per-image C×C confusion matrix, void read off the value map rather
# than a hardcoded ignore id. ``collect_metrics`` calls that helper below.


def collect_metrics(
    run: tlc.Run,
    table: tlc.Table,
    model: nn.Module,
    device: torch.device,
    image_size: int,
    epoch: int,
) -> float:
    """Predict on every sample at original size and write per-sample metrics to the run.

    Per sample: the predicted segmentation (as RLE), mean IoU, foreground (pet) IoU,
    cross-entropy loss and mean prediction entropy (both at original resolution, border
    excluded from the loss), and a pooled bottleneck embedding. The embedding is reduced
    to 2D after the run; ``loss``/``entropy`` are deliberately not proportional to IoU —
    they surface confidently-wrong and uncertain samples that hard-label IoU misses.
    """
    sample_type = SemanticSegmentationSampleType()
    predictions: list[dict] = []
    ious: list[float] = []
    pet_ious: list[float] = []
    confusion_matrices: list[list[int]] = []
    losses: list[float] = []
    entropies: list[float] = []
    embeddings: list[np.ndarray] = []

    # Tap the bottleneck activations; one pooled vector per sample becomes the embedding.
    captured: dict[str, torch.Tensor] = {}
    handle = model.bottleneck.register_forward_hook(lambda _m, _i, out: captured.__setitem__("emb", out))

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
            embeddings.append(captured["emb"].mean(dim=(2, 3)).squeeze(0).cpu().numpy().astype(np.float32))

            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            pred_map = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

            target = seg.mask.copy()
            target[target == IGNORE_CLASS_ID] = IGNORE_INDEX
            target_tensor = torch.from_numpy(target).long()[None].to(device)
            losses.append(F.cross_entropy(logits, target_tensor, ignore_index=IGNORE_INDEX).item())
            probs = logits.softmax(dim=1)
            entropies.append(float((-(probs * probs.clamp_min(1e-12).log()).sum(dim=1)).mean()))

            # Compact storage: background (id 0) is dropped on the wire and recovered as
            # the implicit fill, so only the "pet" layer is stored per row.
            predicted = SemanticSegmentation(
                image_width=width, image_height=height, mask=pred_map, background_id=0,
            )
            predictions.append(sample_type.to_row(predicted))

            # IoU readouts + the per-image C×C confusion matrix come from the core helper.
            # include_background=True averages mIoU over {background, pet} (VOC convention);
            # void ("border") is read off GT_CLASSES and excluded. The flattened per-image
            # matrix is carried so the headline can be computed *cumulatively* — sum the
            # per-image matrices dashboard-side, then derive — rather than the noisier
            # mean-of-per-image (SPEC §4.2).
            m = semantic_segmentation_metrics(pred_map, seg.mask, GT_CLASSES, include_background=True)
            ious.append(m["mean_iou"])
            pet_ious.append(m["per_class_iou"][m["class_ids"].index(FOREGROUND_CLASS_ID)])
            confusion_matrices.append([int(x) for row in m["confusion_matrix"] for x in row])

    handle.remove()

    run.add_metrics(
        {
            "predicted_segmentation": predictions,
            "iou": ious,
            "pet_iou": pet_ious,
            "confusion_matrix": confusion_matrices,
            "loss": losses,
            "entropy": entropies,
            "embedding": embeddings,
            "epoch": [epoch] * len(ious),
        },
        schema={
            "predicted_segmentation": SemanticSegmentationRLESchema(classes=PREDICTED_CLASSES),
            "embedding": tlc.schemas.EmbeddingSchema(shape=len(embeddings[0])),
        },
        foreign_table_url=table.url,
    )
    return float(np.mean(ious))


# TRAINING ##################################################################


def seed_everything(seed: int) -> None:
    """Seed all RNGs so runs are reproducible and differences are attributable to actual changes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN convolutions (slightly slower, no benchmark autotuning).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id: int) -> None:
    """Reseed per-worker RNGs from the worker's (deterministic) torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--collect-frequency", type=int, default=10, help="Collect per-sample metrics every N epochs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-augment", action="store_true", help="Disable training-set augmentation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--description",
        default="TinyUNet on Oxford-IIIT Pets (fixed edge budget)",
        help="Run description; use this to distinguish baseline vs curated runs",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # .latest() picks up the newest revision, so retraining consumes any Dashboard curation
    # (excluded/relabeled samples) without code changes. Val is the fixed ruler — also latest,
    # but normally left uncurated.
    train_table = tlc.Table.from_names(
        table_name="train", dataset_name=DATASET_NAME, project_name=PROJECT_NAME
    ).latest()
    val_table = tlc.Table.from_names(
        table_name="val", dataset_name=DATASET_NAME, project_name=PROJECT_NAME
    ).latest()

    train_dataset = PetsSegDataset(train_table, args.image_size, augment=not args.no_augment)
    val_dataset = PetsSegDataset(val_table, args.image_size)
    print(f"Train: {len(train_dataset)} of {len(train_table)} rows after weight filtering | Val: {len(val_dataset)}")

    run = tlc.init(
        PROJECT_NAME,
        description=args.description,
        parameters={**vars(args), "train_table": train_table.url.to_str(), "n_train": len(train_dataset)},
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    model = TinyUNet(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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

        log_entry = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        scheduler.step()

        is_final = epoch == args.epochs - 1
        if (epoch + 1) % args.collect_frequency == 0 or is_final:
            train_miou = collect_metrics(run, train_table, model, device, args.image_size, epoch=epoch)
            val_miou = collect_metrics(run, val_table, model, device, args.image_size, epoch=epoch)
            log_entry |= {"train_miou": train_miou, "val_miou": val_miou}

        tlc.log(log_entry)
        print("  ".join(f"{k}={v:.4f}" if k != "epoch" else f"epoch {v}" for k, v in log_entry.items()))

    # Reduce embeddings to 2D with PaCMAP, fitting one model on the final-epoch val
    # embeddings and applying it to every metrics table (both splits, all epochs) so
    # they share a single, stable space. delete_source_tables drops the raw 128-d
    # vectors afterwards — only the 2D reduction is kept.
    print("Reducing embeddings (PaCMAP)...")
    run.reduce_embeddings_by_foreign_table_url(val_table.url, method="pacmap", delete_source_tables=True)

    run.set_status_completed()
    print(f"Run: {run.url}")


if __name__ == "__main__":
    main()
