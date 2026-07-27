"""Collect a *traversal index over the original (unreduced) embeddings* with 3LC.

This is a minimal, self-contained demo of the feature discussed in the
accompanying findings notebook (``traversal-coreset-findings.ipynb``): instead of
deriving a farthest-first traversal ordering from a *reduced* projection
(UMAP / PaCMAP), 3LC computes it directly on the original high-dimensional
embeddings and persists only the resulting scalar index column(s) — the full
embeddings never need to be stored in the Table.

The whole feature is exercised by a single call:

    run.reduce_embeddings_by_foreign_table_url(..., add_traversal_index=True)

When the collected metrics also include a confidence and/or a categorical label
column, 3LC additionally emits the *weighted* and *per-category* variants, so a
single reduce step produces up to four ordering columns:

    <emb>_traversal_index
    <emb>_weighted_traversal_index                 (needs confidence)
    <emb>_per_category_traversal_index             (needs a category column)
    <emb>_per_category_weighted_traversal_index    (needs both)

Requires a 3LC build with the traversal-index feature (monorepo branch
``feature/everley/traversal-index-embeddings``) plus the ``umap`` extra.

    python collect_traversal_index.py                 # 2 quick epochs, then collect
    python collect_traversal_index.py --epochs 0      # skip training (random backbone)

NOTE: model accuracy is irrelevant here — the point is the collection/reduction
mechanism. A couple of epochs (or none) is plenty to see the columns appear.
"""

from __future__ import annotations

import argparse

import numpy as np
import tlc
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PROJECT_NAME = "cifar10-traversal-index-demo"
DOWNLOAD_PATH = "../../transient_data"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
IMG_SIZE = 32
EMBEDDING_LAYER_NAME = "backbone"  # its output is the 512-d embedding we reduce


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# A compact CIFAR-style ResNet-9 (backbone -> 512-d feature, then a small head)
# --------------------------------------------------------------------------- #
def _conv_bn(c_in: int, c_out: int, pool: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class _Residual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(_conv_bn(channels, channels), _conv_bn(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNet9(nn.Module):
    """Backbone (-> 512-d pooled feature) + a 3-layer classifier head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _conv_bn(3, 64),
            _conv_bn(64, 128, pool=True), _Residual(128),
            _conv_bn(128, 256, pool=True),
            _conv_bn(256, 512, pool=True), _Residual(512),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),  # -> (B, 512)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# --------------------------------------------------------------------------- #
# Picklable transform (3LC sample dict -> (tensor, int label))
# --------------------------------------------------------------------------- #
_NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_VAL_TF = transforms.Compose([
    transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), _NORMALIZE,
])


class MapFn:
    """Applies the (no-aug) transform; top-level class so DataLoader can pickle it."""

    def __call__(self, sample: dict) -> tuple[torch.Tensor, int]:
        raw = sample["image"]
        img = raw.convert("RGB") if isinstance(raw, Image.Image) else Image.open(raw).convert("RGB")
        return _VAL_TF(img), int(sample["label"])


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def create_train_table() -> tlc.Table:
    """Register CIFAR-10 train as a 3LC Table (reused on subsequent runs)."""
    dataset = torchvision.datasets.CIFAR10(root=DOWNLOAD_PATH, train=True, download=True)
    ################## 3LC ##################
    return tlc.Table.from_torch_dataset(
        dataset=dataset,
        dataset_name="cifar-10-train",
        table_name="train",
        project_name=PROJECT_NAME,
        description="CIFAR-10 training dataset",
        schema={
            "image": tlc.schemas.ImageSchema(),
            "label": tlc.schemas.CategoricalLabelSchema(classes=CLASS_NAMES),
        },
        if_exists="reuse",
    )
    #########################################


def train_briefly(model: nn.Module, table: tlc.Table, device: torch.device, epochs: int) -> None:
    """A few quick epochs so the embeddings are non-trivial. Skipped when epochs == 0."""
    if epochs <= 0:
        print("[train] skipped (using randomly-initialised backbone)")
        return
    loader = torch.utils.data.DataLoader(
        table.with_transform(MapFn()), batch_size=256, shuffle=True, num_workers=0,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optim.step()
        print(f"[train] epoch {epoch + 1}/{epochs} done (last batch loss {loss.item():.3f})")


def collect_embeddings(model: nn.Module, table: tlc.Table) -> None:
    """Collect 512-d backbone embeddings (+ confidence + label) into the active run.

    3LC's reducer auto-discovers the embedding column produced here; the
    confidence and label columns unlock the weighted / per-category variants.
    """
    layer_idx = next(i for i, (name, _) in enumerate(model.named_modules()) if name == EMBEDDING_LAYER_NAME)
    ################## 3LC ##################
    tlc.collect_metrics(
        table.with_transform(MapFn()),
        metrics_collectors=[
            tlc.metrics.EmbeddingsMetricsCollector(layers=[layer_idx]),
            tlc.metrics.ConfidenceMetricsCollector(),
            tlc.metrics.LabelMetricsCollector(classes=CLASS_NAMES),
        ],
        predictor=tlc.metrics.Predictor(model, layers=[layer_idx]),
        split="train",
        dataloader_args={"batch_size": 256, "num_workers": 0},
    )
    #########################################


def reduce_with_traversal_index(run: tlc.Run, table: tlc.Table) -> dict:
    """Reduce the collected embeddings AND compute the traversal index on the
    original (unreduced) embeddings in the same call. Returns {column_name: order}.
    """
    ################## 3LC ##################
    # add_traversal_index=True is the whole feature: the ordering is computed on
    # the original embeddings, not on the reduced (UMAP) projection.
    url_map = run.reduce_embeddings_by_foreign_table_url(
        table.url,
        method="umap",
        n_components=3,
        add_traversal_index=True,
        traversal_distance_metric="euclidean",
        delete_source_tables=False,
    )
    #########################################

    # Read back every column whose name ends in a traversal-index suffix.
    suffixes = (
        "_per_category_weighted_traversal_index",
        "_per_category_traversal_index",
        "_weighted_traversal_index",
        "_traversal_index",
    )
    found: dict[str, np.ndarray] = {}
    for reduced_url in dict(url_map).values():
        reduced = tlc.Table.from_url(reduced_url)
        for col in reduced.columns:
            if col.endswith(suffixes) and col not in found:
                found[col] = np.array(reduced.get_column_as_pyarrow_array(col).to_pylist())
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2, help="Quick training epochs before collection (0 to skip).")
    args = parser.parse_args()

    device = get_device()
    print(f"[setup] device={device}  project={PROJECT_NAME}")

    train_table = create_train_table()
    print(f"[data]  train table: {train_table.url}  ({len(train_table)} rows)")

    model = ResNet9(num_classes=len(CLASS_NAMES)).to(device)
    train_briefly(model, train_table, device, args.epochs)

    run = tlc.init(project_name=PROJECT_NAME, run_name="traversal-index-demo", if_exists="overwrite")
    model.eval()
    collect_embeddings(model, train_table)
    orderings = reduce_with_traversal_index(run, train_table)
    run.set_status_completed()

    print(f"\n[done] run: {run.url}")
    print(f"[done] traversal-index columns produced ({len(orderings)}):")
    for name, order in orderings.items():
        # order[i] = the traversal rank of row i (0 = picked first / most representative)
        first_ten = np.argsort(order)[:10]
        print(f"  - {name}: first 10 rows in traversal order -> {first_ten.tolist()}")


if __name__ == "__main__":
    main()
