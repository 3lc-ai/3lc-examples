# Copyright 2026 3LC Inc. All rights reserved.

"""Ingest Cityscapes (fine annotations) into 3LC tables with semseg-as-RLE.

Sibling of ``ingest_oxford_pets.py`` for a second, larger semseg dataset. Uses the
documented front door, ``Table.from_semantic_segmentation()``, to create train/val
tables with two columns:

- image: the street-scene image (``*_leftImg8bit.png``)
- mask: semantic segmentation over the 19 standard Cityscapes eval classes, derived
  from ``*_gtFine_labelIds.png`` and stored as one RLE per class. All non-eval /
  unlabeled pixels collapse to a single ``void`` class (the standard trainId 255
  ignore set), tagged so it is excluded from metrics downstream.

(The front door writes exactly ``image`` + ``mask``; per-row metadata like a ``city``
categorical would need the manual ``TableWriter`` path, so it is intentionally dropped
here.)

Expects the standard Cityscapes layout under ``--data-root`` (default ``~/Data/cityscapes``):

    <root>/leftImg8bit/{train,val}/<city>/<id>_leftImg8bit.png
    <root>/gtFine/{train,val}/<city>/<id>_gtFine_labelIds.png

i.e. the contents of ``leftImg8bit_trainvaltest.zip`` + ``gtFine_trainvaltest.zip``.
The held-out ``test`` split ships with blank labels, so it is intentionally ignored.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tlc
from PIL import Image

PROJECT_NAME = "cityscapes-semseg-poc"
DATASET_NAME = "cityscapes"

# The 19 standard Cityscapes evaluation classes, in trainId order (0..18). Everything
# else (unlabeled, ego vehicle, rectification border, out-of-roi, the non-eval
# label ids, ...) is the ignore set and maps to a single void class.
TRAIN_CLASS_NAMES = [
    "road",          # 0
    "sidewalk",      # 1
    "building",      # 2
    "wall",          # 3
    "fence",         # 4
    "pole",          # 5
    "traffic light",  # 6
    "traffic sign",   # 7
    "vegetation",    # 8
    "terrain",       # 9
    "sky",           # 10
    "person",        # 11
    "rider",         # 12
    "car",           # 13
    "truck",         # 14
    "bus",           # 15
    "train",         # 16
    "motorcycle",    # 17
    "bicycle",       # 18
]
VOID_ID = len(TRAIN_CLASS_NAMES)  # 19

# Class universe handed to the front door: 19 eval classes + void. Passing void=VOID_ID
# tags it via its reserved internal_name so downstream (metrics, editor) reads the role
# back rather than hardcoding the id.
SEGMENTATION_CLASSES = {**{i: name for i, name in enumerate(TRAIN_CLASS_NAMES)}, VOID_ID: "void"}

# Cityscapes raw label id (0..33) -> trainId. Source of truth is the official
# `labels` table in cityscapesscripts.helpers.labels; inlined here to keep the POC
# dependency-free. Any label id not listed is ignore -> our single void class.
LABELID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}

# 256-entry lookup table: labelId png -> our class ids (eval classes + void). Built
# once and applied per image as a vectorized gather.
_LABELID_LUT = np.full(256, VOID_ID, dtype=np.int32)
for _label_id, _train_id in LABELID_TO_TRAINID.items():
    _LABELID_LUT[_label_id] = _train_id


def collect_samples(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    """Pair every ``*_leftImg8bit.png`` in a split with its ``*_gtFine_labelIds.png``."""
    img_root = data_root / "leftImg8bit" / split
    gt_root = data_root / "gtFine" / split
    if not img_root.is_dir():
        raise FileNotFoundError(f"Missing image split dir: {img_root}")

    pairs = []
    for image_path in sorted(img_root.rglob("*_leftImg8bit.png")):
        city = image_path.parent.name
        base = image_path.name[: -len("_leftImg8bit.png")]
        label_path = gt_root / city / f"{base}_gtFine_labelIds.png"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def load_mask(label_path: Path) -> np.ndarray:
    """Decode a ``*_gtFine_labelIds.png`` and remap raw label ids to our class ids."""
    label_ids = np.asarray(Image.open(label_path), dtype=np.int64)
    return _LABELID_LUT[label_ids]


class LazyMasks(Sequence):
    """A ``Sequence`` of dense label masks that decodes + remaps each PNG on access.

    Cityscapes masks are 2048x1024; materializing all ~3000 as arrays up front would
    cost tens of GB. The front door only needs indexed access, so we load lazily.
    """

    def __init__(self, label_paths: list[Path]) -> None:
        self._label_paths = label_paths

    def __len__(self) -> int:
        return len(self._label_paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return load_mask(self._label_paths[index])


def build_table(data_root: Path, split: str, table_name: str, max_rows: int | None, seed: int) -> tlc.Table:
    pairs = collect_samples(data_root, split)
    if max_rows is not None:
        random.Random(seed).shuffle(pairs)
        pairs = pairs[:max_rows]

    images = [image_path for image_path, _ in pairs]
    masks = LazyMasks([label_path for _, label_path in pairs])

    table = tlc.Table.from_semantic_segmentation(
        images=images,
        masks=masks,
        classes=SEGMENTATION_CLASSES,
        void=VOID_ID,
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=table_name,
        if_exists="overwrite",
    )
    print(f"Wrote {table_name}: {len(table)} rows -> {table.url}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.home() / "Data" / "cityscapes")
    parser.add_argument("--max-train", type=int, default=None, help="Optionally cap train rows for a quick POC.")
    parser.add_argument("--max-val", type=int, default=None, help="Optionally cap val rows for a quick POC.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_table = build_table(args.data_root, "train", "train", args.max_train, args.seed)
    build_table(args.data_root, "val", "val", args.max_val, args.seed)

    # Sanity check: sample view should hand back the SemanticSegmentation dataclass.
    seg = train_table[0]["mask"]
    print(
        f"Sample view round-trip: {type(seg).__name__}, layer universe: {seg.class_ids}, "
        f"classes present: {seg.present_class_ids.tolist()}"
    )


if __name__ == "__main__":
    main()
