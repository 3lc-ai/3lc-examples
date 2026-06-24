# Copyright 2026 3LC Inc. All rights reserved.

"""Ingest the Pascal VOC 2012 segmentation set into 3LC tables with semseg-as-RLE.

Sibling of ``ingest_oxford_pets.py`` / ``ingest_cityscapes.py``. Uses the documented
front door, ``Table.from_semantic_segmentation()``, to create train/val tables with
two columns:

- image: the source image (``JPEGImages/<id>.jpg``)
- mask: semantic segmentation over the 21 VOC classes (background + 20 objects),
  read straight from the paletted ``SegmentationClass/<id>.png``. The boundary
  ``void`` border (palette index 255) is tagged as void so it is excluded from
  metrics; background (index 0) is tagged as background (rendered transparent).

Only the images that have a segmentation annotation are ingested (the ids listed in
``ImageSets/Segmentation/{train,val}.txt`` — the official VOC segmentation split, not
the larger SBD/SegmentationClassAug set).

Unlike Cityscapes, VOC's paletted PNG already stores class ids as the pixel values
(index == class id), so no label remap / LUT is needed: ``np.asarray`` of the PNG is
the dense label map, void included.

Expects the standard VOCdevkit layout under ``--data-root`` (default
``~/Data/VOCdevkit/VOC2012``):

    <root>/JPEGImages/<id>.jpg
    <root>/SegmentationClass/<id>.png
    <root>/ImageSets/Segmentation/{train,val}.txt
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tlc
from PIL import Image

PROJECT_NAME = "pascal-voc-semseg-poc"
DATASET_NAME = "pascal-voc-2012"

# VOC palette index == class id. 0 is background; 1..20 are the object classes in the
# canonical VOC order; 255 is the void / boundary "don't care" border.
VOC_CLASS_NAMES = [
    "background",   # 0
    "aeroplane",    # 1
    "bicycle",      # 2
    "bird",         # 3
    "boat",         # 4
    "bottle",       # 5
    "bus",          # 6
    "car",          # 7
    "cat",          # 8
    "chair",        # 9
    "cow",          # 10
    "diningtable",  # 11
    "dog",          # 12
    "horse",        # 13
    "motorbike",    # 14
    "person",       # 15
    "pottedplant",  # 16
    "sheep",        # 17
    "sofa",         # 18
    "train",        # 19
    "tvmonitor",    # 20
]
BACKGROUND_ID = 0
VOID_ID = 255

# Class universe handed to the front door: background + 20 objects + void. Passing
# background/void tags them via reserved internal_names (background transparent, void
# excluded from metrics) so downstream reads the roles back rather than hardcoding ids.
SEGMENTATION_CLASSES = {**{i: name for i, name in enumerate(VOC_CLASS_NAMES)}, VOID_ID: "void"}


def collect_samples(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    """Pair every id in ``ImageSets/Segmentation/<split>.txt`` with its jpg + mask png."""
    split_file = data_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing segmentation split file: {split_file}")

    pairs = []
    for name in split_file.read_text().split():
        image_path = data_root / "JPEGImages" / f"{name}.jpg"
        label_path = data_root / "SegmentationClass" / f"{name}.png"
        if image_path.exists() and label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def load_mask(label_path: Path) -> np.ndarray:
    """Decode a paletted ``SegmentationClass`` PNG; its indices are the class ids directly."""
    return np.asarray(Image.open(label_path), dtype=np.int32)


class LazyMasks(Sequence):
    """A ``Sequence`` of dense label masks that decodes each PNG on indexed access.

    Keeps memory flat: the front door only needs indexed access, so we never hold all
    masks at once.
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
        background=BACKGROUND_ID,
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
    parser.add_argument("--data-root", type=Path, default=Path.home() / "Data" / "VOCdevkit" / "VOC2012")
    parser.add_argument("--max-train", type=int, default=None, help="Optionally cap train rows for a quick POC.")
    parser.add_argument("--max-val", type=int, default=None, help="Optionally cap val rows for a quick POC.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_table = build_table(args.data_root, "train", "train", args.max_train, args.seed)
    build_table(args.data_root, "val", "val", args.max_val, args.seed)

    # Sanity check: sample view should hand back the SemanticSegmentation dataclass.
    seg = train_table[0]["mask"]
    print(
        f"Sample view round-trip: {type(seg).__name__}, "
        f"classes present: {seg.present_class_ids.tolist()}"
    )


if __name__ == "__main__":
    main()
