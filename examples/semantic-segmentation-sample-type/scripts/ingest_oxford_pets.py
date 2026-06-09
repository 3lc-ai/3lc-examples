# Copyright 2026 3LC Inc. All rights reserved.

"""Ingest a subset of Oxford-IIIT Pets into 3LC tables with semseg-as-RLE.

Creates train/val tables with columns:
- image: the pet image
- segmentation: semantic segmentation (background / pet / border) from the trimap,
  stored as one RLE per class via the "semantic_segmentation" sample type
- species: cat / dog
- breed: one of the 37 breeds
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import tlc
from PIL import Image
from semseg_sample_type import SemanticSegmentation, SemanticSegmentationSampleType

PROJECT_NAME = "oxford-pets-semseg-poc"
DATASET_NAME = "oxford-pets"

# Trimap pixel values: 1 = pet, 2 = background, 3 = border. Remap to 0-based with background = 0.
TRIMAP_REMAP = {2: 0, 1: 1, 3: 2}
SEGMENTATION_CLASSES = {
    0: tlc.schemas.MapElement("background", display_color="#00000000"),
    1: tlc.schemas.MapElement("pet"),
    2: tlc.schemas.MapElement("border"),
}
SPECIES_CLASSES = ["cat", "dog"]


def parse_annotations(data_root: Path) -> list[dict]:
    """Parse trainval.txt: `Image CLASS-ID SPECIES BREED-ID` per line."""
    samples = []
    for line in (data_root / "annotations" / "trainval.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, class_id, species, _breed_id = line.split()
        image_path = data_root / "images" / f"{name}.jpg"
        trimap_path = data_root / "annotations" / "trimaps" / f"{name}.png"
        if not image_path.exists() or not trimap_path.exists():
            continue
        samples.append(
            {
                "name": name,
                "image_path": image_path,
                "trimap_path": trimap_path,
                "breed": int(class_id) - 1,  # 0-based
                "species": int(species) - 1,  # 0: cat, 1: dog
            }
        )
    return samples


def breed_names(samples: list[dict]) -> list[str]:
    """Derive 0-indexed breed names from image file names (name minus trailing _N)."""
    names: dict[int, str] = {}
    for s in samples:
        names[s["breed"]] = s["name"].rsplit("_", 1)[0].replace("_", " ").lower()
    return [names[i] for i in range(max(names) + 1)]


def load_segmentation(trimap_path: Path) -> SemanticSegmentation:
    trimap = np.asarray(Image.open(trimap_path))
    label_map = np.zeros_like(trimap, dtype=np.int32)
    for src, dst in TRIMAP_REMAP.items():
        label_map[trimap == src] = dst
    return SemanticSegmentation(
        image_width=trimap.shape[1],
        image_height=trimap.shape[0],
        label_map=label_map,
    )


def write_table(samples: list[dict], table_name: str, breeds: list[str]) -> tlc.Table:
    writer = tlc.TableWriter(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=table_name,
        schema={
            "image": tlc.schemas.ImageSchema(),
            "segmentation": SemanticSegmentationSampleType.schema(SEGMENTATION_CLASSES),
            "species": tlc.schemas.CategoricalLabelSchema(SPECIES_CLASSES),
            "breed": tlc.schemas.CategoricalLabelSchema(breeds),
        },
        if_exists="overwrite",
    )
    for s in samples:
        writer.add_row(
            {
                "image": tlc.Url(s["image_path"]).to_relative().to_str(),
                "segmentation": load_segmentation(s["trimap_path"]),
                "species": s["species"],
                "breed": s["breed"],
            }
        )
    table = writer.finalize()
    print(f"Wrote {table_name}: {len(table)} rows -> {table.url}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.home() / "data" / "Oxford-IIIT-Pets")
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=680)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = parse_annotations(args.data_root)
    print(f"Found {len(samples)} annotated samples")
    breeds = breed_names(samples)

    random.Random(args.seed).shuffle(samples)
    train_samples = samples[: args.n_train]
    val_samples = samples[args.n_train : args.n_train + args.n_val]

    train_table = write_table(train_samples, "train", breeds)
    write_table(val_samples, "val", breeds)

    # Sanity check: sample view should hand back the dataclass.
    sample = train_table[0]
    seg = sample["segmentation"]
    print(f"Sample view round-trip: {type(seg).__name__}, classes present: {seg.class_ids.tolist()}")


if __name__ == "__main__":
    main()
