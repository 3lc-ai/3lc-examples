# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================

"""Create a 3LC table from a directory of DigiFarm .npz tiles without copying any data.

Each row references one .npz file via three ``digifarm-npz://`` URLs (rgb,
extent_mask, contour_mask) that the custom adapter resolves into PNG bytes on
the fly at read time. Geo coordinates and the original tif/shp filenames are
stored inline.

Usage::

    pip install -e .
    python -m digifarm_virtual_table.create_table /path/to/DigiFarm
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import tlc
from tlc import Schema
from tlc.schemas import Float64Schema, ImageSchema, StringSchema
from tlc.schemas._schema import MapElement
from tlc.schemas.values import SegmentationMaskUrlStringValue
from tqdm import tqdm

# DigiFarm clean_2km_v3 class scheme. Both extent and contour masks use this scheme.
CLASS_MAP: dict[int, str] = {
    0: "background",
    1: "crops",
    2: "pastures",
    3: "tree-crops",
}


def _mask_schema(display_name: str) -> Schema:
    """Build a segmentation-mask schema in URL-passthrough mode with the DigiFarm class map."""
    value_map = {float(k): MapElement(v) for k, v in CLASS_MAP.items()}
    return Schema(
        value=SegmentationMaskUrlStringValue(map=value_map),
        sample_type=None,
        display_name=display_name,
        writable=True,
    )


def _tile_url(npz_path: Path, view: str) -> str:
    return f"digifarm-npz://{npz_path.resolve().as_posix()}?view={view}"


def create_digifarm_table(
    data_root: Path,
    table_name: str = "digifarm_clean_2km_v3",
    dataset_name: str = "DigiFarm",
    project_name: str = "3LC Tutorials - DigiFarm Virtual Table",
    root_url: str | None = None,
    if_exists: Literal["overwrite", "rename", "raise"] = "overwrite",
) -> tlc.Table:
    assert data_root.exists(), f"DigiFarm data root {data_root} does not exist!"

    npz_files = sorted(data_root.glob("*.npz"))
    assert len(npz_files) > 0, f"No .npz files found in {data_root}"

    schema = Schema(
        values={
            "rgb": ImageSchema(sample_type="url", display_name="RGB", writable=False),
            "extent_mask": _mask_schema("Extent mask"),
            "contour_mask": _mask_schema("Contour mask"),
            "lon_lat": Float64Schema(shape=2, display_name="[lon, lat]", writable=False),
            "image_fn": StringSchema(display_name="Source image", writable=False),
            "mask_fn": StringSchema(display_name="Source mask", writable=False),
        }
    )

    table_writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=dataset_name,
        project_name=project_name,
        root_url=root_url,
        if_exists=if_exists,
        schema=schema,
    )

    for npz_path in tqdm(npz_files, desc="Tiles"):
        with np.load(npz_path) as data:
            geo = data["geo"].reshape(-1).tolist()
            image_fn = data["image_fn"].item()
            mask_fn = data["mask_fn"].item()

        table_writer.add_row({
            "rgb": _tile_url(npz_path, "rgb"),
            "extent_mask": _tile_url(npz_path, "extent"),
            "contour_mask": _tile_url(npz_path, "contour"),
            "lon_lat": geo,
            "image_fn": image_fn,
            "mask_fn": mask_fn,
        })

    return table_writer.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a virtual DigiFarm table (no data copy)")
    parser.add_argument("data_root", type=Path, help="Directory containing .npz tiles")
    parser.add_argument("--table-name", default="digifarm_clean_2km_v3")
    parser.add_argument("--dataset-name", default="DigiFarm")
    parser.add_argument("--project-name", default="3LC Tutorials - DigiFarm Virtual Table")
    parser.add_argument("--if-exists", default="overwrite", choices=["overwrite", "rename", "raise"])
    args = parser.parse_args()

    table = create_digifarm_table(
        data_root=args.data_root,
        table_name=args.table_name,
        dataset_name=args.dataset_name,
        project_name=args.project_name,
        if_exists=args.if_exists,
    )

    print(f"\nCreated virtual table with {len(table)} rows")
    print(f"Table URL: {table.url}")
    print("\nNo image data was copied — the table references original .npz tiles")
    print("via digifarm-npz:// URLs that render PNG views at read time.")


if __name__ == "__main__":
    main()
