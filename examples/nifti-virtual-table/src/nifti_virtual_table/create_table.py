# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================

"""Create a 3LC table from BraTS2020 NIfTI data without copying any images.

Each row represents one axial slice from one subject. Image columns store
``nifti-slice://`` URLs that the custom adapter resolves into PNG bytes on
the fly at read time.

Usage::

    pip install -e .
    python -m nifti_virtual_table.create_table /path/to/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData

Options::

    --max-subjects 10        # limit number of subjects
    --skip-empty             # omit slices that are entirely background
    --modalities flair t1ce  # choose which modalities to include
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import nibabel as nib
import numpy as np
from tqdm import tqdm

import tlc
from tlcurl.url_adapter_registry import UrlAdapterRegistry

from .adapter import NiftiSliceUrlAdapter

ALL_MODALITIES = ("flair", "t1", "t1ce", "t2")

SEG_LABELS = {
    0: "background",
    1: "necrotic / non-enhancing tumor",
    2: "peritumoral edema",
    4: "GD-enhancing tumor",
}


def _read_nifti_meta(nii_path: Path) -> dict[str, Any]:
    """Read essential metadata from a NIfTI file without loading voxel data."""
    nii = nib.load(str(nii_path))
    w, h, d = nii.shape[:3]
    return {
        "path": nii_path,
        "w": w,
        "h": h,
        "d": d,
        "dtype": str(nii.get_data_dtype()),
        "offset": int(nii.dataobj.offset),
    }


def _compute_vmax(nii_path: Path) -> float:
    """Compute 99th-percentile of non-zero voxels for windowing."""
    nii = nib.load(str(nii_path))
    data = np.asarray(nii.dataobj).ravel()
    nonzero = data[data > 0]
    if len(nonzero) == 0:
        return 1.0
    return float(np.percentile(nonzero, 99))


def _make_nifti_slice_url(nii_path: Path, z: int, meta: dict[str, Any], vmax: float) -> str:
    """Build a nifti-slice:// URL for a single axial slice."""
    abs_path = nii_path.resolve().as_posix()
    return (
        f"nifti-slice://{abs_path}"
        f"?z={z}&dtype={meta['dtype']}&offset={meta['offset']}"
        f"&w={meta['w']}&h={meta['h']}&vmax={vmax}"
    )


def _get_tumor_slices(seg_path: Path) -> set[int]:
    """Return set of slice indices that contain tumor labels."""
    seg = np.asarray(nib.load(str(seg_path)).dataobj)
    return {z for z in range(seg.shape[2]) if np.any(seg[:, :, z] > 0)}


def create_virtual_brats_table(
    brats_root: Path,
    max_subjects: int | None = None,
    modalities: tuple[str, ...] = ALL_MODALITIES,
    skip_empty: bool = False,
    table_name: str = "BraTS2020_virtual",
    dataset_name: str = "BraTS2020",
    project_name: str = "BraTS_VIRTUAL",
    root_url: str | None = None,
    if_exists: Literal["overwrite", "rename", "raise"] = "overwrite",
) -> tlc.Table:
    assert brats_root.exists(), f"BraTS root {brats_root} does not exist!"

    # Discover subject directories
    subject_dirs = sorted(
        d for d in brats_root.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training")
    )
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    assert len(subject_dirs) > 0, f"No subject directories found in {brats_root}"

    # Build schema
    schema: dict[str, tlc.Schema] = {
        "subject_id": tlc.StringSchema(writable=False),
        "slice_index": tlc.Int32Schema(writable=False),
    }
    for mod in modalities:
        schema[mod] = tlc.ImageUrlSchema()
    schema["has_tumor"] = tlc.BoolSchema(writable=False)

    table_writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=dataset_name,
        project_name=project_name,
        root_url=root_url,
        if_exists=if_exists,
        schema=schema,
    )

    for subject_dir in tqdm(subject_dirs, desc="Subjects"):
        subject_id = subject_dir.name

        # Read metadata and compute vmax for each modality
        mod_info: dict[str, tuple[dict[str, Any], float]] = {}
        num_slices = None
        for mod in modalities:
            nii_path = subject_dir / f"{subject_id}_{mod}.nii"
            if not nii_path.exists():
                # Try .nii.gz fallback
                nii_path = subject_dir / f"{subject_id}_{mod}.nii.gz"
            if not nii_path.exists():
                msg = f"Missing {mod} file for {subject_id}"
                raise FileNotFoundError(msg)

            meta = _read_nifti_meta(nii_path)
            vmax = _compute_vmax(nii_path)
            mod_info[mod] = (meta, vmax)

            if num_slices is None:
                num_slices = meta["d"]
            elif meta["d"] != num_slices:
                msg = f"Slice count mismatch for {subject_id}: {meta['d']} vs {num_slices}"
                raise ValueError(msg)

        assert num_slices is not None

        # Check for segmentation to determine tumor slices
        seg_path = subject_dir / f"{subject_id}_seg.nii"
        if not seg_path.exists():
            seg_path = subject_dir / f"{subject_id}_seg.nii.gz"
        tumor_slices = _get_tumor_slices(seg_path) if seg_path.exists() else set()

        # Determine which slices to include
        if skip_empty:
            # Use any modality to check for non-empty slices
            first_mod = modalities[0]
            meta, _ = mod_info[first_mod]
            nii = nib.load(str(meta["path"]))
            data = np.asarray(nii.dataobj)
            slice_indices = [z for z in range(num_slices) if np.any(data[:, :, z] > 0)]
        else:
            slice_indices = list(range(num_slices))

        for z in slice_indices:
            row: dict[str, Any] = {
                "subject_id": subject_id,
                "slice_index": z,
                "has_tumor": z in tumor_slices,
            }
            for mod in modalities:
                meta, vmax = mod_info[mod]
                row[mod] = _make_nifti_slice_url(meta["path"], z, meta, vmax)

            table_writer.add_row(row)

    table = table_writer.finalize()
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a virtual BraTS2020 table (no data copy)")
    parser.add_argument("brats_root", type=Path, help="Path to MICCAI_BraTS2020_TrainingData directory")
    parser.add_argument("--max-subjects", type=int, default=None, help="Limit number of subjects")
    parser.add_argument("--modalities", nargs="+", default=list(ALL_MODALITIES), help="Modalities to include")
    parser.add_argument("--skip-empty", action="store_true", help="Skip slices with no brain content")
    parser.add_argument("--table-name", default="BraTS2020_virtual")
    parser.add_argument("--dataset-name", default="BraTS2020")
    parser.add_argument("--project-name", default="BraTS_VIRTUAL")
    parser.add_argument("--if-exists", default="overwrite", choices=["overwrite", "rename", "raise"])
    args = parser.parse_args()

    # Register adapter if not already discovered via entry points
    if "nifti-slice" not in UrlAdapterRegistry.get_registered_schemes():
        UrlAdapterRegistry.register_url_adapter_for_scheme("nifti-slice", NiftiSliceUrlAdapter())

    table = create_virtual_brats_table(
        brats_root=args.brats_root,
        max_subjects=args.max_subjects,
        modalities=tuple(args.modalities),
        skip_empty=args.skip_empty,
        table_name=args.table_name,
        dataset_name=args.dataset_name,
        project_name=args.project_name,
        if_exists=args.if_exists,
    )

    print(f"\nCreated virtual table with {len(table)} rows")
    print(f"Table URL: {table.url}")
    print(f"\nNo image data was copied — the table references original .nii files")
    print(f"via nifti-slice:// URLs that render PNG slices at read time.")


if __name__ == "__main__":
    main()
