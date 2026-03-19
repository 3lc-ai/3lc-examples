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
the fly at read time. Segmentation masks are RLE-encoded and stored inline
in parquet (enabling global IoU computation in the Dashboard).

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
import csv
from pathlib import Path
from typing import Any, Literal

import nibabel as nib
import numpy as np
import tlc
from tlc.core.helpers.segmentation_helper import SegmentationHelper
from tlcurl.url_adapter_registry import UrlAdapterRegistry
from tqdm import tqdm

from .adapter import NiftiSliceUrlAdapter

ALL_MODALITIES = ("flair", "t1", "t1ce", "t2")

# BraTS segmentation labels (note: label 3 is intentionally absent)
SEG_LABEL_MAP = {
    1: "necrotic / non-enhancing tumor (NCR/NET)",
    2: "peritumoral edema (ED)",
    4: "GD-enhancing tumor (ET)",
}

# Labels in the order we emit instances
SEG_LABELS = [1, 2, 4]


# ── NIfTI helpers ──


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


def _find_nifti(subject_dir: Path, subject_id: str, suffix: str) -> Path | None:
    """Find a NIfTI file, trying .nii then .nii.gz."""
    for ext in (".nii", ".nii.gz"):
        p = subject_dir / f"{subject_id}_{suffix}{ext}"
        if p.exists():
            return p
    return None


# ── CSV metadata loading ──


def _load_name_mapping(brats_root: Path) -> dict[str, str]:
    """Load name_mapping.csv → {BraTS_2020_subject_ID: Grade}."""
    csv_path = brats_root / "name_mapping.csv"
    if not csv_path.exists():
        return {}
    mapping: dict[str, str] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            mapping[row["BraTS_2020_subject_ID"]] = row["Grade"]
    return mapping


def _load_survival_info(brats_root: Path) -> dict[str, dict[str, Any]]:
    """Load survival_info.csv → {Brats20ID: {age, survival_days, resection}}."""
    csv_path = brats_root / "survival_info.csv"
    if not csv_path.exists():
        return {}
    info: dict[str, dict[str, Any]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            age_str = row.get("Age", "")
            survival_str = row.get("Survival_days", "")
            info[row["Brats20ID"]] = {
                "age": float(age_str) if age_str else None,
                "survival_days": int(survival_str) if survival_str.isdigit() else None,
                "resection": row.get("Extent_of_Resection", ""),
            }
    return info


# ── Segmentation helpers ──


def _seg_slice_to_instance_row(seg_slice: np.ndarray, h: int, w: int) -> dict[str, Any]:
    """Convert a 2D segmentation slice to an instance segmentation row dict.

    Creates one instance per label present (labels 1, 2, 4). Empty slices
    produce a valid row with zero instances.
    """
    labels_present = []
    masks_list = []
    for label in SEG_LABELS:
        mask = (seg_slice == label).astype(np.uint8)
        if mask.any():
            labels_present.append(label)
            masks_list.append(mask)

    if masks_list:
        masks_3d = np.stack(masks_list, axis=-1)  # (H, W, N)
        rles = SegmentationHelper.rles_from_masks(masks_3d)
        rle_bytes = [rle["counts"] for rle in rles]
    else:
        rle_bytes = []

    return {
        "image_height": h,
        "image_width": w,
        "instance_properties": {"label": labels_present},
        "rles": rle_bytes,
    }


# ── Table creation ──


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
    subject_dirs = sorted(d for d in brats_root.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training"))
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    assert len(subject_dirs) > 0, f"No subject directories found in {brats_root}"

    # Load CSV metadata
    grade_map = _load_name_mapping(brats_root)
    survival_map = _load_survival_info(brats_root)

    # Build schema
    schema: dict[str, tlc.Schema] = {
        "subject_id": tlc.StringSchema(writable=False),
        "slice_index": tlc.Int32Schema(writable=False),
    }
    for mod in modalities:
        schema[mod] = tlc.ImageUrlSchema()
    schema["segmentation"] = tlc.SegmentationMasksSchema(
        classes=SEG_LABEL_MAP,
        include_per_instance_label=True,
    )
    schema["has_tumor"] = tlc.BoolSchema(writable=False)

    # CSV-derived metadata columns
    schema["grade"] = tlc.StringSchema(writable=False)
    schema["age"] = tlc.Float32Schema(writable=False)
    schema["survival_days"] = tlc.Int32Schema(writable=False)
    schema["resection"] = tlc.StringSchema(writable=False)

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
        image_w, image_h = None, None
        for mod in modalities:
            nii_path = _find_nifti(subject_dir, subject_id, mod)
            if nii_path is None:
                msg = f"Missing {mod} file for {subject_id}"
                raise FileNotFoundError(msg)

            meta = _read_nifti_meta(nii_path)
            vmax = _compute_vmax(nii_path)
            mod_info[mod] = (meta, vmax)

            if num_slices is None:
                num_slices = meta["d"]
                image_w, image_h = meta["w"], meta["h"]
            elif meta["d"] != num_slices:
                msg = f"Slice count mismatch for {subject_id}: {meta['d']} vs {num_slices}"
                raise ValueError(msg)

        assert num_slices is not None and image_w is not None and image_h is not None

        # Load segmentation volume (needed for inline RLE encoding)
        seg_path = _find_nifti(subject_dir, subject_id, "seg")
        seg_volume = np.asarray(nib.load(str(seg_path)).dataobj) if seg_path else None

        # CSV metadata for this subject
        grade = grade_map.get(subject_id, "")
        surv = survival_map.get(subject_id, {})
        age = surv.get("age")
        survival_days = surv.get("survival_days")
        resection = surv.get("resection", "NA") or "NA"

        # Determine which slices to include
        if skip_empty:
            first_mod = modalities[0]
            meta, _ = mod_info[first_mod]
            nii = nib.load(str(meta["path"]))
            data = np.asarray(nii.dataobj)
            slice_indices = [z for z in range(num_slices) if np.any(data[:, :, z] > 0)]
        else:
            slice_indices = list(range(num_slices))

        for z in slice_indices:
            # Segmentation for this slice
            if seg_volume is not None:
                seg_slice = seg_volume[:, :, z]
                has_tumor = bool(np.any(seg_slice > 0))
                seg_row = _seg_slice_to_instance_row(seg_slice, image_h, image_w)
            else:
                has_tumor = False
                seg_row = _seg_slice_to_instance_row(np.zeros((image_w, image_h), dtype=np.uint8), image_h, image_w)

            row: dict[str, Any] = {
                "subject_id": subject_id,
                "slice_index": z,
                "has_tumor": has_tumor,
                "segmentation": seg_row,
                "grade": grade,
                "age": age if age is not None else 0.0,
                "survival_days": survival_days if survival_days is not None else -1,
                "resection": resection,
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
    print("\nNo image data was copied — the table references original .nii files")
    print("via nifti-slice:// URLs that render PNG slices at read time.")
    print("Segmentation masks are RLE-encoded inline in parquet.")


if __name__ == "__main__":
    main()
