# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================

"""Create a 3LC table from KITTI detection data without copying any LiDAR data.

Instead of reading .bin files into memory and letting the TableWriter write them
to .raw chunk files, this script creates pre-externalized Geometry3D objects whose
vertex and intensity data are ``kitti-velodyne://`` URLs pointing to the original
files. The custom URL adapter de-interleaves the data on-the-fly at read time.

Bounding boxes are small and stored inline as usual.

Usage::

    pip install -e .   # install the adapter first
    python -m kitti_virtual_table.create_table /path/to/kitti/training

Or use the console entry point::

    create-kitti-virtual-table /path/to/kitti/training
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tlc
from tqdm import tqdm

# ── KITTI constants (duplicated here so the example is self-contained) ──

KITTI_DETECTION_VALUE_MAP = {
    "Pedestrian": 0,
    "Truck": 1,
    "Car": 2,
    "Cyclist": 3,
    "DontCare": 4,
    "Misc": 5,
    "Van": 6,
    "Tram": 7,
    "Person_sitting": 8,
}

KITTI_BOUNDS = tlc.GeometryHelper.create_isotropic_bounds_3d(-80, 80, -80, 80, -3, 3, force_z_min=False)


# ── Calibration & bounding-box helpers (from the original KITTI loader) ──


def load_kitti_calib(calib_path: Path) -> tuple:
    with open(calib_path) as f:
        lines = f.readlines()
    P0 = np.array(lines[0].split()[1:], dtype=np.float32).reshape(3, 4)
    P1 = np.array(lines[1].split()[1:], dtype=np.float32).reshape(3, 4)
    P2 = np.array(lines[2].split()[1:], dtype=np.float32).reshape(3, 4)
    P3 = np.array(lines[3].split()[1:], dtype=np.float32).reshape(3, 4)
    R0_rect = np.array(lines[4].split()[1:], dtype=np.float32).reshape(3, 3)
    Tr_velo_to_cam = np.array(lines[5].split()[1:], dtype=np.float32).reshape(3, 4)
    Tr_imu_to_velo = np.array(lines[6].split()[1:], dtype=np.float32).reshape(3, 4)
    return P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo


def kitti_box_to_lidar(label_line: str, calib: tuple) -> tuple[np.ndarray, dict[str, Any]]:
    *_, R0_rect, Tr_velo_to_cam, _ = calib
    vals = label_line.strip().split()
    height, width, length = map(float, vals[8:11])
    x, y, z = map(float, vals[11:14])
    ry = float(vals[14])

    center_cam_rect = np.array([x, y - height / 2.0, z])
    R0_rect_4 = np.eye(4, dtype=np.float32)
    R0_rect_4[:3, :3] = R0_rect
    T_velo_to_cam = np.eye(4, dtype=np.float32)
    T_velo_to_cam[:3, :4] = Tr_velo_to_cam
    T_cam_to_velo = np.linalg.inv(T_velo_to_cam)

    center_cam_rect_h = np.hstack([center_cam_rect, 1.0])
    center_velo = (T_cam_to_velo @ np.linalg.inv(R0_rect_4) @ center_cam_rect_h)[:3]

    size_x, size_y, size_z = length, width, height

    R_cam_to_velo = T_cam_to_velo[:3, :3]
    R_camrect_to_velo = R_cam_to_velo @ R0_rect.T
    forward_camrect = np.array([np.cos(ry), 0.0, np.sin(ry)])
    forward_velo = R_camrect_to_velo @ forward_camrect
    yaw_velo = float(np.arctan2(forward_velo[1], forward_velo[0]))

    obb = np.array(
        [
            float(center_velo[0]),
            float(center_velo[1]),
            float(center_velo[2]),
            float(size_x),
            float(size_y),
            float(size_z),
            -float(yaw_velo),
            np.nan,
            np.nan,
        ],
        dtype=np.float32,
    )

    return obb, {
        "label": vals[0],
        "truncation": float(vals[1]),
        "occlusion": int(vals[2]),
    }


def parse_kitti_3d_obb(label_file: Path, calib_file: Path) -> tlc.OrientedBoundingBoxes3D:
    calib = load_kitti_calib(calib_file)
    x_min, x_max, y_min, y_max, z_min, z_max = KITTI_BOUNDS
    obbs = tlc.OrientedBoundingBoxes3D.create_empty(
        x_min=x_min,
        y_min=y_min,
        z_min=z_min,
        x_max=x_max,
        y_max=y_max,
        z_max=z_max,
    )

    with open(label_file) as f:
        for line in f:
            obb, extras = kitti_box_to_lidar(line, calib)
            obbs.add_instance(
                obb=obb,
                label=KITTI_DETECTION_VALUE_MAP[extras["label"]],
                per_instance_extras={
                    "truncation": extras["truncation"],
                    "occlusion": extras["occlusion"],
                },
            )

    return obbs


# ── Virtual table creation ──


def _make_kitti_velodyne_url(bin_path: Path, component: str, num_bytes: int) -> str:
    """Build a kitti-velodyne:// URL with chunk-compatible :offset-length suffix.

    The URL format is designed so that the BulkDataAccessor can parse the trailing
    :offset-length portion as usual, then the adapter handles the rest.
    """
    abs_path = bin_path.resolve().as_posix()
    return f"kitti-velodyne://{abs_path}?component={component}:0-{num_bytes}"


def create_virtual_kitti_table(
    kitti_det_root: Path,
    max_frames: int | None = None,
    table_name: str = "KITTI_DET_virtual",
    dataset_name: str = "KITTI_DET",
    project_name: str = "3LC Tutorials - KITTI Virtual Table",
    root_url: str | None = None,
    if_exists: Literal["overwrite", "rename", "raise"] = "overwrite",
) -> tlc.Table:
    assert kitti_det_root.exists(), f"KITTI root {kitti_det_root} does not exist!"

    # Schema: geometry with bulk-data vertices + intensity, inline bounding boxes
    lidar_schema = tlc.schemas.Geometry3DSchema(
        per_vertex_schemas={
            "intensity": tlc.schemas.Float32ListSchema(),
        },
        is_bulk_data=True,
    )

    obb_schema = tlc.schemas.OrientedBoundingBoxes3DSchema(
        classes=KITTI_DETECTION_VALUE_MAP.keys(),
        per_instance_schemas={
            "occlusion": tlc.schemas.CategoricalLabelListSchema(
                {0: "fully visible", 1: "partly visible", 2: "largely occluded", 3: "unknown", -1: "unknown"},
                writable=False,
            ),
            "truncation": tlc.schemas.Float32ListSchema(writable=False),
        },
    )

    table_writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=dataset_name,
        project_name=project_name,
        root_url=root_url,
        if_exists=if_exists,
        schema={
            "lidar": lidar_schema,
            "bbs": obb_schema,
            "image_2": tlc.schemas.ImageUrlSchema(),
            "input_file": tlc.schemas.StringSchema(writable=False, default_visible=False),
        },
    )

    input_files = sorted(kitti_det_root.glob("velodyne/*.bin"))
    label_files = sorted(kitti_det_root.glob("label_2/*.txt"))
    image_files = sorted(kitti_det_root.glob("image_2/*.png"))
    calib_files = sorted(kitti_det_root.glob("calib/*.txt"))

    assert len(input_files) == len(label_files) == len(image_files) == len(calib_files), (
        "Number of input, label, image, and calib files must match"
    )
    assert len(input_files) > 0, "No input files found!"

    if max_frames is not None:
        max_frames = min(max_frames, len(input_files))
        input_files = input_files[:max_frames]
        label_files = label_files[:max_frames]
        image_files = image_files[:max_frames]
        calib_files = calib_files[:max_frames]

    for input_file, label_file, image_file, calib_file in tqdm(
        zip(input_files, label_files, image_files, calib_files),
        total=len(input_files),
        desc="Building virtual table",
    ):
        obbs = parse_kitti_3d_obb(label_file, calib_file)

        # Compute the number of points from file size (4 floats × 4 bytes per point)
        file_size = input_file.stat().st_size
        num_points = file_size // 16  # 4 × float32

        # Byte sizes for the de-interleaved components
        vertex_bytes = num_points * 3 * 4  # 3 coords × float32
        intensity_bytes = num_points * 1 * 4  # 1 value × float32

        # Create a Geometry3D in externalized mode — URLs instead of arrays
        x_min, x_max, y_min, y_max, z_min, z_max = KITTI_BOUNDS
        lidar = tlc.Geometry3D.create_empty(
            x_min=x_min,
            y_min=y_min,
            z_min=z_min,
            x_max=x_max,
            y_max=y_max,
            z_max=z_max,
        )
        lidar.add_instance(
            vertices=_make_kitti_velodyne_url(input_file, "vertices", vertex_bytes),
            per_vertex_extras={
                "intensity": _make_kitti_velodyne_url(input_file, "intensity", intensity_bytes),
            },
        )

        row = {
            "input_file": tlc.Url(input_file).to_relative().to_str(),
            "lidar": lidar,
            "image_2": image_file.resolve().as_posix(),
            "bbs": obbs,
        }

        table_writer.add_row(row)

    table = table_writer.finalize()
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a virtual KITTI 3D detection table (no data copy)")
    parser.add_argument("kitti_root", type=Path, help="Path to KITTI training directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames")
    parser.add_argument("--table-name", default="KITTI_DET_virtual", help="Table name")
    parser.add_argument("--dataset-name", default="KITTI_DET", help="Dataset name")
    parser.add_argument("--project-name", default="3LC Tutorials - KITTI Virtual Table", help="Project name")
    parser.add_argument("--if-exists", default="overwrite", choices=["overwrite", "rename", "raise"])
    args = parser.parse_args()

    # The `kitti-velodyne` adapter is registered via the `tlc.url_adapters` entry point declared in pyproject.toml.
    # If you see an "unknown scheme" error from tlc.Url, install this package (e.g. `pip install -e .`).
    table = create_virtual_kitti_table(
        kitti_det_root=args.kitti_root,
        max_frames=args.max_frames,
        table_name=args.table_name,
        dataset_name=args.dataset_name,
        project_name=args.project_name,
        if_exists=args.if_exists,
    )

    print(f"\nCreated virtual table with {len(table)} rows")
    print(f"Table URL: {table.url}")
    print("\nNo LiDAR data was copied — the table references the original .bin files")
    print("via kitti-velodyne:// URLs that are resolved at read time.")


if __name__ == "__main__":
    main()
