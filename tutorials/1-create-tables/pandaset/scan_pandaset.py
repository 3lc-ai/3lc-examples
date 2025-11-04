from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, dict, list

import numpy as np
import pandaset
import transforms3d as t3d
from pandaset.geometry import _heading_position_to_mat


@dataclass
class Bounds3D:
    x_min: float = float("inf")
    x_max: float = float("-inf")
    y_min: float = float("inf")
    y_max: float = float("-inf")
    z_min: float = float("inf")
    z_max: float = float("-inf")

    def update_with_xyz(self, xyz: np.ndarray) -> None:
        if xyz.size == 0:
            return
        # xyz shape: [N,3]
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        self.x_min = min(self.x_min, float(np.min(x)))
        self.x_max = max(self.x_max, float(np.max(x)))
        self.y_min = min(self.y_min, float(np.min(y)))
        self.y_max = max(self.y_max, float(np.max(y)))
        self.z_min = min(self.z_min, float(np.min(z)))
        self.z_max = max(self.z_max, float(np.max(z)))

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "x": {"min": self.x_min, "max": self.x_max},
            "y": {"min": self.y_min, "max": self.y_max},
            "z": {"min": self.z_min, "max": self.z_max},
        }


@dataclass
class ExtrinsicStats:
    count: int
    t_mean: list[float]
    t_std: list[float]
    q_mean_wxyz: list[float]
    rot_mean_err_deg: float
    rot_std_deg: float
    trans_err_norm_mean: float
    trans_err_norm_std: float
    T_cam_from_lidar: list[list[float]]  # 4x4 row-major

    def pretty(self) -> str:
        return (
            f"count={self.count}\n"
            f"t_mean (m)={np.array(self.t_mean)}\n"
            f"t_std (m)={np.array(self.t_std)}\n"
            f"q_mean [w x y z]={np.array(self.q_mean_wxyz)}\n"
            f"rot_err_mean/std (deg)={self.rot_mean_err_deg:.6f}/{self.rot_std_deg:.6f}\n"
            f"trans_err_norm mean/std (m)={self.trans_err_norm_mean:.6f}/{self.trans_err_norm_std:.6f}\n"
            f"T_cam_from_lidar (mean):\n"
            + "\n".join(" ".join(f"{v: .6f}" for v in row) for row in self.T_cam_from_lidar)
        )


def _average_quaternions_wxyz(quats: np.ndarray) -> np.ndarray:
    """
    Average unit quaternions by sign-aligning to the first and normalizing the sum.
    quats: [N,4] with [w,x,y,z]
    Returns: [4] mean quaternion [w,x,y,z]
    """
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError("quats must be [N,4]")
    if len(quats) == 0:
        raise ValueError("cannot average zero quaternions")
    q0 = quats[0]
    acc = np.zeros(4, dtype=np.float64)
    for q in quats:
        if np.dot(q0, q) < 0.0:
            acc += -q
        else:
            acc += q
    norm = np.linalg.norm(acc)
    if norm == 0.0:
        # fallback to first
        return q0.astype(np.float64, copy=True)
    return (acc / norm).astype(np.float64, copy=False)


def _extrinsic_stats(R_list: list[np.ndarray], t_list: list[np.ndarray]) -> ExtrinsicStats:
    """
    Compute mean and spread stats for extrinsics. Rotation is averaged in quaternion space with
    sign disambiguation; spread is reported as angular error stats to the mean.
    Translation stats are per-axis mean/std and norm error stats to the mean.
    """
    if len(R_list) != len(t_list):
        raise ValueError("Rotation and translation lists must have same length")
    n = len(R_list)
    if n == 0:
        raise ValueError("No samples for extrinsic stats")

    quats = np.stack([t3d.quaternions.mat2quat(R) for R in R_list], axis=0)  # [N,4] wxyz
    q_mean = _average_quaternions_wxyz(quats)
    R_mean = t3d.quaternions.quat2mat(q_mean)  # 3x3

    angles_deg: list[float] = []
    for R in R_list:
        R_err = R_mean.T @ R
        angle = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
        angles_deg.append(float(angle))
    rot_mean_err_deg = float(np.mean(angles_deg))
    rot_std_deg = float(np.std(angles_deg))

    t_arr = np.stack(t_list, axis=0)  # [N,3]
    t_mean = np.mean(t_arr, axis=0)
    t_std = np.std(t_arr, axis=0)
    t_err = t_arr - t_mean.reshape(1, 3)
    t_err_norm = np.linalg.norm(t_err, axis=1)
    trans_err_norm_mean = float(np.mean(t_err_norm))
    trans_err_norm_std = float(np.std(t_err_norm))

    T_mean = np.eye(4, dtype=float)
    T_mean[:3, :3] = R_mean
    T_mean[:3, 3] = t_mean

    return ExtrinsicStats(
        count=n,
        t_mean=t_mean.tolist(),
        t_std=t_std.tolist(),
        q_mean_wxyz=q_mean.tolist(),
        rot_mean_err_deg=rot_mean_err_deg,
        rot_std_deg=rot_std_deg,
        trans_err_norm_mean=trans_err_norm_mean,
        trans_err_norm_std=trans_err_norm_std,
        T_cam_from_lidar=T_mean.tolist(),
    )


def scan_pandaset(
    dataset_root: Path,
    max_sequences: int | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    dataset = pandaset.DataSet(dataset_root)
    sequence_ids = dataset.sequences(with_semseg=True)
    if max_sequences is not None:
        sequence_ids = sequence_ids[:max_sequences]

    bounds = Bounds3D()
    unique_cuboid_labels: set[str] = set()

    # camera_name -> sensor_index -> {"R": [..], "t": [..]}
    extrinsics_accum: dict[str, dict[int, dict[str, list[np.ndarray]]]] = {}

    for seq_id in sequence_ids:
        seq = dataset[seq_id]
        seq.load_lidar()
        seq.load_camera()
        seq.load_cuboids()

        # Lidar bounds (world coordinates) across both sensors if present
        for sensor_idx in (0, 1):
            try:
                seq.lidar.set_sensor(sensor_idx)
            except Exception:
                continue
            pc_frames = seq.lidar[:max_frames]
            for pc in pc_frames:
                xyz = pc.values[:, :3].astype(np.float64, copy=False)
                bounds.update_with_xyz(xyz)

        # Unique cuboid labels
        cub_frames = seq.cuboids[:max_frames]
        for df in cub_frames:
            if df is None or "label" not in df or len(df) == 0:
                continue
            labels = df["label"].values
            if labels.size == 0:
                continue
            for lbl in labels:
                if isinstance(lbl, str) and len(lbl) > 0:
                    unique_cuboid_labels.add(lbl)

        # Extrinsics per camera relative to lidar sensors
        camera_names = list(seq.camera.keys())
        lidar_poses_by_sensor: dict[int, list[dict]] = {}
        for sensor_idx in (0, 1):
            try:
                seq.lidar.set_sensor(sensor_idx)
            except Exception:
                continue
            lidar_poses_by_sensor[sensor_idx] = list(seq.lidar.poses[:max_frames])

        for cam_name in camera_names:
            cam_poses = list(seq.camera[cam_name].poses[:max_frames])
            for sensor_idx, lidar_poses in lidar_poses_by_sensor.items():
                n = min(len(cam_poses), len(lidar_poses))
                if n == 0:
                    continue
                acc = extrinsics_accum.setdefault(cam_name, {}).setdefault(sensor_idx, {"R": [], "t": []})
                for i in range(n):
                    cam_pose = cam_poses[i]
                    lidar_pose = lidar_poses[i]
                    world_T_cam = _heading_position_to_mat(cam_pose["heading"], cam_pose["position"])  # 4x4
                    world_T_lidar = _heading_position_to_mat(lidar_pose["heading"], lidar_pose["position"])  # 4x4
                    cam_T_world = np.linalg.inv(world_T_cam)
                    cam_T_lidar = cam_T_world @ world_T_lidar  # camera_from_lidar
                    acc["R"].append(cam_T_lidar[:3, :3])
                    acc["t"].append(cam_T_lidar[:3, 3])

        dataset.unload(seq_id)

    # Finalize extrinsic stats
    extrinsics_stats: dict[str, dict[str, Any]] = {}
    for cam_name, by_sensor in extrinsics_accum.items():
        extrinsics_stats[cam_name] = {}
        for sensor_idx, parts in by_sensor.items():
            stats = _extrinsic_stats(parts["R"], parts["t"]) if len(parts["R"]) > 0 else None
            if stats is not None:
                extrinsics_stats[cam_name][f"lidar_{sensor_idx}"] = asdict(stats)

    result: dict[str, Any] = {
        "bounds_world": bounds.to_dict(),
        "unique_cuboid_labels": sorted(unique_cuboid_labels),
        "extrinsics_cam_from_lidar": extrinsics_stats,
    }
    return result


def _print_report(summary: dict[str, Any]) -> None:
    print("\nGlobal LiDAR bounds (world coords):")
    b = summary["bounds_world"]
    print(f"x: [{b['x']['min']:.3f}, {b['x']['max']:.3f}]")
    print(f"y: [{b['y']['min']:.3f}, {b['y']['max']:.3f}]")
    print(f"z: [{b['z']['min']:.3f}, {b['z']['max']:.3f}]")

    print("\nUnique cuboid labels (sorted):")
    for lbl in summary["unique_cuboid_labels"]:
        print(lbl)

    print("\nCamera extrinsics (mean) per camera and lidar sensor:")
    extr = summary["extrinsics_cam_from_lidar"]
    for cam_name, sensors in extr.items():
        print(f"\nCamera: {cam_name}")
        for sensor_key, stats in sensors.items():
            print(f"  Relative to: {sensor_key}")
            # Pretty-print
            T = np.array(stats["T_cam_from_lidar"])  # 4x4
            t_mean = np.array(stats["t_mean"])  # 3
            q_mean = np.array(stats["q_mean_wxyz"])  # 4
            print(f"    count: {stats['count']}")
            print(f"    t_mean (m): {t_mean}")
            print(f"    t_std (m): {np.array(stats['t_std'])}")
            print(f"    q_mean [w x y z]: {q_mean}")
            print(f"    rot_err_mean/std (deg): {stats['rot_mean_err_deg']:.6f}/{stats['rot_std_deg']:.6f}")
            print(
                f"    trans_err_norm mean/std (m): {stats['trans_err_norm_mean']:.6f}/{stats['trans_err_norm_std']:.6f}"
            )
            print("    T_cam_from_lidar (mean):")
            for r in range(4):
                print("     ", " ".join(f"{v: .6f}" for v in T[r]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Pandaset to compute: (1) global LiDAR bounds across both sensors, "
            "(2) unique cuboid labels observed, (3) per-camera extrinsics (lidar->camera) "
            "aggregated across frames and sequences with mean/std stats."
        )
    )
    parser.add_argument("--root", type=str, default="D:/Data/pandaset", help="Path to Pandaset root directory")
    parser.add_argument("--max-sequences", type=int, default=10, help="Optional limit on number of sequences")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit on frames per sequence")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "pandaset_scan_summary.json"),
        help="Where to write the JSON summary",
    )

    args = parser.parse_args()
    root = Path(args.root)

    summary = scan_pandaset(root, max_sequences=args.max_sequences, max_frames=args.max_frames)

    # Save JSON
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console report
    _print_report(summary)


if __name__ == "__main__":
    main()
