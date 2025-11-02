from pathlib import Path

import numpy as np
import pandaset
import transforms3d as t3d
from pandaset.geometry import _heading_position_to_mat


def _format_T(R: np.ndarray, t: np.ndarray) -> str:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return "\n".join(" ".join(f"{v: .6f}" for v in row) for row in T)


root = Path("D:/Data/pandaset")
dataset = pandaset.DataSet(root)

# Pick a sequence and camera to analyze
sequence_id = "001"
camera_name = "back_camera"  # change to front_camera etc. to test others

seq = dataset[sequence_id]
seq.lidar.set_sensor(0)
seq.load_lidar()
seq.load_camera()

camera_poses = seq.camera[camera_name].poses
lidar_poses = seq.lidar.poses

# n = min(len(camera_poses), len(lidar_poses))
n = 10

R_list = []
t_list = []

for i in range(n):
    cam_pose = camera_poses[i]
    lidar_pose = lidar_poses[i]

    world_T_cam = _heading_position_to_mat(cam_pose["heading"], cam_pose["position"])  # 4x4
    world_T_lidar = _heading_position_to_mat(lidar_pose["heading"], lidar_pose["position"])  # 4x4

    cam_T_world = np.linalg.inv(world_T_cam)
    cam_T_lidar = cam_T_world @ world_T_lidar

    R_rel = cam_T_lidar[:3, :3]
    t_rel = cam_T_lidar[:3, 3]

    R_list.append(R_rel)
    t_list.append(t_rel)

R_arr = np.stack(R_list, axis=0)  # [n,3,3]
t_arr = np.stack(t_list, axis=0)  # [n,3]

print(f"Sequence: {sequence_id}, Camera: {camera_name}, Frames: {n}")

# Baseline (frame 0) as candidate constant extrinsic
R0 = R_arr[0]
t0 = t_arr[0]
q0 = t3d.quaternions.mat2quat(R0)
print("\nBaseline lidar->camera (frame 0):")
print(f"t0 (m): {t0}")
print(f"q0 [w x y z]: {q0}")
print("T_cam_lidar (frame 0):\n" + _format_T(R0, t0))

print("\nPer-frame lidar->camera and delta to baseline:")
for i in range(n):
    R = R_arr[i]
    t = t_arr[i]
    q = t3d.quaternions.mat2quat(R)

    # Rotation delta to baseline
    R_err = R0.T @ R
    angle_deg = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))

    dt = t - t0
    dt_norm = float(np.linalg.norm(dt))

    print(
        f"i={i:03d} | t (m): {t} | dt: {dt} | |dt|: {dt_norm:.6f} | q [w x y z]: {q} | dR_to_baseline (deg): {angle_deg:.6f}"
    )
