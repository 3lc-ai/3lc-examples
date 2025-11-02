# uv add --group pandaset git+https://github.com/scaleapi/pandaset-devkit.git@master#subdirectory=python

# pip install git+https://github.com/scaleapi/pandaset-devkit.git

from pathlib import Path

import numpy as np
import pandaset
import tlc
import tqdm

# Each pandaset sequence contains 80 frames (8 seconds at 10 fps)
frames_per_sequence = 80
bounds = tlc.GeometryHelper.create_isotropic_bounds_3d(-175, 175, -175, 175, -10, 20)

semseg_classes = {
    1: "Smoke",
    2: "Exhaust",
    3: "Spray or rain",
    4: "Reflection",
    5: "Vegetation",
    6: "Ground",
    7: "Road",
    8: "Lane Line Marking",
    9: "Stop Line Marking",
    10: "Other Road Marking",
    11: "Sidewalk",
    12: "Driveway",
    13: "Car",
    14: "Pickup Truck",
    15: "Medium-sized Truck",
    16: "Semi-truck",
    17: "Towed Object",
    18: "Motorcycle",
    19: "Other Vehicle - Construction Vehicle",
    20: "Other Vehicle - Uncommon",
    21: "Other Vehicle - Pedicab",
    22: "Emergency Vehicle",
    23: "Bus",
    24: "Personal Mobility Device",
    25: "Motorized Scooter",
    26: "Bicycle",
    27: "Train",
    28: "Trolley",
    29: "Tram / Subway",
    30: "Pedestrian",
    31: "Pedestrian with Object",
    32: "Animals - Bird",
    33: "Animals - Other",
    34: "Pylons",
    35: "Road Barriers",
    36: "Signs",
    37: "Cones",
    38: "Construction Signs",
    39: "Temporary Construction Barriers",
    40: "Rolling Containers",
    41: "Building",
    42: "Other Static Object",
}

cuboid_classes = {
    "Animals - Bird": 0,
    "Animals - Other": 1,
    "Bicycle": 2,
    "Bus": 3,
    "Car": 4,
    "Cones": 5,
    "Construction Signs": 6,
    "Emergency Vehicle": 7,
    "Medium-sized Truck": 8,
    "Motorcycle": 10,
    "Motorized Scooter": 11,
    "Other Vehicle - Construction Vehicle": 12,
    "Other Vehicle - Pedicab": 13,
    "Other Vehicle - Uncommon": 14,
    "Pedestrian": 15,
    "Pedestrian with Object": 16,
    "Personal Mobility Device": 17,
    "Pickup Truck": 18,
    "Pylons": 19,
    "Road Barriers": 20,
    "Rolling Containers": 21,
    "Semi-truck": 22,
    "Signs": 23,
    "Temporary Construction Barriers": 24,
    "Towed Object": 25,
    "Train": 26,
    "Tram / Subway": 27,
}


def get_lidar_schema() -> tlc.Schema:
    schema = tlc.Geometry3DSchema(
        include_3d_vertices=True,
        is_bulk_data=True,
        per_vertex_schemas={
            "intensity": tlc.Float32ListSchema(),
            "distance": tlc.Float32ListSchema(),
            "semseg": tlc.CategoricalLabelListSchema(semseg_classes),
        },
    )

    return schema


def get_bb_schema() -> tlc.Schema:
    schema = tlc.OrientedBoundingBoxes3DSchema(
        classes=cuboid_classes.keys(),
    )

    return schema


def load_intrinsics(dataset: pandaset.DataSet) -> dict[str, dict[str, float]]:
    sequence = dataset.sequences(with_semseg=True)[0]
    dataset[sequence].load_camera()
    return {
        name: {
            "fx": cam.intrinsics.fx,
            "fy": cam.intrinsics.fy,
            "cx": cam.intrinsics.cx,
            "cy": cam.intrinsics.cy,
        }
        for name, cam in dataset[sequence].camera.items()
    }


def load_car() -> tuple[dict, tlc.Schema]:
    car_obj_path = tlc.Url("<TEST_DATA>/data/car/NormalCar2.obj").to_absolute().to_str()
    scale = 1.25
    transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # x' = x
            [0.0, 0.0, 1.0, 0.0],  # y' = -z (rotate 90 deg around x)
            [0.0, 1.0, 0.0, 0.0],  # z' = y
            [0.0, 0.0, 0.0, 1.0],  # homogeneous row
        ],
        dtype=np.float32,
    )
    car_geometry = tlc.GeometryHelper.load_obj_geometry(car_obj_path, scale, transform, bounds)

    car_schema = tlc.Geometry3DSchema(
        include_3d_vertices=True,
        include_triangles=True,
        per_triangle_schemas={
            "red": tlc.Float32ListSchema(),
            "green": tlc.Float32ListSchema(),
            "blue": tlc.Float32ListSchema(),
        },
        is_bulk_data=True,
    )
    return car_geometry.to_row(), car_schema


def load_pandaset(
    dataset_root: Path,
    max_sequences: int | None = None,
    max_frames: int | None = None,
    table_name: str = "pandaset",
    dataset_name: str = "pandaset",
    project_name: str = "pandaset",
    tlc_project_root: str | None = None,
) -> tlc.Table:
    dataset = pandaset.DataSet(dataset_root)
    car, car_schema = load_car()
    intrinsics = load_intrinsics(dataset)

    table_writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=dataset_name,
        project_name=project_name,
        column_schemas={
            "lidar": get_lidar_schema(),
            "bbs": get_bb_schema(),
            "back_camera": tlc.ImageUrlSchema(metadata={"intrinsics": intrinsics["back_camera"]}),
            "front_camera": tlc.ImageUrlSchema(metadata={"intrinsics": intrinsics["front_camera"]}),
            "front_left_camera": tlc.ImageUrlSchema(metadata={"intrinsics": intrinsics["front_left_camera"]}),
            "front_right_camera": tlc.ImageUrlSchema(metadata={"intrinsics": intrinsics["front_right_camera"]}),
            "left_camera": tlc.ImageUrlSchema(metadata={"intrinsics": intrinsics["left_camera"]}),
            "right_camera": tlc.ImageUrlSchema(metadata={"intrinsics": intrinsics["right_camera"]}),
            "car": car_schema,
        },
        root_url=tlc_project_root,
    )

    sequences = dataset.sequences(with_semseg=True)[:max_sequences]
    total_sequences = len(sequences)

    # Progress bar uses sequences as the main unit for stable ETA
    frames_per_seq_effective = frames_per_sequence if max_frames is None else min(frames_per_sequence, max_frames)
    pbar = tqdm.tqdm(total=total_sequences, desc="Sequences", unit="seq")

    for sequence_idx, sequence_id in enumerate(sequences):
        # Update description per sequence and show loading state
        pbar.set_description(f"Seq {sequence_idx + 1}/{total_sequences} ({sequence_id})")
        pbar.set_postfix_str(f"loading… 0/{frames_per_seq_effective} frames")
        sequence = dataset[sequence_id]
        sequence.lidar.set_sensor(0)
        sequence.load_lidar()
        sequence.load_cuboids()
        sequence.load_semseg()
        sequence.load_camera()

        # LiDAR 0 (mechanical 360° LiDAR)
        pc_all = sequence.lidar[:max_frames]
        lidar_poses_all = sequence.lidar.poses[:max_frames]

        # Semantic Segmentation
        semseg_all = sequence.semseg[:max_frames]

        # Cuboids
        cuboids_all = sequence.cuboids[:max_frames]

        # Cameras
        back_camera_all = list(Path(sequence.camera["back_camera"]._directory).glob("*.jpg"))[:max_frames]
        front_camera_all = list(Path(sequence.camera["front_camera"]._directory).glob("*.jpg"))[:max_frames]
        front_left_camera_all = list(Path(sequence.camera["front_left_camera"]._directory).glob("*.jpg"))[:max_frames]
        front_right_camera_all = list(Path(sequence.camera["front_right_camera"]._directory).glob("*.jpg"))[:max_frames]
        left_camera_all = list(Path(sequence.camera["left_camera"]._directory).glob("*.jpg"))[:max_frames]
        right_camera_all = list(Path(sequence.camera["right_camera"]._directory).glob("*.jpg"))[:max_frames]

        frame_iter = zip(
            pc_all,
            lidar_poses_all,
            semseg_all,
            cuboids_all,
            back_camera_all,
            front_camera_all,
            front_left_camera_all,
            front_right_camera_all,
            left_camera_all,
            right_camera_all,
        )
        frames_total = len(pc_all)
        pbar.set_postfix_str(f"frames 0/{frames_total}")

        for frame_id, (
            pc,
            lidar_pose,
            semseg,
            cuboids,
            back_camera_path,
            front_camera_path,
            front_left_camera_path,
            front_right_camera_path,
            left_camera_path,
            right_camera_path,
        ) in enumerate(frame_iter):
            # Create world to ego transform from position and heading
            pose_mat = pandaset.geometry._heading_position_to_mat(lidar_pose["heading"], lidar_pose["position"])
            T_inv = np.linalg.inv(pose_mat)
            R_inv = T_inv[:3, :3]  # 3x3 world to ego rotation matrix
            t_inv = T_inv[:3, 3]  # 3x1 world to ego translation vector

            # Transform LiDAR points from world to ego coordinates
            verts = pc.values[:, :3].astype(np.float32, copy=False)
            verts = (R_inv @ verts.T + t_inv.reshape(3, 1)).T.astype(np.float32, copy=False)

            # Extract intensity, distance, and semantic segmentation values (per-vertex)
            intensities = pc.values[:, 3].astype(np.float32, copy=False)
            distances = pc.values[:, 5].astype(np.float32, copy=False)
            semseg_values = semseg.values.astype(np.int32, copy=False).reshape(-1)[: len(verts)]

            # Create a new geometry object to store the transformed LiDAR points
            geometry = tlc.Geometry3DInstances.create_empty(
                *bounds,
                per_vertex_extras_keys=["intensity", "distance", "semseg"],
            )
            geometry.add_instance(
                verts,
                per_vertex_extras={
                    "intensity": intensities,
                    "distance": distances,
                    "semseg": semseg_values,
                },
            )

            # Transform cuboids from world to ego coordinates and prepare for table writing
            obbs = load_cuboids(cuboids, R_inv, t_inv)

            table_writer.add_row(
                {
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "lidar": geometry.to_row(),
                    "bbs": obbs.to_row(),
                    "back_camera": back_camera_path.as_posix(),
                    "front_camera": front_camera_path.as_posix(),
                    "front_left_camera": front_left_camera_path.as_posix(),
                    "front_right_camera": front_right_camera_path.as_posix(),
                    "left_camera": left_camera_path.as_posix(),
                    "right_camera": right_camera_path.as_posix(),
                    "car": car,
                }
            )

            # Update frame progress within the current sequence (does not advance the bar)
            pbar.set_postfix_str(f"frames {frame_id + 1}/{frames_total}")

        dataset.unload(sequence_id)
        # Advance the bar by one completed sequence
        pbar.update(1)

    pbar.close()

    table = table_writer.finalize()

    return table


def load_cuboids(cuboids, R_inv, t_inv) -> tlc.OBB3DInstances:
    # Vectorized world->ego transform for centers and yaw
    labels_int = [cuboid_classes.get(str(lbl)) for lbl in cuboids["label"].values]

    centers_world = np.stack(
        [cuboids["position.x"].values, cuboids["position.y"].values, cuboids["position.z"].values],
        axis=1,
    )
    sizes = np.stack(
        [cuboids["dimensions.x"].values, cuboids["dimensions.y"].values, cuboids["dimensions.z"].values],
        axis=1,
    )
    yaw_world = cuboids["yaw"].values.astype(np.float32, copy=False)

    # Apply world->ego: X_e = R_inv * X_w + t_inv (vectorized as row-vectors)
    centers_ego = centers_world @ R_inv.T + t_inv.reshape(1, 3)

    # Yaw in ego: yaw_e = yaw_w + yaw(R_inv)
    yaw_offset = float(np.arctan2(R_inv[1, 0], R_inv[0, 0]))
    yaw_ego = yaw_world + yaw_offset

    obbs = tlc.OBB3DInstances.create_empty(
        x_min=bounds[0],
        x_max=bounds[1],
        y_min=bounds[2],
        y_max=bounds[3],
        z_min=bounds[4],
        z_max=bounds[5],
        include_instance_labels=True,
    )

    # Pack dictionaries
    for (cx, cy, cz), (sx, sy, sz), yaw_val, label_val in zip(centers_ego, sizes, yaw_ego, labels_int):
        obbs.add_instance(
            obb=np.array([cx, cy, cz, sx, sy, sz, yaw_val, np.nan, np.nan]),
            label=label_val,
        )

    return obbs


if __name__ == "__main__":
    TLC_PROJECT_ROOT = "D:/3LC-projects"
    DATASET_ROOT = Path("D:/Data/pandaset")
    table = load_pandaset(
        dataset_root=DATASET_ROOT,
        tlc_project_root=TLC_PROJECT_ROOT,
        max_sequences=1,
        max_frames=10,
        table_name="pandaset",
        dataset_name="pandaset",
        project_name="pandaset",
    )
    print(table)
