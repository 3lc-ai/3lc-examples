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


def rotmat_from_yaw(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


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
    tlc_project_root: str | None = None,
) -> tlc.Table:
    dataset = pandaset.DataSet(dataset_root)
    car, car_schema = load_car()

    table_writer = tlc.TableWriter(
        table_name="pandaset",
        dataset_name="pandaset-test",
        project_name="pandaset",
        column_schemas={
            "lidar": get_lidar_schema(),
            "bbs": get_bb_schema(),
            "back_camera": tlc.ImageUrlSchema(),
            "front_camera": tlc.ImageUrlSchema(),
            "front_left_camera": tlc.ImageUrlSchema(),
            "front_right_camera": tlc.ImageUrlSchema(),
            "left_camera": tlc.ImageUrlSchema(),
            "right_camera": tlc.ImageUrlSchema(),
            "car": car_schema,
        },
        root_url=tlc_project_root,
    )

    sequences = dataset.sequences(with_semseg=True)[:max_sequences]
    total_sequences = len(sequences)

    # Global progress bar with known total frames
    frames_per_seq_effective = frames_per_sequence if max_frames is None else min(frames_per_sequence, max_frames)
    total_frames = total_sequences * frames_per_seq_effective
    pbar = tqdm.tqdm(total=total_frames, desc="Processing sequences", unit="frame")

    for sequence_idx, sequence_id in enumerate(sequences):
        # Update description per sequence
        pbar.set_description(f"Seq {sequence_idx + 1}/{total_sequences} ({sequence_id})")
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

        # Define the iterator separately for readability

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
            lidar_position = lidar_pose["position"]
            lidar_heading = lidar_pose["heading"]
            pose_mat = pandaset.geometry._heading_position_to_mat(lidar_heading, lidar_position)
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
            cuboid_dict = load_cuboids(cuboids, R_inv, t_inv)

            # table_writer.add_row(
            #     {
            #         "sequence_id": sequence_id,
            #         "frame_id": frame_id,
            #         "lidar": geometry.to_row(),
            #         "bbs": cuboid_dict,
            #         "back_camera": back_camera_path.as_posix(),
            #         "front_camera": front_camera_path.as_posix(),
            #         "front_left_camera": front_left_camera_path.as_posix(),
            #         "front_right_camera": front_right_camera_path.as_posix(),
            #         "left_camera": left_camera_path.as_posix(),
            #         "right_camera": right_camera_path.as_posix(),
            #         "car": car,
            #     }
            # )

            # Advance global progress
            pbar.update(1)

        dataset.unload(sequence_id)

    pbar.close()

    # table = table_writer.finalize()
    table = None  # table writing disabled

    return table


def load_cuboids(cuboids, R_inv, t_inv):
    # cuboids = tlc.OrientedBoundingBoxes3DInstances.create_empty(*bounds, per_instance_extras_keys=["label"])
    cuboid_dict = {
        "instances": [],
        "instances_additional_data": {
            "label": [],
        },
        "x_min": bounds[0],
        "x_max": bounds[1],
        "y_min": bounds[2],
        "y_max": bounds[3],
        "z_min": bounds[4],
        "z_max": bounds[5],
    }

    for label, x, y, z, length, width, height, yaw in zip(
        cuboids["label"].values,
        cuboids["position.x"].values,
        cuboids["position.y"].values,
        cuboids["position.z"].values,
        cuboids["dimensions.x"].values,
        cuboids["dimensions.y"].values,
        cuboids["dimensions.z"].values,
        cuboids["yaw"].values,
    ):
        # world -> ego for centers, using the same transform as lidar_points_to_ego
        x, y, z = (
            R_inv[0, 0] * x + R_inv[0, 1] * y + R_inv[0, 2] * z + t_inv[0],
            R_inv[1, 0] * x + R_inv[1, 1] * y + R_inv[1, 2] * z + t_inv[1],
            R_inv[2, 0] * x + R_inv[2, 1] * y + R_inv[2, 2] * z + t_inv[2],
        )

        R_box_w = rotmat_from_yaw(float(yaw))
        R_box_e = R_inv @ R_box_w
        yaw = float(np.arctan2(R_box_e[1, 0], R_box_e[0, 0]))
        bb = {
            "center_x": x,
            "center_y": y,
            "center_z": z,
            "size_x": length,
            "size_y": width,
            "size_z": height,
            "yaw": yaw,
            "pitch": np.nan,
            "roll": np.nan,
        }
        cuboid_dict["instances"].append({"oriented_bbs_3d": [bb]})
        label_int = cuboid_classes.get(label, -1)
        if label_int == -1:
            raise ValueError(f"Label {label} not found in cuboid_classes")
        cuboid_dict["instances_additional_data"]["label"].append(label_int)
    return cuboid_dict


if __name__ == "__main__":
    TLC_PROJECT_ROOT = "D:/Data/3LC-projects"
    DATASET_ROOT = Path("D:/Data/pandaset")
    table = load_pandaset(
        dataset_root=DATASET_ROOT,
        tlc_project_root=TLC_PROJECT_ROOT,
        max_sequences=None,
        max_frames=None,
    )
    print(table)
