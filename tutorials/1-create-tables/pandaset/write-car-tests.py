import numpy as np
import tlc

bounds = (-10, 10, -10, 10, 0, 20)


def load_car(transform) -> tuple[dict, tlc.Schema]:
    car_obj_path = tlc.Url("<TEST_DATA>/data/car/NormalCar2.obj").to_absolute().to_str()
    scale = 1.0

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


if __name__ == "__main__":
    # car 1. natural pose (front +z, wheels -y)
    transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    car_1_geometry, car_1_schema = load_car(transform)

    # car 2. front +x, wheels -z
    R2 = np.array(
        [
            [0.0, 0.0, 1.0],  # e_x_s -> e_y_t, e_y_s -> e_z_t, e_z_s -> e_x_t
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    transform2 = np.array(
        [
            [R2[0, 0], R2[0, 1], R2[0, 2], 0.0],
            [R2[1, 0], R2[1, 1], R2[1, 2], 0.0],
            [R2[2, 0], R2[2, 1], R2[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    car_2_geometry, _ = load_car(transform2)

    table_writer = tlc.TableWriter(
        table_name="car_tests",
        project_name="car_tests",
        dataset_name="car_tests",
        column_schemas={
            "car": car_1_schema,
        },
    )
    table_writer.add_row({"car": car_2_geometry})

    table = table_writer.finalize()
    print(table)
