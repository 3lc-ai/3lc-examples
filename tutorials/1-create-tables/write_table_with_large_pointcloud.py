import numpy as np
import tlc

# Define the bounds of the point cloud
bounds = (0, 1, 0, 1, 0, 1)

# Create a point cloud
num_rows = 2
num_points = 500_000

# Define a schema for the column containing the point cloud
schema = tlc.Geometry3DSchema(
    include_3d_vertices=True,
    is_bulk_data=True,
)

# Create a TableWriter for writing the point cloud
writer = tlc.TableWriter(
    table_name="point_cloud",
    project_name="point_cloud_project",
    column_schemas={"points": schema},
)

for _ in range(num_rows):
    instances = tlc.Geometry3DInstances.create_empty(*bounds)
    instances.add_instance(np.random.rand(num_points, 3).astype(np.float32))
    writer.add_row({"points": instances.to_row()})

table = writer.finalize()

# The bulk data is stored in the Table's bulk_data_url property.
print(table.bulk_data_url)
# Url('relative://../../bulk_data/samples/point_cloud')
