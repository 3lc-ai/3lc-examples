# kitti-virtual-table

Example showing how to create a 3LC table from KITTI LiDAR point clouds **without copying any data**.

A custom URL adapter reads the original `.bin` files on the fly, de-interleaves them into vertex coordinates and intensity values, and applies the KITTI alignment matrix — all at read time.

## Data preparation

This example requires the KITTI 3D Object Detection dataset. Download the following from the
[KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):

- **Velodyne point clouds** (29 GB)
- **Training labels**
- **Camera calibration matrices**

Unpack them so the directory looks like:

```
kitti/training/
├── calib/        # 000000.txt, 000001.txt, ...
├── label_2/      # 000000.txt, 000001.txt, ...
└── velodyne/     # 000000.bin, 000001.bin, ...
```

## Quick start

```bash
# Install the adapter plugin (editable mode recommended for development)
pip install -e .

# Create a virtual table from KITTI training data
create-kitti-virtual-table /path/to/kitti/training
```

After running the command, open the 3LC Dashboard to browse the table. LiDAR point clouds are resolved from the original `.bin` files whenever 3LC reads a row.

## How it works

### Custom URL scheme

The adapter registers the `kitti-velodyne` scheme. URLs encode the path to a `.bin` file and a query parameter selecting the component:

```
kitti-velodyne:///path/to/velodyne/003526.bin?component=vertices
kitti-velodyne:///path/to/velodyne/003526.bin?component=intensity
```

### Virtual table creation

The `create_table.py` script builds a table with pre-externalized `Geometry3D` objects whose vertex and intensity columns point to `kitti-velodyne://` URLs. No LiDAR data is copied — the adapter de-interleaves the original `.bin` files at read time.

### Entry-point discovery

The plugin declares an entry point in `pyproject.toml`:

```toml
[project.entry-points."tlc.url_adapters"]
kitti-velodyne = "kitti_virtual_table.adapter:KittiVelodyneUrlAdapter"
```

Installing the package is all that's needed — 3LC discovers and registers the adapter automatically.
