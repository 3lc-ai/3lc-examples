# nifti-virtual-table

Example showing how to create a 3LC table from BraTS2020 NIfTI brain MRI volumes **without copying any images**.

A custom URL adapter extracts individual 2D axial slices from uncompressed `.nii` files on the fly, normalizes them to uint8, and returns PNG bytes — all without loading the full 3D volume.

## Quick start

```bash
# Install the adapter plugin (editable mode recommended for development)
pip install -e .

# Create a virtual table from BraTS2020 training data
create-nifti-virtual-table /path/to/BraTS2020_TrainingData
```

Options:

```bash
create-nifti-virtual-table /path/to/data --max-subjects 10 --skip-empty --modalities flair t1ce
```

After running the command, open the 3LC Dashboard to browse the table. Slice images are rendered from the original `.nii` files whenever 3LC reads a row.

## How it works

### Custom URL scheme

The adapter registers the `nifti-slice` scheme. All parameters needed for raw byte access are encoded in the URL query string:

```
nifti-slice:///path/to/volume.nii?z=77&dtype=int16&offset=2880&w=240&h=240&vmax=386
```

The adapter seeks directly to the slice bytes, reads only what is needed, and encodes the result as a grayscale PNG.

### Virtual table creation

The `create_table.py` script scans NIfTI volumes, computes per-volume metadata (data offset, dimensions, intensity range), and creates a table where each row is one axial slice. Segmentation masks are RLE-encoded and stored inline in parquet, enabling global IoU computation in the Dashboard.

### Entry-point discovery

The plugin declares an entry point in `pyproject.toml`:

```toml
[project.entry-points."tlc.url_adapters"]
nifti-slice = "nifti_virtual_table.adapter:NiftiSliceUrlAdapter"
```

Installing the package is all that's needed — 3LC discovers and registers the adapter automatically.
