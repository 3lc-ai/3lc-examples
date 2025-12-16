# Image Normalization Utility

This utility helps normalize images in the `tutorials/images/` directory by:

- Resizing images to a maximum dimension while preserving aspect ratio
- Re-compressing images with appropriate quality settings based on file type
- Operating in dry-run mode to preview changes before applying them
- Being idempotent (safe to run multiple times)

## Features

- **Max dimension constraint**: Resize images that exceed a maximum width or height
- **Smart compression**: Apply appropriate compression settings for JPEG and PNG files
- **Dry-run mode**: Preview what changes would be made without modifying files
- **Idempotent**: Running multiple times produces the same result
- **Backup safety**: Creates backups before processing and restores if processing fails

## Requirements

The utility requires the following dependencies (already included in the project):

- `opencv-python>=4.10.0.84` (for image processing)
- `numpy` (dependency of opencv)

## Usage

### Quick Start (Recommended)

Use the convenience script to process images in `tutorials/images/`:

```bash
# Dry run (preview changes without modifying files)
python3 normalize_images.py

# Actually process the images
python3 normalize_images.py --process
```

### Advanced Usage

Use the CLI module for more control:

```bash
# Dry run with default settings
python3 -m utils tutorials/images --dry-run

# Process with custom settings
python3 -m utils tutorials/images --max-dimension 1000 --jpeg-quality 80

# Get help
python3 -m utils --help
```

### Programmatic Usage

```python
from utils.image_normalizer import ImageNormalizer, NormalizationSettings
from pathlib import Path

# Create custom settings
settings = NormalizationSettings(
    max_dimension=1000,    # Maximum width or height
    jpeg_quality=85,       # JPEG quality (1-100)
    png_compression=6      # PNG compression (0-9)
)

# Create normalizer and process directory
normalizer = ImageNormalizer(settings)
results = normalizer.normalize_directory(
    Path("tutorials/images"), 
    dry_run=True  # Set to False to actually process
)

# Print summary
normalizer.print_summary(results, dry_run=True)
```

## Default Settings

- **Max dimension**: 1000 pixels (optimized for notebook titles while keeping good quality)
- **JPEG quality**: 85 (high quality with reasonable compression)
- **PNG compression**: 6 (good balance of size and speed)

## How It Works

### Idempotency

The utility is designed to be idempotent, meaning you can run it multiple times safely:

1. **Backup mechanism**: Before processing, creates a `.backup` file
2. **Skip processed files**: If a backup exists, the file is considered already processed
3. **Quality checks**: Only processes if the new file would be smaller or similar size

### Processing Logic

1. **Scan directory**: Finds all `.jpg`, `.jpeg`, and `.png` files recursively
2. **Check if processing needed**: 
   - Image exceeds max dimension, OR
   - File size is over 500KB (indicating potential for compression)
3. **Resize if needed**: Uses high-quality Lanczos interpolation
4. **Recompress**: Applies appropriate compression settings based on file type
5. **Validate result**: Ensures processed file isn't significantly larger than original

### Dry Run Mode

Dry run mode shows you exactly what would happen without modifying any files:

- Shows original vs. new dimensions
- Estimates file size savings
- Lists which files would be processed
- Safe to run anytime to preview changes

## Example Output

```
DRY RUN: Normalizing images in tutorials/images/
Settings: max_dimension=1000, jpeg_quality=85, png_compression=6

=== IMAGE NORMALIZATION DRY RUN SUMMARY ===
Total images: 74
Processed: 52
Skipped: 22
Errors: 0
Original total size: 52.3 MB
Estimated new size: 24.8 MB
Estimated savings: 27.5 MB

Images that would be processed:
  add-classification-metrics.png: 3840x2160 → 1000x563, 898.6KB → 51.2KB
  add-embeddings.jpg: 3840x2160 → 1000x563, 906.9KB → 52.1KB
  ...
```

## Safety Features

- **Automatic backups**: Original files are backed up before processing
- **Rollback on failure**: If processing fails, original file is restored
- **Size validation**: Rejects processed images that are significantly larger
- **Error handling**: Graceful handling of corrupted or unsupported images

## Troubleshooting

### Module not found errors

Make sure you're in the project root directory and have the dependencies installed:

```bash
# If using pip
pip install opencv-python

# If using uv (recommended for this project)
uv sync
```

### Permission errors

Ensure you have write permissions to the `tutorials/images/` directory.

### Large file warnings

If you see warnings about files being larger after processing, this is normal behavior. The utility will automatically revert to the original file in such cases.
