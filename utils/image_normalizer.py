"""
Image normalization utility for ensuring consistent image sizes and compression.

This module provides functionality to normalize images in a directory by:
- Resizing images to a maximum dimension while preserving aspect ratio
- Re-compressing images with appropriate quality settings
- Operating in dry-run mode to preview changes
- Being idempotent (safe to run multiple times)
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


@dataclass
class ImageProcessingResult:
    """Result of processing a single image."""

    path: Path
    original_size: tuple[int, int]  # (width, height)
    new_size: tuple[int, int] | None  # (width, height) or None if not changed
    original_file_size: int  # bytes
    new_file_size: int | None  # bytes or None if not changed
    processed: bool
    error: str | None = None


@dataclass
class NormalizationSettings:
    """Settings for image normalization."""

    max_dimension: int = 1280  # Maximum width or height (good for notebook titles + galleries)
    jpeg_quality: int = 95  # JPEG compression quality (1-100)
    png_compression: int = 6  # PNG compression level (0-9)


class ImageNormalizer:
    """
    Image normalization utility that ensures consistent sizing and compression.

    Features:
    - Resizes images to max dimension while preserving aspect ratio
    - Re-compresses with appropriate quality settings
    - Dry-run mode for previewing changes
    - Idempotent operation
    - Supports JPEG and PNG formats
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, settings: NormalizationSettings | None = None):
        """Initialize the image normalizer.

        Args:
            settings: Normalization settings. Uses defaults if None.
        """
        self.settings = settings or NormalizationSettings()

    def _get_image_hash(self, image_path: Path) -> str:
        """Get a hash of the image content for idempotency checking."""
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _needs_processing(self, image_path: Path) -> bool:
        """Check if an image needs processing based on size and quality."""
        try:
            # Read image to get dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                return False

            height, width = img.shape[:2]
            max_dim = max(width, height)

            # Check if image exceeds max dimension
            if max_dim > self.settings.max_dimension:
                return True

            # Check file size - if it's very large, we might benefit from recompression
            file_size = image_path.stat().st_size
            return file_size > 500 * 1024

        except Exception as e:
            logger.warning(f"Error checking if {image_path} needs processing: {e}")
            return False

    def _calculate_new_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Calculate new dimensions while preserving aspect ratio."""
        max_dim = max(width, height)

        if max_dim <= self.settings.max_dimension:
            return width, height

        # Calculate scaling factor
        scale = self.settings.max_dimension / max_dim

        new_width = int(width * scale)
        new_height = int(height * scale)

        return new_width, new_height

    def _process_image(self, image_path: Path, dry_run: bool = False) -> ImageProcessingResult:
        """Process a single image.

        Args:
            image_path: Path to the image file
            dry_run: If True, only calculate what would be done without modifying files

        Returns:
            ImageProcessingResult with details of the processing
        """
        try:
            # Get original file size
            original_file_size = image_path.stat().st_size

            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return ImageProcessingResult(
                    path=image_path,
                    original_size=(0, 0),
                    new_size=None,
                    original_file_size=original_file_size,
                    new_file_size=None,
                    processed=False,
                    error="Could not read image",
                )

            height, width = img.shape[:2]
            original_size = (width, height)

            # Check if processing is needed
            if not self._needs_processing(image_path):
                return ImageProcessingResult(
                    path=image_path,
                    original_size=original_size,
                    new_size=None,
                    original_file_size=original_file_size,
                    new_file_size=None,
                    processed=False,
                )

            # Calculate new dimensions
            new_width, new_height = self._calculate_new_dimensions(width, height)
            new_size = (new_width, new_height)

            if dry_run:
                # Estimate new file size (rough approximation)
                scale_factor = (new_width * new_height) / (width * height)
                estimated_size = int(original_file_size * scale_factor * 0.7)  # Assume some compression

                return ImageProcessingResult(
                    path=image_path,
                    original_size=original_size,
                    new_size=new_size,
                    original_file_size=original_file_size,
                    new_file_size=estimated_size,
                    processed=True,
                )

            # Resize image if needed
            if new_size != original_size:
                img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)

            # Prepare encoding parameters based on file type
            ext = image_path.suffix.lower()
            if ext in {".jpg", ".jpeg"}:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality]
            elif ext == ".png":
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, self.settings.png_compression]
            else:
                return ImageProcessingResult(
                    path=image_path,
                    original_size=original_size,
                    new_size=None,
                    original_file_size=original_file_size,
                    new_file_size=None,
                    processed=False,
                    error=f"Unsupported file extension: {ext}",
                )

            # Create backup path
            backup_path = image_path.with_suffix(image_path.suffix + ".backup")

            # Backup original file
            if not backup_path.exists():
                image_path.replace(backup_path)

                # Encode and save the processed image
                success = cv2.imwrite(str(image_path), img, encode_params)

                if not success:
                    # Restore backup if encoding failed
                    backup_path.replace(image_path)
                    return ImageProcessingResult(
                        path=image_path,
                        original_size=original_size,
                        new_size=None,
                        original_file_size=original_file_size,
                        new_file_size=None,
                        processed=False,
                        error="Failed to encode processed image",
                    )

                # Get new file size
                new_file_size = image_path.stat().st_size

                # Remove backup if processing was successful and file is smaller or similar quality
                if new_file_size <= original_file_size * 1.1:  # Allow 10% tolerance
                    backup_path.unlink()
                else:
                    # Restore original if new file is significantly larger
                    image_path.unlink()
                    backup_path.replace(image_path)
                    return ImageProcessingResult(
                        path=image_path,
                        original_size=original_size,
                        new_size=None,
                        original_file_size=original_file_size,
                        new_file_size=None,
                        processed=False,
                        error="Processed image was larger than original",
                    )
            else:
                # Backup already exists, which means we've processed this before
                return ImageProcessingResult(
                    path=image_path,
                    original_size=original_size,
                    new_size=None,
                    original_file_size=original_file_size,
                    new_file_size=None,
                    processed=False,
                )

            return ImageProcessingResult(
                path=image_path,
                original_size=original_size,
                new_size=new_size,
                original_file_size=original_file_size,
                new_file_size=new_file_size,
                processed=True,
            )

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return ImageProcessingResult(
                path=image_path,
                original_size=(0, 0),
                new_size=None,
                original_file_size=original_file_size if "original_file_size" in locals() else 0,
                new_file_size=None,
                processed=False,
                error=str(e),
            )

    def normalize_directory(self, directory: Path, dry_run: bool = False) -> list[ImageProcessingResult]:
        """Normalize all images in a directory.

        Args:
            directory: Path to directory containing images
            dry_run: If True, only show what would be done without modifying files

        Returns:
            List of ImageProcessingResult for each processed image
        """
        results = []

        # Find all image files recursively
        image_files: list[Path] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_files.extend(directory.rglob(f"*{ext}"))

        logger.info(f"Found {len(image_files)} image files in {directory}")

        for image_file in sorted(image_files):
            logger.debug(f"Processing {image_file}")
            result = self._process_image(image_file, dry_run=dry_run)
            results.append(result)

        return results

    def print_summary(self, results: list[ImageProcessingResult], dry_run: bool = False) -> None:
        """Print a summary of processing results.

        Args:
            results: List of processing results
            dry_run: Whether this was a dry run
        """
        mode = "DRY RUN" if dry_run else "PROCESSED"

        processed_count = sum(1 for r in results if r.processed)
        error_count = sum(1 for r in results if r.error)
        skipped_count = len(results) - processed_count - error_count

        total_original_size = sum(r.original_file_size for r in results)
        total_new_size = sum(r.new_file_size for r in results if r.new_file_size is not None)

        if not dry_run:
            total_new_size = sum(
                r.new_file_size if r.new_file_size is not None else r.original_file_size for r in results
            )

        print(f"\n=== IMAGE NORMALIZATION {mode} SUMMARY ===")
        print(f"Total images: {len(results)}")
        print(f"Processed: {processed_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Errors: {error_count}")

        if dry_run and processed_count > 0:
            print(f"Original total size: {total_original_size / (1024 * 1024):.1f} MB")
            print(f"Estimated new size: {total_new_size / (1024 * 1024):.1f} MB")
            print(f"Estimated savings: {(total_original_size - total_new_size) / (1024 * 1024):.1f} MB")
        elif not dry_run and processed_count > 0:
            print(f"Original total size: {total_original_size / (1024 * 1024):.1f} MB")
            print(f"New total size: {total_new_size / (1024 * 1024):.1f} MB")
            print(f"Actual savings: {(total_original_size - total_new_size) / (1024 * 1024):.1f} MB")

        # Show errors if any
        if error_count > 0:
            print("\nErrors encountered:")
            for result in results:
                if result.error:
                    print(f"  {result.path}: {result.error}")

        # Show detailed results for processed images
        if processed_count > 0:
            print(f"\nImages that {'would be' if dry_run else 'were'} processed:")
            for result in results:
                if result.processed:
                    old_size = f"{result.original_size[0]}x{result.original_size[1]}"
                    new_size = f"{result.new_size[0]}x{result.new_size[1]}" if result.new_size else old_size

                    old_file_size = result.original_file_size / 1024
                    new_file_size = result.new_file_size / 1024 if result.new_file_size else old_file_size

                    print(
                        f"  {result.path.name}: {old_size} → {new_size}, {old_file_size:.1f}KB → {new_file_size:.1f}KB"
                    )
