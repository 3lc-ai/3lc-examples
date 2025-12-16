#!/usr/bin/env python3
"""
Command-line interface for image normalization utility.
"""

import argparse
import logging
import sys
from pathlib import Path

from .image_normalizer import ImageNormalizer, NormalizationSettings


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Normalize images in a directory by resizing and recompressing them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on tutorials/images directory
  python -m utils.cli --dry-run tutorials/images
  
  # Actually process images with custom settings
  python -m utils.cli --max-dimension 1000 --jpeg-quality 80 tutorials/images
  
  # Verbose output
  python -m utils.cli --verbose --dry-run tutorials/images
        """,
    )

    parser.add_argument("directory", type=Path, help="Directory containing images to normalize")

    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without actually modifying files"
    )

    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1280,
        help="Maximum width or height for images (default: 1000, optimized for notebook titles)",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        choices=range(1, 101),
        metavar="1-100",
        help="JPEG compression quality 1-100 (default: 85)",
    )

    parser.add_argument(
        "--png-compression",
        type=int,
        default=6,
        choices=range(0, 10),
        metavar="0-9",
        help="PNG compression level 0-9 (default: 6)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate directory
    if not args.directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist.", file=sys.stderr)
        return 1

    if not args.directory.is_dir():
        print(f"Error: '{args.directory}' is not a directory.", file=sys.stderr)
        return 1

    # Create settings
    settings = NormalizationSettings(
        max_dimension=args.max_dimension, jpeg_quality=args.jpeg_quality, png_compression=args.png_compression
    )

    # Create normalizer and process
    normalizer = ImageNormalizer(settings)

    print(f"{'DRY RUN: ' if args.dry_run else ''}Normalizing images in: {args.directory}")
    print(
        f"Settings: max_dimension={settings.max_dimension}, "
        f"jpeg_quality={settings.jpeg_quality}, "
        f"png_compression={settings.png_compression}"
    )

    try:
        results = normalizer.normalize_directory(args.directory, dry_run=args.dry_run)
        normalizer.print_summary(results, dry_run=args.dry_run)

        if args.dry_run:
            print("\nTo actually process the images, run the same command without --dry-run")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
