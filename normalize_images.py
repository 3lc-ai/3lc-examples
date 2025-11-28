#!/usr/bin/env python3
"""
Convenience script to normalize images in the tutorials/images directory.

This script provides an easy way to run the image normalization utility
specifically on the tutorials/images directory.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.cli import main
from utils.image_normalizer import ImageNormalizer, NormalizationSettings

def normalize_tutorials_images(dry_run: bool = True) -> None:
    """Normalize images in the tutorials/images directory.
    
    Args:
        dry_run: If True, only show what would be done without modifying files
    """
    # Set up the image directory
    tutorials_images_dir = project_root / "tutorials" / "images"
    
    if not tutorials_images_dir.exists():
        print(f"Error: Directory '{tutorials_images_dir}' does not exist.")
        return
    
    # Use default settings optimized for tutorial images (gallery + notebook cards)
    settings = NormalizationSettings(
        max_dimension=1280,  # Good balance for notebook titles and gallery use
        jpeg_quality=85,     # High quality compression
        png_compression=6    # Good balance of size and quality
    )
    
    normalizer = ImageNormalizer(settings)
    
    print(f"{'DRY RUN: ' if dry_run else ''}Normalizing images in tutorials/images/")
    print(f"Settings: max_dimension={settings.max_dimension}, "
          f"jpeg_quality={settings.jpeg_quality}, "
          f"png_compression={settings.png_compression}")
    print()
    
    try:
        results = normalizer.normalize_directory(tutorials_images_dir, dry_run=dry_run)
        normalizer.print_summary(results, dry_run=dry_run)
        
        if dry_run:
            print("\nTo actually process the images, run:")
            print("python normalize_images.py --process")
            
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    # Simple argument handling
    if len(sys.argv) > 1 and sys.argv[1] in ["--process", "-p"]:
        normalize_tutorials_images(dry_run=False)
    else:
        normalize_tutorials_images(dry_run=True)
