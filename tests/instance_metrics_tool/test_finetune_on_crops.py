"""Basic integration tests for finetune_on_crops functionality"""

import os
import tempfile
from pathlib import Path

import tlc
from tlc_tools.augment_bbs.finetune_on_crops import train_model

test_data_dir = Path(__file__).parent.parent.parent / "data"

coco128_annotations_file = test_data_dir / "coco128" / "annotations.json"
coco128_images_dir = test_data_dir / "coco128" / "images"

balloons_train_annotation_file = test_data_dir / "balloons" / "train" / "train-annotations.json"
balloons_train_images_dir = test_data_dir / "balloons" / "train"
balloons_val_annotation_file = test_data_dir / "balloons" / "val" / "val-annotations.json"
balloons_val_images_dir = test_data_dir / "balloons" / "val"


def test_train_model_detection_basic():
    """Test basic training functionality with detection tables"""
    # Create small train and val tables
    train_table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="detect",
        project_name="Test Training",
        dataset_name="COCO128 Test",
        table_name="detection_train_test",
    )

    val_table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="detect",
        project_name="Test Training",
        dataset_name="COCO128 Test",
        table_name="detection_val_test",
    )

    # Use temporary directory for cleaner cleanup
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "test_model.pth")

        # Test with minimal training (just 1 epoch, small batch)
        model, saved_path = train_model(
            train_table_url=train_table.url,
            val_table_url=val_table.url,
            model_name="efficientnet_b0",
            model_checkpoint=checkpoint_path,
            epochs=1,  # Very fast training
            batch_size=2,  # Small batch
            include_background=False,
            num_workers=0,  # No multiprocessing for testing
        )

        # Should return a model and checkpoint path
        assert model is not None
        assert saved_path == checkpoint_path
        assert os.path.exists(checkpoint_path)
        # File cleanup happens automatically when temp_dir context exits


def test_train_model_segmentation_basic():
    """Test basic training functionality with segmentation tables"""
    train_table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="segment",
        project_name="Test Training",
        dataset_name="COCO128 Test",
        table_name="segmentation_train_test",
    )

    val_table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="segment",
        project_name="Test Training",
        dataset_name="COCO128 Test",
        table_name="segmentation_val_test",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "test_model.pth")

        # Test with segmentation data
        model, saved_path = train_model(
            train_table_url=train_table.url,
            val_table_url=val_table.url,
            epochs=1,
            batch_size=2,
            num_workers=0,
            model_checkpoint=checkpoint_path,
        )

        assert model is not None
        assert saved_path == checkpoint_path
        assert os.path.exists(checkpoint_path)
        # File cleanup happens automatically when temp_dir context exits


def test_train_model_detection_single_class():
    """Single class table should add background class"""
    train_table = tlc.Table.from_coco(
        balloons_train_annotation_file,
        balloons_train_images_dir,
        task="detect",
        project_name="Test Training",
        dataset_name="Balloons",
        table_name="detection_train_test",
    )
    val_table = tlc.Table.from_coco(
        balloons_val_annotation_file,
        balloons_val_images_dir,
        task="detect",
        project_name="Test Training",
        dataset_name="Balloons",
        table_name="detection_val_test",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "test_model.pth")
        model, saved_path = train_model(
            train_table_url=train_table.url,
            val_table_url=val_table.url,
            model_name="efficientnet_b0",
            model_checkpoint=checkpoint_path,
            epochs=1,
            batch_size=2,
            num_workers=0,
        )
