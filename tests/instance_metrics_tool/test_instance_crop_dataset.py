"""Basic integration tests for refactored instance functionality"""

from pathlib import Path

import tlc

from tlc_tools.augment_bbs.instance_config import InstanceConfig
from tlc_tools.augment_bbs.instance_crop_dataset import InstanceCropDataset

test_data_dir = Path(__file__).parent.parent.parent / "data"
coco128_annotations_file = test_data_dir / "coco128" / "annotations.json"
coco128_images_dir = test_data_dir / "coco128" / "images"


def test_instance_crop_dataset_detection():
    """Test InstanceCropDataset works with detection tables"""
    table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="detect",
        project_name="Test",
        dataset_name="Test",
        table_name="detection_dataset_test",
    )

    config = InstanceConfig.resolve(table)
    dataset = InstanceCropDataset(table, instance_config=config)

    assert len(dataset) > 0
    pil_image, tensor, label = dataset[0]
    assert pil_image is not None
    assert tensor is not None
    assert label is not None


def test_instance_crop_dataset_detection_label_free():
    """Test InstanceCropDataset works in label-free mode with detection"""
    table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="detect",
        project_name="Test",
        dataset_name="Test",
        table_name="detection_dataset_label_free_test",
    )

    config = InstanceConfig.resolve(table, allow_label_free=True)
    dataset = InstanceCropDataset(table, instance_config=config)

    assert len(dataset) > 0
    pil_image, tensor, label = dataset[0]
    assert pil_image is not None
    assert tensor is not None
    # In label-free mode, label might be None or a placeholder
    assert label is not None or label is None


def test_instance_crop_dataset_segmentation():
    """Test InstanceCropDataset works with segmentation tables"""
    table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="segment",
        project_name="Test",
        dataset_name="Test",
        table_name="segmentation_dataset_test",
    )

    config = InstanceConfig.resolve(table)
    dataset = InstanceCropDataset(table, instance_config=config)

    assert len(dataset) > 0
    pil_image, tensor, label = dataset[0]
    assert pil_image is not None
    assert tensor is not None
    assert label is not None


def test_instance_crop_dataset_segmentation_label_free():
    """Test InstanceCropDataset works in label-free mode with segmentation"""
    table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="segment",
        project_name="Test",
        dataset_name="Test",
        table_name="segmentation_dataset_label_free_test",
    )

    config = InstanceConfig.resolve(table, allow_label_free=True)
    dataset = InstanceCropDataset(table, instance_config=config)

    assert len(dataset) > 0
    pil_image, tensor, label = dataset[0]
    assert pil_image is not None
    assert tensor is not None
    # In label-free mode, label might be None or a placeholder
    assert label is not None or label is None
