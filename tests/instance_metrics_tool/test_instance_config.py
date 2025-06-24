"""Basic integration tests for refactored instance functionality"""

from pathlib import Path

import tlc
from tlc_tools.augment_bbs.instance_config import InstanceConfig

test_data_dir = Path(__file__).parent.parent.parent / "data"
coco128_annotations_file = test_data_dir / "coco128" / "annotations.json"
coco128_images_dir = test_data_dir / "coco128" / "images"


def test_instance_config_basic_functionality():
    """Test basic InstanceConfig.resolve() works for both BB and segmentation tables"""

    # Test with detection table
    det_table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="detect",
        project_name="3LC Integration Test",
        dataset_name="COCO128 Test",
        table_name="detection_test",
    )

    config = InstanceConfig.resolve(det_table)
    assert config.instance_column == "bbs"
    assert config.instance_type == "bounding_boxes"
    assert config.instance_properties_column == "bb_list"

    # Test with segmentation table
    seg_table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="segment",
        project_name="3LC Integration Test",
        dataset_name="COCO128 Test",
        table_name="segmentation_test",
    )

    config = InstanceConfig.resolve(seg_table)
    assert config.instance_column == "segmentations"
    assert config.instance_type == "segmentations"
    assert config.instance_properties_column == "instance_properties"
