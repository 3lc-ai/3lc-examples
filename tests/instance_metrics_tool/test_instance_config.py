from pathlib import Path
from typing import Callable

import pytest
import tlc

from tlc_tools.common import resolve_instance_config

test_data_dir = Path(__file__).parent.parent.parent / "data"

coco128_annotations_file = test_data_dir / "coco128" / "annotations.json"
coco128_images_dir = test_data_dir / "coco128" / "images"


def coco_128_segmentation_table() -> tlc.Table:
    return (
        tlc.Table.from_coco(
            coco128_annotations_file,
            coco128_images_dir,
            task="segment",
            project_name="3LC Tutorials Testing",
            dataset_name="COCO128 Testing",
            table_name="segmentation",
        ),
        "segmentations",
        "segmentations.instance_properties.label",
        "segment",
    )


def coco_128_detection_table() -> tlc.Table:
    return (
        tlc.Table.from_coco(
            coco128_annotations_file,
            coco128_images_dir,
            task="detect",
            project_name="3LC Tutorials Testing",
            dataset_name="COCO128 Testing",
            table_name="detection",
        ),
        "bbs",
        "bbs.bb_list.label",
        "detect",
    )


@pytest.mark.parametrize("table_factory", [coco_128_segmentation_table, coco_128_detection_table])
def test_resolve_instance_config(table_factory: Callable[[], tlc.Table]) -> None:
    table, instance_column, label_column_path, task = table_factory()
    instance_config = resolve_instance_config(table)

    assert instance_config.instance_column == instance_column
    assert instance_config.label_column_path == label_column_path
    assert instance_config.instance_type == "segmentations" if task == "segment" else "bounding_boxes"
