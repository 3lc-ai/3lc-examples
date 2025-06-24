"""Tests for extend_table_with_instance_metrics.py"""

from pathlib import Path

import tlc
from tlc_tools.augment_bbs.extend_table_with_metrics import extend_table_with_metrics
from tlc_tools.augment_bbs.instance_config import InstanceConfig

test_data_dir = Path(__file__).parent.parent.parent / "data"
coco128_annotations_file = test_data_dir / "coco128" / "annotations.json"
coco128_images_dir = test_data_dir / "coco128" / "images"


def test_extend_table_with_metrics_detection() -> None:
    """Test extend_table_with_metrics works with detection tables"""
    table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="detect",
        project_name="Test",
        dataset_name="Test",
        table_name="detection_metrics_test",
    )

    config = InstanceConfig.resolve(table)

    # Test with just image metrics (simpler than embeddings)
    output_url, _, _ = extend_table_with_metrics(
        input_table=table,
        output_table_name="detection_with_metrics",
        add_embeddings=False,
        add_image_metrics=True,
        batch_size=2,  # Small batch for testing
        instance_config=config,
    )

    assert output_url is not None
    # Verify output table was created
    output_table = tlc.Table.from_url(output_url)
    assert len(output_table) > 0


def test_extend_table_with_metrics_detection_label_free() -> None:
    """Test extend_table_with_metrics works in label-free mode with detection"""
    table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="detect",
        project_name="Test",
        dataset_name="Test",
        table_name="detection_metrics_label_free_test",
    )

    config = InstanceConfig.resolve(table, allow_label_free=True)

    # Test with embeddings in pretrained mode (label-free)
    output_url, _, _ = extend_table_with_metrics(
        input_table=table,
        output_table_name="detection_with_embeddings_pretrained",
        add_embeddings=True,
        add_image_metrics=False,
        batch_size=2,
        num_components=3,
        model_checkpoint=None,  # Use pretrained
        instance_config=config,
    )

    assert output_url is not None
    output_table = tlc.Table.from_url(output_url)
    assert len(output_table) > 0


def test_extend_table_with_metrics_segmentation() -> None:
    """Test extend_table_with_metrics works with segmentation tables"""
    table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="segment",
        project_name="Test",
        dataset_name="Test",
        table_name="segmentation_metrics_test",
    )

    config = InstanceConfig.resolve(table)

    # Test with just image metrics
    output_url, _, _ = extend_table_with_metrics(
        input_table=table,
        output_table_name="segmentation_with_metrics",
        add_embeddings=False,
        add_image_metrics=True,
        batch_size=2,
        instance_config=config,
    )

    assert output_url is not None
    output_table = tlc.Table.from_url(output_url)
    assert len(output_table) > 0


def test_extend_table_with_metrics_segmentation_label_free() -> None:
    """Test extend_table_with_metrics works in label-free mode with segmentation"""
    table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="segment",
        project_name="Test",
        dataset_name="Test",
        table_name="segmentation_metrics_label_free_test",
    )

    config = InstanceConfig.resolve(table, allow_label_free=True)

    # Test with embeddings in pretrained mode
    output_url, _, _ = extend_table_with_metrics(
        input_table=table,
        output_table_name="segmentation_with_embeddings_pretrained",
        add_embeddings=True,
        add_image_metrics=False,
        batch_size=2,
        num_components=3,
        model_checkpoint=None,  # Use pretrained
        instance_config=config,
    )

    assert output_url is not None
    output_table = tlc.Table.from_url(output_url)
    assert len(output_table) > 0


def test_extend_table_with_metrics_misc() -> None:
    """Test extend_table_with_metrics works with misc tables"""
    table = tlc.Table.from_coco(
        coco128_annotations_file,
        coco128_images_dir,
        task="detect",
        project_name="Test",
        dataset_name="Test",
        table_name="misc_metrics_test",
    )

    instance_config = InstanceConfig.resolve(table, allow_label_free=True)

    # Test with embeddings in pretrained mode
    output_url, pacmap_reducer, fit_embeddings = extend_table_with_metrics(
        input_table=table,
        output_table_name="misc_with_metrics",
        add_embeddings=True,
        add_image_metrics=True,
        reduce_last_dims=2,
        batch_size=2,
        instance_config=instance_config,
    )

    # Use existing reducer and fit embeddings
    output_url, _, _ = extend_table_with_metrics(
        input_table=table,
        output_table_name="misc_with_metrics_transformed",
        add_embeddings=True,
        add_image_metrics=True,
        batch_size=2,
        reduce_last_dims=2,
        pacmap_reducer=pacmap_reducer,
        fit_embeddings=fit_embeddings,
        instance_config=instance_config,
    )
