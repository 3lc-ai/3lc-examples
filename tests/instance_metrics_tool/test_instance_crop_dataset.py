"""Basic integration tests for refactored instance functionality"""

from pathlib import Path

import pytest
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


# ============================================================================
# DEBUGGING/VISUAL INSPECTION TEST
# Run with: pytest -k debug -s --tb=short
# ============================================================================
@pytest.mark.skip(reason="Interactive debugging test - run manually with pytest -k debug_interactive_crop_inspector -s")
def test_debug_interactive_crop_inspector():
    """
    Interactive debugging test: Shows crops one by one with keyboard navigation
    Use arrow keys for navigation: ←/→ for next/prev sample, ↑/↓ for scenario, 'q' to quit

    Usage: pytest tests/instance_metrics_tool/test_instance_crop_dataset.py::test_debug_interactive_crop_inspector -s
    """
    import matplotlib.pyplot as plt
    from tlc_tools.augment_bbs.label_utils import get_label_name

    # Configuration - removed segmentation with background
    scenarios = [
        ("detect", False, False, "Detection No-BG"),
        ("detect", False, True, "Detection With-BG"),
        ("detect", True, False, "Detection Label-Free No-BG"),
        ("segment", False, False, "Segmentation No-BG"),
        ("segment", True, False, "Segmentation Label-Free No-BG"),
    ]

    # State variables
    current_scenario = 0
    current_sample = 0
    datasets = []

    # Load all datasets
    for task, label_free, add_background, scenario_name in scenarios:
        try:
            table = tlc.Table.from_coco(
                coco128_annotations_file,
                coco128_images_dir,
                task=task,
                project_name="Interactive",
                dataset_name="Interactive",
                table_name=f"interactive_{task}_{'lf' if label_free else 'normal'}_{'bg' if add_background else 'nobg'}",
            )

            config = InstanceConfig.resolve(
                table,
                allow_label_free=label_free,
            )

            # Get label map for string conversion
            label_map = table.get_simple_value_map(config.label_column_path)
            dataset = InstanceCropDataset(table, instance_config=config, add_background=add_background)
            datasets.append((dataset, scenario_name, label_map))
            print(f"Loaded {scenario_name}: {len(dataset)} samples")

        except Exception as e:
            print(f"Failed to load {scenario_name}: {e}")
            datasets.append((None, scenario_name, {}))

    if not any(ds[0] for ds in datasets):
        print("No datasets loaded successfully!")
        return

    # Create interactive plot
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)

    def update_display():
        ax.clear()

        dataset, scenario_name, label_map = datasets[current_scenario]
        if dataset is None or len(dataset) == 0:
            ax.text(
                0.5, 0.5, f"No data for\n{scenario_name}", ha="center", va="center", transform=ax.transAxes, fontsize=16
            )
            ax.set_title(f"{scenario_name} - No Data")
            plt.draw()
            return

        try:
            # Ensure sample index is valid
            actual_sample = current_sample % len(dataset)
            pil_image, _, label = dataset[actual_sample]  # Ignore the middle return value

            # Convert label tensor to string representation
            label_int = label.item() if hasattr(label, "item") else label

            if dataset.instance_config.allow_label_free:
                label_str = "label-free"
            elif (
                label_map
                and dataset.background_label is not None
                and label_int == dataset.label_2_contiguous_idx.get(dataset.background_label, -1)
            ):
                label_str = "background"
            elif label_map and hasattr(dataset, "label_2_contiguous_idx"):
                # Find original label from contiguous index
                original_label = None
                for orig_label, cont_idx in dataset.label_2_contiguous_idx.items():
                    if cont_idx == label_int:
                        original_label = orig_label
                        break

                if original_label is not None:
                    label_str = get_label_name(original_label, label_map, dataset.background_label)
                else:
                    label_str = f"unknown_{label_int}"
            else:
                label_str = str(label_int)

            # Display image
            ax.imshow(pil_image)
            ax.set_title(f"{scenario_name}\nSample {actual_sample}/{len(dataset) - 1} | {label_str}")
            ax.axis("off")

            # Add some info text (without type information)
            info_text = f"Image size: {pil_image.size}\nLabel: {label_str} (raw: {label_int})"
            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=10,
            )

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error loading sample {current_sample}:\n{e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{scenario_name} - Error")

        plt.draw()

    def on_key(event):
        nonlocal current_sample, current_scenario

        if event.key == "right":  # Next sample
            current_sample += 1
            update_display()
        elif event.key == "left":  # Previous sample
            current_sample = max(0, current_sample - 1)
            update_display()
        elif event.key == "up":  # Previous scenario
            current_scenario = (current_scenario - 1) % len(scenarios)
            current_sample = 0
            update_display()
        elif event.key == "down":  # Next scenario
            current_scenario = (current_scenario + 1) % len(scenarios)
            current_sample = 0
            update_display()
        elif event.key == "q":  # Quit
            plt.close()

    # Connect keyboard events
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Add instruction text
    instructions = "Navigation: ←/→ Sample, ↑/↓ Scenario, 'q' Quit"
    fig.suptitle(instructions, fontsize=12)

    # Initial display
    update_display()

    print("\n" + "=" * 60)
    print("INTERACTIVE CROP INSPECTOR")
    print("=" * 60)
    print("Controls:")
    print("  ← / → - Previous/Next sample")
    print("  ↑ / ↓ - Previous/Next scenario")
    print("  'q'   - Quit")
    print("=" * 60)

    plt.show()
