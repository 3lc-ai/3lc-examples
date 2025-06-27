"""Basic integration tests for refactored instance functionality"""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import tlc

from tlc_tools.augment_bbs.instance_config import InstanceConfig
from tlc_tools.augment_bbs.instance_crop_dataset import InstanceCropDataset
from tlc_tools.augment_bbs.label_utils import get_label_name

test_data_dir = Path(__file__).parent.parent.parent / "data"
coco128_annotations_file = test_data_dir / "coco128" / "annotations.json"
coco128_images_dir = test_data_dir / "coco128" / "images"


class DatasetExplorer:
    """Interactive dataset explorer for visualizing instance crops."""

    def __init__(self, datasets):
        """
        Initialize the dataset explorer.

        Args:
            datasets: List of tuples containing (dataset, name, label_map)
        """
        self.datasets = datasets
        self.current_scenario = 0
        self.current_sample = 0
        self.fig = None
        self.ax = None

    def _update_display(self):
        """Update the display with the current dataset and sample."""
        self.ax.clear()

        dataset, scenario_name, label_map = self.datasets[self.current_scenario]
        if dataset is None or len(dataset) == 0:
            self.ax.text(
                0.5,
                0.5,
                f"No data for\n{scenario_name}",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=16,
            )
            self.ax.set_title(f"{scenario_name} - No Data")
            plt.draw()
            return

        try:
            # Ensure sample index is valid
            actual_sample = self.current_sample % len(dataset)
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
            self.ax.imshow(pil_image)
            self.ax.set_title(f"{scenario_name}\nSample {actual_sample}/{len(dataset) - 1} | {label_str}")
            self.ax.axis("off")

            # Add some info text (without type information)
            info_text = f"Image size: {pil_image.size}\nLabel: {label_str} (raw: {label_int})"
            self.ax.text(
                0.02,
                0.98,
                info_text,
                transform=self.ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=10,
            )

        except Exception as e:
            self.ax.text(
                0.5,
                0.5,
                f"Error loading sample {self.current_sample}:\n{e}",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            self.ax.set_title(f"{scenario_name} - Error")

        plt.draw()

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == "right":  # Next sample
            self.current_sample += 1
            self._update_display()
        elif event.key == "left":  # Previous sample
            self.current_sample = max(0, self.current_sample - 1)
            self._update_display()
        elif event.key == "up":  # Previous scenario
            self.current_scenario = (self.current_scenario - 1) % len(self.datasets)
            self.current_sample = 0
            self._update_display()
        elif event.key == "down":  # Next scenario
            self.current_scenario = (self.current_scenario + 1) % len(self.datasets)
            self.current_sample = 0
            self._update_display()
        elif event.key == "q":  # Quit
            plt.close()

    def show(self):
        """Show the interactive explorer."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Add instruction text
        instructions = "Navigation: ←/→ Sample, ↑/↓ Scenario, 'q' Quit"
        self.fig.suptitle(instructions, fontsize=12)

        # Initial display
        self._update_display()

        print("\n" + "=" * 60)
        print("INTERACTIVE CROP INSPECTOR")
        print("=" * 60)
        print("Controls:")
        print("  ← / → - Previous/Next sample")
        print("  ↑ / ↓ - Previous/Next scenario")
        print("  'q'   - Quit")
        print("=" * 60)

        plt.show()


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
    # Configuration - removed segmentation with background
    scenarios = [
        ("detect", False, False, "Detection No-BG"),
        ("detect", False, True, "Detection With-BG"),
        ("detect", True, False, "Detection Label-Free No-BG"),
        ("segment", False, False, "Segmentation No-BG"),
        ("segment", True, False, "Segmentation Label-Free No-BG"),
    ]

    datasets = []
    for task, label_free, add_background, scenario_name in scenarios:
        try:
            table = tlc.Table.from_coco(
                coco128_annotations_file,
                coco128_images_dir,
                task=task,
                project_name="Interactive",
                dataset_name="Interactive",
                table_name=f"interactive_{task}_{'lf' if label_free else 'norm'}_{'bg' if add_background else 'nobg'}",
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

    explorer = DatasetExplorer(datasets)
    explorer.show()


@pytest.mark.skip(reason="Interactive debugging test - run manually with pytest -k debug_custom_dataset_explorer -s")
def test_debug_custom_dataset_explorer():
    """
    Interactive debugging test for custom datasets: Shows crops one by one with keyboard navigation
    Use arrow keys for navigation: ←/→ for next/prev sample, ↑/↓ for scenario, 'q' to quit

    Usage: pytest tests/instance_metrics_tool/test_instance_crop_dataset.py::test_debug_custom_dataset_explorer -s
    """
    # Example table URLs
    table_urls = [
        # Add table URLs here
    ]

    datasets = []
    for i, table_url in enumerate(table_urls):
        try:
            table = tlc.Table.from_url(table_url)
            config = InstanceConfig.resolve(table)
            label_map = table.get_simple_value_map(config.label_column_path)
            dataset = InstanceCropDataset(table, instance_config=config)
            datasets.append((dataset, f"Dataset {i + 1}", label_map))
            print(f"Loaded Dataset {i + 1}: {len(dataset)} samples")

        except Exception as e:
            print(f"Failed to load Dataset {i + 1}: {e}")
            datasets.append((None, f"Dataset {i + 1}", {}))

    if not any(ds[0] for ds in datasets):
        print("No datasets loaded successfully!")
        return

    explorer = DatasetExplorer(datasets)
    explorer.show()
