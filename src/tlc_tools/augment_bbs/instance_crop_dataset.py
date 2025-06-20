from __future__ import annotations

import random
import warnings
from io import BytesIO

import numpy as np
import tlc
import torch
from PIL import Image
from tlc.core.builtins.types.segmentation_helper import SegmentationHelper
from torch.utils.data import Dataset

from tlc_tools.augment_bbs.instance_config import InstanceConfig

from .label_utils import create_label_mappings


class InstanceCropDataset(Dataset):
    """Clean dataset for cropping instances (bounding boxes or segmentations) and generating background patches."""

    def __init__(
        self,
        table: tlc.Table,
        transform=None,
        add_background: bool = False,
        background_freq: float = 0.5,
        image_column_name: str = "image",
        x_max_offset: float = 0.0,
        y_max_offset: float = 0.0,
        y_scale_range: tuple[float, float] = (1.0, 1.0),
        x_scale_range: tuple[float, float] = (1.0, 1.0),
        instance_config: InstanceConfig | None = None,
        include_background_in_labels: bool | None = None,
    ):
        """
        :param table: The input table containing image and instance data.
        :param transform: Transformations to apply to cropped images.
        :param add_background: Whether to include background patches (sampling
            behavior).
        :param background_freq: Probability of sampling a background patch.
        :param image_column_name: Name of the image column.
        :param x_max_offset: Maximum offset in the x direction for instance
            cropping.
        :param y_max_offset: Maximum offset in the y direction for instance
            cropping.
        :param y_scale_range: Range of scaling factors in the y direction for
            instance cropping.
        :param x_scale_range: Range of scaling factors in the x direction for
            instance cropping.
        :param instance_config: Instance configuration object with
            column/type/label info.
        :param include_background_in_labels: Whether to include background class
            in label mapping (defaults to add_background).
        """
        self.table = table
        self.transform = transform
        self.image_column_name = image_column_name

        # Resolve instance configuration
        if instance_config is None:
            self.instance_config = InstanceConfig.resolve(
                input_table=table,
                allow_label_free=False,  # Dataset creation usually requires labels
            )
        else:
            # Ensure config is validated for this table (cached, so safe to call multiple times)
            instance_config._ensure_validated_for_table(table)
            self.instance_config = instance_config

        # Check for unsupported combination: segmentation with background
        if self.instance_config.instance_type == "segmentations" and add_background:
            raise ValueError(
                "Background generation is not supported for segmentation instances. "
                "Background patches can only be generated for bounding box instances."
            )

        # Determine whether to include background in label mapping
        if include_background_in_labels is None:
            include_background_in_labels = add_background

        # Get label mappings if labels are available
        if self.instance_config.label_column_path and not self.instance_config.allow_label_free:
            self.label_map = table.get_simple_value_map(self.instance_config.label_column_path)
            if not self.label_map:
                raise ValueError(f"No label map found at path: {self.instance_config.label_column_path}")

            # Create label mappings (separate from sampling behavior)
            self.label_2_contiguous_idx, _, self.background_label, label_mapping_has_background = create_label_mappings(
                self.label_map,
                include_background=include_background_in_labels,
            )
        else:
            # Label-free mode
            self.label_map = None
            self.label_2_contiguous_idx = {}
            self.background_label = None
            label_mapping_has_background = False
            add_background = False  # Force disable background for label-free

        # Sampling behavior: only generate background crops if requested AND label mapping supports it
        self.add_background = add_background and label_mapping_has_background
        self.background_freq = background_freq if self.add_background else 0
        self.random_gen = random.Random(42)  # Fixed seed for reproducibility
        self.x_max_offset = x_max_offset
        self.y_max_offset = y_max_offset
        self.y_scale_range = y_scale_range
        self.x_scale_range = x_scale_range

        # Collect all instances from the table
        self.all_instances = self._collect_instances()

    def _collect_instances(self) -> list[tuple[int, dict]]:
        """Collect all instances from the table based on instance type."""
        instances = []

        for row_idx, row in enumerate(self.table.table_rows):
            if self.instance_config.instance_type == "bounding_boxes":
                # Handle bounding boxes
                bbs = row[self.instance_config.instance_column][self.instance_config.instance_properties_column]
                for bb in bbs:
                    instances.append((row_idx, {"type": "bbox", "data": bb}))

            elif self.instance_config.instance_type == "segmentations":
                # Handle segmentation instances
                instance_data = row[self.instance_config.instance_column]

                if "rles" in instance_data:
                    # RLE format
                    rles = instance_data["rles"]
                    if self.instance_config.label_column_path and not self.instance_config.allow_label_free:
                        labels = instance_data["instance_properties"]["label"]
                        for rle, label in zip(rles, labels):
                            instances.append((row_idx, {"type": "rle", "rle": rle, "label": label}))
                    else:
                        # Label-free mode
                        for rle in rles:
                            instances.append((row_idx, {"type": "rle", "rle": rle, "label": None}))
                else:
                    raise ValueError(
                        f"Unsupported segmentation format in column {self.instance_config.instance_column}"
                    )

            else:
                raise ValueError(f"Unknown instance type: {self.instance_config.instance_type}")

        return instances

    def __len__(self) -> int:
        return len(self.all_instances)

    def __getitem__(self, idx: int):
        """
        Fetch a sample from the dataset.

        :param idx: int, index of the specific instance.
        :returns: (original_pil_image, transformed_tensor, label) where label is a tensor (or -1 for label-free mode).
        """
        # Determine if a background patch should be generated
        is_background = self.add_background and self.random_gen.random() < self.background_freq

        if is_background:
            crop, label = self._generate_background()
        else:
            crop, label = self._get_instance_crop(idx)

        # Keep original PIL image for metrics
        original_pil = crop.copy()

        # Apply transform for model
        transformed_tensor = self.transform(crop) if self.transform else crop

        return original_pil, transformed_tensor, label

    def _get_instance_crop(self, idx: int):
        """Get a crop for a specific instance."""
        row_idx, instance_data = self.all_instances[idx]
        row = self.table.table_rows[row_idx]
        image = self._load_image_data(row)

        if instance_data["type"] == "bbox":
            # Handle bounding box instances
            bb = instance_data["data"]
            bb_schema = self.table.schema.values["rows"].values[self.instance_config.instance_column].values["bb_list"]

            crop = tlc.BBCropInterface.crop(
                image,
                bb,
                bb_schema,
                x_max_offset=self.x_max_offset,
                y_max_offset=self.y_max_offset,
                y_scale_range=self.y_scale_range,
                x_scale_range=self.x_scale_range,
            )

            # All bounding boxes should have labels in normal training
            if "label" not in instance_data["data"]:
                raise ValueError(
                    f"Bounding box missing 'label' key. Available keys: {list(instance_data['data'].keys())}"
                )

            label = instance_data["data"]["label"]

        elif instance_data["type"] == "rle":
            # Handle RLE segmentation instances
            crop, label = self._process_rle_instance(image, instance_data)

        else:
            raise ValueError(f"Unknown instance type: {instance_data['type']}")

        # Convert label to tensor
        if self.instance_config.allow_label_free:
            # Label-free mode - always use PyTorch's default ignore_index, regardless of actual label value
            label_tensor = torch.tensor(-100, dtype=torch.long)
        elif label is not None and self.label_2_contiguous_idx:
            if label not in self.label_2_contiguous_idx:
                raise KeyError(
                    f"Label {label} not found in mapping. Available labels: {list(self.label_2_contiguous_idx.keys())}"
                )
            label_tensor = torch.tensor(self.label_2_contiguous_idx[label], dtype=torch.long)
        else:
            raise ValueError(
                f"Invalid state: label={label}, label_2_contiguous_idx={bool(self.label_2_contiguous_idx)}, "
                f"allow_label_free={self.instance_config.allow_label_free}"
            )

        return crop, label_tensor

    def _process_rle_instance(self, image, instance_data):
        """Process an RLE segmentation instance."""
        w, h = image.size
        rle = instance_data["rle"]
        label = instance_data.get("label")

        # Convert RLE to mask and bbox
        coco = {"size": [h, w], "counts": rle}
        bbox = SegmentationHelper.bbox_from_rle(coco)
        mask = SegmentationHelper.mask_from_rle(coco)

        # Create bbox dict for cropping
        bb_dict = {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}

        # Apply mask to image
        image_array = np.array(image.convert("RGB"))
        mask = mask[:, :, np.newaxis]  # Shape becomes (h, w, 1)
        mask = np.repeat(mask, 3, axis=2)  # Shape becomes (h, w, 3)
        masked_image = image_array * mask
        masked_image = Image.fromarray(masked_image.astype(np.uint8), mode="RGB")

        # Create temporary schema for cropping
        bb_schema = tlc.BoundingBoxListSchema(
            {},
            x1_number_role=tlc.NUMBER_ROLE_BB_SIZE_X,
            y1_number_role=tlc.NUMBER_ROLE_BB_SIZE_Y,
        )["bb_list"]

        # Crop the masked image
        crop = tlc.BBCropInterface.crop(
            masked_image,
            bb_dict,
            bb_schema,
            x_max_offset=self.x_max_offset,
            y_max_offset=self.y_max_offset,
            y_scale_range=self.y_scale_range,
            x_scale_range=self.x_scale_range,
        )

        return crop, label

    def _generate_background(self):
        """Generate a background patch (only works with bounding box instances)."""
        if self.instance_config.instance_type != "bounding_boxes":
            warnings.warn(
                "Background generation only supported for bounding boxes. Returning random instance.",
                stacklevel=2,
            )
            return self._get_instance_crop(self.random_gen.randint(0, len(self.all_instances) - 1))

        # Select a random row for background
        row_idx = self.random_gen.randint(0, len(self.table) - 1)
        row = self.table.table_rows[row_idx]
        image = self._load_image_data(row)
        bbs = row[self.instance_config.instance_column]["bb_list"]

        crop, label = self._generate_background_crop(image, bbs)
        return crop, label

    def _generate_background_crop(self, image, bbs, max_attempts=100):
        """Generate a background patch from the image."""
        image_width, image_height = image.size
        bb_schema = self.table.schema.values["rows"].values[self.instance_config.instance_column].values["bb_list"]
        bb_factory = tlc.BoundingBox.from_schema(bb_schema)

        gt_boxes_xywh = [
            bb_factory([bb["x0"], bb["y0"], bb["x1"], bb["y1"]])
            .to_top_left_xywh()
            .denormalize(image_width, image_height)
            for bb in bbs
        ]

        for _attempt_idx in range(max_attempts):
            # Pick a random bounding box from all_instances (only BB instances)
            bbox_instances = [inst for inst in self.all_instances if inst[1]["type"] == "bbox"]
            if not bbox_instances:
                break

            _, random_instance = self.random_gen.choice(bbox_instances)
            random_bb = random_instance["data"]

            # Convert random bb to xywh format using the factory
            proposed_box = (
                bb_factory([random_bb["x0"], random_bb["y0"], random_bb["x1"], random_bb["y1"]])
                .to_top_left_xywh()
                .denormalize(image_width, image_height)
            )

            # Ensure the proposed box does not intersect any ground truth boxes
            if not any(self._intersects(proposed_box, gt_box) for gt_box in gt_boxes_xywh):
                break

        if _attempt_idx == max_attempts - 1:
            # Return a 100x100 black square if no valid background patch is found
            warnings.warn(
                "No valid background patch found. Returning a black square. Please check your data.",
                stacklevel=2,
            )
            return Image.new("RGB", (100, 100), (0, 0, 0)), self.background_label

        # Crop the background patch
        crop = tlc.BBCropInterface.crop(
            image,
            random_bb,
            bb_schema,
            x_max_offset=self.x_max_offset,
            y_max_offset=self.y_max_offset,
            y_scale_range=self.y_scale_range,
            x_scale_range=self.x_scale_range,
        )
        return crop, self.background_label

    def _load_image_data(self, row):
        """Load image data from a table row."""
        image_bytes = tlc.Url(row[self.image_column_name]).read()
        image = Image.open(BytesIO(image_bytes))
        return image

    @staticmethod
    def _intersects(box1: list[int], box2: list[int]) -> bool:
        """Check if two boxes intersect (assumes [x, y, w, h] format)."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    def get_row_instance_mapping(self) -> list[tuple[int, int]]:
        """Get mapping from dataset index to (row_index, instance_index_within_row).

        This provides the mapping needed to assign embeddings and predictions back to the
        correct instances in the original table structure.

        :return: List where index is dataset index, value is (row_index, instance_index_within_row)
        """
        mapping = []

        # Group instances by row to determine instance_index_within_row
        row_instance_counts: dict[int, int] = {}

        for row_idx, _ in self.all_instances:
            # Get the instance index within this row (0-based)
            instance_index_within_row = row_instance_counts.get(row_idx, 0)
            row_instance_counts[row_idx] = instance_index_within_row + 1

            mapping.append((row_idx, instance_index_within_row))

        return mapping

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for this dataset to handle PIL images alongside tensors.

        :param batch: List of (pil_image, tensor, label) tuples
        :return: (pil_images_list, tensor_batch, label_batch)
        """
        pil_images = [item[0] for item in batch]  # Keep PIL images in a list
        tensors = torch.stack([item[1] for item in batch])  # Stack tensors normally
        labels = torch.tensor([item[2] for item in batch])  # Convert labels to tensor

        return pil_images, tensors, labels
