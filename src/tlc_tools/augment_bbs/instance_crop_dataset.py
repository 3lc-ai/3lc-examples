from __future__ import annotations

import random
import warnings
from io import BytesIO

import numpy as np
import tlc
import torch
from PIL import Image
from tlc.core.data_formats.bb_conversions import legacy_bb_row_to_bounding_boxes_2d
from tlc.core.helpers.segmentation_helper import SegmentationHelper
from tlc.core.sample_types.registry import SampleTypeRegistry
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
                raise ValueError(
                    f"No label map found at path: {self.instance_config.label_column_path}. Edit your label column "
                    "path or set allow_label_free=True in the InstanceConfig."
                )

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
        ic = self.instance_config

        for row_idx, row in enumerate(self.table.table_rows):
            if ic.instance_type == "bounding_boxes":
                bb2d = self._get_bounding_boxes_2d(row)
                for i in range(bb2d.num_instances):
                    label = int(bb2d.instance_labels[i]) if bb2d.instance_labels is not None else None
                    instances.append(
                        (
                            row_idx,
                            {
                                "type": "bbox",
                                "xyxy": bb2d.bbs[i],  # (4,) float32 array [x_min, y_min, x_max, y_max]
                                "label": label,
                            },
                        )
                    )

            elif ic.instance_type == "segmentations":
                instance_data = row[ic.instance_column]

                if "rles" in instance_data:
                    rles = instance_data["rles"]
                    if ic.label_column_path and not ic.allow_label_free:
                        labels = instance_data["instance_properties"]["label"]
                        for rle, label in zip(rles, labels):
                            instances.append((row_idx, {"type": "rle", "rle": rle, "label": label}))
                    else:
                        for rle in rles:
                            instances.append((row_idx, {"type": "rle", "rle": rle, "label": None}))
                else:
                    raise ValueError(f"Unsupported segmentation format in column {ic.instance_column}")

            else:
                raise ValueError(f"Unknown instance type: {ic.instance_type}")

        return instances

    def _get_bounding_boxes_2d(self, row):
        """Get BoundingBoxes2D from a row, handling both legacy and new format."""
        raw = row[self.instance_config.instance_column]
        if self.instance_config.is_legacy_bb:
            schema = self.table.rows_schema.values[self.instance_config.instance_column]
            return legacy_bb_row_to_bounding_boxes_2d(raw, schema)
        else:
            return SampleTypeRegistry.get("bounding_boxes_2d").from_row(raw)

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
            xyxy = instance_data["xyxy"]  # absolute [x_min, y_min, x_max, y_max]
            crop = self._crop_from_xyxy(image, xyxy)
            label = instance_data["label"]

        elif instance_data["type"] == "rle":
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
        bbox = SegmentationHelper.bbox_from_rle(coco)  # [x, y, w, h] format
        mask = SegmentationHelper.mask_from_rle(coco)

        # Apply mask to image
        image_array = np.array(image.convert("RGB"))
        mask = mask[:, :, np.newaxis]
        mask = np.repeat(mask, 3, axis=2)
        masked_image = image_array * mask
        masked_image = Image.fromarray(masked_image.astype(np.uint8), mode="RGB")

        # Convert bbox from [x, y, w, h] to [x_min, y_min, x_max, y_max]
        xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)
        crop = self._crop_from_xyxy(masked_image, xyxy)

        return crop, label

    def _crop_from_xyxy(self, image: Image.Image, xyxy: np.ndarray) -> Image.Image:
        """Crop an image using absolute XYXY coordinates with optional augmentation.

        :param image: Source PIL image.
        :param xyxy: Array of [x_min, y_min, x_max, y_max] in absolute pixel coords.
        :returns: Cropped PIL image.
        """
        image_width, image_height = image.size
        x_min, y_min, x_max, y_max = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        w = x_max - x_min
        h = y_max - y_min

        # Apply random offset augmentation
        x_offset = random.uniform(0, self.x_max_offset) * w
        y_offset = random.uniform(0, self.y_max_offset) * h
        x_min -= x_offset
        y_min -= y_offset

        # Apply random scale augmentation
        x_scale = random.uniform(*self.x_scale_range)
        y_scale = random.uniform(*self.y_scale_range)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w *= x_scale
        h *= y_scale
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2

        # Clamp to image bounds
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(image_width, int(x_max))
        y_max = min(image_height, int(y_max))

        if x_max <= x_min or y_max <= y_min:
            return Image.new("RGB", (1, 1), (0, 0, 0))

        return image.crop((x_min, y_min, x_max, y_max)).convert("RGB")

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
        bb2d = self._get_bounding_boxes_2d(row)

        crop, label = self._generate_background_crop(image, bb2d)
        return crop, label

    def _generate_background_crop(self, image, bb2d, max_attempts=100):
        """Generate a background patch from the image."""
        from tlc.core.data_formats.bb_conversions import xyxy_to_xywh

        # Convert GT boxes to xywh for intersection check
        gt_boxes_xywh = xyxy_to_xywh(bb2d.bbs) if bb2d.num_instances > 0 else np.empty((0, 4), dtype=np.float32)

        for _attempt_idx in range(max_attempts):
            bbox_instances = [inst for inst in self.all_instances if inst[1]["type"] == "bbox"]
            if not bbox_instances:
                break

            _, random_instance = self.random_gen.choice(bbox_instances)
            proposed_xyxy = random_instance["xyxy"]
            proposed_xywh = xyxy_to_xywh(proposed_xyxy)

            if not any(self._intersects(proposed_xywh, gt_boxes_xywh[i]) for i in range(len(gt_boxes_xywh))):
                break

        if _attempt_idx == max_attempts - 1:
            warnings.warn(
                "No valid background patch found. Returning a black square. Please check your data.",
                stacklevel=2,
            )
            return Image.new("RGB", (100, 100), (0, 0, 0)), self.background_label

        crop = self._crop_from_xyxy(image, random_instance["xyxy"])
        return crop, self.background_label

    def _load_image_data(self, row):
        """Load image data from a table row."""
        image_bytes = tlc.Url(row[self.image_column_name]).read_bytes()
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
