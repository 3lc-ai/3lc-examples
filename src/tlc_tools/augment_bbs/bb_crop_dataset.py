from __future__ import annotations

import random
import warnings
from collections.abc import Mapping
from io import BytesIO

import tlc
import torch
from PIL import Image
from torch.utils.data import Dataset


class BBCropDataset(Dataset):
    """Custom dataset for cropping bounding boxes and generating background patches."""

    def __init__(
        self,
        table: tlc.Table,
        transform=None,
        label_map: Mapping | None = None,
        add_background: bool = False,
        background_freq: float = 0.5,
        image_column_name: str = "image",
        x_max_offset: float = 0.0,
        y_max_offset: float = 0.0,
        y_scale_range: tuple[float, float] = (1.0, 1.0),
        x_scale_range: tuple[float, float] = (1.0, 1.0),
    ):
        """
        :param table: The input table containing image and bounding box data.
        :param transform: Transformations to apply to cropped images.
        :param label_map: Mapping from original labels to contiguous integer labels.
        :param add_background: Whether to include background patches.
        :param background_freq: Probability of sampling a background patch.
        :param x_max_offset: Maximum offset in the x direction for bounding box cropping.
        :param y_max_offset: Maximum offset in the y direction for bounding box cropping.
        :param y_scale_range: Range of scaling factors in the y direction for bounding box cropping.
        :param x_scale_range: Range of scaling factors in the x direction for bounding box cropping.
        """
        self.table = table
        self.transform = transform
        self.label_map = label_map or table.get_value_map("bbs.bb_list.label")
        self.image_column_name = image_column_name

        if not self.label_map:
            raise ValueError("No label map found. Expecting label map under the key 'bbs.bb_list.label'.")

        self.bb_schema = table.schema.values["rows"].values["bbs"].values["bb_list"]

        self.add_background = add_background
        self.background_freq = background_freq

        self.background_label = int(max(self.label_map.keys()) + 1) if add_background else None
        self.label_2_contiguous_idx = {label: idx for idx, label in enumerate(self.label_map.keys())}
        self.label_2_contiguous_idx[self.background_label] = len(self.label_2_contiguous_idx)

        self.random_gen = random.Random(42)  # Fixed seed for reproducibility
        self.x_max_offset = x_max_offset
        self.y_max_offset = y_max_offset
        self.y_scale_range = y_scale_range
        self.x_scale_range = x_scale_range

        # Create a list of (image_idx, bb) pairs for all bounding boxes
        self.all_bbs = []
        for idx, row in enumerate(self.table.table_rows):
            bbs = row["bbs"]["bb_list"]
            for bb in bbs:
                self.all_bbs.append((idx, bb))

    def __len__(self) -> int:
        return len(self.all_bbs)  # Return total number of bounding boxes

    def __getitem__(self, idx: int):
        """
        Fetch a sample from the dataset.

        :param idx: int, index of the specific bounding box.
        :returns: (cropped image, label) where label is a tensor.
        """
        # Determine if a background patch should be generated
        is_background = self.add_background and self.random_gen.random() < self.background_freq

        if is_background:
            # Select a random row for background
            row_idx = self.random_gen.randint(0, len(self.table) - 1)
            row = self.table.table_rows[row_idx]
            image = self.load_image_data(row)
            bbs = row["bbs"]["bb_list"]
            crop, label = self.generate_background(image, bbs)
        else:
            # Get the specific bounding box and its image
            image_idx, bb = self.all_bbs[idx]
            row = self.table.table_rows[image_idx]
            image = self.load_image_data(row)
            crop = tlc.BBCropInterface.crop(
                image,
                bb,
                self.bb_schema,
                x_max_offset=self.x_max_offset,
                y_max_offset=self.y_max_offset,
                y_scale_range=self.y_scale_range,
                x_scale_range=self.x_scale_range,
            )
            label = torch.tensor(self.label_2_contiguous_idx[bb["label"]], dtype=torch.long)

        if self.transform:
            crop = self.transform(crop)

        return crop, label

    def load_image_data(self, row):
        image_bytes = tlc.Url(row[self.image_column_name]).read()
        image = Image.open(BytesIO(image_bytes))
        return image

    def generate_background(self, image, bbs, max_attempts=100):
        """
        Generate a background patch from the image.

        :param image: The input image.
        :param bbs: Bounding boxes associated with the image.

        Returns:
            tuple: (background patch, background label) where label is a tensor.
        """
        image_width, image_height = image.size
        bb_factory = tlc.BoundingBox.from_schema(self.bb_schema)
        gt_boxes_xywh = [
            bb_factory([bb["x0"], bb["y0"], bb["x1"], bb["y1"]])
            .to_top_left_xywh()
            .denormalize(image_width, image_height)
            for bb in bbs
        ]

        for i in range(max_attempts):  # noqa: B007
            # Pick a random bounding box from all_bbs
            _, random_bb = self.random_gen.choice(self.all_bbs)

            # Convert random bb to xywh format using the factory
            proposed_box = (
                bb_factory([random_bb["x0"], random_bb["y0"], random_bb["x1"], random_bb["y1"]])
                .to_top_left_xywh()
                .denormalize(image_width, image_height)
            )

            # Ensure the proposed box does not intersect any ground truth boxes
            if not any(self._intersects(proposed_box, gt_box) for gt_box in gt_boxes_xywh):
                break

        if i == max_attempts - 1:
            # Return a 100x100 black square if no valid background patch is found
            warnings.warn(
                "No valid background patch found. Returning a black square. Please check your data.",
                stacklevel=2,
            )
            return Image.new("RGB", (100, 100), color=(0, 0, 0)), torch.tensor(
                self.label_2_contiguous_idx[self.background_label], dtype=torch.long
            )

        # Crop the background patch from the image
        background_patch = image.crop(
            (proposed_box[0], proposed_box[1], proposed_box[0] + proposed_box[2], proposed_box[1] + proposed_box[3])
        )
        return background_patch, torch.tensor(self.label_2_contiguous_idx[self.background_label], dtype=torch.long)

    @staticmethod
    def _intersects(box1: list[int], box2: list[int]) -> bool:
        """
        Check if two bounding boxes intersect.

        :param box1: First bounding box [x, y, w, h].
        :param box2: Second bounding box [x, y, w, h].
        :returns: True if boxes intersect, otherwise False.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)
