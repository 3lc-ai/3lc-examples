import random
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
        is_train: bool = True,
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
        :param is_train: Whether the dataset is used for training (affects background generation).
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
        self.is_train = is_train
        self.background_label = len(self.label_map) if add_background else None
        self.random_gen = random.Random(42)  # Fixed seed for reproducibility
        self.x_max_offset = x_max_offset
        self.y_max_offset = y_max_offset
        self.y_scale_range = y_scale_range
        self.x_scale_range = x_scale_range

    def __len__(self) -> int:
        return len(self.table)  # Dataset length tied to the number of table rows

    def __getitem__(self, idx: int):
        """
        Fetch a sample from the dataset.

        :param idx: int, index provided by the sampler.
        :returns: (cropped image, label) where label is a tensor.
        """
        # Determine if a background patch should be generated
        is_background = self.add_background and self.is_train and self.random_gen.random() < self.background_freq

        # Select a random row for background or use the given index
        row_idx = self.random_gen.randint(0, len(self.table) - 1) if is_background else idx

        row = self.table.table_rows[row_idx]
        image = self.load_image_data(row)

        bbs = row["bbs"]["bb_list"]
        while not is_background and len(bbs) == 0:
            row_idx = self.random_gen.randint(0, len(self.table) - 1)
            row = self.table.table_rows[row_idx]
            image = self.load_image_data(row)
            bbs = row["bbs"]["bb_list"]

        if is_background:
            crop, label = self.generate_background(image, bbs)
        else:
            crop, label = self.generate_bb_crop(image, bbs)

        if self.transform:
            crop = self.transform(crop)

        return crop, label

    def load_image_data(self, row):
        image_bytes = tlc.Url(row[self.image_column_name]).read()
        image = Image.open(BytesIO(image_bytes))
        return image

    def generate_bb_crop(self, image, bbs):
        """
        Crop a bounding box from the image.

        :param image: PIL.Image, the input image.
        :param bbs: list, bounding boxes associated with the image.
        :returns: (cropped image, label) where label is a tensor.
        """
        if not bbs:
            raise ValueError("No bounding boxes found. Check your sampler.")

        random_bb = random.choice(bbs)

        crop = tlc.BBCropInterface.crop(
            image,
            random_bb,
            self.bb_schema,
            x_max_offset=self.x_max_offset,
            y_max_offset=self.y_max_offset,
            y_scale_range=self.y_scale_range,
            x_scale_range=self.x_scale_range,
        )

        label = random_bb["label"]
        return crop, torch.tensor(label, dtype=torch.long)

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
            # Generate a random box
            x = max(
                min(int(self.random_gen.normalvariate(mu=image_width // 2, sigma=image_width // 6)), image_width - 1), 0
            )
            y = max(
                min(
                    int(self.random_gen.normalvariate(mu=image_height // 2, sigma=image_height // 6)),
                    image_height - 1,
                ),
                0,
            )
            w = max(
                min(int(self.random_gen.normalvariate(mu=image_width // 8, sigma=image_width // 16)), image_width - x),
                1,
            )
            h = max(
                min(
                    int(self.random_gen.normalvariate(mu=image_height // 8, sigma=image_height // 16)), image_height - y
                ),
                1,
            )
            proposed_box = [x, y, w, h]

            # Ensure the proposed box does not intersect any ground truth boxes
            if not any(self._intersects(proposed_box, gt_box) for gt_box in gt_boxes_xywh):
                break

        if i == max_attempts - 1:
            # Return a 100x100 black square if no valid background patch is found
            return Image.new("RGB", (100, 100), color=(0, 0, 0)), torch.tensor(self.background_label, dtype=torch.long)

        # Crop the background patch from the image
        background_patch = image.crop((x, y, x + w, y + h))
        return background_patch, torch.tensor(self.background_label, dtype=torch.long)

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
