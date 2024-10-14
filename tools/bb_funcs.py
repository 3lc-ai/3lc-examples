from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Generator

from PIL import Image

import tlc
from tlc.core.builtins.types.bb_crop_interface import BBCropInterface
from tlc.core.objects.table import Table
from tlc.core.schema import Schema
from tlc.core.schema_helper import SchemaHelper
from tlc.core.url import Url

random.seed(42)


class BBAccessor:
    def __init__(self, table: Table, row_filter: Callable | None = None) -> None:
        """
        Initializes the BBAccessor with a table of bounding boxes.

        :param table: The table containing rows with bounding boxes.
        :param row_filter: A filter function to filter rows (optional).
        """
        self.table = table
        self.row_filter = row_filter
        self._indices = self._build_indices()
        self._bb_schema = table.rows_schema.values["bbs"].values["bb_list"]

    def _build_indices(self) -> list[dict[str, Any]]:
        """
        Precomputes the indices for quick access.

        :returns: A list of (row_idx, bb_idx) pairs representing available bounding boxes.
        """
        indices = []
        for row_idx, row in enumerate(self.table.table_rows):
            if self.row_filter and not self.row_filter(row):
                continue
            # image_url = row["image"]
            image_bbs = row["bbs"]["bb_list"]
            if len(image_bbs) == 0:
                continue
            for bb_idx in range(len(image_bbs)):
                indices.append(
                    {
                        "row_idx": row_idx,
                        "bb_idx": bb_idx,
                        "image_url": row["image"],
                        "bb": row["bbs"]["bb_list"][bb_idx],
                    }
                )
        return indices

    @property
    def bb_schema(self) -> Schema:
        """
        Exposes the bounding box schema.

        :returns: The bounding box schema.
        """
        return self._bb_schema

    def __len__(self) -> int:
        """
        Returns the total number of bounding boxes available.

        :returns: The length of the bounding box indices.
        """
        return len(self._indices)

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """
        Iterates over the bounding boxes by yielding (row_idx, bb_idx) pairs.

        :yields: tuples of (row_idx, bb_idx) representing the location of bounding boxes.
        """
        yield from self._indices

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Single-index access to bounding boxes by index.

        :param idx: The index of the bounding box to retrieve.
        :returns: A tuple (row_idx, bb_idx) representing the bounding box.
        :raises IndexError: If the index is out of range.
        """
        if idx < 0 or idx >= len(self._indices):
            msg = "Index out of range"
            raise IndexError(msg)
        return self._indices[idx]


class ResizeWithPadding:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # Get the original dimensions
        w, h = img.size

        # Determine scale factor to maintain aspect ratio
        scale = min(self.target_size / w, self.target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image
        resized_img = img.resize((new_w, new_h), Image.BILINEAR)

        # Create a new black image of the target size
        new_img = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))

        # Paste the resized image onto the black canvas (center it)
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        new_img.paste(resized_img, (paste_x, paste_y))

        return new_img


def write_bb_crops_to_image_folders(
    table: Table,
    root_url: Url | str | Path,
    row_filter: Callable | None = None,
    bb_filter: Callable | None = None,
    train_split: float = 0.8,
    max_imgs_per_folder: dict[str, int] = {"train": 1000, "val": 100},
    crop_strategy: str = "resize",
) -> None:
    """
    Write all the bounding box crops to image folders.

    :param table: The table containing rows with bounding boxes.
    :param root_url: The root path where the folders should be created.
    :param row_filter: A filter function to filter rows (optional).
    :param bb_filter: A filter function to filter bounding boxes (optional).
    :param train_split: Ratio of images going to train folder (rest to val folder).
    :param max_imgs_per_folder: Max images to save in each label's folder.
    :returns: None
    """
    # Initialize BBAccessor to iterate over bounding boxes
    accessor = BBAccessor(table, row_filter=row_filter)
    value_map = table.get_value_map("bbs.bb_list.label")
    if not value_map:
        msg = "No value map found for 'bbs.bb_list.label'."
        raise ValueError(msg)

    simple_value_map = SchemaHelper.to_simple_value_map(value_map)

    # Ensure root directory exists
    root_path = Path(root_url)

    # Counters to track the number of images per label folder
    folder_counters: dict[str, dict[str, int]] = {"val": {}, "train": {}}

    # Iterate over train set
    for bb_struct in accessor:
        split = "train" if random.random() < train_split else "val"
        row_idx = bb_struct["row_idx"]
        bb_idx = bb_struct["bb_idx"]
        image_url = Url(bb_struct["image_url"]).to_absolute().to_str()
        bb = bb_struct["bb"]
        if bb_filter and not bb_filter(bb):
            continue

        label_float = bb["label"]
        label_str = simple_value_map[label_float]
        filename = f"{row_idx}_{bb_idx}.jpg"
        crop_path = root_path / split / label_str / filename

        if crop_path.exists():
            print("BB already written")
            continue

        folder_counters[split][label_str] = folder_counters[split].get(label_str, 0) + 1
        counter = folder_counters[split][label_str]

        if split == "train" and counter > max_imgs_per_folder["train"]:
            continue
        if split == "val" and counter > max_imgs_per_folder["val"]:
            continue

        # Early exit if all folders are full:
        if all(
            counter > max_imgs_per_folder[split]
            for split in ["train", "val"]
            for counter in folder_counters[split].values()
        ):
            print("All folders full")
            break

        image = Image.open(image_url)
        crop = BBCropInterface.crop(image, bb, accessor.bb_schema)

        if crop_strategy == "resize_with_padding":
            crop = ResizeWithPadding(224)(crop)
        else:
            msg = f"Unknown crop strategy: {crop_strategy}"
            raise NotImplementedError(msg)

        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path)
