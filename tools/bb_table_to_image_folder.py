from __future__ import annotations

import os
import random
from io import BytesIO

import tlc
from PIL import Image


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


def write_bb_crop_tables_from_bb_table(table, save_images_dir, value_map, class_names):
    bb_schema = table.schema.values["rows"].values["bbs"].values["bb_list"]
    resize = ResizeWithPadding(224)
    for row_idx, row in enumerate(table):
        image_filename = row["image"]
        image_bbs = row["bbs"]["bb_list"]

        if len(image_bbs) == 0:
            continue

        image_bytes = tlc.Url(image_filename).read()
        image = Image.open(BytesIO(image_bytes))
        w, h = image.size

        for idx, bb in enumerate(image_bbs):
            label = bb["label"]
            if label not in value_map:
                continue
            crop = tlc.BBCropInterface.crop(image, bb, bb_schema, image_height=h, image_width=w)
            crop = resize(crop)

            if save_images_dir:
                dir = "train" if random.random() < 0.8 else "val"
                cropped_image_filename = f"{row_idx}_{idx}.jpg"
                cropped_image_path = os.path.join(
                    save_images_dir, dir, class_names[value_map[bb["label"]]], cropped_image_filename
                )
                os.makedirs(os.path.dirname(cropped_image_path), exist_ok=True)
                crop.save(
                    cropped_image_path,
                )


if __name__ == "__main__":
    # input_table = tlc.Table.from_names(project_name="DCVAI", dataset_name="dcvai_train", table_name="initial")
    input_table = tlc.Table.from_url(
        "C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/DCVAI/datasets/dcvai_train/tables/set-all-weights-1-4-real"
    )

    GLASSES = 4.0
    SUNGLASSES = 5.0
    MAN = 9.0
    WOMAN = 10.0
    GIRL = 11.0
    BOY = 12.0

    def has_girl_boy_man_woman(row):
        bb_list = row["bbs"]["bb_list"]
        return any(bb["label"] in [MAN, WOMAN, GIRL, BOY] for bb in bb_list)

    def has_weight_one(row):
        return row["weight"] == 1.0

    # value_map = {MAN: 0.0, WOMAN: 1.0, GIRL: 2.0, BOY: 3.0}
    value_map = {GLASSES: 0.0, SUNGLASSES: 1.0}
    class_names = {
        0.0: "Glasses",
        1.0: "Sunglasses",
    }

    # class_names = {
    #     0.0: "Man",
    #     1.0: "Woman",
    #     2.0: "Girl",
    #     3.0: "Boy",
    # }
    write_bb_crop_tables_from_bb_table(
        input_table,
        save_images_dir="./glasses-dataset",
        value_map=value_map,
        class_names=class_names,
    )
