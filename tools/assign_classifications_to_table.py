from __future__ import annotations

from io import BytesIO
from typing import Generator

import tlc
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_WRITE_TO_FILE = True
_READ_FROM_FILE = False

value_mapping = {
    0: 12,  # boy
    1: 11,  # girl
    2: 9,  # man
    3: 10,  # woman
    None: 43,  # person
}

value_mapping = {
    0: 4,  # glasses
    1: 5,  # sunglasses
}


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


def get_yolo_classifier(model_path):
    from ultralytics import YOLO

    return YOLO(model_path)


def get_classifier(model_path):
    import timm

    model = timm.create_model("resnet50", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def get_imagenet_classifier():
    import timm

    model = timm.create_model("convnext_nano.in12k_ft_in1k", pretrained=True)
    model = model.eval()
    return model


def get_bb_iterator(
    table: tlc.Table, row_filter=None, bb_filter=None
) -> Generator[tuple[int, int, Image.Image], None, None]:
    """An iterator of BBs in a table

    Returns (row_idx, bb_idx, image)
    """
    bb_schema = table.rows_schema.values["bbs"].values["bb_list"]

    for row_idx, row in tqdm(enumerate(table), total=len(table), desc="Fetching crops..."):
        if row_filter and not row_filter(row):
            continue

        image_filename = row["image"]
        image_bbs = row["bbs"]["bb_list"]

        if len(image_bbs) == 0:
            continue

        image_bytes = tlc.Url(image_filename).read()
        image = Image.open(BytesIO(image_bytes))
        w, h = image.size

        for idx, bb in enumerate(image_bbs):
            if bb_filter and not bb_filter(bb):
                continue

            crop = tlc.BBCropInterface.crop(image, bb, bb_schema, image_height=h, image_width=w)
            yield row_idx, idx, crop


def batched(iterator, batch_size):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def apply_classifier_to_bb_table(table, classifier, row_filter, bb_filter, transform) -> list[tuple[int, int, int]]:
    edits = []  # (row_idx, bb_idx, new predicted label)

    bb_iterator = get_bb_iterator(table, row_filter, bb_filter)

    batch_size = 16
    num_bbs = sum(len(row["bbs"]["bb_list"]) for row in table)
    num_batches = num_bbs // batch_size

    for batch in tqdm(batched(bb_iterator, batch_size=8), desc="Classifying batches...", total=num_batches):
        row_idx_batch, bb_idx_batch, crop_batch = zip(*batch)
        crops = [transform(crop) for crop in crop_batch]
        crops = torch.stack(crops)
        predicted_labels = classifier(crops)

        for row_idx, bb_idx, predicted_label in zip(row_idx_batch, bb_idx_batch, predicted_labels):
            if predicted_label != -1:
                edits.append((row_idx, bb_idx, predicted_label))

    return edits


def translate_edits(original_table: tlc.Table, edits: list[tuple[int, int, int]]) -> dict:
    runs_and_values = []

    # Group edits by row_idx
    edits_by_row: dict[int, list[tuple[int, int]]] = {}
    for row_idx, bb_idx, new_label in edits:
        if row_idx not in edits_by_row:
            edits_by_row[row_idx] = []
        edits_by_row[row_idx].append((bb_idx, new_label))

    keep_originals = False
    for row_idx in range(len(original_table)):
        if row_idx in edits_by_row:
            # If the row has edits, process it
            if keep_originals:
                # Start with a copy of the original bounding boxes if we want to keep them
                bbs_copy = original_table[row_idx]["bbs"].copy()
            else:
                # Otherwise, start with an empty bb structure
                bbs_copy = {"bb_list": []}

            # Loop through the edits to update or add the new bounding boxes
            for bb_idx, new_label in edits_by_row[row_idx]:
                original_bb = original_table[row_idx]["bbs"]["bb_list"][bb_idx].copy()
                if original_bb["label"] == new_label:
                    continue
                original_bb["label"] = new_label

                # If keeping originals, just update the relevant box, otherwise add only updated boxes
                if keep_originals:
                    bbs_copy["bb_list"][bb_idx] = original_bb  # Update the original copy
                else:
                    bbs_copy["bb_list"].append(original_bb)  # Append only the updated box
        else:
            # If no edits, set an empty bb_list if keep_originals is False
            if keep_originals:
                bbs_copy = original_table[row_idx]["bbs"].copy()
            else:
                bbs_copy = {"bb_list": []}

        # Append the row and bounding boxes (either original + new, or only new/empty)
        runs_and_values.append([row_idx])
        runs_and_values.append(bbs_copy)

    translated_edits = {"bbs": {"runs_and_values": runs_and_values}}
    return translated_edits


def create_edited_table(original_table: tlc.Table, edits: list[tuple[int, int, int]]) -> tlc.Table:
    # Need to translate edits into a true "edits" object for EditedTable

    edited_table_url = original_table.url.create_sibling("assigned-cls-labels").create_unique()

    translated_edits = translate_edits(original_table, edits)

    edited_table = tlc.EditedTable(
        edited_table_url,
        input_table_url=original_table.url.to_relative(edited_table_url),
        edits=translated_edits,
        row_cache_url="./row_cache.parquet",
    )
    edited_table.get_rows_as_binary()
    return edited_table


class TimmModelWrapper:
    def __init__(self, model, value_mapping):
        self.model = model
        self.value_mapping = value_mapping

    def __call__(self, x):
        output = self.model(x)
        predicted_labels = [p.argmax().item() for p in output]
        return [self.value_mapping[label] for label in predicted_labels]


class YOLOModelWrapper:
    def __init__(self, model, value_mapping):
        self.model = model
        self.value_mapping = value_mapping

    def __call__(self, x):
        output = self.model(x)
        predicted_labels = self.get_yolo_labels(output)
        return [self.value_mapping[label] for label in predicted_labels]

    def get_yolo_labels(self, predictions):
        return [p.probs.top1 for p in predictions]


class ImageNet2DCVAIModelWrapper:
    label_maps = [
        [770],  # Footwear
        [608],  # Jeans
        None,  # Trousers
        None,  # Shorts
        None,  # Glasses
        [837],  # Sunglasses
        None,  # High heels
        [514],  # Boot
        [774],  # Sandal
        None,  # Man
        None,  # Woman
        None,  # Girl
        None,  # Boy
        [895],  # Warplane, military plane
        [751, 817],  # Race car, sports car
        [864, 555, 569, 867],  # Tow truck, fire truck, garbage truck, trailer truck
        [656, 675, 734],  # Minivan, moving van, police van
        [407],  # Ambulance
        None,  # Helicopter
        [670, 665],  # Motor scooter, moped
        [671],  # Mountain bike
        [880],  # Unicycle
        [654, 779, 874],  # Minibus, school bus, trolleybus
        [468],  # Cab, taxi
        [851],  # Television system
        [527, 664, 782],  # Desktop computer, monitor, CRT screen
        [620],  # Laptop
        [487],  # Mobile phone
        [528, 707],  # Dial phone, pay-phone
        [852],  # Tennis ball
        None,  # Tennis racket
        None,  # Table tennis racket
        [574],  # Golf ball
        None,  # Ball
        [768],  # Rugby ball
        None,  # Football
        [21],  # Kite
        [890],  # Volleyball
        [417],  # Balloon
        None,  # Hamburger
        None,  # Sandwich
        None,  # Submarine sandwich
        [934],  # Hot dog
    ]

    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        output = self.model(x)
        import torch.nn.functional as F

        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)  # shape: (batch_size, 1000)

        batch_size = probabilities.size(0)
        custom_class_probs = torch.zeros(batch_size, len(self.label_maps))  # shape: (batch_size, 43)

        # For each custom class, map to corresponding ImageNet classes
        for custom_class_idx, class_map in enumerate(self.label_maps):
            if class_map is not None:  # Ignore if no corresponding ImageNet classes
                # Fetch activations for the corresponding ImageNet classes and take max
                custom_class_probs[:, custom_class_idx] = probabilities[:, class_map].max(dim=1).values

        # Apply the threshold: For each batch element, get the index of the custom class with the max activation
        max_vals, max_indices = custom_class_probs.max(dim=1)

        # Mask out the classes where the maximum activation is below the threshold
        max_indices[max_vals < 0.1] = -1  # Set to -1 if below threshold

        # Count non -1 values in max_indices
        num_non_neg_ones = (max_indices != -1).sum().item()

        if num_non_neg_ones > 0:
            print(f"Found {num_non_neg_ones} relevant activations in batch")
        return max_indices  # shape: (batch_size,)


if __name__ == "__main__":
    original_table = tlc.Table.from_url(
        "c:/Users/gudbrand/AppData/Local/3LC/3LC/projects/DCVAI/datasets/dcvai_eval/tables/initial"
        # "C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/DCVAI/datasets/dcvai_train/tables/set-all-weights-1-4-real"
    )
    # c:/Users/gudbrand/AppData/Local/3LC/3LC/projects/DCVAI/datasets/dcvai_eval/tables/initial
    # original_table.ensure_fully_defined()

    def row_filter(row):
        return True

    unmatched_labels = [i for i, label in enumerate(ImageNet2DCVAIModelWrapper.label_maps) if label is None]

    def bb_filter(bb):
        # return bb["label"] not in unmatched_labels
        return True

    transform = transforms.Compose(
        [
            ResizeWithPadding(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # classifier = YOLOModelWrapper(
    #     get_yolo_classifier("C:/Project/ultralytics-3lc/runs/classify/train25/weights/best.pt"), value_mapping
    # )
    # classifier = TimmModelWrapper(get_classifier("C:/Project/voxel51-challenge/trained_model.pth"), value_mapping)
    classifier = ImageNet2DCVAIModelWrapper(get_imagenet_classifier())

    predicted_labels = apply_classifier_to_bb_table(
        original_table,
        classifier,
        row_filter,
        bb_filter,
        transform,
    )

    # DEBUG write predicted_labels to a file:
    file_path = tlc.Url("predicted_labels.txt").create_unique().to_str()
    if _WRITE_TO_FILE:
        with open(file_path, "w") as f:
            for row_idx, bb_idx, label in predicted_labels:
                f.write(f"{row_idx},{bb_idx},{label}\n")

    # DEBUG read predicted_labels from a file:
    if False:  # _READ_FROM_FILE:
        predicted_labels = []
        with open("predicted_labels_0000.txt", "r") as f:
            for line in f.readlines():
                row_idx, bb_idx, label = line.strip().split(",")
                predicted_labels.append((int(row_idx), int(bb_idx), int(label)))

    edited_table = create_edited_table(original_table, predicted_labels)
    print(edited_table.url)
    # assert True
