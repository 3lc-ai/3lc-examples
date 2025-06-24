import io

import numpy as np
import torch
import tqdm
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

import tlc
from tlc.core.builtins.types.bounding_box import XYXYBoundingBox
from tlc_tools.add_columns_to_table import add_columns_to_table

from .common import check_is_bb_column, infer_torch_device

MODEL_TYPE = "vit_h"
CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"


def bbs_to_segments(
    input_table: tlc.Table,
    bb_column: str = "bbs",
    bb_list_column: str = "bb_list",
    image_column: str = "image",
    sam_model_type: str = MODEL_TYPE,
    checkpoint: str = CHECKPOINT,
    description: str = "Segmentation of bounding boxes",
):
    device = infer_torch_device()

    # Checks that the bb_column contains a "bb_list" sub-column with "label", "x0", "y0", "x1", "y1" fields
    check_is_bb_column(input_table, bb_column)

    # Use BoundingBox type to allow arbitrary bounding box types as input
    bb_type = tlc.BoundingBox.from_schema(input_table.rows_schema.values[bb_column].values[bb_list_column])

    # The value map from the bb column will be used in the output table's segmentation column
    value_map = input_table.get_value_map(f"{bb_column}.{bb_list_column}.{tlc.LABEL}")

    if not value_map:
        raise ValueError(f"Could not find value map for column {bb_column}.{bb_list_column}.{tlc.LABEL}")

    # Load the SAM model
    sam_model = sam_model_registry[sam_model_type](checkpoint=checkpoint)
    sam_model.to(device)
    sam_model.eval()

    # The SAM predictor provides a convenient wrapper
    sam_predictor = SamPredictor(sam_model)

    segmentations = []

    for row in tqdm.tqdm(input_table, desc="Predicting with SAM", total=len(input_table)):
        buffer = io.BytesIO(tlc.Url(row[image_column]).read())
        image = np.array(Image.open(buffer).convert("RGB"))
        h, w, _ = image.shape

        sam_predictor.set_image(image)

        boxes = []
        labels = []

        # Gather all bounding boxes from the current image into a (num_bbs, 4) array
        for bb in row[bb_column][bb_list_column]:
            box_arr = np.array(
                XYXYBoundingBox.from_top_left_xywh(
                    bb_type([bb[tlc.X0], bb[tlc.Y0], bb[tlc.X1], bb[tlc.Y1]])
                    .to_top_left_xywh()
                    .denormalize(image_height=h, image_width=w)
                )
            )
            boxes.append(box_arr)
            labels.append(bb[tlc.LABEL])

        # Call Predictor's predict_torch instead of predict, to allow multiple box prompts in a single call
        if len(boxes):
            boxes_np = sam_predictor.transform.apply_boxes(np.array(boxes), sam_predictor.original_size)
            boxes_torch = torch.as_tensor(boxes_np, dtype=torch.float, device=device)

            masks, scores, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=boxes_torch,
                multimask_output=False,
            )

            # Convert masks from (num_bbs, 1, h, w) to (h, w, num_bbs)
            segments = masks.squeeze(1).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            scores = scores.squeeze(1).cpu().numpy().tolist()
        else:
            segments = np.zeros((0, 0, 0), dtype=np.uint8)
            scores = []

        output_row = {
            "image_height": h,
            "image_width": w,
            "masks": segments,
            "instance_properties": {
                "label": labels,
                "score": scores,
            },
        }

        segmentations.append(output_row)

    out_table = add_columns_to_table(
        input_table,
        columns={
            "segments": segmentations,
        },
        schemas={
            "segments": tlc.InstanceSegmentationMasks(
                "segmentations",
                instance_properties_structure={
                    tlc.LABEL: tlc.CategoricalLabel("label", classes=value_map),
                    "score": tlc.Schema(value=tlc.Float32Value(0, 1), writable=False),
                },
            ),
        },
        output_table_name=f"{input_table.name}-with-sam-segmentations",
        description=description,
    )

    return out_table
