import io

import numpy as np
import tlc
import torch
import tqdm
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from tlc import BoundingBoxes2D
from tlc._core.data_formats.bb_conversions import legacy_bb_row_to_bounding_boxes_2d
from tlc.helpers.annotation_helper import get_label_path, is_legacy_bb_column

from tlc_tools.add_columns_to_table import add_columns_to_table

from .common import check_is_bb_column, infer_torch_device

MODEL_TYPE = "vit_h"
CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"


def bbs_to_segments(
    input_table: tlc.Table,
    bb_column: str = "bbs",
    image_column: str = "image",
    sam_model_type: str = MODEL_TYPE,
    checkpoint: str = CHECKPOINT,
    description: str = "Segmentation of bounding boxes",
):
    device = infer_torch_device()

    check_is_bb_column(input_table, bb_column)

    # Detect format and get value map
    column_schema = input_table.rows_schema.values[bb_column]
    is_legacy = is_legacy_bb_column(column_schema)

    label_path = get_label_path(input_table, bb_column)
    if not label_path:
        raise ValueError(f"Could not find label path for column {bb_column}")
    value_map = input_table.get_value_map(label_path)
    if not value_map:
        raise ValueError(f"Could not find value map for label path {label_path}")

    # Load the SAM model
    sam_model = sam_model_registry[sam_model_type](checkpoint=checkpoint)
    sam_model.to(device)
    sam_model.eval()

    sam_predictor = SamPredictor(sam_model)

    segmentations = []

    for row in tqdm.tqdm(input_table.table_rows, desc="Predicting with SAM", total=len(input_table)):
        buffer = io.BytesIO(tlc.Url(row[image_column]).read_bytes())
        image = np.array(Image.open(buffer).convert("RGB"))
        h, w, _ = image.shape

        sam_predictor.set_image(image)

        # Get BoundingBoxes2D — handles both legacy and new format
        raw = row[bb_column]
        bb2d = legacy_bb_row_to_bounding_boxes_2d(raw, column_schema) if is_legacy else BoundingBoxes2D.from_row(raw)

        # bb2d.bboxes is (N, 4) in absolute XYXY — exactly what SAM expects
        labels = bb2d.instance_labels.tolist() if bb2d.instance_labels is not None else []

        if bb2d.num_instances > 0:
            boxes_np = sam_predictor.transform.apply_boxes(bb2d.bboxes, sam_predictor.original_size)
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
            "segments": tlc.schemas.SegmentationMasksSchema(
                classes=value_map,
                per_instance_schemas={
                    "score": tlc.schemas.ConfidenceSchema(writable=False),
                },
            ),
        },
        output_table_name=f"{input_table.name}-with-sam-segmentations",
        description=description,
    )

    return out_table
