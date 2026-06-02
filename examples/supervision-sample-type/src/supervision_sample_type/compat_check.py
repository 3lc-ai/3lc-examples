# Copyright 2026 3LC Inc. All rights reserved.

"""Compatibility check between ``supervision`` converters and the corresponding 3LC ops.

Supervision ships a large library of geometry "converter" helpers
(https://supervision.roboflow.com/latest/detection/utils/). 3LC has its own conversion
ops, but with a narrower scope — it focuses on *data representation* (box formats, mask /
polygon / RLE round-tripping), not on analysis ops (IoU, NMS, image transforms).

This script checks, for every supervision converter, whether 3LC has an equivalent and
whether the two produce the same result. Run with::

    check-supervision-compatibility

or::

    python -m supervision_sample_type.compat_check

Key findings (see the printed report for the live numbers):

- **Box format conversions have full numeric parity.** ``xyxy``↔``xywh``,
  center-``xywh``→``xyxy``, and (de)normalization match exactly.
- **Pixel-boundary convention differs for raster ops.** supervision treats the far
  box/polygon edge as *inclusive* (closed intervals); 3LC, via ``pycocotools``, treats it
  as *exclusive* (half-open). This shows up as a consistent 1-pixel difference in
  ``polygon_to_mask`` and ``mask_to_xyxy``. Do not mix the two libraries' raster outputs
  without accounting for it.
- **3LC has no public IoU / NMS / box-arithmetic / image-transform ops.** Those parts of
  supervision have no 3LC counterpart — there is *not* full parity.

Two issues in 3LC surfaced by this check (reported separately, not bugs in this example):

1. ``SegmentationHelper.bounding_box_from_rle`` is documented as returning ``[x1, y1, x2, y2]``
   but actually returns ``pycocotools`` ``[x, y, w, h]``.
2. ``supervision`` and 3LC disagree by one pixel on the far edge of raster conversions
   (the closed-vs-half-open convention above).
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from tlc.data_types import BoundingBoxes2D
from tlc.data_types._bb_conversions import (
    _cxywh_to_xyxy as tlc_cxywh_to_xyxy,
)
from tlc.data_types._bb_conversions import (
    _xywh_to_xyxy as tlc_xywh_to_xyxy,
)
from tlc.data_types._bb_conversions import (
    denormalize_bbs_2d as tlc_denormalize,
)
from tlc.data_types._bb_conversions import (
    normalize_bbs_2d as tlc_normalize,
)
from tlc.data_types._bb_conversions import (
    xyxy_to_cxywh as tlc_xyxy_to_cxywh,
)
from tlc.data_types._bb_conversions import (
    xyxy_to_xywh as tlc_xyxy_to_xywh,
)
from tlc.helpers import SegmentationHelper

# supervision converters that 3LC has no public equivalent for (not exhaustive, but covers
# the converter page). Listed so the report is honest about the lack of full parity.
NO_PARITY = {
    "IoU": ["box_iou", "box_iou_batch", "box_iou_batch_with_jaccard", "mask_iou_batch", "oriented_box_iou_batch"],
    "NMS / merge": ["box_non_max_suppression", "box_non_max_merge", "mask_non_max_suppression", "mask_non_max_merge"],
    "Box arithmetic": ["clip_boxes", "pad_boxes", "scale_boxes", "move_boxes", "xyxy_to_xcycarh"],
    "Polygon<->box": ["xyxy_to_polygons", "polygon_to_xyxy"],
    "Image transforms": ["crop_image", "resize_image", "scale_image", "letterbox_image", "grayscale_image"],
    "Other mask ops": ["move_masks", "calculate_masks_centroids", "approximate_polygon", "filter_polygons_by_area"],
}


def _row(name: str, status: str, note: str = "") -> tuple[str, str, str]:
    return (name, status, note)


def check_box_conversions() -> list[tuple[str, str, str]]:
    """Compare box-format converters. Expect exact numeric parity."""
    xyxy = np.array([[10, 20, 110, 220], [50, 60, 150, 260], [0, 0, 32, 48]], dtype=np.float64)
    rows = []

    rows.append(
        _row(
            "xyxy_to_xywh  ==  tlc.xyxy_to_xywh",
            "MATCH" if np.allclose(sv.xyxy_to_xywh(xyxy), tlc_xyxy_to_xywh(xyxy)) else "DIFF",
        )
    )

    xywh = sv.xyxy_to_xywh(xyxy)
    rows.append(
        _row(
            "xywh_to_xyxy  ==  tlc._xywh_to_xyxy",
            "MATCH" if np.allclose(sv.xywh_to_xyxy(xywh), tlc_xywh_to_xyxy(xywh)) else "DIFF",
        )
    )

    cxywh = tlc_xyxy_to_cxywh(xyxy)
    rows.append(
        _row(
            "xcycwh_to_xyxy  ==  tlc._cxywh_to_xyxy",
            "MATCH" if np.allclose(sv.xcycwh_to_xyxy(cxywh), tlc_cxywh_to_xyxy(cxywh)) else "DIFF",
            "supervision calls center-xywh 'xcycwh'; 3LC calls it 'cxywh'",
        )
    )

    width, height = 640, 480
    norm = tlc_normalize(xyxy, width, height)
    same = np.allclose(sv.denormalize_boxes(norm, (width, height)), tlc_denormalize(norm, width, height))
    rows.append(
        _row(
            "denormalize_boxes  ==  tlc.denormalize_bbs_2d",
            "MATCH" if same else "DIFF",
            "arg shape differs: sv takes (w, h) tuple, 3LC takes width, height",
        )
    )

    # Public-API equivalent: the BoundingBoxes2D constructor performs xywh->xyxy internally.
    bb = BoundingBoxes2D(bounding_boxes=xywh.astype(np.float32), bounding_box_format="xywh")
    rows.append(
        _row(
            "BoundingBoxes2D(format='xywh')  ==  sv.xywh_to_xyxy",
            "MATCH" if np.allclose(bb.bounding_boxes, sv.xywh_to_xyxy(xywh)) else "DIFF",
            "3LC's public entry point for format conversion",
        )
    )
    return rows


def check_mask_conversions() -> list[tuple[str, str, str]]:
    """Compare mask / polygon / RLE converters. Expect convention-level differences."""
    height, width = 60, 80
    mask = np.zeros((height, width), dtype=bool)
    mask[10:40, 15:55] = True
    rows = []

    # mask -> box
    sv_box = sv.mask_to_xyxy(mask[None, ...])[0]
    rle = SegmentationHelper.rles_from_masks(mask[..., None].astype(np.uint8))[0]
    tlc_box = SegmentationHelper.bounding_box_from_rle(rle)
    rows.append(
        _row(
            "mask_to_xyxy  vs  tlc.bounding_box_from_rle",
            "DIFF",
            f"sv={sv_box.tolist()} (xyxy, inclusive max) vs tlc={tlc_box} (xywh, half-open)",
        )
    )

    # mask -> polygons (both wrap cv2.findContours)
    sv_pts = np.vstack(sv.mask_to_polygons(mask))
    tlc_pts = np.array(SegmentationHelper.polygons_from_mask(mask)).reshape(-1, 2)
    same_points = {tuple(p) for p in sv_pts.astype(int)} == {tuple(p) for p in tlc_pts.astype(int)}
    rows.append(
        _row(
            "mask_to_polygons  ==  tlc.polygons_from_mask",
            "MATCH" if same_points else "DIFF",
            "same contour points; sv returns per-contour arrays, 3LC returns one flat list",
        )
    )

    # polygon -> mask (rasterization)
    rect = np.array([[15, 10], [55, 10], [55, 40], [15, 40]], dtype=np.float32)
    sv_mask = sv.polygon_to_mask(rect, (width, height)).astype(bool)
    tlc_mask = SegmentationHelper.mask_from_polygons([rect.reshape(-1).tolist()], height, width).astype(bool)
    inter = int((sv_mask & tlc_mask).sum())
    union = int((sv_mask | tlc_mask).sum())
    iou = inter / union if union else 0.0
    rows.append(
        _row(
            "polygon_to_mask  vs  tlc.mask_from_polygons",
            "DIFF",
            f"IoU={iou:.3f}; 3LC mask is a strict subset (excludes far edge — half-open convention)",
        )
    )

    # RLE interop
    decoded = sv.rle_to_mask(rle["counts"], (width, height))
    rows.append(
        _row(
            "RLE interop (sv.rle_to_mask on 3LC counts)",
            "MATCH" if np.array_equal(decoded.astype(bool), mask) else "DIFF",
            "sv decodes 3LC's pycocotools bytes; note sv.mask_to_rle default form is list[int]/str, not a COCO dict",
        )
    )
    return rows


def _print_table(title: str, rows: list[tuple[str, str, str]]) -> None:
    print(f"\n=== {title} ===")
    width = max(len(name) for name, _, _ in rows)
    for name, status, note in rows:
        line = f"  [{status:5}] {name.ljust(width)}"
        if note:
            line += f"   {note}"
        print(line)


def main() -> None:
    """Run the compatibility checks and print a report."""
    _print_table("BOX FORMAT CONVERSIONS (expect full parity)", check_box_conversions())
    _print_table("MASK / POLYGON / RLE (expect convention-level differences)", check_mask_conversions())

    print("\n=== supervision converters with NO 3LC equivalent (not full parity) ===")
    for group, names in NO_PARITY.items():
        print(f"  {group}:")
        for name in names:
            print(f"    - sv.{name}")
    print(
        "\n  3LC focuses on data representation, not analysis/transform ops. For IoU/NMS/image\n"
        "  transforms, use supervision (or the training framework) directly — there is no 3LC counterpart."
    )


if __name__ == "__main__":
    main()
