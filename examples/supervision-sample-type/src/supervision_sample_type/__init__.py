# Copyright 2026 3LC Inc. All rights reserved.

"""Example: a custom 3LC sample type that accepts Roboflow ``supervision`` detections.

This package shows how to interface 3LC with the ``supervision`` library by building a
custom sample type on top of the builtin :class:`tlc.BoundingBoxes2D` annotation dataclass.
Columns using it accept :class:`supervision.Detections` on write and return them on read,
while storing data in the standard 2D bounding-box format the Dashboard understands.
"""

from supervision_sample_type.sample_type import (
    SupervisionDetectionsSampleType,
    bounding_boxes_2d_to_detections,
    detections_to_bounding_boxes_2d,
)

__all__ = [
    "SupervisionDetectionsSampleType",
    "bounding_boxes_2d_to_detections",
    "detections_to_bounding_boxes_2d",
]
