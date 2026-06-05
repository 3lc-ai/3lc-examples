# Copyright 2026 3LC Inc. All rights reserved.

"""POC: semantic segmentation as a distinct, RLE-backed sample type.

Reuses the instance-segmentation RLE *wire format* byte-for-byte (one RLE per
class, ``instance_properties.label`` = class id) but is its own sample type
(``"semantic_segmentation"``): the sample form is a dense ``(H, W)`` integer
label map — an exhaustive partition of the image — not a set of potentially
overlapping instances.

Registered as ``"semantic_segmentation"`` via the ``tlc.sample_types`` entry
point in ``pyproject.toml`` and via the decorator below (so a plain import of
this module is enough when the package is not installed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tlc.constants import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INSTANCE_PROPERTIES,
    LABEL,
    NUMBER_ROLE_LABEL,
    RLES,
)
from tlc import Schema
from tlc.sample_types import SampleType, register_sample_type
from tlc.schemas import MapElement
from tlc.schemas.values import (
    DimensionNumericValue,
    InstanceSegmentationRleBytesStringValue,
    Int32Value,
)


@dataclass
class SemanticSegmentation:
    """Sample form for semantic segmentation: a dense integer label map.

    ``label_map`` is a ``(H, W)`` integer array where each pixel holds a class id.
    """

    image_width: int
    image_height: int
    label_map: np.ndarray

    def __post_init__(self) -> None:
        self.label_map = np.asarray(self.label_map)
        if self.label_map.ndim != 2:
            raise ValueError(f"label_map must be 2D (H, W), got {self.label_map.ndim}D")
        h, w = self.label_map.shape
        if (h, w) != (self.image_height, self.image_width):
            raise ValueError(
                f"label_map shape {(h, w)} does not match (image_height, image_width) "
                f"{(self.image_height, self.image_width)}"
            )

    @property
    def class_ids(self) -> np.ndarray:
        """Class ids present in the label map, sorted ascending."""
        return np.unique(self.label_map)


@register_sample_type("semantic_segmentation")
class SemanticSegmentationSampleType(SampleType):
    """SampleType for semantic segmentation, stored as per-class RLEs on the wire.

    Sample form: :class:`SemanticSegmentation` dataclass (dense ``(H, W)`` label map).
    Storable form: dict with keys ``{image_height, image_width, instance_properties, rles}``
    — identical to the instance segmentation wire format, with one RLE per class
    present in the label map and ``instance_properties.label`` holding the class ids.
    """

    def to_row(self, sample: SemanticSegmentation) -> dict[str, Any]:
        """Encode a dense label map as one RLE per class present."""
        from tlc.helpers.segmentation_helper import SegmentationHelper

        class_ids = sample.class_ids
        masks = np.stack(
            [(sample.label_map == class_id) for class_id in class_ids],
            axis=-1,
        ).astype(np.uint8)
        rles = SegmentationHelper.rles_from_masks(masks)

        return {
            IMAGE_HEIGHT: sample.image_height,
            IMAGE_WIDTH: sample.image_width,
            INSTANCE_PROPERTIES: {LABEL: [int(class_id) for class_id in class_ids]},
            RLES: [rle["counts"] for rle in rles],
        }

    def from_row(self, data: dict[str, Any]) -> SemanticSegmentation:
        """Decode per-class RLEs back into a dense label map."""
        from tlc.helpers.segmentation_helper import SegmentationHelper

        image_height = data[IMAGE_HEIGHT]
        image_width = data[IMAGE_WIDTH]
        class_ids = data.get(INSTANCE_PROPERTIES, {}).get(LABEL, [])

        label_map = np.zeros((image_height, image_width), dtype=np.int32)
        if data[RLES]:
            rles = [{"size": [image_height, image_width], "counts": rle} for rle in data[RLES]]
            masks = SegmentationHelper.masks_from_rles(rles)
            for i, class_id in enumerate(class_ids):
                label_map[masks[:, :, i] > 0] = class_id

        return SemanticSegmentation(
            image_width=image_width,
            image_height=image_height,
            label_map=label_map,
        )

    def accepts(self, value: Any) -> bool:
        """Auto-detection: accept :class:`SemanticSegmentation` instances."""
        return isinstance(value, SemanticSegmentation)

    @classmethod
    def schema(
        cls,
        classes: Any,
        *,
        display_name: str = "",
        description: str = "",
        writable: bool = True,
        default_visible: bool = True,
        display_importance: float = 0,
    ) -> Schema:
        """Build the column schema for semantic segmentation.

        Args:
            classes: A class map (list of names, or dict of numeric keys to names or MapElements).
            display_name: Column display name in the Dashboard.
            description: Column description.
            writable: Whether the column is editable in the Dashboard.
            default_visible: Whether the column is visible by default.
            display_importance: Ordering weight for column display.

        """
        label_map = MapElement._construct_value_map(classes)

        schema = Schema(
            display_name=display_name,
            description=description,
            writable=writable,
            default_visible=default_visible,
            display_importance=display_importance,
            sample_type="semantic_segmentation",  # type: ignore[arg-type]
        )

        schema.add_sub_value(IMAGE_HEIGHT, Int32Value(), computable=False)
        schema.add_sub_value(IMAGE_WIDTH, Int32Value(), computable=False)

        instance_properties_schema = Schema()
        label_schema = Schema(
            value=Int32Value(
                value_map=label_map,
                value_min=int(min(label_map.keys())),
                value_max=int(max(label_map.keys())),
                number_role=NUMBER_ROLE_LABEL,
            ),
            computable=False,
            writable=writable,
            size0=DimensionNumericValue(),
        )
        instance_properties_schema.add_sub_schema(LABEL, label_schema)
        schema.add_sub_schema(INSTANCE_PROPERTIES, instance_properties_schema)

        rle_schema = Schema(
            value=InstanceSegmentationRleBytesStringValue(),
            size0=DimensionNumericValue(),
        )
        schema.add_sub_schema(RLES, rle_schema)

        return schema
