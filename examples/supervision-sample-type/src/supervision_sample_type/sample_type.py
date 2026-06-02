# Copyright 2026 3LC Inc. All rights reserved.

"""Custom sample type that accepts Roboflow ``supervision`` detections.

This module demonstrates an **inline** sample type built on top of the 3LC
annotation dataclasses. The sample type accepts :class:`supervision.Detections`
objects on write, serializes them to the standard ``bounding_boxes_2d`` wire
format (so the 3LC Dashboard renders them as ordinary bounding boxes), and
returns :class:`supervision.Detections` objects again in sample view.

The trick is that we do *not* invent a new wire format. We convert
``sv.Detections`` to a :class:`BoundingBoxes2D` (the builtin 2D bounding-box
annotation dataclass) and reuse its ``to_row()`` / ``from_row()`` and ``schema()``.
The only thing that makes this a *custom* sample type is that the schema's
``sample_type`` points at us instead of ``"bounding_boxes_2d"``, so reads come
back as ``sv.Detections`` rather than ``BoundingBoxes2D``.

What round-trips:

- ``xyxy`` bounding boxes (absolute pixels) ↔ ``BoundingBoxes2D.bounding_boxes``
- ``class_id`` ↔ per-instance labels (rendered against the schema's class map)
- ``confidence`` ↔ per-instance confidences
- ``tracker_id`` ↔ a per-instance extra (opt in via ``schema(include_tracker_id=True)``)
- image dimensions ↔ scene bounds, carried on ``Detections.metadata`` under
  ``"image_width"`` / ``"image_height"`` (needed for the Dashboard to place boxes)

The sample type is registered as ``"supervision_detections"`` via the
``tlc.sample_types`` entry point in ``pyproject.toml`` — ``pip install`` makes it
available. It can also be registered explicitly with
``@tlc.sample_types.register_sample_type("supervision_detections")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import tlc
from tlc.data_types import BoundingBoxes2D
from tlc.sample_types import SampleType, ValidationError

if TYPE_CHECKING:
    import supervision as sv
    from tlc.schemas._schema import Schema, ValueMapLike

SAMPLE_TYPE_NAME = "supervision_detections"

# Key under which we stash image dimensions on ``Detections.metadata``. Supervision
# treats ``metadata`` as collection-level (not per-instance), which is exactly right
# for image width/height. 3LC needs these as scene bounds so the Dashboard knows the
# coordinate frame the boxes live in.
IMAGE_WIDTH_KEY = "image_width"
IMAGE_HEIGHT_KEY = "image_height"

# Per-instance extra key used to round-trip supervision tracker ids.
TRACKER_ID_KEY = "tracker_id"


def _detections_class() -> type[sv.Detections] | None:
    """Return ``supervision.Detections`` if supervision is importable, else ``None``."""
    try:
        import supervision as sv
    except ImportError:
        return None
    return sv.Detections


def detections_to_bounding_boxes_2d(detections: sv.Detections) -> BoundingBoxes2D:
    """Convert a :class:`supervision.Detections` into a :class:`BoundingBoxes2D`.

    Args:
        detections: The supervision detections to convert. ``xyxy`` is interpreted as
            absolute-pixel ``[x_min, y_min, x_max, y_max]`` boxes.

    Returns:
        A 2D bounding-box annotation dataclass holding the same boxes, labels,
        confidences, tracker ids, and image bounds.

    """
    num_instances = len(detections)
    boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(num_instances, 4)

    labels = None if detections.class_id is None else np.asarray(detections.class_id).astype(np.int64)
    confidences = None if detections.confidence is None else np.asarray(detections.confidence, dtype=np.float32)

    per_instance_extras: dict[str, np.ndarray] = {}
    if detections.tracker_id is not None:
        per_instance_extras[TRACKER_ID_KEY] = np.asarray(detections.tracker_id).astype(np.int64)

    metadata = getattr(detections, "metadata", {}) or {}
    image_width = metadata.get(IMAGE_WIDTH_KEY)
    image_height = metadata.get(IMAGE_HEIGHT_KEY)

    return BoundingBoxes2D(
        bounding_boxes=boxes,
        labels=labels,
        confidences=confidences,
        per_instance_extras=per_instance_extras,
        x_max=float(image_width) if image_width is not None else None,
        y_max=float(image_height) if image_height is not None else None,
    )


def bounding_boxes_2d_to_detections(bounding_boxes: BoundingBoxes2D) -> sv.Detections:
    """Convert a :class:`BoundingBoxes2D` back into a :class:`supervision.Detections`.

    Args:
        bounding_boxes: The 2D bounding-box annotation dataclass to convert.

    Returns:
        A supervision detections object with image dimensions restored on ``metadata``.

    Raises:
        ImportError: If the ``supervision`` package is not installed.

    """
    detections_cls = _detections_class()
    if detections_cls is None:
        msg = "The 'supervision' package is required to read this column. Install it with `pip install supervision`."
        raise ImportError(msg)

    num_instances = bounding_boxes.num_instances
    xyxy = bounding_boxes.bounding_boxes.astype(np.float32).reshape(num_instances, 4)

    class_id = None if bounding_boxes.labels is None else np.asarray(bounding_boxes.labels).astype(int)
    confidence = (
        None if bounding_boxes.confidences is None else np.asarray(bounding_boxes.confidences, dtype=np.float32)
    )

    extras = bounding_boxes.per_instance_extras or {}
    tracker_id = np.asarray(extras[TRACKER_ID_KEY]).astype(int) if TRACKER_ID_KEY in extras else None

    metadata: dict[str, Any] = {}
    if bounding_boxes.x_max is not None:
        metadata[IMAGE_WIDTH_KEY] = float(bounding_boxes.x_max)
    if bounding_boxes.y_max is not None:
        metadata[IMAGE_HEIGHT_KEY] = float(bounding_boxes.y_max)

    return detections_cls(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        tracker_id=tracker_id,
        metadata=metadata,
    )


class SupervisionDetectionsSampleType(SampleType):
    """Inline sample type that stores Roboflow ``supervision`` detections as 3LC bounding boxes.

    On write, accepts :class:`supervision.Detections` and serializes them via the
    builtin ``bounding_boxes_2d`` wire format. In sample view, returns
    :class:`supervision.Detections` rebuilt from the stored boxes.
    """

    def to_row(self, sample: Any) -> Any:
        """Convert a :class:`supervision.Detections` to the 3LC bounding-box row format."""
        return detections_to_bounding_boxes_2d(sample).to_row()

    def from_row(self, data: Any) -> Any:
        """Rebuild a :class:`supervision.Detections` from a 3LC bounding-box row."""
        return bounding_boxes_2d_to_detections(BoundingBoxes2D.from_row(data))

    def accepts(self, value: Any) -> bool:
        """Return ``True`` for live :class:`supervision.Detections` instances."""
        detections_cls = _detections_class()
        return detections_cls is not None and isinstance(value, detections_cls)

    def validate_sample(self, sample: Any) -> list[ValidationError]:
        """Check that the sample is a :class:`supervision.Detections` with a well-formed ``xyxy``."""
        detections_cls = _detections_class()
        if detections_cls is None or not isinstance(sample, detections_cls):
            got = type(sample).__name__
            return [ValidationError("", f"Expected supervision.Detections, got {got}")]

        xyxy = np.asarray(sample.xyxy)
        n = len(sample)
        errors: list[ValidationError] = []
        if not (xyxy.ndim == 2 and xyxy.shape[1] == 4) and n > 0:
            errors.append(ValidationError("xyxy", f"Expected shape (N, 4), got {xyxy.shape}"))
        if sample.class_id is not None and len(sample.class_id) != n:
            errors.append(ValidationError("class_id", f"Length {len(sample.class_id)} != {n} detections"))
        if sample.confidence is not None and len(sample.confidence) != n:
            errors.append(ValidationError("confidence", f"Length {len(sample.confidence)} != {n} detections"))
        return errors

    @classmethod
    def schema(
        cls,
        classes: ValueMapLike | None = None,
        *,
        include_confidence: bool = True,
        include_tracker_id: bool = False,
        image_width: float | None = None,
        image_height: float | None = None,
        display_name: str = "",
        description: str = "",
        writable: bool = True,
        default_visible: bool = True,
    ) -> Schema:
        """Build a 2D bounding-box {py:class}`~tlc.Schema` wired to this sample type.

        Reuses {py:meth}`BoundingBoxes2D.schema` for the column structure (so the
        Dashboard renders boxes), then points ``sample_type`` at
        ``"supervision_detections"`` so reads return :class:`supervision.Detections`.

        Args:
            classes: Class map for per-instance labels — a list of class names, a dict of
                ``class_id`` to name, or ``None`` to omit labels. ``class_id`` values on the
                detections index into this map.
            include_confidence: Whether to add a per-instance confidence field.
            include_tracker_id: Whether to round-trip ``Detections.tracker_id`` as a
                per-instance integer extra.
            image_width: Default scene width (used when a sample omits image dimensions).
            image_height: Default scene height.
            display_name: Column display name in the Dashboard.
            description: Column description.
            writable: Whether the column is editable in the Dashboard.
            default_visible: Whether the column is visible by default.

        Returns:
            A column schema that accepts ``sv.Detections`` on write and returns them on read.

        """
        per_instance_schemas: dict[str, Schema] | None = None
        if include_tracker_id:
            per_instance_schemas = {TRACKER_ID_KEY: tlc.schemas.Int32Schema(shape=(-1,))}

        schema = BoundingBoxes2D.schema(
            classes=classes,
            include_per_instance_confidence=include_confidence,
            x_max_default=image_width,
            y_max_default=image_height,
            display_name=display_name,
            description=description,
            writable=writable,
            default_visible=default_visible,
            per_instance_schemas=per_instance_schemas,
        )
        # Route serialization through this custom sample type instead of the builtin one.
        schema.sample_type = SAMPLE_TYPE_NAME
        return schema
