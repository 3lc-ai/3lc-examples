# Copyright 2026 3LC Inc. All rights reserved.

"""Ergonomic class-map surface for semantic segmentation (strawman).

Special classes (background, void/ignore) are marked by **reserved
``internal_name`` constants** on their ``MapElement`` — see SPEC.md §3. Users
should never type those constants; this module is the front door:

- :func:`semseg_classes` — build a value map, tagging which ids are background /
  void (their human label is preserved as ``display_name``; the reserved string
  goes to ``internal_name``).
- :func:`background_id` / :func:`void_id` / :func:`real_class_ids` — read the
  roles back off a value map (used by metrics helpers, training, the UI bridge)
  without anyone re-specifying which class is which.

The reserved names are underscore-namespaced because ``.``/``:`` are illegal in
map-element names (``_DISALLOWED_MAP_ELEMENT_NAME_CHARS`` in
``tlc._core.utils.string_validation``), and the ``__tlc_…__`` shape signals
"reserved, internal" and avoids collisions with a user's genuine "background"
class.

Example::

    from semseg_sample_type import semseg_classes, SemanticSegmentationSampleType

    classes = semseg_classes(
        {0: "background", 1: "pet", 2: "border"},
        background=0,
        void=2,
    )
    schema = SemanticSegmentationSampleType.schema(classes)

    # ...later, in a metrics helper or the training loop:
    void = void_id(classes)          # -> 2 (instead of a hardcoded IGNORE_CLASS_ID)
    reals = real_class_ids(classes)  # -> [0, 1]
"""

from __future__ import annotations

from typing import Any

from tlc.schemas import MapElement

# Reserved internal_name constants. Underscore-namespaced (dots/colons are
# disallowed in map-element names) and double-underscore-wrapped to read as
# "reserved/internal" and not collide with user class names.
TLC_SEMSEG_BACKGROUND = "__tlc_semseg_background__"
TLC_SEMSEG_VOID = "__tlc_semseg_void__"

_RESERVED = (TLC_SEMSEG_BACKGROUND, TLC_SEMSEG_VOID)

# Default render hint for background: fully transparent (it's the "absence of a
# labeled object" fill). Overridable by passing an explicit display_color.
_BACKGROUND_TRANSPARENT = "#00000000"


def semseg_classes(
    classes: Any,
    *,
    background: int | None = None,
    void: int | None = None,
) -> dict[int, MapElement]:
    """Build a semantic-segmentation value map with reserved special classes tagged.

    Args:
        classes: Anything accepted by the schema today — a list of names, or a dict
            of id -> name / MapElement. The human label of a special class is kept as
            its ``display_name``.
        background: Id of the background class, if any. Its ``internal_name`` is set
            to :data:`TLC_SEMSEG_BACKGROUND` and it defaults to a transparent color.
        void: Id of the void / ignore / border class, if any. Its ``internal_name``
            is set to :data:`TLC_SEMSEG_VOID`; excluded from metrics downstream.

    Returns:
        ``dict[int, MapElement]`` ready to pass to
        :meth:`SemanticSegmentationSampleType.schema`.

    Raises:
        ValueError: if a named id is missing from ``classes``, or background == void.
    """
    # Reuse the exact normalization the schema already uses, then re-key to int.
    normalized = {int(k): v for k, v in MapElement._construct_value_map(classes).items()}

    if background is not None and background == void:
        msg = f"background and void cannot be the same class id ({background})"
        raise ValueError(msg)

    for role_id, reserved in ((background, TLC_SEMSEG_BACKGROUND), (void, TLC_SEMSEG_VOID)):
        if role_id is None:
            continue
        if role_id not in normalized:
            msg = f"class id {role_id} (marked {reserved}) is not present in the class map"
            raise ValueError(msg)
        normalized[role_id] = _tag(normalized[role_id], reserved)

    return normalized


def _tag(element: MapElement, reserved: str) -> MapElement:
    """Return a copy of ``element`` with its reserved internal_name set.

    The original human label (whichever of internal_name / display_name was set)
    is preserved as ``display_name`` so the Dashboard still shows it.
    """
    label = element.display_name or element.internal_name
    display_color = element.display_color
    if reserved == TLC_SEMSEG_BACKGROUND and not display_color:
        display_color = _BACKGROUND_TRANSPARENT
    return MapElement(
        internal_name=reserved,
        display_name=label,
        description=element.description,
        display_color=display_color,
        url=element.url,
    )


# --- read-side helpers ------------------------------------------------------
# Consumers (metrics, training, the UI bridge) ask "which id plays this role?"
# by scanning for the reserved internal_name — no out-of-band config.


def _value_map(classes: Any) -> dict[int, MapElement]:
    """Coerce a class map (or an already-built value map) to ``dict[int, MapElement]``."""
    return {int(k): v for k, v in MapElement._construct_value_map(classes).items()}


def _id_with_internal_name(classes: Any, reserved: str) -> int | None:
    for class_id, element in _value_map(classes).items():
        if element.internal_name == reserved:
            return class_id
    return None


def background_id(classes: Any) -> int | None:
    """Id of the background class, or ``None`` if unset."""
    return _id_with_internal_name(classes, TLC_SEMSEG_BACKGROUND)


def void_id(classes: Any) -> int | None:
    """Id of the void / ignore class, or ``None`` if unset."""
    return _id_with_internal_name(classes, TLC_SEMSEG_VOID)


def real_class_ids(classes: Any) -> list[int]:
    """Class ids that are neither background nor void, sorted ascending.

    These are the rows/columns kept in the confusion matrix and the classes mIoU
    is averaged over (background inclusion is a separate, documented toggle).
    """
    value_map = _value_map(classes)
    special = {bid for bid in (background_id(value_map), void_id(value_map)) if bid is not None}
    return sorted(cid for cid in value_map if cid not in special)
