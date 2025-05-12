"""Utilities for handling label mapping in bounding box classification."""

from __future__ import annotations

from copy import deepcopy

import tlc


def create_label_mappings(
    label_map: dict, include_background: bool = False, force_no_background: bool = False
) -> tuple[dict[int, int], dict[int, int], int | None, bool]:
    """Create bidirectional mappings between original labels and contiguous indices.

    Args:
        label_map: Original label map from the table
        include_background: Whether to add a background class (will be forced True for single-class tables)
        force_no_background: If True, never add background regardless of other settings (used for validation)

    Returns:
        tuple of:
        - label_to_contiguous: Mapping from original labels to contiguous indices
        - contiguous_to_label: Mapping from contiguous indices back to original labels
        - background_label: The background label value if using background, else None
        - add_background: Whether background class was actually added
    """
    # Force background for single class, unless explicitly forbidden
    add_background = (len(label_map) == 1 or include_background) and not force_no_background

    # Find the next available label value (handle non-contiguous labels)
    background_label = max(label_map.keys()) + 1 if add_background else None

    # Create mapping from original labels to contiguous indices
    label_to_contiguous = {label: idx for idx, label in enumerate(label_map.keys())}
    if add_background:
        label_to_contiguous[background_label] = len(label_to_contiguous)

    # Create reverse mapping
    contiguous_to_label = {idx: label for label, idx in label_to_contiguous.items()}

    return label_to_contiguous, contiguous_to_label, background_label, add_background


def get_label_name(label: int, label_map: dict, background_label: int | None = None) -> str:
    """Get a human-readable name for a label.

    Args:
        label: The label value
        label_map: The original label map from the table
        background_label: The background label value if using background class

    Returns:
        A human-readable name for the label
    """
    if background_label is not None and label == background_label:
        return "background"
    return str(label_map.get(label, f"unknown_{label}"))


def setup_label_schema(
    bb_list_schema: tlc.Schema, background_label: int | None = None
) -> tuple[tlc.Schema, tlc.Schema]:
    """Set up label and confidence schemas for the table.

    Args:
        bb_list_schema: The bounding box list schema
        background_label: The background label value if using background class

    Returns:
        tuple of:
        - label_schema: Schema for the label column
        - confidence_schema: Schema for the confidence column
    """
    # Create label and confidence schemas
    label_schema = deepcopy(bb_list_schema.values["label"])
    if background_label is not None:
        assert hasattr(label_schema.value, "map") and label_schema.value.map is not None
        label_schema.value.map[background_label] = tlc.MapElement("background")
    label_schema.writable = False

    confidence_schema = tlc.Schema(value=tlc.Float32Value(), writable=False)

    return label_schema, confidence_schema
