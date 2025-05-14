"""Utilities for handling label mapping in bounding box classification."""

from __future__ import annotations


def create_label_mappings(
    label_map: dict,
    include_background: bool = False,
) -> tuple[dict[int, int], dict[int, int], int | None, bool]:
    """Create bidirectional mappings between original labels and contiguous indices.

    :param label_map: Original label map from the table
    :param include_background: Whether to add a background class (will be forced True for single-class tables)

    :returns:
        tuple of:
        - label_to_contiguous: Mapping from original labels to contiguous indices
        - contiguous_to_label: Mapping from contiguous indices back to original labels
        - background_label: The background label value if using background, else None
        - add_background: Whether background class was actually added
    """
    # Force background for single class, unless explicitly forbidden
    add_background = len(label_map) == 1 or include_background

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

    :param label: The label value
    :param label_map: The original label map from the table
    :param background_label: The background label value if using background class

    :returns: A human-readable name for the label
    """
    if background_label is not None and label == background_label:
        return "background"
    return str(label_map.get(label, f"unknown_{label}"))
