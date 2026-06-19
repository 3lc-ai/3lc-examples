# Copyright 2026 3LC Inc. All rights reserved.

"""POC: semantic segmentation as an RLE-backed 3LC sample type.

The sample type and its ergonomic class-map API **graduated into core ``tlc``**
(``tlc.sample_types``) — see SPEC.md §10. This package now re-exports the core
implementation so the example scripts keep working against the dev ``tlc`` while
the original staging implementation is preserved (but no longer registered) in
``sample_type.py`` / ``class_map.py`` for reference.
"""

from tlc.sample_types import (
    TLC_SEMSEG_BACKGROUND,
    TLC_SEMSEG_VOID,
    SemanticSegmentation,
    SemanticSegmentationSampleType,
    background_id,
    real_class_ids,
    semseg_classes,
    void_id,
)

__all__ = [
    "SemanticSegmentation",
    "SemanticSegmentationSampleType",
    "semseg_classes",
    "background_id",
    "void_id",
    "real_class_ids",
    "TLC_SEMSEG_BACKGROUND",
    "TLC_SEMSEG_VOID",
]
