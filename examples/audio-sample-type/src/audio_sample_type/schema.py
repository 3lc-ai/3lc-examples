# Copyright 2026 3LC Inc. All rights reserved.

"""Convenience schema for audio columns.

This module shows how to wrap a raw :class:`~tlc.Schema` into a
user-friendly convenience class — the same pattern used by the builtin
``ImageSchema``, ``NumpyArraySchema``, etc.
"""

from __future__ import annotations

from typing import Any

from tlc import Schema
from tlc._core.schema import StringValue


class AudioSchema(Schema):
    """Schema for audio waveform data stored as WAV files.

    In sample view, values are loaded as 1D ``numpy.ndarray`` (float32).
    On write via :class:`~tlc.TableWriter`, NumPy
    arrays are saved as WAV files and the Table stores URL references.

    Args:
        sample_rate: Audio sample rate in Hz. Defaults to 16000.
        display_name: Column display name in the Dashboard.
        description: Column description.
        writable: Whether the column is editable in the Dashboard.
        visible: Whether the column is visible by default.
        display_importance: Ordering weight for column display.
        default_value: Default value for new rows.
        bulk_data_location: URL override for bulk data storage.

    Example::

        import tlc
        from audio_sample_type import AudioSchema

        writer = tlc.TableWriter(
            project_name="Audio Project",
            schema={"audio": AudioSchema(sample_rate=22050)},
        )

    """

    def __init__(
        self,
        sample_rate: int = 16000,
        display_name: str = "",
        description: str = "",
        writable: bool = True,
        visible: bool = True,
        display_importance: float = 0,
        default_value: Any = None,
        bulk_data_location: str | None = None,
    ) -> None:
        super().__init__(
            value=StringValue(string_role="URL/Audio"),
            sample_type={"name": "wav_audio", "sample_rate": sample_rate},
            display_name=display_name,
            description=description,
            writable=writable,
            default_visible=visible,
            display_importance=display_importance,
            default_value=default_value,
            bulk_data_location=bulk_data_location,
        )
