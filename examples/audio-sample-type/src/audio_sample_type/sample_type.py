# Copyright 2026 3LC Inc. All rights reserved.

"""Custom sample type that stores audio waveforms as WAV files.

This module demonstrates a file-backed sample type: the Table stores URL strings
pointing to WAV files, and the sample type handles saving/loading NumPy arrays.

The sample type is registered as ``"wav_audio"`` via the ``tlc.sample_types``
entry point in ``pyproject.toml`` — ``pip install`` is all that's needed to make
it available. It can also be registered explicitly with
``@tlc.sample_types.register_sample_type("wav_audio")`` if entry points are not desired.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from tlc import Schema, Url
from tlc.sample_types import EncodedSample, ExternalSampleType, ValidationError
from tlc.schemas import StringSchema


class WavAudioSampleType(ExternalSampleType):
    """Sample type that stores audio waveforms as WAV files.

    In sample view, columns using this sample type return 1D NumPy arrays
    (float32 waveforms). On write, the arrays are saved as WAV files. The
    Table stores only URL references to these files.

    Args:
        sample_rate: Audio sample rate in Hz. Used when saving WAV files.
            Defaults to 16000 (common for speech models).

    """

    file_extension = ".wav"

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate

    def save(self, sample: np.ndarray, url: Url) -> None:
        """Save a NumPy waveform as a WAV file."""
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, sample, self._sample_rate, format="WAV")
        url.write_bytes(buf.getvalue())

    def load(self, url: Url) -> np.ndarray:
        """Load a WAV file and return a float32 NumPy array."""
        import soundfile as sf

        data: np.ndarray = sf.read(io.BytesIO(url.read_bytes()), dtype="float32")[0]
        return data

    def validate_sample(self, sample: Any) -> list[ValidationError]:
        """Check that the sample is a 1D NumPy array, raw bytes, or EncodedSample."""
        if isinstance(sample, (bytes, EncodedSample)):
            return []
        if not isinstance(sample, np.ndarray):
            return [ValidationError("", f"Expected numpy.ndarray, got {type(sample).__name__}")]
        if sample.ndim != 1:
            return [ValidationError("", f"Expected 1D array, got {sample.ndim}D")]
        return []

    def accepts(self, value: Any) -> bool:
        """Auto-detection: accept 1D NumPy arrays, raw bytes, or EncodedSample.

        Raw bytes / EncodedSample inputs flow through the
        :meth:`ExternalSampleType.externalize` pre-encoded-bytes fastpath, which
        writes them verbatim without a decode/re-encode cycle. This is what the
        Hugging Face import pipeline relies on when ``datasets.Audio(decode=False)``
        yields raw WAV bytes from parquet.
        """
        if isinstance(value, (bytes, EncodedSample)):
            return True
        return isinstance(value, np.ndarray) and value.ndim == 1

    @classmethod
    def schema(
        cls,
        sample_rate: int = 16000,
        display_name: str = "",
        description: str = "",
        writable: bool = True,
        visible: bool = True,
        display_importance: float = 0,
        default_value: Any = None,
        bulk_data_location: str | None = None,
    ) -> Schema:
        """Build a :class:`~tlc.schemas.StringSchema` configured for this sample type.

        Args:
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            display_name: Column display name in the Dashboard.
            description: Column description.
            writable: Whether the column is editable in the Dashboard.
            visible: Whether the column is visible by default.
            display_importance: Ordering weight for column display.
            default_value: Default value for new rows.
            bulk_data_location: URL override for bulk data storage.

        Returns:
            A schema that stores WAV-file URLs and decodes to NumPy arrays in sample view.

        Example::

            import tlc
            from audio_sample_type import WavAudioSampleType

            writer = tlc.TableWriter(
                project_name="Audio Project",
                schema={"audio": WavAudioSampleType.schema(sample_rate=22050)},
            )

        """
        return StringSchema(
            string_role="URL/Audio",
            sample_type={"name": "wav_audio", "sample_rate": sample_rate},
            display_name=display_name,
            description=description,
            writable=writable,
            default_visible=visible,
            display_importance=display_importance,
            default_value=default_value,
            bulk_data_location=bulk_data_location,
        )
