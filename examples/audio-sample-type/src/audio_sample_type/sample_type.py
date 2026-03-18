# Copyright 2026 3LC Inc. All rights reserved.

"""Custom sample type that stores audio waveforms as WAV files.

This module demonstrates a file-backed sample type: the Table stores URL strings
pointing to WAV files, and the sample type handles saving/loading NumPy arrays.

The sample type is registered as ``"wav_audio"`` via the ``tlc.sample_types``
entry point in ``pyproject.toml`` — ``pip install`` is all that's needed to make
it available. It can also be registered explicitly with
``@tlc.register_sample_type("wav_audio")`` if entry points are not desired.
"""

from __future__ import annotations

import io
from typing import Any, ClassVar

import numpy as np

import tlc
from tlc.core.sample_types.registry import SampleType
from tlc.core.url import Url


class WavAudioSampleType(SampleType):
    """Sample type that stores audio waveforms as WAV files.

    In sample view, columns using this sample type return 1D NumPy arrays
    (float32 waveforms). On write, the arrays are saved as WAV files. The
    Table stores only URL references to these files.

    Args:
        sample_rate: Audio sample rate in Hz. Used when saving WAV files.
            Defaults to 16000 (common for speech models).

    """

    is_leaf: ClassVar[bool] = True
    default_storage: ClassVar[str] = "file"
    default_file_extension: ClassVar[str] = ".wav"

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

        data, _ = sf.read(io.BytesIO(url.read_bytes()), dtype="float32")
        return data

    def validate_sample(self, sample: Any) -> list[tlc.ValidationError]:
        """Check that the sample is a 1D NumPy array."""
        if not isinstance(sample, np.ndarray):
            return [tlc.ValidationError("", f"Expected numpy.ndarray, got {type(sample).__name__}")]
        if sample.ndim != 1:
            return [tlc.ValidationError("", f"Expected 1D array, got {sample.ndim}D")]
        return []

    def accepts(self, value: Any) -> bool:
        """Auto-detection: accept 1D NumPy arrays."""
        return isinstance(value, np.ndarray) and value.ndim == 1
