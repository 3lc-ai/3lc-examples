# Copyright 2026 3LC Inc. All rights reserved.

"""Custom sample type that stores audio waveforms as WAV files.

This module demonstrates a file-backed sample type: the Table stores URL strings
pointing to WAV files, and the sample type handles saving/loading
:class:`AudioWaveform` instances (waveform + sample rate, packaged together).

The sample type is registered as ``"wav_audio"`` via the ``tlc.sample_types``
entry point in ``pyproject.toml`` — ``pip install`` is all that's needed to make
it available. It can also be registered explicitly with
``@tlc.sample_types.register_sample_type("wav_audio")`` if entry points are not desired.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from tlc import Schema, Url
from tlc.sample_types import EncodedSample, ExternalSampleType, ValidationError
from tlc.schemas import StringSchema


@dataclass
class AudioWaveform:
    """Audio sample form — waveform plus sample rate.

    Sample rate travels with the data, so a single column can hold mixed-rate audio
    and downstream consumers always know the rate of the array they are looking at.
    """

    waveform: np.ndarray
    sample_rate: int = 16000


class WavAudioSampleType(ExternalSampleType):
    """Sample type that stores audio waveforms as WAV files.

    In sample view, columns using this sample type return :class:`AudioWaveform`
    instances. On write, the waveform is saved as a WAV file at its accompanying
    sample rate. The Table stores only URL references to these files.
    """

    file_extension = ".wav"

    def save(self, sample: AudioWaveform, url: Url) -> None:
        """Save an :class:`AudioWaveform` as a WAV file."""
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, sample.waveform, sample.sample_rate, format="WAV")
        url.write_bytes(buf.getvalue())

    def load(self, url: Url) -> AudioWaveform:
        """Load a WAV file as an :class:`AudioWaveform`."""
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(url.read_bytes()), dtype="float32")
        return AudioWaveform(waveform=data, sample_rate=int(sr))

    def validate_sample(self, sample: Any) -> list[ValidationError]:
        """Check that the sample is an :class:`AudioWaveform`, raw bytes, or :class:`EncodedSample`."""
        if isinstance(sample, (bytes, EncodedSample)):
            return []
        if not isinstance(sample, AudioWaveform):
            return [ValidationError("", f"Expected AudioWaveform, got {type(sample).__name__}")]
        if sample.waveform.ndim != 1:
            return [ValidationError("", f"Expected 1D waveform, got {sample.waveform.ndim}D")]
        return []

    def accepts(self, value: Any) -> bool:
        """Auto-detection: accept :class:`AudioWaveform`, raw bytes, or :class:`EncodedSample`.

        Raw bytes / :class:`EncodedSample` inputs flow through the
        :meth:`ExternalSampleType.externalize` pre-encoded-bytes fastpath, which
        writes them verbatim without a decode/re-encode cycle. This is what the
        Hugging Face import pipeline relies on when ``datasets.Audio(decode=False)``
        yields raw WAV bytes from parquet.
        """
        if isinstance(value, (bytes, EncodedSample)):
            return True
        return isinstance(value, AudioWaveform) and value.waveform.ndim == 1

    @classmethod
    def schema(
        cls,
        *,
        display_name: str = "",
        description: str = "",
        writable: bool = True,
        visible: bool = True,
        default_value: Any = None,
        bulk_data_location: str | None = None,
    ) -> Schema:
        """Build a :class:`~tlc.schemas.UrlSchema` configured for this sample type.

        Args:
            display_name: Column display name in the Dashboard.
            description: Column description.
            writable: Whether the column is editable in the Dashboard.
            visible: Whether the column is visible by default.
            default_value: Default value for new rows.
            bulk_data_location: URL override for bulk data storage.

        Returns:
            A schema that stores WAV-file URLs and decodes to :class:`AudioWaveform`
            in sample view.

        Example::

            import tlc
            from audio_sample_type import AudioWaveform, WavAudioSampleType

            writer = tlc.TableWriter(
                project_name="Audio Project",
                schema={"audio": WavAudioSampleType.schema()},
            )
            writer.add_row({"audio": AudioWaveform(waveform=arr, sample_rate=22050)})

        """
        return StringSchema(
            string_role="URL/Audio",
            sample_type="wav_audio",  # type: ignore[arg-type]
            display_name=display_name,
            description=description,
            writable=writable,
            default_visible=visible,
            default_value=default_value,
            bulk_data_location=bulk_data_location,
        )
