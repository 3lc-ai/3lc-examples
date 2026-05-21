# Copyright 2026 3LC Inc. All rights reserved.

"""Example: Custom file-backed sample type for WAV audio data in 3LC.

This package demonstrates how to build a custom sample type that stores audio
waveforms as WAV files and returns :class:`AudioWaveform` instances (waveform +
sample rate) in sample view.
"""

from audio_sample_type.sample_type import AudioWaveform, WavAudioSampleType

__all__ = ["AudioWaveform", "WavAudioSampleType"]
