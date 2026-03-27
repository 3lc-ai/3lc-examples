# Copyright 2026 3LC Inc. All rights reserved.

"""Create a demo 3LC Table with synthetic audio data.

Generates sine-wave audio clips at different frequencies and stores them in a
Table using the custom ``wav_audio`` sample type. Run with::

    create-audio-demo-table

or::

    python -m audio_sample_type.create_demo_table

"""

from __future__ import annotations

import numpy as np
import tlc

# Importing the package registers the sample type
from audio_sample_type import AudioSchema

SAMPLE_RATE = 16000
DURATION_S = 1.0
NOTES = [
    ("C4", 261.63),
    ("D4", 293.66),
    ("E4", 329.63),
    ("F4", 349.23),
    ("G4", 392.00),
    ("A4", 440.00),
    ("B4", 493.88),
    ("C5", 523.25),
]


def generate_sine(frequency: float, duration: float, sample_rate: int) -> np.ndarray:
    """Generate a sine wave as a float32 NumPy array."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    return np.asarray(0.5 * np.sin(2 * np.pi * frequency * t))


def main() -> None:
    writer = tlc.TableWriter(
        project_name="3LC Tutorials - Audio Sample Type",
        dataset_name="sine-waves",
        table_name="initial",
        schema={
            "audio": AudioSchema(sample_rate=SAMPLE_RATE),
            "note": tlc.StringSchema(),
            "frequency_hz": tlc.Float32Schema(writable=False),
            "duration_s": tlc.Float32Schema(writable=False),
        },
        if_exists="overwrite",
    )

    for note_name, freq in NOTES:
        waveform = generate_sine(freq, DURATION_S, SAMPLE_RATE)
        writer.add_row(
            {
                "audio": waveform,
                "note": note_name,
                "frequency_hz": freq,
                "duration_s": DURATION_S,
            }
        )

    table = writer.finalize()

    print(f"Created table with {len(table)} rows")
    print(f"Table URL: {table.url}")
    print()

    # Show round-trip: row view has URLs, sample view has arrays
    print("Row view (stored data):")
    print(f"  audio = {table.table_rows[0]['audio']}")
    print()
    print("Sample view (loaded data):")
    sample = table[0]
    print(f"  audio = numpy array, shape={sample['audio'].shape}, dtype={sample['audio'].dtype}")
    print(f"  note  = {sample['note']}")


if __name__ == "__main__":
    main()
