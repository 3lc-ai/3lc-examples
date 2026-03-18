# Audio Sample Type Example

Demonstrates how to build a **custom file-backed sample type** for 3LC. The sample type stores audio waveforms as WAV
files and returns NumPy arrays in sample view.

This is a teaching example — it shows the same pattern used internally by `ImageSchema` and `NumpyArraySchema`, applied
to a data type that 3LC doesn't have builtin support for.

## What it demonstrates

1. **Custom sample type** (`WavAudioSampleType`): A `SampleType` subclass with `save()`/`load()` for WAV I/O
2. **Convenience schema** (`AudioSchema`): Wraps `Schema(value=UrlStringValue(), sample_type=...)` into a clean API
3. **Round-trip**: Write NumPy arrays → stored as WAV files → read back as NumPy arrays

## Quick start

```bash
pip install -e .
create-audio-demo-table
```

This creates a Table with 8 synthetic sine-wave audio clips. Open the 3LC Dashboard to browse it.

## Dashboard behavior

Since there is no builtin audio widget in the 3LC Dashboard, the audio column displays URL strings pointing to the WAV
files. This is the expected behavior for custom sample types — the Dashboard stores and displays your data, but
specialized features (rendering, editing, etc.) are only available for builtin schema types. Any metrics you collect on
the table still work normally, so you can identify interesting samples through the Dashboard even without a dedicated
audio player.

## Structure

```
src/audio_sample_type/
├── __init__.py           # Package exports
├── sample_type.py        # WavAudioSampleType — the core save/load logic
├── schema.py             # AudioSchema — convenience wrapper
└── create_demo_table.py  # Demo script generating synthetic audio
```

## Using in your own project

```python
import tlc
from audio_sample_type import AudioSchema

writer = tlc.TableWriter(
    project_name="My Audio Project",
    schema={
        "audio": AudioSchema(sample_rate=22050),
        "label": tlc.CategoricalLabelSchema(classes=["speech", "music", "noise"]),
    },
)

for waveform, label in my_audio_data:
    writer.add_row({"audio": waveform, "label": label})

table = writer.finalize()

# Sample view: audio column returns numpy arrays
sample = table[0]
sample["audio"]  # numpy.ndarray, shape=(num_samples,), dtype=float32
```
