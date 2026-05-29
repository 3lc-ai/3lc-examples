# Audio Sample Type Example

Demonstrates how to build a **custom file-backed sample type** for 3LC. The sample type stores audio waveforms as WAV
files and returns `AudioWaveform` instances (waveform + sample rate, packaged together) in sample view.

This is a teaching example — it shows the same pattern used internally by `ImageSchema` and the
`Float32Schema(sample_type="numpy_array")` shorthand, applied to a data type that 3LC doesn't have builtin
support for.

## What it demonstrates

1. **Custom sample type** (`WavAudioSampleType`): An `ExternalSampleType` subclass with `save()`/`load()` for WAV I/O
2. **Sample dataclass** (`AudioWaveform`): A small dataclass that bundles the waveform with its sample rate, so the
   rate travels with the data instead of being baked into the column's sample-type config
3. **Schema factory** (`WavAudioSampleType.schema()`): Classmethod that returns a configured `UrlSchema`
4. **Round-trip**: Write `AudioWaveform` → stored as WAV files → read back as `AudioWaveform`

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
├── sample_type.py        # AudioWaveform dataclass + WavAudioSampleType (save/load + schema() factory)
└── create_demo_table.py  # Demo script generating synthetic audio
```

## Using in your own project

```python
import tlc
from audio_sample_type import AudioWaveform, WavAudioSampleType

writer = tlc.TableWriter(
    project_name="My Audio Project",
    schema={
        "audio": WavAudioSampleType.schema(),
        "label": tlc.schemas.CategoricalLabelSchema(classes=["speech", "music", "noise"]),
    },
)

for waveform, sample_rate, label in my_audio_data:
    writer.add_row({"audio": AudioWaveform(waveform=waveform, sample_rate=sample_rate), "label": label})

table = writer.finalize()

# Sample view: audio column returns AudioWaveform instances
sample = table[0]
sample["audio"].waveform      # numpy.ndarray, shape=(num_samples,), dtype=float32
sample["audio"].sample_rate   # int
```
