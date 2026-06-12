# Audio Classification POC

End-to-end audio classification with 3LC tables, using the `rfcx/frugalai`
binary classification dataset (chainsaw vs. forest, 3-second clips at 12 kHz).

This is an **experimental POC**. It exercises a few moving pieces that are
not yet officially supported in 3LC:

1. The **HF → 3LC audio bytes fastpath**. When importing a Hugging Face audio
   dataset, raw WAV bytes embedded in parquet are written verbatim to disk —
   no soundfile/librosa decode + re-encode round trip during table creation.
   Mirrors the existing image fastpath in `_TableFromHuggingFaceBase`.
2. The **`wav_audio` sample type** from the sibling `audio-sample-type`
   example, extended to also accept raw bytes / `EncodedSample` so the
   fastpath above can flow through it.
3. A **plain PyTorch training loop** over the 3LC table.

## Layout

- `01-hf-audio-explore.ipynb` — Mirror of the image-bytes walkthrough but
  for audio. Shows that `datasets.Audio(decode=False)` returns raw WAV bytes
  straight from parquet.
- `02-create-3lc-table.ipynb` — Build a 3LC table over a small subset of
  `rfcx/frugalai` using `Table.from_hugging_face_dataset`. Verify that the
  parquet representation is just a string URL with `string_role="URL/Audio"`,
  and that the materialized WAV files are byte-identical to the source.
- `03-pytorch-train.ipynb` — Wrap the 3LC table as a `torch.utils.data.Dataset`,
  compute mel-spectrograms on the fly, train a small CNN.

## Setup

This POC depends on temporary changes in `tlc-monorepo` (HF Audio fastpath)
that are not yet shipped in the released `3lc` package. Run the notebooks
from a venv that has the local checkout installed in editable mode.
