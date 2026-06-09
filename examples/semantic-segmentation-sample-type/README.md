# Semantic Segmentation Sample Type (POC)

POC of semantic segmentation as a distinct, RLE-backed 3LC sample type
(`"semantic_segmentation"`).

The sample form is a dense `(H, W)` integer label map
([`SemanticSegmentation`](src/semseg_sample_type/sample_type.py)) — an
exhaustive partition of the image. The wire format reuses the instance
segmentation RLE layout byte-for-byte: one RLE per class present, with
`instance_properties.label` holding the class ids. This means existing RLE
rendering works unchanged, while the sample type (and eventually metrics/UX)
dispatches on `sample_type == "semantic_segmentation"`.

## Layout

- `src/semseg_sample_type/sample_type.py` — dataclass, sample type, schema
- `scripts/ingest_oxford_pets.py` — ingest an Oxford-IIIT Pets subset
  (image, trimap-derived semseg-as-RLE, species + breed categoricals)
- `scripts/train_unet.py` — train a tiny UNet (random flip / affine / color-jitter
  augmentation, cosine LR schedule) and collect per-sample metrics into a Run every
  `--collect-frequency` epochs (final epoch always collected): predicted-segmentation-as-RLE,
  mean IoU, foreground (pet) IoU, cross-entropy loss, mean prediction entropy, and a
  pooled bottleneck embedding. After training, the embeddings are reduced to 2D with
  PaCMAP (one model fit on the final-epoch val embeddings, applied to all metrics tables
  for a shared space) and the raw embedding vectors are dropped.

## Usage

```bash
pip install -e .[train]

# Expects the Oxford-IIIT Pets dataset (images/ + annotations/) at ~/data/Oxford-IIIT-Pets
python scripts/ingest_oxford_pets.py  # full trainval split: 3000 train / 680 val
python scripts/train_unet.py --epochs 40 --collect-frequency 10
```

The sample type is registered via the `tlc.sample_types` entry point on
install; a plain `import semseg_sample_type` also registers it (via the
`@register_sample_type` decorator) for uninstalled / from-source use. When
both fire, the registry treats the duplicate as a no-op.

## Classes

Trimap pixel values are remapped to `{0: background, 1: pet, 2: border}`.
Background/border are ordinary `MapElement`s with dedicated internal names —
no `role` mechanism in this POC.
