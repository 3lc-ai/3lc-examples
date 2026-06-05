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
- `scripts/train_unet.py` — train a tiny UNet, collect per-sample
  predicted-segmentation-as-RLE and mean IoU into a Run

## Usage

```bash
pip install -e .[train]

# Expects the Oxford-IIIT Pets dataset (images/ + annotations/) at ~/Data/Oxford-IIIT-Pets
python scripts/ingest_oxford_pets.py --n-train 200 --n-val 50
python scripts/train_unet.py --epochs 10
```

The sample type is registered via the `tlc.sample_types` entry point on
install; a plain `import semseg_sample_type` also registers it.

## Classes

Trimap pixel values are remapped to `{0: background, 1: pet, 2: border}`.
Background/border are ordinary `MapElement`s with dedicated internal names —
no `role` mechanism in this POC.
