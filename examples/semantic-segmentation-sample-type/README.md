# Semantic Segmentation on Oxford-IIIT Pets

Worked example for the **`semantic_segmentation`** sample type. The sample type
itself, its ergonomic class-map API, and the metrics helper now live in **core
`tlc`** — this directory holds only the application: table generation, training,
and metrics collection.

A semantic segmentation is a dense `(H, W)` integer label map — an exhaustive
partition of the image. It is stored in the instance-segmentation RLE wire format
(one RLE per class, `instance_properties.label` holding the class ids), so existing
RLE rendering works unchanged, while semseg-aware behavior dispatches on the
schema's `composite_role == "SemanticSegmentation"` (the `sample_type` stays
`"semantic_segmentation"` and names the Python (de)serializer). Each row is stored
as exactly `C` class-ordered layers (empty RLE for absent classes) so layer ↔ class
alignment is stable across the dataset. The full design doc lives with the frontend
work in the `tlc-ui` repo, branch `feature/gudbrand/semseg-sample-type`
(`src/7_Interfaces/ListInterfaces/ShapeListInterfaces/ImageInterface/SEMANTIC_SEGMENTATION_SPEC.md`).

## What lives where

- **Core `tlc`** (not in this repo):
  - `tlc.sample_types` — `SemanticSegmentation`, `SemanticSegmentationSampleType`,
    `semseg_classes` / `background_id` / `void_id` / `real_class_ids`.
  - `tlc.helpers.semantic_segmentation_metrics` — per-image C×C confusion matrix
    and the mIoU / pixel-accuracy / per-class IoU / Dice readouts.
- **This example:**
  - `scripts/ingest_oxford_pets.py` — table generation: ingest an Oxford-IIIT Pets
    subset (image, trimap-derived semseg-as-RLE, species + breed categoricals).
    Trimap pixels remap to `{0: background, 1: pet, 2: border}`, with `background`
    and `border` (void) tagged via `semseg_classes(..., background=0, void=2)`.
  - `scripts/train_unet.py` — training + metrics collection: train a tiny UNet
    (random flip / affine / color-jitter augmentation, cosine LR) and collect
    per-sample metrics into a Run every `--collect-frequency` epochs (final epoch
    always): predicted-segmentation-as-RLE, mean IoU, pet IoU, the per-image
    confusion matrix (via the core helper), cross-entropy loss, mean prediction
    entropy, and a pooled bottleneck embedding (PaCMAP-reduced to 2D after training).

## Usage

```bash
# Provisions dependencies (3lc, numpy, pycocotools, torch, torchvision, tqdm);
# the semantic_segmentation sample type is provided by the installed `tlc`.
pip install -e .

# Expects the Oxford-IIIT Pets dataset (images/ + annotations/) at ~/data/Oxford-IIIT-Pets
python scripts/ingest_oxford_pets.py                       # 3000 train / 680 val
python scripts/train_unet.py --epochs 40 --collect-frequency 10
```
