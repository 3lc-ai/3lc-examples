# Tutorial Quality Parameters: Test vs Publish Mode

This document lists variables in parameter cells that affect run quality.
Use **Test** values for quick nightly runs; use **Publish** values for production-quality outputs.

---

## Classification Training

### pytorch-cifar10-resnet-training.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `EPOCHS` | 5 | 20 | Training iterations |
| `BATCH_SIZE` | 32 | 128 | Larger = more stable gradients |
| `MODEL_NAME` | resnet18 | resnet50 | Model depth/capacity |
| `INITIAL_LR` | 0.01 | 0.01 | Keep same, adjust if needed |

### lightning-cifar10-resnet-training.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `EPOCHS` | 5 | 20 | Training iterations |
| `BATCH_SIZE` | 32 | 128 | |

### ultralytics-cifar10-yolo11-cls-training.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL_NAME` | yolov8n-cls.pt | yolov8m-cls.pt | n=nano, s=small, m=medium, l=large |
| `EPOCHS` | 5 | 20 | |
| `BATCH_SIZE` | 32 | 128 | |
| `IMAGE_SIZE` | 32 | 224 | Higher resolution |

### pytorch-fashion-mnist-resnet-training.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `EPOCHS` | 5 | 20 | |
| `TRAIN_BATCH_SIZE` | 64 | 256 | |

### pytorch-mnist-resnet-training.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `EPOCHS` | 5 | 20 | |
| `TRAIN_BATCH_SIZE` | 64 | 256 | |

---

## Object Detection Training

### supergradients-detection-yolonas-training.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL` | YOLO_NAS_S | YOLO_NAS_L | S=small, M=medium, L=large |
| `MAX_EPOCHS` | 5 | 20 | |
| `BATCH_SIZE` | 4 | 16 | |
| `INPUT_DIM` | (640, 640) | (640, 640) | Keep same unless memory allows higher |
| `INITIAL_LR` | 5e-5 | 1e-4 | Learning rate |

### detectron2-balloons-detection-finetuning.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MAX_ITERS` | 200 | 1000 | Training iterations |
| `BATCH_SIZE` | 2 | 8 | |
| `MODEL_CONFIG` | faster_rcnn_R_50_FPN_3x | faster_rcnn_R_101_FPN_3x | R_50 vs R_101 backbone |
| `BASE_LR` | 0.00025 | 0.0005 | Learning rate |
| `ROI_BATCH_SIZE_PER_IMAGE` | 128 | 512 | RoI head batch size |

### detectron2-coco128-detection-collection.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL_CONFIG` | faster_rcnn_R_50_FPN_3x | faster_rcnn_R_101_FPN_3x | R_50 vs R_101 backbone |
| `ROI_BATCH_SIZE_PER_IMAGE` | 512 | 512 | Keep same for inference |

### ultralytics-animalpose-yolo11-pose-training.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL_NAME` | yolo11n-pose.pt | yolo11m-pose.pt | n/s/m/l variants |
| `EPOCHS` | 10 | 50 | |

### ultralytics-cell-segmentation-yolo11-seg.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL_NAME` | yolo11n-seg.pt | yolo11m-seg.pt | |
| `EPOCHS` | 10 | 50 | |
| `BATCH_SIZE` | 4 | 16 | |

### ultralytics-hrsc-2016-yolo-obb-training.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL_NAME` | yolo11n-obb.pt | yolo11m-obb.pt | Oriented bounding box |
| `EPOCHS` | 10 | 50 | |

---

## Semantic/Instance Segmentation

### lightning-balloons-segformer-training.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL` | nvidia/mit-b0 | nvidia/mit-b2 | b0/b1/b2/b3/b4/b5 scale |
| `EPOCHS` | 10 | 50 | |
| `BATCH_SIZE` | 8 | 32 | |
| `LEARNING_RATE` | 2e-05 | 2e-05 | Keep same, adjust if needed |

### huggingface-ade20k-segformer-finetuning.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL` | nvidia/mit-b0 | nvidia/mit-b2 | |
| `EPOCHS` | 200 | 500 | Already high, increase for more polish |
| `BATCH_SIZE` | 2 | 8 | |
| `LEARNING_RATE` | 0.00006 | 0.00006 | Keep same |

### huggingface-coco128-segformer-collection.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `HF_MODEL_ID` | mask2former-swin-tiny-coco-instance | mask2former-swin-base-coco-instance | tiny/small/base |
| `BATCH_SIZE` | 4 | 16 | |

---

## Embeddings / Feature Extraction

### pytorch-cifar10-train-autoencoder.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `BACKBONE` | resnet50 | resnet101 | |
| `EPOCHS` | 10 | 50 | |
| `BATCH_SIZE` | 64 | 256 | |
| `EMBEDDING_DIM` | 512 | 512 | Keep same |

### huggingface-cifar100-collect-embeddings.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL` | google/vit-base-patch16-224 | google/vit-large-patch16-224 | base/large |
| `BATCH_SIZE` | 32 | 128 | |

### add-instance-embeddings/1-train-crop-model.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL_NAME` | efficientnet_b0 | efficientnet_b2 | b0-b7 scale |
| `EPOCHS` | 10 | 50 | |

---

## Video / Temporal Processing

### sam2-video-segmentation.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL_SIZE` | small | large | small/base_plus/large |
| `MAX_FRAMES_PER_VIDEO` | 10 | 60 | More frames processed |
| `POINTS_PER_SIDE` | 32 | 64 | Denser point grid |
| `CONFIDENCE_THRESHOLD` | 0.7 | 0.8 | Higher = cleaner masks |

---

## NLP / Text Classification

### huggingface-imdb-finetuning.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `MODEL` | distilbert-base-uncased | bert-base-uncased | distilbert is smaller/faster |
| `EPOCHS` | 10 | 30 | |
| `TRAIN_BATCH_SIZE` | 16 | 64 | |
| `LEARNING_RATE` | 2e-5 | 2e-5 | Keep same |
| `WEIGHT_DECAY` | 0.01 | 0.01 | Keep same |

---

## Data Curation

### 1-weight-coreset.ipynb

| Variable | Test | Publish | Notes |
|----------|------|---------|-------|
| `CORESET_SIZE` | 0.01 | 0.1 | Fraction of data (1% vs 10%) |

---

## Quick Reference: Model Size Progressions

| Framework | Test (Fast) | Publish (Quality) |
|-----------|-------------|-------------------|
| **YOLO/Ultralytics** | yolo11n / yolov8n | yolo11m / yolov8m |
| **YOLO-NAS** | YOLO_NAS_S | YOLO_NAS_L |
| **ResNet** | resnet18 | resnet50 or resnet101 |
| **EfficientNet** | efficientnet_b0 | efficientnet_b2 |
| **SegFormer/MiT** | nvidia/mit-b0 | nvidia/mit-b2 |
| **ViT** | vit-base | vit-large |
| **BERT** | distilbert-base | bert-base |
| **SAM2** | small | large |
| **Mask2Former** | swin-tiny | swin-base |
| **Detectron2** | R_50_FPN | R_101_FPN |

---

## General Guidelines

| Parameter | Test Mode | Publish Mode |
|-----------|-----------|--------------|
| **EPOCHS** | 5-10 | 20-50 |
| **BATCH_SIZE** | 2-32 | 64-256 (memory permitting) |
| **Model variant** | nano/tiny/small/b0 | medium/base/b2+ |
| **Image resolution** | Default/small | Higher if supported |
| **Learning rate** | Default | Default (usually no change needed) |

---

## Recently Parameterized Values

The following values were moved from hardcoded locations to parameter cells:

| Notebook | New Parameters Added |
|----------|---------------------|
| detectron2-balloons-detection-finetuning | `BASE_LR`, `ROI_BATCH_SIZE_PER_IMAGE` |
| detectron2-coco128-detection-collection | `ROI_BATCH_SIZE_PER_IMAGE` |
| huggingface-ade20k-segformer-finetuning | `MODEL`, `LEARNING_RATE` |
| huggingface-imdb-finetuning | `MODEL`, `LEARNING_RATE`, `WEIGHT_DECAY` |
| huggingface-coco128-segformer-collection | `BATCH_SIZE` |
| lightning-balloons-segformer-training | `LEARNING_RATE` |
| supergradients-detection-yolonas-training | `MODEL`, `BATCH_SIZE`, `INPUT_DIM`, `MAX_EPOCHS`, `INITIAL_LR` |

---

## Model File Downloads

This section lists all model downloads across notebooks for future implementation of
"use local if present, download if necessary" caching.

### HuggingFace Models (`from_pretrained`)

| Notebook | Model ID | Type |
|----------|----------|------|
| huggingface-cifar100-collect-embeddings | `google/vit-base-patch16-224` | ViT |
| huggingface-coco128-segformer-collection | `facebook/mask2former-swin-tiny-coco-instance` | Mask2Former |
| huggingface-imdb-finetuning | `distilbert-base-uncased` | DistilBERT |
| huggingface-ade20k-segformer-finetuning | `nvidia/mit-b0` | SegFormer |
| huggingface-mrcp-bert-finetuning | `bert-base-uncased` | BERT |
| lightning-balloons-segformer-training | `nvidia/mit-b0` | SegFormer |
| huggingface-ade20k-segformer-finetuning | `huggingface/label-files` (ade20k-id2label.json) | Label map |

### timm Models (`timm.create_model`)

| Notebook | Model ID | Notes |
|----------|----------|-------|
| pytorch-cifar10-resnet-training | `resnet18` | Via `MODEL_NAME` param |
| pytorch-cifar10-collect-metrics | `hf_hub:FredMell/resnet18-cifar10` | HF Hub reference |

### torchvision Models

| Notebook | Model ID | Notes |
|----------|----------|-------|
| lightning-cifar10-resnet-training | `resnet18` | `torchvision.models.resnet18()` |

### Detectron2 Models (`model_zoo`)

| Notebook | Config | Download Method |
|----------|--------|-----------------|
| detectron2-coco128-detection-collection | `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml` | `model_zoo.get_checkpoint_url()` |
| detectron2-balloons-detection-finetuning | `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml` | `model_zoo.get_checkpoint_url()` |
| detectron2-balloons-segmentation-finetuning | `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` | `model_zoo.get_checkpoint_url()` |
| detectron2-coco128-segmentation-collection | `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` | `model_zoo.get_checkpoint_url()` |

### Ultralytics Models (`YOLO()`)

| Notebook | Model ID | Type |
|----------|----------|------|
| ultralytics-cifar10-yolo11-cls-training | `yolov8n-cls.pt` | Classification |
| ultralytics-cell-segmentation-yolo11-seg | `yolo11n-seg.pt` | Segmentation |
| ultralytics-animalpose-yolo11-pose-training | `yolo11n-pose.pt` | Pose |
| ultralytics-hrsc-2016-yolo-obb-training | `yolo11n-obb.pt` | Oriented BB |
| hard-hat (end-to-end) | YOLO model | Detection |

### SuperGradients Models (`models.get`)

| Notebook | Model ID | Download Method |
|----------|----------|-----------------|
| supergradients-detection-yolonas-training | `YOLO_NAS_S` | `models.get(..., pretrained_weights="coco")` |
| supergradients-animalpose-yolonas-training | `yolo_nas_pose_n` | Direct HTTP + `models.get()` |

**Note:** The animalpose notebook downloads directly from S3:
```
https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_pose_n_coco_pose.pth
```

### SAM2 Models (`from_pretrained`)

| Notebook | Model ID | Variants Available |
|----------|----------|-------------------|
| sam2-video-segmentation | `facebook/sam2-hiera-small` | tiny, small, base_plus, large |

Model configs defined in notebook:
```python
MODEL_CONFIGS = {
    "tiny": ("facebook/sam2-hiera-tiny", "sam2_hiera_t.yaml"),
    "small": ("facebook/sam2-hiera-small", "sam2_hiera_s.yaml"),
    "base_plus": ("facebook/sam2-hiera-base-plus", "sam2_hiera_b+.yaml"),
    "large": ("facebook/sam2-hiera-large", "sam2_hiera_l.yaml"),
}
```

### SAM v1 Models (`torch.hub.download_url_to_file`)

| Notebook | Model URL | Local Path |
|----------|-----------|------------|
| sam-coco128-collect-embeddings | `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` | `{TMP_PATH}/sam_vit_b_01ec64.pth` |

**Note:** This notebook already implements local caching:
```python
if not Path(CHECKPOINT).exists():
    torch.hub.download_url_to_file(MODEL_URL, CHECKPOINT)
```

---

### Download Mechanisms Summary

| Library | Download Method | Cache Location |
|---------|-----------------|----------------|
| HuggingFace | `from_pretrained()` | `~/.cache/huggingface/` |
| timm | `timm.create_model(pretrained=True)` | `~/.cache/torch/hub/` |
| torchvision | `torchvision.models.*()` | `~/.cache/torch/hub/` |
| Detectron2 | `model_zoo.get_checkpoint_url()` | `~/.torch/fvcore_cache/` |
| Ultralytics | `YOLO()` | Downloads to CWD or `~/.cache/` |
| SuperGradients | `models.get()` | `~/.cache/torch/hub/` |
| SAM2 | `from_pretrained()` | `~/.cache/huggingface/` |
| torch.hub | `download_url_to_file()` | Custom path |

### Total Model Downloads

- **27 model downloads** across **20 notebooks**
- **8 different download mechanisms**
