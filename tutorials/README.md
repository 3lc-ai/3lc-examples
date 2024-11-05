# Tutorial

This folder contains simple tutorials showcasing basic features of the `3lc` Python package.

## Overview

+ [Register Image Classification Dataset](write-image-classification-table.ipynb) Write a table consisting of images and labels.
+ [Add Image Metrics](add-image-metrics.ipynb) Extend an existing table with image metrics.
+ [Add Embeddings](add-embeddings.ipynb) Extend an existing table with embeddings from a pre-trained model.
+ [Register Augmented Samples](write-augmented-samples.ipynb) Register images as metrics on-the-fly, using image augmentation as an example.
+ [Collect Metrics from a Pre-trained Model](collect_metrics_only) Collect classification metrics from a pre-trained model.
+ [Write Video Thumbnails](write-video-thumbnails.ipynb) Write a table consisting of video thumbnails and paths to videos.
+ [Tutorial 2](write-bb-table.ipynb) Write a table consisting of images and random bounding boxes.
+ [Finetune Segment-Anything-Model (SAM)](sam) Prepare a dataset for SAM and finetune the model.
+ [Mammoth Dataset](mammoth) Writes a 3D point cloud dataset and uses UMAP and PaCMAP to reduce the dimensionality.
# Tutorials

## Level 0: Creating Tables

Simple examples of creating tables from various sources.
Gently introduces the concept of schema.
Cover the most common use cases.

```
|-- create-table.ipynb
|-- create-table-from-coco.ipynb
|-- create-table-from-yolo.ipynb
|-- create-semantic-segmentation-table.ipynb
|-- create-bounding-box-table.ipynb
|-- create-image-classification-table.ipynb
|-- create-video-thumbnails-table.ipynb
```

## Level 1: Modifying and Extending Tables

Examples of modifying and extending tables. 
Create splits, add new columns, (get the latest?) etc.
(Each example could, if appropriate, contain several sub-examples.)


```
|-- split-table.ipynb
|-- add-image-metrics.ipynb
|-- add-embeddings.ipynb
|-- add-new-data-to-table.ipynb
|-- use-latest-table.ipynb
```

## Level 2: Training and Metrics Collection

Small, to-the-point examples of metrics collection.
Introduces the concept of runs and metrics.

```
|-- collect_metrics_only
|   |-- README.md
|   |-- collect_metrics_only.py
|   | requirements.txt
|-- train-classifier.ipynb
```

## Level 3: Complete Examples

Complete examples of training and evaluation. Includes examples of integrating with different frameworks (huggingface, sam, etc.)

```

|-- fine-tune-sam
|   |-- 1-create-sam-dataset.ipynb
|   |-- 2-fine-tune-sam.ipynb
|   | README.md

|-- huggingface-segmentation-example.ipynb
```

## Level 4. Advanced Examples

```
|-- bb-embeddings
|   |-- 1-fine-tune-on-crops.ipynb
|   |-- 2-collect-embeddings.ipynb
|   |-- 3-add-embeddings-to-table.ipynb
|   |-- 3b-add-embeddings-to-run.ipynb
```

## Level 5: Misc.

```
|-- write-augmented-samples.ipynb
|-- mammoth
|   |-- 1-write-mammoth-table.ipynb
|   |-- 2-flatten-mammoth.ipynb
|   | README.md
```

## Problems

+ `tools` is to a large degree a collection of functions either hiding functionality that is overly complex, or that is not yet implemented (but should be) in the main package.
+ It has been deemed a necessary evil to go with the TableWriter + for-loop pattern in order to get balls rolling. This is bad, it is not procedural, as our ethos demands.

## Design decisions

Yay or Nay?`

+ include general requirements in top-level?
+ install specific requirements in notebook cells?
+ include specific requirements.txt in folders with notebooks/python scripts?
+ have several "extra" dependency groups in pyproject.toml, just import early in notebooks?
+ have "open in github" and "open in colab" badges in the notebooks?
+ have inter-notebook / inter-level dependencies?
+ should it always be notebooks?

## Potential work in core

Table.get_column full implementation
Table.add_column full implementation
MapElement extensions
JoinedTable add source_table column (off by default)

## Default Deps

3lc + basic deps
torch / torchvision
sklearn
..

## Non-default Deps

sam2 !!??
segment_anything
huggingface (transformers, datasets)
ultralytics
umap-learn
pacmap
joblib
cv2
