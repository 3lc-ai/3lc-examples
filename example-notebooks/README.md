# Public Examples

This folder contains "official" public examples of 3LC.

## Notebooks Overview

Here's a brief overview of the example notebooks included in this directory:

+ [MNIST](./pytorch-mnist.ipynb): Train a simple model on the MNIST dataset and collect classification metrics.
+ [CIFAR-10](./pytorch-cifar10.ipynb): Train a model on the CIFAR10 dataset and collect classification metrics.
+ [Hugging Face IMDB Reviews](./huggingface-imdb.ipynb): Use our Hugging Face `Trainer` integration to train a language
  model on the IMDB Reviews dataset.
+ [Hugging Face Fine-tuning with BERT](./huggingface-finetuning.ipynb): Use our Hugging Face `Trainer` integration to train BERT model on the glue/mrpc dataset.
+ [Hugging Face CIFAR-100](./huggingface-cifar100.ipynb): Loads the CIFAR-100 dataset from Hugging Face dataset
  and computes dimensionality reduced 2D image embeddings.
+ [Detectron2 Balloons](./detectron2-balloons.ipynb): Trains an object detection model and gathers bounding box metrics
  with detectron2.
+ [Detectron2 COCO128](./detectron2-coco128.ipynb): Executes inference and gathers bounding box metrics using
  detectron2.
+ [Per Bounding Box Metrics](./calculate-luminosity.ipynb): Describes metric collection for individual bounding boxes in
  images.
+ [PyTorch Lightning SegFormer](./pytorch-lightning-segformer.ipynb): Demonstrates how to use a custom metrics collector
  for collecting predicted masks from a semantic segmentation model.

Each notebook is self-contained and declares and installs its required dependencies.
