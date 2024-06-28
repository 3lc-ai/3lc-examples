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
+ [Per Bounding Box Embeddings](./add-bb-embeddings.ipynb): Covers embedding collection for bounding boxes and uses UMAP
  for dimensionality reduction.
+ [Bounding Box Classifier](./train-bb-classifier.ipynb): Details an advanced workflow where a model is trained to
  classify bounding boxes in an image, which can be used in conjunction with an object detection model to find bounding
  boxes of special interest.
+ [PyTorch Lightning SegFormer](./pytorch-lightning-segformer.ipynb): Demonstrates how to use a custom metrics collector
  for collecting predicted masks from a semantic segmentation model.

Each notebook is self-contained and declares and installs its required dependencies.
