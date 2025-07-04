# Create 3LC Tables

This folder contains notebooks demonstrating how to create Tables in 3LC. Tables are the primary data structure in 3LC, and are used to store and manage data for various machine learning tasks.

The `tlc` Python package provides several helper functions for creating Tables from different commonly used data sources. For custom data sources, you can create Tables from scratch using the `TableWriter` class. This allows specifying a custom schema, and adding data row-by-row or in batches.

|                        |                        |                        |
|:----------:|:----------:|:----------:|
| **Image classification** | **Custom Table** | **Bounding boxes** |
| [![img][image-classification-img]][image-classification-link] | [![custom][custom-img]][custom-link]   | [![bb][bb-img]][bb-link]                 |
| Create a Table from a folder of subfolders, each subfolder containing images belonging to a certain class. | Write a custom Table by specifying a schema and adding data row-by-row. | Write a custom Table containing images and bounding boxes, using custom bounding boxes of an arbitrary format. |
| **YOLO** | **Semantic segmentation**  | **Video thumbnails** |
| [![yolo][yolo-img]][yolo-link] | [![semseg][semseg-img]][semseg-link] | [![video][video-img]][video-link] |
| Load images and bounding boxes from the YOLO YAML file format. | Write a Table from images with PNG grayscale masks. | Work with video datasets in 3LC by creating thumbnails and storing URLs to video files. |
| **From PyTorch Dataset** | **COCO** | **Instance segmentation - Polygons** |
| [![torch][torch-img]][torch-link] | [![coco][coco-img]][coco-link] | [![instance-segmentation][instance-segmentation-img]][instance-segmentation-polygons-link] |
| Create a Table directly from a PyTorch Dataset (CIFAR-10). | Create a Table from a folder of images and a COCO format JSON file. | Create a Table from instance segmentation polygons.|
| **Instance segmentation - Masks** | **Instance segmentation - Bitmaps** | **Instance segmentation - Custom RLE** |
| [![instance-segmentation][instance-segmentation-img]][instance-segmentation-masks-link] | [![instance-segmentation][instance-segmentation-bitmaps-img]][instance-segmentation-bitmaps-link] | [![instance-segmentation][instance-segmentation-custom-rle-img]][instance-segmentation-custom-rle-link] |
| Create a Table from instance segmentation masks. | Create a Table a set of PNG grayscale masks for each class. | Create a Table from masks in a custom RLE format. |

[image-classification-img]: ../images/create-image-classification-table.jpg
[image-classification-link]: create-image-classification-table.ipynb
[custom-img]: ../images/create-custom-table.jpg
[custom-link]: create-custom-table.ipynb
[bb-img]: ../images/create-bb-table.jpg
[bb-link]: create-bb-table.ipynb
[yolo-img]: ../images/create-yolo-table.png
[yolo-link]: create-yolo-table.ipynb
[semseg-img]: ../images/ade-20-semseg.jpg
[semseg-link]: create-semantic-segmentation-table.ipynb
[video-img]: ../images/create-video-thumbnail-table.jpg
[video-link]: create-video-thumbnail-table.ipynb
[torch-img]: ../images/from-torch.png
[torch-link]: create-table-from-torch.ipynb
[coco-img]: ../images/coco.jpg
[coco-link]: create-table-from-coco.ipynb
[instance-segmentation-img]: ../images/instance-segmentation.jpg
[instance-segmentation-polygons-link]: create-instance-segmentation-polygons-table.ipynb
[instance-segmentation-masks-link]: create-instance-segmentation-masks-table.ipynb
[instance-segmentation-bitmaps-link]: create-instance-segmentations-from-masks.ipynb
[instance-segmentation-custom-rle-link]: create-instance-segmentations-from-custom.ipynb
[instance-segmentation-custom-rle-img]: ../images/cell-segmentations.jpg
[instance-segmentation-bitmaps-img]: ../images/LIACi.jpg
