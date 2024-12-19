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
| Load images and bounding boxes from the YOLO YAML file format. | Write a Table from images and one or more masks. | Work with video datasets in 3LC by creating thumbnails and storing URLs to video files. |
| **From PyTorch Dataset** | **COCO** |    |
| [![torch][torch-img]][torch-link] | [![coco][coco-img]][coco-link] |    |
| Create a Table directly from a PyTorch Dataset (CIFAR-10). | Create a Table from a folder of images and a COCO format JSON file. | |

[image-classification-img]: ../images/create-image-classification-table.png
[image-classification-link]: create-image-classification-table.ipynb
[custom-img]: ../images/create-custom-table.png
[custom-link]: create-custom-table.ipynb
[bb-img]: ../images/create-bb-table.png
[bb-link]: create-bb-table.ipynb
[yolo-img]: ../images/create-yolo-table.png
[yolo-link]: create-yolo-table.ipynb
[semseg-img]: ../images/semseg.png
[semseg-link]: create-semantic-segmentation-dataset.ipynb
[video-img]: ../images/create-video-thumbnail-table.png
[video-link]: create-video-thumbnail-table.ipynb
[torch-img]: ../images/from-torch.png
[torch-link]: create-table-from-torch.ipynb
[coco-img]: ../images/coco.png
[coco-link]: create-table-from-coco.ipynb
