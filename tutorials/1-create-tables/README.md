# Create 3LC Tables

This folder contains notebooks demonstrating how to create tables in 3LC. Tables are the primary data structure in 3LC, and are used to store and manage data for various machine learning tasks.

The `tlc` Python package provides several helper functions for creating tables from different commonly used data sources. For custom data sources, you can create tables from scratch using the `TableWriter` class with a specified schema, and adding data row-by-row or in bulk.

|                        |                        |                        |
|------------------------|------------------------|------------------------|
| Image Classification   | Custom Table           | Bounding Boxes         |
| [![img][image-classification-img]][image-classification-link] <br> Create a table from a folder of subfolders, each subfolder containing images belonging to a certain class. | [![custom][custom-img]][custom-link] <br> Write a custom Table by specifying a schema and adding data row-by-row. | [![bb][bb-img]][bb-link] <br> Write a custom Table containing images and bounding boxes, using custom bounding boxes of an arbitrary format. |
| YOLO                   | Semantic Segmentation  | Video Thumbnails       |
| [![yolo][yolo-img]][yolo-link] <br> Load images and bounding boxes from the YOLO YAML file format. | [![semseg][semseg-img]][semseg-link] <br> Write a Table from images and one or more masks | [![video][video-img]][video-link] <br> Write a Table with video thumbnails. |
| From Torch Dataset      |  |    |
| [![torch][torch-img]][torch-link] <br> Create a Table directly from a PyTorch Dataset (CIFAR-10). |  |    |

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
