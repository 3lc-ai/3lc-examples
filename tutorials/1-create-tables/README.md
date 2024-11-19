# Create 3LC Tables

This folder contains notebooks demonstrating how to create tables in 3LC. Tables are the primary data structure in 3LC, and are used to store and manage data for various machine learning tasks.

The `tlc` Python package provides several helper functions for creating tables from different commonly used data sources. For custom data sources, you can create tables from scratch using the `TableWriter` class with a specified schema, and adding data row-by-row or in bulk.

|                        |                        |                        |
|------------------------|------------------------|------------------------|
| <div align="center">Image Classification</div>                | <div align="center">Custom Table</div> | <div align="center">Bounding Boxes</div> |
| [![img][image-classification-img]][image-classification-link] | [![custom][custom-img]][custom-link]   | [![bb][bb-img]][bb-link]                 |
| Create a table from a folder of subfolders, each subfolder containing images belonging to a certain class. | Write a custom Table by specifying a schema and adding data row-by-row. | Write a custom Table containing images and bounding boxes, using custom bounding boxes of an arbitrary format. |
| <div align="center">YOLO</div>                   | <div align="center">Semantic Segmentation</div>  | <div align="center">Video</div>       |
| [![yolo][yolo-img]][yolo-link] | [![semseg][semseg-img]][semseg-link] | [![video][video-img]][video-link] |
| Load images and bounding boxes from the YOLO YAML file format. | Write a Table from images and one or more masks. | Work with video datasets in 3LC by creating thumbnails. |
| <div align="center">From Torch Dataset</div>      |  |    |
| [![torch][torch-img]][torch-link] |  |    |
| Create a Table directly from a PyTorch Dataset (CIFAR-10). | | |

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
