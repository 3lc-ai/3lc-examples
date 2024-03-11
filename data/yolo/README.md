# YOLO Object Detection Dataset Format

The YOLO dataset format for object detection has three main parts:
- The dataset yaml file which defines the path to the dataset root, the train/val/test splits and the categories/classes.
- One or more folders of images
- One or more corresponding folders of labels

The YAML file can specify the locations in three different ways
1. A single directory
2. A list of directories
3. A path to a single .txt file specifying the paths

We add examples here for all three, one yaml for each.

For more details see: https://docs.ultralytics.com/datasets/detect/