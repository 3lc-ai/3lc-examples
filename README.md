# `tlc` Python Package Example Notebooks

Welcome to our collection of example notebooks and tutorials for the `tlc`
Python package! This repository contains various Jupyter notebooks and Python
scripts that demonstrate how to use the `tlc` Python package across different
scenarios and use cases.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing
purposes.

### Prerequisites

You will need the following tools installed on your system:

+ A suitable version of Python (See [documentation](https://docs.3lc.ai/3lc/latest/quickstart/quickstart.html#requirements) for supported versions)
+ Access to the `tlc` Python package

### Installation

Clone this repository to your local machine:

```bash
# Copy code
git clone https://github.com/3lc-ai/3lc-examples.git

# Navigate to the cloned directory:
cd 3lc-examples

# Activate your Python environment (if applicable)

# Open the Jupyter notebook interface:
jupyter notebook

#From the Jupyter interface, open any notebook from the list to get started.
```

## Contributing

We welcome contributions to this repository! If you have a suggestion for an
additional example or improvement, please open an issue or create a pull
request.

Any contributions should be made in the `tutorials` and `data` folders only,
other files and folders are maintained by the 3LC team.

### Data

All required data for running the notebooks is either contained in the `./data`
folder, or is downloaded from the internet during the notebook execution.

When contributing new notebooks/scripts, it is preferable to have the notebook
download any required data from the internet, rather than including them in the
repository. If however, this is not possible, small contributions of data files
are accepted in the `./data` folder, but please do not include large files or
datasets. Ensure also that the data files are not restricted by any licensing
agreements.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

We use two versions of the Balloons dataset:

**Title**: Balloons Dataset  
**Author**: Paul Guerrie  
**Publisher**: Roboflow  
**Year**: 2024  
**URL**: [Balloons Dataset on Roboflow Universe](https://universe.roboflow.com/paul-guerrie-tang1/balloons-geknh)  
**Note**: Visited on 2024-03-15

**Title**: Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow  
**Author**: Waleed Abdulla  
**Year**: 2017  
**Publisher**: Github  
**URL**: [Releases](https://github.com/matterport/Mask_RCNN/releases)  
**Repository**: [GitHub repository](https://github.com/matterport/Mask_RCNN)

**Title**: cat-and-dog-small  
**Author**: Hongwei Cao  
**Publisher**: Kaggle  
**Year**: 2020  
**URL**: [Kaggle Dataset](https://www.kaggle.com/datasets/hongweicao/catanddogsmall)  

**Title**: Recognizing realistic actions from videos "in the wild"
**Author**: J. Liu, J. Luo and M. Shah
**Publisher**: CVPR
**Year**: 2009
**URL**: [Website](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php)
**Note**: Visited on 2024-06-25

We also use the first 128 images from the [COCO](https://cocodataset.org/#home) dataset.
