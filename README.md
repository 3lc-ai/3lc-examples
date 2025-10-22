# `tlc` Python Package Example Notebooks, Tutorials, and Tools

Welcome to our collection of example notebooks and tutorials for the `tlc`
Python package! This repository contains various Jupyter notebooks and Python
scripts that demonstrate how to use the `tlc` Python package across different
scenarios and use cases.

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.

### Prerequisites

You will need the following tools installed on your system:

+ A suitable version of Python (See
  [documentation](https://docs.3lc.ai/3lc/latest/quickstart/quickstart.html#requirements)
  for supported versions)
+ Access to the `tlc` Python package

### Installation

Follow these steps to set up the repository and environment for running the notebooks.

1. Clone the Repository

    ```bash
    git clone https://github.com/3lc-ai/3lc-examples.git
    cd 3lc-examples
    ```

2. Python Environment Setup

    It is recommended to use a virtual environment or Conda
    environment to avoid conflicts with system-wide packages. Activate your
    environment before proceeding.

3. Install Required Python Packages

    Install the required Python packages with:

    ```bash
    pip install -e .
    ```

    This will install the main package, `tlc_tools`, along with its dependencies. Additionally, the following optional extras are provided for specific notebooks:

    + **huggingface**: For transformer models and hugging face datasets.
    + **umap**: For dimensionality reduction.
    + **pacmap**: For dimensionality reduction.
    + **sam**: For segmentation tasks (Segment Anything Model).
    + **timm**: For pretrained vision models.
    + **kaggle**: For downloading datasets.
    + **ultralytics**: For YOLOv8/YOLO11 object detection.
    + **lightning**: For PyTorch Lightning tasks.

    To install extras required for a specific notebook, use the following syntax:

    ```bash
      pip install -e .[extra_name]
    ```

    For example, to install packages required for notebooks using Hugging Face and UMAP:

    ```bash
    pip install -e .[huggingface,umap]
    ```

    If you want to install all extras at once, use:

    ```bash
    pip install -e .[all]
    ```

    To install the dev dependencies for formatting code and running tests, use:

    ```bash
    pip install -e .[dev]
    ```

4. Running the Notebooks

    You can open and run the notebooks with your preferred notebook editor or interface, such as Jupyter Notebook, JupyterLab, or VS Code. Start your notebook interface in the top-level of the repository to ensure relative paths in the notebooks work correctly.

    Example for Jupyter:

    ```bash
    jupyter lab
    ```

5. Get Started

    From your notebook interface, open any notebook from the repository to get started. Be sure to select the correct kernel/environment where the packages have been installed.

## CLI

The `tlc_tools` package includes a CLI for running some tools directly from the command line. To see the available tools, run:

```bash
3lc-tools list
```

To run a tool, use the following syntax:

```bash
3lc-tools run <tool-name> <tool-args>
```

The tool name is the name presented in the `3lc-tools list` command, and the tool-args are forwarded to the tool.

## Contributing

We welcome contributions to this repository! If you have a suggestion for an
additional example or improvement, please open an issue or create a pull
request.

Any contributions should be made in the `tools`, `tutorials` and `data` folders
only, other files and folders are maintained by the 3LC team. See the
[CONTRIBUTING.md](CONTRIBUTING.md) file for more details.

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

This project is licensed under the Apache 2.0 License - see the LICENSE file for
details.

## Acknowledgments

We would like to thank the authors and publishers of the datasets used in these
notebooks. The datasets are listed below:

**Title**: Balloons Dataset  
**Author**: Paul Guerrie  
**Publisher**: Roboflow  
**Year**: 2024  
**URL**: [Balloons Dataset on Roboflow
Universe](https://universe.roboflow.com/paul-guerrie-tang1/balloons-geknh)  
**Note**: Visited on 2024-03-15

**Title**: Mask R-CNN for object detection and instance segmentation on Keras
and TensorFlow  
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

**Title**: ADE20K Dataset
**Author**: Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso and Antonio Torralba
**Publisher**: MIT
**Year**: 2017
**URL**: [ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/index.html)

**Title**: LIACi semantic segmentation dataset for underwater ship inspections
**Author**: Maryna Waszak, Alexandre Cardaillac, Brian Elvesæter, Frode Rødølen, and Martin Ludvigsen
**Publisher**: SINTEF
**Year**: 2023
**URL**: [Website](https://data.sintef.no/product/details/dp-9e112cec-3a59-4b58-86b3-ecb1f2878c60)

**Title**: MSSDet: Multi-Scale Ship-Detection Framework in Optical Remote-Sensing Images and New Benchmark  
**Authors**: W. Chen, B. Han, Z. Yang, X. Gao  
**Journal**: Remote Sensing  
**Year**: 2022  
**Volume**: 14  
**Article**: 5460  
**URL**: [https://doi.org/10.3390/rs14215460](https://doi.org/10.3390/rs14215460)

**Title**: Cross-Domain Adaptation for Animal Pose Estimation  
**Authors**: J. Cao, H. Tang, H.-S. Fang, X. Shen, C. Lu, and Y.-W. Tai  
**Journal**: arXiv preprint  
**Year**: 2019  
**URL**: [https://arxiv.org/abs/1908.05806](https://arxiv.org/abs/1908.05806)

We also use the first 128 images from the [COCO](https://cocodataset.org/#home)
dataset.
