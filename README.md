# `tlc` Python Package Examples, Tutorials, and Tools

Welcome to our collection of examples and tutorial notebooks for the `tlc`
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

    To install the development dependencies for formatting code and running tests (maintainers and contributors only), use [uv](https://docs.astral.sh/uv/):

    ```bash
    uv sync --all-extras --dev
    ```

4. Running the Notebooks

    You can open and run the notebooks with your preferred notebook editor or interface, such as Jupyter Notebook, JupyterLab, or VS Code. Start your notebook interface in the top-level of the repository to ensure relative paths in the notebooks work correctly.

    Example for Jupyter:

    ```bash
    jupyter lab
    ```

5. Get Started

    From your notebook interface, open any notebook from the repository to get started. Be sure to select the correct kernel/environment where the packages have been installed.

## Running the Notebooks

The notebooks are designed to run top-to-bottom on a fresh clone with no setup beyond the installation steps above. A few things are useful to know:

+ **Data**: Most notebooks either use small datasets bundled in the `data/` folder or download what they need at runtime (to the `transient_data/` folder, which is ignored by git). A few notebooks require extra steps, and say so in their introduction:
  + Kaggle-hosted datasets require [Kaggle API credentials](https://www.kaggle.com/docs/api) (`~/.kaggle/kaggle.json`).
  + Some datasets (e.g. FHIBE, LIACi) must be downloaded manually due to licensing; the notebook explains where to get them.
+ **Prerequisite notebooks**: Some notebooks build on 3LC Tables created by earlier notebooks (for example, training notebooks reuse tables from the `1-create-tables` tutorials). When a notebook has a prerequisite, it is linked in the notebook's introduction — run the linked notebook first.
+ **Hardware**: Training notebooks run fastest with a GPU (CUDA or Apple Silicon). The Detectron2 notebooks require a CUDA GPU and a Linux-like environment.

### Repository Layout

+ `tutorials/` — the main collection, organized as `1-create-tables`, `2-modify-tables`, `3-training-and-metrics`, and `4-end-to-end-examples`.
+ `examples/` — additional standalone examples, including custom sample types and integrations.
+ `data/` — small datasets bundled for the tutorials.
+ `src/tlc_tools/` — the `tlc_tools` Python package and CLI installed by `pip install -e .`.
+ `tutorials/notebooks.yaml` and `utils/` — **maintainer-facing infrastructure** used by the 3LC team's documentation and CI pipelines. You can ignore these entirely when running the notebooks.

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

+ **Title**: Balloons Dataset  
  **Author**: Paul Guerrie  
  **Publisher**: Roboflow  
  **Year**: 2024  
  **URL**: [Balloons Dataset on Roboflow Universe](https://universe.roboflow.com/paul-guerrie-tang1/balloons-geknh)  
  **Note**: Visited on 2024-03-15

+ **Title:** Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow  
  **Author:** Waleed Abdulla  
  **Year:** 2017  
  **Publisher:** Github  
  **URL:** [Releases](https://github.com/matterport/Mask_RCNN/releases)  
  **Repository:** [GitHub repository](https://github.com/matterport/Mask_RCNN)

+ **Title:** cat-and-dog-small  
  **Author:** Hongwei Cao  
  **Publisher:** Kaggle  
  **Year:** 2020  
  **URL:** [Kaggle Dataset](https://www.kaggle.com/datasets/hongweicao/catanddogsmall)

+ **Title:** Recognizing realistic actions from videos "in the wild"  
  **Author:** J. Liu, J. Luo and M. Shah  
  **Publisher:** CVPR  
  **Year:** 2009  
  **URL:** [Website](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php)  
  **Note:** Visited on 2024-06-25

+ **Title:** ADE20K Dataset  
  **Author:** Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso and Antonio Torralba  
  **Publisher:** MIT  
  **Year:** 2017  
  **URL:** [ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/index.html)

+ **Title:** LIACi semantic segmentation dataset for underwater ship inspections  
  **Author:** Maryna Waszak, Alexandre Cardaillac, Brian Elvesæter, Frode Rødølen, and Martin Ludvigsen  
  **Publisher:** SINTEF  
  **Year:** 2023  
  **URL:** [Website](https://data.sintef.no/product/details/dp-9e112cec-3a59-4b58-86b3-ecb1f2878c60)

+ **Title:** MSSDet: Multi-Scale Ship-Detection Framework in Optical Remote-Sensing Images and New Benchmark  
  **Authors:** W. Chen, B. Han, Z. Yang, X. Gao  
  **Journal:** Remote Sensing  
  **Year:** 2022  
  **Volume:** 14  
  **Article:** 5460  
  **URL:** [https://doi.org/10.3390/rs14215460](https://doi.org/10.3390/rs14215460)

+ **Title:** Cross-Domain Adaptation for Animal Pose Estimation  
  **Authors:** J. Cao, H. Tang, H.-S. Fang, X. Shen, C. Lu, and Y.-W. Tai  
  **Journal:** arXiv preprint  
  **Year:** 2019  
  **URL:** [https://arxiv.org/abs/1908.05806](https://arxiv.org/abs/1908.05806)

+ **Title:** LowPoly Cars  
  **Author:** Quaternius  
  **Publisher:** itch.io  
  **Year:** 2018  
  **URL:** [LowPoly Cars](https://quaternius.itch.io/lowpoly-cars)

+ **Title:** Fair human-centric image dataset for ethical AI benchmarking  
  **Author:** Sony AI  
  **Journal:** Nature  
  **Year:** 2025  
  **URL:** [https://www.nature.com/articles/s41586-025-09716-2](https://www.nature.com/articles/s41586-025-09716-2)

We also use the first 128 images from the [COCO](https://cocodataset.org/#home)
dataset.
