"""Common utilities for the tools."""

from __future__ import annotations

import io
import platform
import subprocess
import sys
import zipfile
from typing import Literal, cast

import requests
import tlc
import torch
from packaging import version


def infer_torch_device() -> torch.device:
    """Infer the device to use for the computation.

    :returns: The device to use for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def check_tlc_package_version() -> str:
    """Check the installed version of the tlc package.

    :returns: Version of the tlc package if installed, otherwise a message indicating it's not installed.
    """
    try:
        import tlc
    except ImportError:
        return "tlc package is not installed."
    else:
        return f"tlc version: {tlc.__version__}"


def check_package_version(package_name: str, required_min_version: str) -> None:
    """Check if the installed version of a package meets the required version.

    :param package_name: The name of the package to check.
    :param required_version: The minimum required version of the package.
    :raises ValueError: If the installed version of the package is less than the required version.
    """

    package = __import__(package_name)
    installed_version = package.__version__
    if version.parse(installed_version) < version.parse(required_min_version):
        raise ValueError(
            f"The installed version of {package_name} is {installed_version}, "
            f"but {required_min_version} or higher is required."
        )


def install_package(package_name: str) -> None:
    """
    Install a package using pip.

    :param package_name: The name of the package to install.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def run_command(command) -> str:
    """Run a command in the system shell and return the output.

    :param command: The command to run.
    :returns: The output from the command.

    """
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True)
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        return f"Error running command: {e.stderr.decode().strip()}"


def check_python_version() -> str:
    """Check the version of Python currently running.

    :returns: Python version.
    """
    return sys.version


def is_package_installed(package_name: str) -> bool:
    """Check if a specific package is installed.

    :param package_name: The name of the package to check.
    :returns: True if the package is installed, False otherwise.
    """
    try:
        __import__(package_name)
    except ImportError:
        return False
    else:
        return True


def is_windows() -> bool:
    return platform.system() == "Windows"


def check_is_bb_column(
    input_table: tlc.Table,
    bb_column: str = "bbs",
    bb_list_column: str = "bb_list",
) -> None:
    """Check that a column conforms to 3LC's bounding box format.

    Ensures that the `bb_list_column` is present in the `bb_column`'s schema.

    Ensures the required "label", "x1", "y1", "x0", "y0" sub-columns are present
    in the `bb_list_column`'s schema.

    :param input_table: The table to check.
    :param bb_column: The name of the column to check.
    :param bb_list_column: The name of the sub-column in the column to check.

    :raises ValueError: If the column is missing the `bb_list_column` or any of
        the required sub-columns.
    """
    if bb_column not in input_table.columns:
        raise ValueError(f"Column {bb_column} not found in table {input_table.name}")

    if bb_list_column not in input_table.rows_schema.values[bb_column].values:
        raise ValueError(f"Column {bb_column} is missing the {bb_list_column} sub-column")

    bb_list_schema = input_table.rows_schema.values[bb_column].values[bb_list_column]

    for column in ["label", "x1", "y1", "x0", "y0"]:
        if column not in bb_list_schema.values:
            raise ValueError(f"Column {bb_column} is missing the {column} sub-column")


def check_is_segmentation_column(
    input_table: tlc.Table,
    segmentation_column: str = tlc.SEGMENTATIONS,
    sample_type: Literal["instance_segmentation_masks", "instance_segmentation_polygons", ""] = "",
) -> None:
    """Check that a column conforms to 3LC's segmentation format.

    Ensures that the `segmentation_column` is present in the `input_table`'s schema.

    Ensures that the `segmentation_column`'s schema is a `tlc.InstanceSegmentationMasks`
    or `tlc.InstanceSegmentationPolygons` sample type.

    :param input_table: The table to check.
    :param segmentation_column: The name of the column to check.
    :param sample_type: The sample type of the segmentation column.

    :raises ValueError: If the column is missing the `segmentation_column` or
        the `segmentation_column`'s schema is not a `tlc.InstanceSegmentationMasks`
        or `tlc.InstanceSegmentationPolygons` sample type.
    """
    if segmentation_column not in input_table.columns:
        raise ValueError(f"Column {segmentation_column} not found in table {input_table.name}")

    if segmentation_column not in input_table.rows_schema.values:
        raise ValueError(f"Column {segmentation_column} not found in table {input_table.name}")

    if sample_type and input_table.rows_schema.values[segmentation_column].sample_type != sample_type:
        raise ValueError(f"Column {segmentation_column} is not a {sample_type} sample type")


def download_and_extract_zipfile(url: str, location: str = "."):
    """Download a zipfile and extract it to a specified location.

    :param url: The URL of the zipfile to download.
    :param location: The location to extract the zipfile to.
    """
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(location)
