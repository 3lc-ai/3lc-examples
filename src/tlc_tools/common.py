"""Common utilities for the tools."""

from __future__ import annotations

import platform
import subprocess
import sys

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

    Returns:
        str: A message indicating whether the installed version is sufficient or not.

    """

    package = __import__(package_name)
    installed_version = package.__version__
    assert version.parse(installed_version) >= version.parse(required_min_version)


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


def check_is_bb_column(input_table: tlc.Table, bb_column: str) -> None:
    if bb_column not in input_table.columns:
        raise ValueError(f"Column {bb_column} not found in table {input_table.name}")

    if "bb_list" not in input_table.rows_schema.values[bb_column].values:
        raise ValueError(f"Column {bb_column} is missing the bb_list sub-column")

    if "label" not in input_table.rows_schema.values[bb_column].values["bb_list"].values:
        raise ValueError(f"Column {bb_column} is missing the label sub-column")

    if "x1" not in input_table.rows_schema.values[bb_column].values["bb_list"].values:
        raise ValueError(f"Column {bb_column} is missing the x1 sub-column")

    if "y1" not in input_table.rows_schema.values[bb_column].values["bb_list"].values:
        raise ValueError(f"Column {bb_column} is missing the y1 sub-column")

    if "x0" not in input_table.rows_schema.values[bb_column].values["bb_list"].values:
        raise ValueError(f"Column {bb_column} is missing the x0 sub-column")

    if "y0" not in input_table.rows_schema.values[bb_column].values["bb_list"].values:
        raise ValueError(f"Column {bb_column} is missing the y0 sub-column")
