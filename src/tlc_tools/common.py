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

    Returns:
        torch.device: The device to use for computation.

    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def check_tlc_package_version() -> str:
    """Check the installed version of the tlc package.

    Returns:
        str: Version of the tlc package if installed, otherwise a message indicating it's not installed.

    """
    try:
        import tlc
    except ImportError:
        return "tlc package is not installed."
    else:
        return f"tlc version: {tlc.__version__}"


def check_package_version(package_name: str, required_version: str) -> None:
    """Check if the installed version of a package meets the required version.

    Args:
        package_name (str): The name of the package to check.
        required_version (str): The minimum required version of the package.

    Returns:
        str: A message indicating whether the installed version is sufficient or not.

    """

    package = __import__(package_name)
    installed_version = package.__version__
    assert version.parse(installed_version) >= version.parse(required_version)


def install_package(package_name: str):
    """Install a package using pip.

    Args:
        package_name (str): The name of the package to install.

    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def run_command(command) -> str:
    """Run a command in the system shell and return the output.

    Args:
        command (str): The command to run.

    Returns:
        str: The output from the command.

    """
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True)
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        return f"Error running command: {e.stderr.decode().strip()}"


def check_python_version():
    """Check the version of Python currently running.

    Returns:
        str: Python version.
    """
    return sys.version


def is_package_installed(package_name: str) -> bool:
    """Check if a specific package is installed.

    Args:
        package_name (str): The name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.

    """
    try:
        __import__(package_name)
    except ImportError:
        return False
    else:
        return True


def is_windows() -> bool:
    return platform.system() == "Windows"


def keep_indices(table: tlc.Table, indices: list[int], table_name: str | None = None) -> tlc.Table:
    """Keep only the rows with the specified indices in the table.

    Args:
        table (tlc.Table): The table to filter.
        indices (list[int]): The indices to keep.

    Returns:
        tlc.Table: The filtered table.
    """

    all_indices = list(range(len(table)))
    indices_to_remove = list(set(all_indices) - set(indices))
    runs_and_values = []
    for index in indices_to_remove:
        runs_and_values.extend([[index], True])
    edits = {
        tlc.SHOULD_DELETE: {"runs_and_values": runs_and_values},
    }
    edited_table = tlc.EditedTable(
        url=table.url.create_sibling(table_name or "remove").create_unique(),
        input_table_url=table,
        edits=edits,
        row_cache_url="./row_cache.parquet",
    )
    edited_table.ensure_fully_defined()
    return edited_table
