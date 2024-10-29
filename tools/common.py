"""Common utilities for the tools."""

import json
import platform
import subprocess
import sys

import numpy as np
import tlc
import torch
from packaging import version
from tlc.core.objects.tables.from_url.table_from_row_cache import TableFromRowCache


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


def check_package_version(package_name: str, required_version: str) -> str:
    """Check if the installed version of a package meets the required version.

    Args:
        package_name (str): The name of the package to check.
        required_version (str): The minimum required version of the package.

    Returns:
        str: A message indicating whether the installed version is sufficient or not.

    """
    try:
        package = __import__(package_name)
        installed_version = package.__version__
        if version.parse(installed_version) >= version.parse(required_version):
            return f"{package_name} version {installed_version} is sufficient (required: {required_version})"
        else:
            return f"{package_name} version {installed_version} is not sufficient (required: {required_version})"
    except ImportError:
        return f"{package_name} package is not installed."


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


def get_column_from_table(table: tlc.Table, column_name: str) -> np.ndarray:
    if isinstance(table, tlc.core.objects.tables.in_memory_columns_table._InMemoryColumnsTable):
        table_with_columns = table

    elif table.row_cache_populated:
        init_params = json.loads((table.url / "object.3lc.json").read())
        init_params["row_cache_url"] = tlc.Url(init_params["row_cache_url"]).to_absolute(table.url)
        table_with_columns = TableFromRowCache(init_params)
    else:
        raise ValueError("Table must have columns to extract.")

    pa_column = table_with_columns.get_column(column_name)
    return np.array(pa_column.to_numpy().tolist(), dtype=np.float32)


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


if __name__ == "__main__":
    # Example usage
    print(check_tlc_package_version())
    print(f"Python version: {check_python_version()}")
    if not is_package_installed("tlc"):
        print("Installing tlc package...")
        install_package("tlc")
        print(check_tlc_package_version())
    else:
        print("tlc package is already installed.")
    print(check_package_version("tlc", "1.0.0"))
