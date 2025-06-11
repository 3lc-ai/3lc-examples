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


# === INSTANCE HANDLING - NEW FUNCTIONALITY ===


class InstanceConfig:
    """Configuration class for instance handling arguments."""

    def __init__(
        self,
        instance_column: str | None = None,
        instance_type: Literal["bounding_boxes", "segmentations"] | None = None,
        label_column_path: str | None = None,
        allow_label_free: bool = False,
    ):
        """Initialize instance configuration.

        :param instance_column: Name of the column containing instances (e.g., "bbs", "segmentations")
        :param instance_type: Type of instances - auto-detect by default
        :param label_column_path: Path to labels within instance column (e.g., "bbs.bb_list.label")
        :param allow_label_free: Whether to allow processing without labels
        """
        self.instance_column = instance_column
        self.instance_type = instance_type
        self.label_column_path = label_column_path
        self.allow_label_free = allow_label_free

    @property
    def instance_properties_column(self) -> str:
        """Get the instance properties column."""
        if self.label_column_path is None:
            raise ValueError("Label column path is not set")
        return self.label_column_path.split(".")[-2]


def resolve_label_column_path(
    instance_column: str,
    instance_type: Literal["bounding_boxes", "segmentations"],
) -> str | None:
    """Resolve label column path intelligently.

    Handles both full paths and relative paths:
    - Full path: "bbs.bb_list.label" -> use as-is
    - Relative path: "bb_list.label" -> prepend instance_column
    - None: auto-infer based on instance_type

    :param instance_column: Name of the instance column
    :param instance_type: Type of instances
    :param label_column_path: Provided label path (full, relative, or None)
    :return: Resolved full label column path, or None if no labels
    """
    if instance_type == "bounding_boxes":
        return f"{instance_column}.bb_list.label"
    elif instance_type == "segmentations":
        return f"{instance_column}.instance_properties.label"
    else:
        raise ValueError(f"Invalid instance type: {instance_type}")


def detect_instance_column_and_type(
    input_table: tlc.Table,
    instance_type: Literal["bounding_boxes", "segmentations", "auto"] = "auto",
) -> tuple[str, Literal["bounding_boxes", "segmentations"]]:
    """Auto-detect the instance column in a table.

    :param input_table: The table to analyze
    :param instance_type: Type of instances
    :return: Name of the detected instance column and type of instances
    :raises ValueError: If no valid instance column found
    """
    if instance_type == "bounding_boxes":
        if "bbs" in input_table.columns:
            return "bbs", "bounding_boxes"
        else:
            raise ValueError(f"No bounding boxes column found in table {input_table.name}")
    elif instance_type == "segmentations":
        if "segmentations" in input_table.columns:
            return "segmentations", "segmentations"
        elif "segments" in input_table.columns:
            return "segments", "segmentations"
        else:
            raise ValueError(f"No segmentation column found in table {input_table.name}")
    elif instance_type == "auto":
        if "bbs" in input_table.columns:
            return "bbs", "bounding_boxes"
        elif "segmentations" in input_table.columns:
            return "segmentations", "segmentations"
        elif "segments" in input_table.columns:
            return "segments", "segmentations"
        else:
            raise ValueError(f"No valid instance column found in table {input_table.name}")


def validate_instance_config(input_table: tlc.Table, instance_config: InstanceConfig) -> None:
    """Validate the instance configuration.

    :param instance_config: The instance configuration to validate.
    """
    pass


def resolve_instance_config(
    input_table: tlc.Table,
    instance_column: str | None = None,
    instance_type: Literal["bounding_boxes", "segmentations", "auto"] = "auto",
    allow_label_free: bool = False,
) -> InstanceConfig:
    """Resolve and validate instance configuration for a table.

    This is the main entry point for instance configuration. It will:
    1. Auto-detect instance column if not provided
    2. Auto-detect instance type if set to "auto"
    3. Intelligently resolve label column path (full/relative/auto-infer)
    4. Validate the final configuration

    :param input_table: The table to configure for
    :param instance_column: Instance column name (auto-detect if None)
    :param instance_type: Instance type ("auto" to detect)
    :param allow_label_free: Whether to allow label-free operation
    :return: Resolved and validated InstanceConfig
    :raises ValueError: If configuration cannot be resolved or is invalid
    """
    # Step 1: Detect instance column if needed
    if instance_column is None:
        instance_column, instance_type = detect_instance_column_and_type(input_table, instance_type)

    assert instance_type in ["bounding_boxes", "segmentations"]
    assert instance_column != ""
    instance_type = cast(Literal["bounding_boxes", "segmentations"], instance_type)

    # Step 3: Intelligently resolve label column path
    resolved_label_path = resolve_label_column_path(instance_column=instance_column, instance_type=instance_type)

    # Step 4: Create and validate configuration
    config = InstanceConfig(
        instance_column=instance_column,
        instance_type=instance_type,
        label_column_path=resolved_label_path,
        allow_label_free=allow_label_free,
    )

    # # Step 5: Validate the configuration
    validate_instance_config(input_table, config)

    return config
