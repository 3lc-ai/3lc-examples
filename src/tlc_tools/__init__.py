"""Tools for working with the 3lc package."""

import tlc_tools.experimental  # noqa: F401

from .add_columns_to_table import add_columns_to_table, add_image_metrics_to_table
from .common import check_package_version
from .split import split_table

try:
    required_min_version = "2.11"
    check_package_version("tlc", required_min_version)
    import tlc
except Exception as e:
    msg = f"tlc_tools requires tlc version {required_min_version} or higher (found {tlc.__version__})."
    raise ImportError(msg) from e


__all__ = ["add_columns_to_table", "split_table", "add_image_metrics_to_table"]
