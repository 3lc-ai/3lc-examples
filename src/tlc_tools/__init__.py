"""Tools for working with the 3lc package."""

from .add_columns_to_table import add_columns_to_table, add_image_metrics_to_table
from .common import check_package_version
from .split import split_table

try:
    check_package_version("tlc", "2.10")
    import tlc
except Exception as e:
    raise ImportError(f"tlc_tools requires tlc version 2.10 or higher (found {tlc.__version__}).") from e


__all__ = ["add_columns_to_table", "split_table", "add_image_metrics_to_table"]
