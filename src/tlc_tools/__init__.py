"""Tools for working with the 3lc package."""

try:
    import tlc
except ImportError:
    raise ImportError("3lc is not installed. Please install it with `pip install 3lc` or equivalent.") from None

from .add_columns_to_table import add_columns_to_table, add_image_metrics_to_table
from .common import check_package_version
from .split import split_table

required_min_version = "2.14"
check_package_version("tlc", required_min_version)


__all__ = ["add_columns_to_table", "split_table", "add_image_metrics_to_table"]
