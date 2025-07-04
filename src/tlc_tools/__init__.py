"""Tools for working with the 3lc package."""

import sys
from typing import Any

# Check 3lc availability but don't import it yet
def _check_tlc_availability():
    """Check if 3lc is available without importing it."""
    try:
        import tlc  # noqa: F401
    except ImportError:
        raise ImportError("3lc is not installed. Please install it with `pip install 3lc` or equivalent.") from None

def _check_package_version_lazy():
    """Check package version lazily."""
    from .common import check_package_version
    required_min_version = "2.14"
    check_package_version("tlc", required_min_version)

# Module-level __getattr__ for lazy imports (Python 3.7+)
def __getattr__(name: str) -> Any:
    """Lazy import attributes when accessed."""
    _check_tlc_availability()
    _check_package_version_lazy()
    
    if name == "split_table":
        from .split import split_table
        return split_table
    elif name == "add_columns_to_table":
        from .add_columns_to_table import add_columns_to_table
        return add_columns_to_table
    elif name == "add_image_metrics_to_table":
        from .add_columns_to_table import add_image_metrics_to_table
        return add_image_metrics_to_table
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Explicitly define what's available for introspection
__all__ = ["add_columns_to_table", "split_table", "add_image_metrics_to_table"]

# For backwards compatibility, still allow direct imports in interactive sessions
if hasattr(sys, 'ps1'):  # Interactive mode
    _check_tlc_availability()
    _check_package_version_lazy()
