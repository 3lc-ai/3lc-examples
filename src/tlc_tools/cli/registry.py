from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple


class ToolInfo(NamedTuple):
    callable: Callable
    module_path: str
    description: str


# Global registry
_TOOLS: dict[str, ToolInfo] = {}


def register_tool(*, name: str | None = None, description: str = ""):
    """Decorator to register a CLI tool, making it available through the 3lc-tools CLI.

    Args:
        name: Optional custom name for the tool. If not provided, defaults to the module name.
        description: A description of the tool, displayed when listing tools.
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name if name is not None else func.__module__.split(".")[-1]

        # Don't set prog when script is run directly, forward command that is
        # used to run the tool when using 3lc-tools run
        callable = func if func.__module__ == "__main__" else partial(func, prog=f"3lc-tools run {tool_name}")

        _TOOLS[tool_name] = ToolInfo(callable=callable, module_path=func.__module__, description=description)
        return callable

    return decorator


def get_registered_tools() -> dict[str, ToolInfo]:
    """Get all registered tools.

    :returns: A dictionary of tool names to tool info.
    """
    return _TOOLS.copy()
