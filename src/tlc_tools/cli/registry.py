from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple


class ToolInfo(NamedTuple):
    callable: Callable
    is_experimental: bool
    module_path: str
    description: str


# Global registry
_TOOLS: dict[str, ToolInfo] = {}


def register_tool(*, experimental: bool = False, description: str = ""):
    """Decorator to register a CLI tool, making it available through the 3lc-tools CLI.

    Args:
        experimental: Whether this is an experimental tool.
        description: A description of the tool, displayed when listing tools.
    """

    def decorator(func: Callable) -> Callable:
        tool_name = func.__module__.split(".")[-1]
        if experimental:
            # Remove experimental prefix from module path for tool name
            tool_name = tool_name.replace("experimental.", "")

        # Don't set prog when script is run directly
        if func.__module__ == "__main__":
            callable = func
        # Forward command that is used to run the tool when using 3lc-tools run
        else:
            callable = partial(func, prog=f"3lc-tools run {'--exp' if experimental else ''} {tool_name}")

        _TOOLS[tool_name] = ToolInfo(
            callable=callable, is_experimental=experimental, module_path=func.__module__, description=description
        )
        return callable

    return decorator


def get_registered_tools() -> dict[str, ToolInfo]:
    """Get all registered tools.

    :returns: A dictionary of tool names to tool info.
    """
    return _TOOLS.copy()
