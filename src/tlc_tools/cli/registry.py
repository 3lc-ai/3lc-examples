from __future__ import annotations

import importlib
from functools import partial
from typing import Callable, NamedTuple


class LazyToolInfo(NamedTuple):
    """Tool info that loads the actual function lazily."""
    module_path: str
    function_name: str
    description: str
    
    @property
    def callable(self) -> Callable:
        """Lazily import and return the tool function."""
        if not hasattr(self, '_callable'):
            try:
                module = importlib.import_module(self.module_path)
                func = getattr(module, self.function_name)
                
                # Don't set prog when script is run directly
                if func.__module__ == "__main__":
                    self._callable = func
                else:
                    tool_name = self.function_name.replace('_cli', '').replace('_', '-')
                    self._callable = partial(func, prog=f"3lc-tools run {tool_name}")
                    
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Failed to load tool {self.function_name} from {self.module_path}: {e}") from e
                
        return self._callable


class ToolInfo(NamedTuple):
    """Legacy tool info for backward compatibility."""
    callable: Callable
    module_path: str
    description: str


# Global registry - tools are registered lazily
_TOOLS: dict[str, LazyToolInfo | ToolInfo] = {}


def register_tool(*, name: str | None = None, description: str = ""):
    """Decorator to register a CLI tool, making it available through the 3lc-tools CLI.

    Args:
        name: Optional custom name for the tool. If not provided, defaults to the module name.
        description: A description of the tool, displayed when listing tools.
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name if name is not None else func.__module__.split(".")[-1]

        # For immediate registration (backward compatibility)
        callable = func if func.__module__ == "__main__" else partial(func, prog=f"3lc-tools run {tool_name}")
        _TOOLS[tool_name] = ToolInfo(callable=callable, module_path=func.__module__, description=description)
        return callable

    return decorator


def register_lazy_tool(name: str, module_path: str, function_name: str, description: str = ""):
    """Register a tool lazily without importing it immediately.
    
    Args:
        name: The name of the tool for the CLI
        module_path: Python module path containing the tool
        function_name: Name of the function in the module
        description: Description of the tool
    """
    _TOOLS[name] = LazyToolInfo(
        module_path=module_path,
        function_name=function_name, 
        description=description
    )


def get_registered_tools() -> dict[str, ToolInfo]:
    """Get all registered tools, converting lazy tools to regular ToolInfo.

    :returns: A dictionary of tool names to tool info.
    """
    result = {}
    for name, tool in _TOOLS.items():
        if isinstance(tool, LazyToolInfo):
            # Convert to ToolInfo for backward compatibility
            result[name] = ToolInfo(
                callable=tool.callable,  # This triggers lazy loading
                module_path=tool.module_path,
                description=tool.description
            )
        else:
            result[name] = tool
    return result


# Pre-register known tools lazily to avoid imports
_LAZY_TOOL_REGISTRY = {
    "alias": ("tlc_tools.cli.commands.alias", "main", "Create aliases for datasets and tables"),
    "augment-instance-table": ("tlc_tools.cli.commands.augment_instance_table", "main", "Augment instance table with additional data"),
    "metric-jumps": ("tlc_tools.cli.commands.metric_jumps_cli", "main", "Compute metric jumps across time steps"),
}

# Register all known tools lazily
for tool_name, (module_path, function_name, description) in _LAZY_TOOL_REGISTRY.items():
    register_lazy_tool(tool_name, module_path, function_name, description)
