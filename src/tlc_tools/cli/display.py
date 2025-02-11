from __future__ import annotations

from tlc_tools.cli.registry import ToolInfo


def normalize_tool_name(name: str) -> str:
    """Convert dashes to underscores in tool name"""
    return name.replace("-", "_")


def display_name(name: str) -> str:
    """Convert underscores to dashes for display"""
    return name.replace("_", "-")


def display_tools(tools: dict[str, ToolInfo]) -> None:
    """Display available tools"""
    print("Available tools:")
    if tools:
        for tool_name, tool in sorted(tools.items()):
            experimental_marker = " (experimental)" if tool.is_experimental else ""
            print(f"  - {display_name(tool_name)}{experimental_marker} : {tool.description}")
    else:
        print("  No tools available")
