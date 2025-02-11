from __future__ import annotations

import argparse
import sys

from .display import display_name, display_tools, normalize_tool_name
from .registry import ToolInfo, get_registered_tools


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser.

    :returns: The argument parser.
    """
    parser = argparse.ArgumentParser(description="3LC Tools CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    subparsers.add_parser("list", help="List available tools")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a tool")
    run_parser.add_argument("tool", help="Tool to run")
    run_parser.add_argument("--exp", "--experimental", action="store_true", help="Allow running experimental tools")

    # Pass remaining args through to the tool
    run_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments to pass to the tool")

    return parser


def run_tool(tools: dict[str, ToolInfo], tool_name: str, args: list, allow_experimental: bool) -> None:
    """Run the specified tool with given arguments.

    :param tools: A dictionary of tool names to tool info.
    :param tool_name: The name of the tool to run.
    :param args: The arguments to pass to the tool.
    :param allow_experimental: Whether to allow running experimental tools.
    """
    normalized_name = normalize_tool_name(tool_name)

    if normalized_name not in tools:
        print(f"Error: Tool '{tool_name}' not found")
        sys.exit(1)

    tool = tools[normalized_name]
    if tool.is_experimental and not allow_experimental:
        print(
            f"Error: '{display_name(normalized_name)}' is an experimental tool. "
            f"Use '3lc-tools run --exp {display_name(normalized_name)}' to run it."
        )
        sys.exit(1)

    # Pass the remaining arguments directly to the tool
    tool.callable(args)


def main() -> int:
    """Main entry point for the CLI"""
    parser = create_parser()
    args = parser.parse_args()

    tools = get_registered_tools()

    if args.command == "list":
        display_tools(tools)
        return 0
    elif args.command == "run":
        run_tool(tools, args.tool, args.args, args.exp)
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
