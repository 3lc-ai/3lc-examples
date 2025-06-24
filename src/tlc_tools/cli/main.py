from __future__ import annotations

import argparse
import sys
from importlib.metadata import version

from .display import display_name, display_tools, normalize_tool_name
from .registry import ToolInfo, get_registered_tools


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser.

    :returns: The argument parser.
    """
    parser = argparse.ArgumentParser(description="List and run 3lc tools from the command line.")
    parser.add_argument("--version", action="store_true", help="show installed versions")

    subparsers = parser.add_subparsers(dest="command")

    # List command
    subparsers.add_parser("list", help="list available tools")

    # Run command
    run_parser = subparsers.add_parser("run", help="run a tool with arguments")
    run_parser.add_argument("tool", help="tool to run")

    # Pass remaining args through to the tool
    run_parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments to pass to the tool")

    return parser


def run_tool(tools: dict[str, ToolInfo], tool_name: str, args: list) -> None:
    """Run the specified tool with given arguments.

    :param tools: A dictionary of tool names to tool info.
    :param tool_name: The name of the tool to run.
    :param args: The arguments to pass to the tool.
    """
    normalized_name = normalize_tool_name(tool_name)

    if normalized_name not in tools:
        print(f"Error: Tool '{tool_name}' not found")
        sys.exit(1)

    tool = tools[normalized_name]

    # Pass the remaining arguments directly to the tool
    try:
        tool.callable(args)
    except Exception as e:
        print(f"Tool '{display_name(normalized_name)}' failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main() -> int:
    """Main entry point for the CLI"""
    parser = create_parser()

    try:
        args = parser.parse_args()
    except Exception:
        # If parsing fails, show help and return error code
        parser.print_help()
        return 1

    # Handle help explicitly
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        parser.print_help()
        sys.exit(0)

    if args.version:
        version_str = f"3lc-tools: {version('3lc-tools')}, 3lc: {version('3lc')}"
        print(version_str)
        sys.exit(0)

    tools = get_registered_tools()

    if args.command == "list":
        display_tools(tools)
        sys.exit(0)
    elif args.command == "run":
        try:
            run_tool(tools, args.tool, args.args)
            sys.exit(0)
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
