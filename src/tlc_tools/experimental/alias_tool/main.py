from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from tlc.core import Url

from tlc_tools.cli import register_tool

from .common import get_input_object, setup_logging
from .list_aliases import list_aliases
from .replace_aliases import replace_aliases

logger = logging.getLogger(__name__)


def create_argument_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """Create and configure the argument parser for the alias tool.

    Args:
        prog: Optional program name to use in help text.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(prog=prog, description="List, rewrite, and create URL aliases in 3LC objects")

    # Verbosity control for all commands
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (use -v for debug output)",
    )
    verbosity_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output except for warnings and errors",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command - for finding aliases in files
    list_parser = subparsers.add_parser("list", help="List all aliases found in files")
    list_parser.add_argument("input_path", help="The input object to process")
    list_parser.add_argument("--columns", help="Comma-separated list of columns to process")

    # Replace command - for applying aliases and path rewrites
    replace_parser = subparsers.add_parser("replace", help="Replace paths with aliases in files")
    replace_parser.add_argument("input_path", help="The input object to process")
    replace_parser.add_argument("--columns", help="Comma-separated list of columns to process")
    replace_parser.add_argument(
        "--no-process-parents",
        action="store_true",
        help="Do not process parent tables when handling Table objects",
    )

    # Replacement specification options
    replace_mode = replace_parser.add_mutually_exclusive_group(required=True)
    replace_mode.add_argument(
        "--apply",
        metavar="ALIAS[,ALIAS,...]",
        help="Apply existing aliases to matching paths (comma-separated list)",
    )
    replace_mode.add_argument(
        "--from",
        dest="from_paths",
        metavar="PATH",
        action="append",
        help="Replace occurrences of this path (can be specified multiple times)",
    )

    # Required when using --from
    replace_parser.add_argument(
        "--to",
        dest="to_paths",
        metavar="PATH",
        action="append",
        help="Replace with this path (must match number of --from arguments)",
    )

    return parser


def parse_columns(columns_arg: str | None) -> list[str]:
    """Parse the columns argument into a list of column names.

    Args:
        columns_arg: Comma-separated string of column names or None.

    Returns:
        List of column names, or empty list if no columns specified.
    """
    return [col.strip() for col in columns_arg.split(",")] if columns_arg else []


def create_rewrites_from_aliases(alias_names: Sequence[str]) -> list[tuple[str, str]]:
    """Create rewrite pairs from alias names.

    Args:
        alias_names: List of alias names to look up.

    Returns:
        List of (value, alias) pairs for rewriting.

    Raises:
        ValueError: If an alias is not found in registered aliases.
    """
    import tlc

    registered_aliases = tlc.get_registered_url_aliases()
    rewrites = []

    for name in alias_names:
        bracketed_name = f"<{name}>" if not name.startswith("<") else name
        if bracketed_name not in registered_aliases:
            raise ValueError(f"Alias '{name}' not found in registered aliases")
        alias_value = registered_aliases[bracketed_name]
        rewrites.append((alias_value, bracketed_name))

    return rewrites


def create_rewrites_from_paths(from_paths: Sequence[str], to_paths: Sequence[str] | None) -> list[tuple[str, str]]:
    """Create rewrite pairs from explicit path mappings.

    Args:
        from_paths: Source paths to replace.
        to_paths: Target paths to replace with.

    Returns:
        List of (from, to) pairs for rewriting.

    Raises:
        ValueError: If to_paths is missing or length doesn't match from_paths.
    """
    if not to_paths:
        raise ValueError("--to PATH is required when using --from")
    if len(from_paths) != len(to_paths):
        raise ValueError("Number of --from and --to arguments must match")
    return list(zip(from_paths, to_paths))


def handle_list_command(input_url: Url, columns: list[str]) -> None:
    """Handle the list command.

    Args:
        input_url: URL of the input object to process.
        columns: List of columns to process.
    """
    obj = get_input_object(input_url)
    list_aliases([input_url], obj, columns)


def handle_replace_command(
    input_url: Url,
    columns: list[str],
    rewrites: list[tuple[str, str]],
    process_parents: bool,
) -> None:
    """Handle the replace command.

    Args:
        input_url: URL of the input object to process.
        columns: List of columns to process.
        rewrites: List of (from, to) pairs for rewriting.
        process_parents: Whether to process parent tables.
    """
    obj = get_input_object(input_url)
    replace_aliases([input_url], obj, columns, rewrites, process_parents=process_parents)


@register_tool(name="alias", experimental=True, description="List, rewrite, and create URL aliases in 3LC objects")
def main(tool_args: list[str] | None = None, prog: str | None = None) -> None:
    """Main function to process aliases in 3LC objects.

    Args:
        tool_args: List of arguments. If None, will parse from command line.
        prog: Program name. If None, will use the tool name.
    """
    parser = create_argument_parser(prog)
    args = parser.parse_args(tool_args)

    # Setup logging based on verbosity flags
    setup_logging(verbosity=args.verbose, quiet=args.quiet)
    logger.debug("Debug logging enabled")

    try:
        # Parse common arguments
        columns = parse_columns(args.columns)
        input_url = Url(args.input_path)

        if args.command == "list":
            handle_list_command(input_url, columns)
        elif args.command == "replace":
            # Create rewrites based on command mode
            if args.apply:
                alias_names = [name.strip() for name in args.apply.split(",")]
                rewrites = create_rewrites_from_aliases(alias_names)
            else:
                rewrites = create_rewrites_from_paths(args.from_paths, args.to_paths)

            handle_replace_command(input_url, columns, rewrites, not args.no_process_parents)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
