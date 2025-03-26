from __future__ import annotations

import argparse
import logging

from tlc.core import Url

from tlc_tools.cli import register_tool

from .common import get_input_object, setup_logging
from .list_aliases import list_aliases
from .replace_aliases import replace_aliases

logger = logging.getLogger(__name__)


@register_tool(experimental=True, description="List, rewrite, and create URL aliases in 3LC objects")
def main(tool_args: list[str] | None = None, prog: str | None = None) -> None:
    """Main function to process aliases in 3LC objects.

    Args:
        tool_args: List of arguments. If None, will parse from command line.
        prog: Program name. If None, will use the tool name.
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

    args = parser.parse_args(tool_args)

    # Setup logging based on verbosity flags
    setup_logging(verbosity=args.verbose, quiet=args.quiet)
    logger.debug("Debug logging enabled")

    # Parse columns if specified (shared between commands)
    columns = [col.strip() for col in args.columns.split(",")] if args.columns else []

    try:
        if args.command == "replace":
            # Validate arguments first
            rewrites = []
            if args.apply:
                # Parse comma-separated aliases
                alias_names = [name.strip() for name in args.apply.split(",")]
                import tlc

                registered_aliases = tlc.get_registered_url_aliases()
                for alias_name in alias_names:
                    bracketed_name = f"<{alias_name}>" if not alias_name.startswith("<") else alias_name
                    if bracketed_name not in registered_aliases:
                        raise ValueError(f"Alias '{alias_name}' not found in registered aliases")
                    alias_value = registered_aliases[bracketed_name]
                    rewrites.append((alias_value, bracketed_name))
            elif args.from_paths:
                if not args.to_paths:
                    raise ValueError("--to PATH is required when using --from")
                if len(args.from_paths) != len(args.to_paths):
                    raise ValueError("Number of --from and --to arguments must match")
                rewrites = list(zip(args.from_paths, args.to_paths))

        # Only load input object after argument validation
        input_url = Url(args.input_path)
        try:
            obj = get_input_object(input_url)
        except Exception as e:
            logger.error(f"Failed to load input object: {e}")
            raise

        if args.command == "list":
            list_aliases([input_url], obj, columns)
        elif args.command == "replace":
            replace_aliases(
                [input_url],
                obj,
                columns,
                rewrites,
                process_parents=not args.no_process_parents,
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
