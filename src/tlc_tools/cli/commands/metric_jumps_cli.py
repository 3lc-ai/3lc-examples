"""CLI command for computing metric jumps."""

from __future__ import annotations

import argparse
import logging

import tlc

from tlc_tools.cli import register_tool
from tlc_tools.cli.logging import setup_logging
from tlc_tools.metric_jumps import compute_metric_jumps, compute_metric_jumps_on_run

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the metric jumps tool.

    Returns:
        Configured argument parser with options.
    """
    parser = argparse.ArgumentParser(
        prog="metric-jumps",
        description="""Compute metric jumps per example per metric across the temporal column and add the results as new metrics tables on the run.

Examples:
    # Compute metric jumps for multiple metrics
    3lc metric-jumps <run_url> --metric-column-names "loss,accuracy" --temporal-column-name epoch

    # Use a different distance metric
    3lc metric-jumps <run_url> --metric-column-names "loss" --metric cosine

    # Use default settings (euclidean distance, epoch as temporal column)
    3lc metric-jumps <run_url> --metric-column-names "loss" """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("run_url", help="URL of the run to compute metric jumps for")
    parser.add_argument(
        "--metric-column-names",
        "-m",
        required=True,
        help="Comma-separated list of metric column names to compute metric jumps for",
    )

    # Optional arguments
    parser.add_argument(
        "--temporal-column-name",
        "-t",
        default="epoch",
        help="Name of the temporal column (default: epoch)",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine", "l1", "l2"],
        default="euclidean",
        help="Distance metric to use (default: euclidean)",
    )

    # Verbosity control
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

    return parser


def parse_metric_columns(columns_arg: str) -> list[str]:
    """Parse the metric columns argument into a list of column names.

    Args:
        columns_arg: Comma-separated string of metric column names.

    Returns:
        List of metric column names.

    Raises:
        ValueError: If no metric columns are provided.
    """
    columns = [col.strip() for col in columns_arg.split(",")]
    if not columns:
        raise ValueError("At least one metric column name must be provided")
    return columns


@register_tool(name="metric_jumps", description="Compute metric jumps of metrics across time steps")
def main(tool_args: list[str] | None = None, prog: str | None = None) -> None:
    """Main function to compute metric jumps.

    Args:
        tool_args: List of arguments. If None, will parse from command line.
        prog: Program name. If None, will use the tool name.
    """
    parser = create_argument_parser()
    args = parser.parse_args(tool_args)

    # Setup logging based on verbosity flags
    setup_logging(verbosity=args.verbose, quiet=args.quiet)
    logger.debug("Debug logging enabled")
    try:
        # Parse metric column names
        metric_names = parse_metric_columns(args.metric_column_names)

        # Get the run
        run = tlc.Run.from_url(args.run_url)

        # Compute metric jumps
        compute_metric_jumps_on_run(
            run=run,
            metric_column_names=metric_names,
            temporal_column_name=args.temporal_column_name,
            distance_fn=args.metric,
        )

        logger.info(f"Successfully computed metric jumps for metrics: {', '.join(metric_names)}")

    except Exception as e:
        logger.error(f"Error computing metric jumps: {str(e)}")
        raise


if __name__ == "__main__":
    main()
