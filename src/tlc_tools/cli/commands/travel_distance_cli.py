"""CLI command for computing travel distances."""

from __future__ import annotations

import argparse
import logging

import tlc
from tlc_tools.travel_distance import compute_metric_travel_distances
from tlc_tools.cli import register_tool

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the travel distance tool.

    Args:
        prog: Optional program name to use in help text.

    Returns:
        Configured argument parser with options.
    """
    parser = argparse.ArgumentParser(
        prog="travel-distance",
        description="""Compute travel distances per example per metric across the temporal column and add the results as new metrics tables on the run.

Examples:
    # Compute travel distances for multiple metrics
    3lc travel-distance <run_url> --metric-column-names "loss,accuracy" --temporal-column-name epoch

    # Use a different distance metric
    3lc travel-distance <run_url> --metric-column-names "loss" --metric cosine

    # Use default settings (euclidean distance, epoch as temporal column)
    3lc travel-distance <run_url> --metric-column-names "loss" """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("run_url", help="URL of the run to compute travel distances for")
    parser.add_argument(
        "--metric-column-names",
        "-m",
        required=True,
        help="Comma-separated list of metric column names to compute travel distances for",
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


@register_tool(name="travel_distance", description="Compute travel distances of metrics across time steps")
def main(tool_args: list[str] | None = None, prog: str | None = None) -> None:
    """Main function to compute travel distances.

    Args:
        tool_args: List of arguments. If None, will parse from command line.
        prog: Program name. If None, will use the tool name.
    """
    parser = create_argument_parser()
    args = parser.parse_args(tool_args)

    try:
        # Parse metric column names
        metric_names = parse_metric_columns(args.metric_column_names)

        # Get the run
        run = tlc.Run.from_url(args.run_url)

        # Compute travel distances
        compute_metric_travel_distances(
            run=run,
            metric_column_names=metric_names,
            temporal_column_name=args.temporal_column_name,
            distance_fn=args.metric,
        )

        logger.info(f"Successfully computed travel distances for metrics: {', '.join(metric_names)}")

    except Exception as e:
        logger.error(f"Error computing travel distances: {str(e)}")
        raise


if __name__ == "__main__":
    main()
