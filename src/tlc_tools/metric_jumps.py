"""Compute metric jumps of metrics across time steps."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from tlc.client.reduce.reduce import _unique_datasets
from tlc.core.builtins.constants.column_names import EXAMPLE_ID, RUN_STATUS, RUN_STATUS_COMPLETED
from tlc.core.objects.mutable_objects.run import Run
from tlc.core.objects.table import Table
from tlc.core.schema import Float32Value, Schema

logger = logging.getLogger(__name__)


@dataclass
class MetricJumpsResult:
    """Result of metric jumps computation.

    The structure supports multiple metrics being computed in the same call.
    While we use "epoch" in variable names for clarity, this actually represents
    any temporal column (e.g., step, iteration, etc.).
    """

    # Maps metric name to its jumps array
    metric_jumps: dict[str, np.ndarray]  # shape: (n_examples, n_epochs)
    # Maps example_id to its index in the jumps array
    example_id_to_idx: dict[int, int]
    # Maps epoch to its index in the jumps array
    epoch_to_idx: dict[int, int]
    # List of example IDs in order
    example_ids: list[int]
    # List of epochs in order
    epochs: list[int]
    # The actual temporal column name used (e.g., "epoch", "step", etc.)
    temporal_column_name: str


def compute_metric_jumps_on_run(
    run: Run,
    metric_column_names: str | list[str],
    temporal_column_name: str = "epoch",
    distance_fn: Callable[[Any, Any], float] | Literal["euclidean", "cosine", "l1", "l2"] = "euclidean",
) -> None:
    """Compute the jumps of metrics across time in their original space.

    Args:
        metric_column_names: Name of the columns containing the metric values
        temporal_column_name: Name of the column containing the temporal information
        distance_fn: Either a custom distance function or a predefined metric name
    """

    # Convert single metric name to list for consistent handling
    if isinstance(metric_column_names, str):
        metric_column_names = [metric_column_names]

    # Compute jumps
    results = compute_metric_jumps(
        run.metrics_tables,
        metric_column_names,
        temporal_column_name,
        distance_fn,
    )

    if not results:
        raise ValueError("No metric jumps computed")

    urls = []
    for foreign_table_url, result in results.items():
        # Create a table with the jumps for each example and epoch
        data: dict[str, list[Any]] = {
            "example_id": [],
            temporal_column_name: [],
        }
        # Add columns for each metric's jumps
        for metric_name in metric_column_names:
            data[f"{metric_name}_jump"] = []

        # Add jumps for each example and epoch
        for example_id in result.example_ids:
            example_idx = result.example_id_to_idx[example_id]
            for epoch in result.epochs:
                epoch_idx = result.epoch_to_idx[epoch]
                # Add jumps for each metric
                for metric_name in metric_column_names:
                    jump = result.metric_jumps[metric_name][example_idx, epoch_idx]
                    data[f"{metric_name}_jump"].append(jump)
                data[EXAMPLE_ID].append(example_id)
                data[temporal_column_name].append(epoch)

        # Create schemas for each metric's jumps
        column_schemas = {
            f"{metric_name}_jump": Schema(
                f"{metric_name}_jump",
                value=Float32Value(),
                description=f"Jump in {metric_name} value from previous {temporal_column_name}",
            )
            for metric_name in metric_column_names
        }

        # Write the results to a new table
        metric_infos = run.add_metrics(
            data,
            column_schemas=column_schemas,
            foreign_table_url=foreign_table_url,
        )
        urls.append(metric_infos[0]["url"])

    logger.info(
        f"Metric jumps of {', '.join(metric_column_names)} over {temporal_column_name} computed for {len(urls)} streams"
    )
    run.update_attribute(RUN_STATUS, RUN_STATUS_COMPLETED)


def compute_metric_jumps(
    metrics_tables: list[Table],
    metric_column_names: str | Sequence[str],
    temporal_column_name: str = "epoch",
    distance_fn: Callable[[Any, Any], float] | Literal["euclidean", "cosine", "l1", "l2"] = "euclidean",
) -> dict[str, MetricJumpsResult]:
    """Compute metric jumps for metrics across time steps.

    :param metrics_tables: List of tables containing the metrics data
    :param metric_column_names: Name or list of names of columns containing the metric values
    :param temporal_column_name: Name of the column containing the temporal information
    :param distance_fn: Either a custom distance function or a predefined metric name

    :returns: A dictionary mapping foreign table URLs to MetricJumpsResult objects containing
        the computed jumps and necessary mappings for example IDs and epochs.
    """
    if isinstance(metric_column_names, str):
        metric_column_names = [metric_column_names]

    # Get the distance function
    dist_fn = _get_distance_function(distance_fn)

    results: dict[str, MetricJumpsResult] = {}

    for foreign_table_url, tables in _unique_datasets(metrics_tables):
        # Filter tables to only those containing required columns
        tables = [
            table
            for table in tables
            if all(metric in table.columns for metric in metric_column_names) and temporal_column_name in table.columns
        ]
        if len(tables) < 2:
            logger.warning(
                f"Skipping stream for table '{foreign_table_url}' because it has less "
                f"than 2 tables with required columns"
            )
            continue

        # Get columns from first table for validation
        first_table = tables[0]
        reference_example_ids = first_table.get_column(EXAMPLE_ID)
        if len(pc.unique(reference_example_ids)) != len(reference_example_ids):  # type: ignore
            msg = f"Table {first_table.url} has repeated example IDs"
            raise ValueError(msg)

        # Validate that all tables have the same example IDs
        for table in tables[1:]:
            example_ids = table.get_column(EXAMPLE_ID)
            if not np.array_equal(example_ids, reference_example_ids):
                msg = f"Table {table.url} has different example IDs than the reference table {first_table.url}"
                raise ValueError(msg)

        # Validate temporal column is constant within each table and collect unique epochs
        epochs = set()
        valid_tables = []
        for table in tables:
            temporal_values = table.get_column(temporal_column_name)
            first_value = temporal_values[0]
            if not all(value == first_value for value in temporal_values):
                msg = f"Table {table.url} has non-constant temporal values in column {temporal_column_name}"
                raise ValueError(msg)
            epoch = first_value.as_py()
            if epoch in epochs:
                logger.warning(f"Multiple tables found for epoch {epoch} in stream {foreign_table_url}")
                continue
            epochs.add(epoch)
            valid_tables.append(table)

        if len(valid_tables) < 2:
            logger.warning(
                f"Skipping stream '{foreign_table_url.name}' because it has less than 2 valid tables after filtering"
            )
            continue

        # Sort tables by temporal column after filtering out duplicates
        valid_tables = sort_tables_by_constant_column(valid_tables, temporal_column_name)

        # Create mappings for example IDs and epochs
        example_ids = [int(x) for x in cast(pa.Array, reference_example_ids).to_numpy()]
        example_id_to_idx = {example_id: idx for idx, example_id in enumerate(example_ids)}

        # Sort epochs to ensure consistent ordering
        epochs_list = sorted(epochs)
        epoch_to_idx = {int(epoch): idx for idx, epoch in enumerate(epochs_list)}

        # Initialize arrays for each metric
        n_examples = len(example_ids)
        n_epochs = len(epochs_list)
        metric_jumps = {
            metric_name: np.zeros((n_examples, n_epochs), dtype=np.float32) for metric_name in metric_column_names
        }

        # Compute jumps for each example and metric
        for i in range(len(valid_tables) - 1):
            current_table = valid_tables[i]
            next_table = valid_tables[i + 1]
            current_epoch = current_table.get_column(temporal_column_name)[0].as_py()
            next_epoch = next_table.get_column(temporal_column_name)[0].as_py()
            assert current_epoch < next_epoch, f"Tables are not sorted chronologically: {current_epoch} >= {next_epoch}"

            for metric_name in metric_column_names:
                current_metrics = current_table.get_column(metric_name)
                next_metrics = next_table.get_column(metric_name)

                # Compute jumps between consecutive points
                for example_idx, example_id in enumerate(reference_example_ids):
                    example_id = example_id.as_py()
                    current_value = current_metrics[example_idx].as_py()
                    next_value = next_metrics[example_idx].as_py()
                    jump = dist_fn(current_value, next_value)
                    metric_jumps[metric_name][example_id_to_idx[example_id], epoch_to_idx[next_epoch]] = jump

        results[str(foreign_table_url)] = MetricJumpsResult(
            metric_jumps=metric_jumps,
            example_id_to_idx=example_id_to_idx,
            epoch_to_idx=epoch_to_idx,
            example_ids=example_ids,
            epochs=epochs_list,
            temporal_column_name=temporal_column_name,
        )

    return results


def _get_distance_function(
    distance_fn: Callable[[Any, Any], float] | Literal["euclidean", "cosine", "l1", "l2"],
) -> Callable[[Any, Any], float]:
    """Get the distance function to use.

    Args:
        distance_fn: Either a custom distance function or a predefined metric name

    Returns:
        A callable that computes the distance between two points

    Raises:
        ValueError: If the distance function name is not recognized
    """
    if callable(distance_fn):
        return distance_fn

    if distance_fn == "euclidean":
        return _euclidean_distance
    elif distance_fn == "cosine":
        return _cosine_distance
    elif distance_fn == "l1":
        return _l1_distance
    elif distance_fn == "l2":
        return _l2_distance
    else:
        msg = f"Unknown distance function: {distance_fn}"  # type: ignore[unreachable]
        raise ValueError(msg)


def _euclidean_distance(a: Any, b: Any) -> float:
    """Compute Euclidean distance between two points.

    Args:
        a: First point
        b: Second point

    Returns:
        Euclidean distance between a and b
    """
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _cosine_distance(a: Any, b: Any) -> float:
    """Compute cosine distance between two points.

    Args:
        a: First point
        b: Second point

    Returns:
        Cosine distance between a and b
    """
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    a = np.array(a)
    b = np.array(b)
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _l1_distance(a: Any, b: Any) -> float:
    """Compute L1 (Manhattan) distance between two points.

    Args:
        a: First point
        b: Second point

    Returns:
        L1 distance between a and b
    """
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    return float(np.sum(np.abs(np.array(a) - np.array(b))))


def _l2_distance(a: Any, b: Any) -> float:
    """Compute L2 (Euclidean) distance between two points.

    Args:
        a: First point
        b: Second point

    Returns:
        L2 distance between a and b
    """
    return _euclidean_distance(a, b)


def sort_tables_by_constant_column(tables: list[Table], column_name: str, reverse: bool = False) -> list[Table]:
    """Sort a list of tables by a constant column.

    :param tables: A list of tables to sort by a constant column.
    :param column_name: The name of the column to sort by.
    :param reverse: Whether to sort in reverse order.
    """
    return sorted(tables, key=lambda table: table.table_rows[0][column_name], reverse=reverse)
