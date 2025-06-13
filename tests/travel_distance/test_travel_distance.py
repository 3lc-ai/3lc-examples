"""Tests for travel distance computation."""

from typing import Any, cast

import numpy as np
import pytest

from tlc_tools.travel_distance import (
    TravelDistanceResult,
    _cosine_distance,
    _euclidean_distance,
    _l1_distance,
    _l2_distance,
    compute_travel_distances,
)
from tlc.core.builtins.schemas.schemas import ForeignTableIdSchema
from tlc.core.objects.table import Table


def create_test_table(
    epoch: int,
    values: list[Any],
    example_ids: list[int],
    foreign_table_url: str = "",
) -> Table:
    """Create a test table with given epoch, values and example IDs."""

    data = {
        "epoch": [epoch] * len(values),
        "metric": values,
        "example_id": example_ids,
    }
    if foreign_table_url:
        data["input_table_id"] = [0] * len(values)

    schemas = {"input_table_id": ForeignTableIdSchema(foreign_table_url)} if foreign_table_url else {}

    return cast(
        Table,
        Table.from_dict(
            data=data,
            table_name="test_table",
            structure=schemas,
            if_exists="rename",
        ),
    )


def test_distance_functions() -> None:
    """Test basic distance functions."""
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]

    # Test Euclidean distance
    assert _euclidean_distance(a, b) == pytest.approx(5.196152422706632)

    # Test cosine distance
    assert _cosine_distance(a, b) == pytest.approx(0.025368153802923787)

    # Test L1 distance
    assert _l1_distance(a, b) == pytest.approx(9.0)

    # Test L2 distance (should be same as Euclidean)
    assert _l2_distance(a, b) == pytest.approx(5.196152422706632)


def test_compute_travel_distances_basic() -> None:
    """Test basic travel distance computation."""
    # Create test tables with increasing values
    tables = [
        create_test_table(1, [1.0, 2.0], [1, 2], "../train"),
        create_test_table(2, [2.0, 3.0], [1, 2], "../train"),
        create_test_table(3, [3.0, 4.0], [1, 2], "../train"),
    ]

    results = compute_travel_distances(tables, "metric", "epoch", "euclidean")

    # Check that we got results for each example
    assert len(results) == 1  # One stream
    result = next(iter(results.values()))
    assert isinstance(result, TravelDistanceResult)
    assert result.temporal_column_name == "epoch"
    assert result.example_ids == [1, 2]
    assert result.epochs == [1, 2, 3]

    # For each example, we should have jumps for each epoch
    # Example 1: epoch 1 -> 0.0, epoch 2 -> 1.0, epoch 3 -> 1.0
    # Example 2: epoch 1 -> 0.0, epoch 2 -> 1.0, epoch 3 -> 1.0
    metric_jumps = result.metric_jumps["metric"]
    assert metric_jumps[0, 0] == 0.0  # Example 1, epoch 1
    assert metric_jumps[0, 1] == 1.0  # Example 1, epoch 2
    assert metric_jumps[0, 2] == 1.0  # Example 1, epoch 3
    assert metric_jumps[1, 0] == 0.0  # Example 2, epoch 1
    assert metric_jumps[1, 1] == 1.0  # Example 2, epoch 2
    assert metric_jumps[1, 2] == 1.0  # Example 2, epoch 3

    # Check the mappings
    assert result.example_id_to_idx == {1: 0, 2: 1}  # Maps example IDs to array indices
    assert result.epoch_to_idx == {1: 0, 2: 1, 3: 2}  # Maps epochs to array indices


def test_compute_travel_distances_skips() -> None:
    """Test that travel distance computation skips invalid tables."""
    # Create test tables with some invalid ones
    tables = [
        create_test_table(1, [1.0, 2.0], [1, 2], "../train"),
        create_test_table(1, [2.0, 3.0], [1, 2], "../train"),  # Duplicate epoch
        create_test_table(2, [3.0, 4.0], [1, 2], "../train"),
    ]

    results = compute_travel_distances(tables, "metric", "epoch", "euclidean")

    # Should only compute jumps between epochs 1 and 2
    result = next(iter(results.values()))
    assert isinstance(result, TravelDistanceResult)
    assert result.example_ids == [1, 2]
    assert result.epochs == [1, 2]

    metric_jumps = result.metric_jumps["metric"]
    assert metric_jumps[0, 0] == 0.0  # Example 1, epoch 1
    assert metric_jumps[0, 1] == 2.0  # Example 1, epoch 2 (3-1)
    assert metric_jumps[1, 0] == 0.0  # Example 2, epoch 1
    assert metric_jumps[1, 1] == 2.0  # Example 2, epoch 2 (4-2)


def test_compute_travel_distances_missing_columns() -> None:
    """Test that travel distance computation handles missing columns."""
    # Create test tables with missing columns
    tables = [
        create_test_table(1, [1.0, 2.0], [1, 2], "../train"),
        Table.from_dict({}, table_name="empty_table", if_exists="rename"),
        create_test_table(2, [3.0, 4.0], [1, 2], "../train"),
    ]

    results = compute_travel_distances(tables, "metric", "epoch", "euclidean")

    # Should skip the empty table and only compute jumps between epochs 1 and 2
    result = next(iter(results.values()))
    assert isinstance(result, TravelDistanceResult)
    assert result.example_ids == [1, 2]
    assert result.epochs == [1, 2]

    metric_jumps = result.metric_jumps["metric"]
    assert metric_jumps[0, 0] == 0.0  # Example 1, epoch 1
    assert metric_jumps[0, 1] == 2.0  # Example 1, epoch 2 (3-1)
    assert metric_jumps[1, 0] == 0.0  # Example 2, epoch 1
    assert metric_jumps[1, 1] == 2.0  # Example 2, epoch 2 (4-2)


def test_compute_travel_distances_different_example_ids() -> None:
    """Test that travel distance computation handles different example IDs."""
    # Create test tables with different example IDs
    tables = [
        create_test_table(1, [1.0, 2.0], [1, 2], "../train"),
        create_test_table(2, [2.0, 3.0], [1, 3], "../train"),  # Different example IDs
    ]

    with pytest.raises(ValueError, match="different example IDs"):
        compute_travel_distances(tables, "metric", "epoch", "euclidean")


def test_compute_travel_distances_non_constant_temporal() -> None:
    """Test that travel distance computation handles non-constant temporal values."""
    # Create a table with non-constant temporal values
    table = cast(
        Table,
        Table.from_dict(
            {
                "epoch": [1, 2],  # Non-constant
                "metric": [1.0, 2.0],
                "example_id": [1, 2],
                "input_table_id": [0, 0],
            },
            table_name="non_constant_temporal",
            structure={"input_table_id": ForeignTableIdSchema("../train")},
        ),
    )
    tables = [table, create_test_table(1, [1.0, 2.0], [1, 2], "../train")]  # Need at least 2 tables

    with pytest.raises(ValueError, match="non-constant temporal values"):
        compute_travel_distances(tables, "metric", "epoch", "euclidean")


def test_compute_travel_distances_missing_foreign_table_url() -> None:
    """Test that travel distance computation handles missing foreign table URLs."""
    # Create test tables without foreign table URLs
    tables = [
        create_test_table(1, [1.0, 2.0], [1, 2]),  # No foreign_table_url
        create_test_table(2, [2.0, 3.0], [1, 2]),  # No foreign_table_url
    ]

    results = compute_travel_distances(tables, "metric", "epoch", "euclidean")
    assert len(results) == 0  # No results because no foreign table URLs


def test_compute_travel_distances_repeated_example_ids() -> None:
    """Test that travel distance computation handles repeated example IDs."""
    # Create test tables with repeated example IDs
    tables = [
        create_test_table(1, [1.0, 2.0], [1, 1], "../foreign_table"),  # Repeated example ID
        create_test_table(2, [2.0, 3.0], [1, 1], "../foreign_table"),  # Repeated example ID
    ]

    with pytest.raises(ValueError, match="has repeated example IDs"):
        compute_travel_distances(tables, "metric", "epoch", "euclidean")
