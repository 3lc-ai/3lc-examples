from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tlc
from tlc.core import Table, Url

from tlc_tools.experimental.alias_tool.alias import (
    handle_object,
    handle_pa_table,
    handle_table,
    parse_alias_pair_string,
)


@pytest.fixture
def sample_table(tmp_path: Path) -> Table:
    """Create a sample table with some test data."""
    # Create a test parquet file with some paths
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = {
        "image_path": [
            "/data/project/images/001.jpg",
            "/data/project/images/002.jpg",
            "/data/project/images/003.jpg",
        ],
        "mask_path": [
            "/data/project/masks/001.png",
            "/data/project/masks/002.png",
            "/data/project/masks/003.png",
        ],
    }
    table = pa.Table.from_pydict(data)
    pq.write_table(table, tmp_path / "test.parquet")

    # Create a Table object
    return Table.from_url(Url(str(tmp_path / "test.parquet")))


@pytest.fixture
def sample_run(tmp_path: Path) -> Any:
    """Create a sample run with some test data."""
    # Create a test run with input tables
    run = tlc.Run.create(
        name="test_run",
        project_name="test_project",
        description="Test run for alias tool",
    )
    return run


def test_parse_alias_pair_string():
    """Test parsing alias pair strings."""
    # Test valid cases
    assert parse_alias_pair_string("old::new", "::") == ("old", "new")
    assert parse_alias_pair_string("<DATA_PATH>::/data/project", "::") == ("<DATA_PATH>", "/data/project")

    # Test invalid cases
    with pytest.raises(ValueError):
        parse_alias_pair_string("invalid", "::")


def test_handle_pa_table_basic(sample_table, tmp_path: Path):
    """Test basic parquet table handling."""
    # Test listing aliases
    handle_pa_table(
        [sample_table.url],
        sample_table.to_pa_table(),
        "",
        [],
        False,
        None,
    )

    # Test creating alias
    output_path = tmp_path / "output.parquet"
    handle_pa_table(
        [sample_table.url],
        sample_table.to_pa_table(),
        "",
        [("/data/project", "<DATA_PATH>")],
        False,
        output_path,
        create_alias=True,
        persist_config=False,
    )

    # Verify output
    output_table = sample_table.to_pa_table()
    assert "<DATA_PATH>/images/001.jpg" in output_table.column("image_path").to_pylist()


def test_handle_table_basic(sample_table, tmp_path: Path):
    """Test basic table handling."""
    # Test listing aliases
    handle_table(
        [sample_table.url],
        sample_table,
        "",
        [],
        False,
        None,
    )

    # Test creating alias
    output_path = tmp_path / "output.parquet"
    handle_table(
        [sample_table.url],
        sample_table,
        "",
        [("/data/project", "<DATA_PATH>")],
        False,
        output_path,
        create_alias=True,
        persist_config=False,
    )

    # Verify output
    output_table = Table.from_url(Url(str(output_path)))
    assert "<DATA_PATH>/images/001.jpg" in output_table.to_pa_table().column("image_path").to_pylist()


def test_handle_object_basic(sample_table, tmp_path: Path):
    """Test basic object handling."""
    # Test listing aliases
    handle_object(
        [sample_table.url],
        sample_table,
        "",
        [],
        False,
        None,
    )

    # Test creating alias
    output_path = tmp_path / "output.parquet"
    handle_object(
        [sample_table.url],
        sample_table,
        "",
        [("/data/project", "<DATA_PATH>")],
        False,
        output_path,
        create_alias=True,
        persist_config=False,
    )

    # Verify output
    output_table = Table.from_url(Url(str(output_path)))
    assert "<DATA_PATH>/images/001.jpg" in output_table.to_pa_table().column("image_path").to_pylist()


def test_column_specific_handling(sample_table, tmp_path: Path):
    """Test handling specific columns."""
    output_path = tmp_path / "output.parquet"
    handle_object(
        [sample_table.url],
        sample_table,
        "image_path",  # Only process image_path column
        [("/data/project", "<DATA_PATH>")],
        False,
        output_path,
        create_alias=True,
        persist_config=False,
    )

    # Verify output
    output_table = Table.from_url(Url(str(output_path)))
    assert "<DATA_PATH>/images/001.jpg" in output_table.to_pa_table().column("image_path").to_pylist()
    assert "/data/project/masks/001.png" in output_table.to_pa_table().column("mask_path").to_pylist()


def test_inplace_modification(sample_table):
    """Test inplace modification of tables."""
    handle_object(
        [sample_table.url],
        sample_table,
        "",
        [("/data/project", "<DATA_PATH>")],
        True,
        sample_table.url,
        create_alias=True,
        persist_config=False,
    )

    # Verify output
    assert "<DATA_PATH>/images/001.jpg" in sample_table.to_pa_table().column("image_path").to_pylist()


def test_persist_config(sample_table, tmp_path: Path):
    """Test persisting aliases to config."""
    # Register a test alias
    tlc.register_project_url_alias("TEST_DATA", "/data/project", project="test_project")

    # Test applying existing alias
    output_path = tmp_path / "output.parquet"
    handle_object(
        [sample_table.url],
        sample_table,
        "",
        [("TEST_DATA", "")],  # Empty string as placeholder
        False,
        output_path,
        apply_alias=True,
        persist_config=False,
    )

    # Verify output
    output_table = Table.from_url(Url(str(output_path)))
    assert "<TEST_DATA>/images/001.jpg" in output_table.to_pa_table().column("image_path").to_pylist()
