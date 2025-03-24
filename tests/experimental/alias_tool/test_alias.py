from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tlc.core import Table, Url

from tlc_tools.experimental.alias_tool.alias import (
    apply_rewrites,
    find_aliases,
    handle_pa_table,
)


@pytest.fixture
def sample_table(tmp_path: Path) -> Generator[Table, None, None]:
    """Create a sample table with some test data.

    Note: This creates both a parquet file and a tlc.Table object.
    The tlc.Table object maintains its own reference to the underlying pa.Table
    via the _pa_table property. When the parquet file is modified, the tlc.Table
    object needs to be recreated to see the changes.
    """
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
    # Create the underlying pa.Table
    pa_table = pa.Table.from_pydict(data)
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa_table, parquet_path)

    # Create a tlc.Table object that wraps the parquet file
    table = Table.from_parquet(Url(str(parquet_path)), if_exists="overwrite")
    table.ensure_fully_defined()
    yield table

    # Delete the parquet file
    parquet_path.unlink()

    # Delete the tlc.Table object
    table.url.delete()


@pytest.fixture
def tmp_parquet(tmp_path):
    """Create a sample parquet file with string columns containing paths."""
    # Create a simple table with paths
    data = {
        "image_path": pa.array(["/data/images/001.jpg", "/data/images/002.jpg", "/other/path/003.jpg"]),
        "mask_path": pa.array(["/data/masks/001.png", "/data/masks/002.png", "/data/masks/003.png"]),
        "label": pa.array([1, 2, 3]),  # Non-string column should be ignored
    }
    table = pa.table(data)

    # Write to temporary file
    path = tmp_path / "test.parquet"
    pq.write_table(table, path)
    return str(path)


@pytest.fixture
def array_types():
    """Fixture providing different array types with the same test data."""
    # Basic array
    basic = pa.array(
        [
            "<DATA_PATH>/images/001.jpg",  # Valid alias at start
            "<MODEL_PATH>/weights.pt",  # Valid alias at start
            "/path/to/<CACHE_PATH>/file.txt",  # Invalid - alias in middle
            "/other/path/003.jpg",  # No alias
        ]
    )

    # Chunked array (split into two chunks)
    chunked = pa.chunked_array(
        [
            basic[:2],  # First chunk with valid aliases
            basic[2:],  # Second chunk with invalid/no aliases
        ]
    )

    # Struct array
    struct_type = pa.struct([("id", pa.int32()), ("path", pa.string())])
    struct_data = [
        {"id": 1, "path": "<DATA_PATH>/images/001.jpg"},
        {"id": 2, "path": "<MODEL_PATH>/weights.pt"},
        {"id": 3, "path": "/path/to/<CACHE_PATH>/file.txt"},
        {"id": 4, "path": "/other/path/003.jpg"},
    ]
    struct = pa.array(struct_data, type=struct_type)

    return [
        ("basic", basic, ["col"]),
        ("chunked", chunked, ["col"]),
        ("struct", struct, ["metadata"]),
    ]


def test_find_aliases(array_types):
    """Test that find_aliases works correctly for different array types."""
    for type_name, array, col_names in array_types:
        # For struct arrays, we expect the path field to be appended to col_names
        expected_col_path = ".".join(col_names + (["path"] if type_name == "struct" else []))

        aliases = find_aliases(col_names, array)

        # Should only find aliases that appear at the start of paths
        assert len(aliases) == 2, f"Failed for {type_name} array"
        assert (expected_col_path, "<DATA_PATH>") in aliases
        assert (expected_col_path, "<MODEL_PATH>") in aliases
        # <CACHE_PATH> should not be detected since it's in the middle of the path
        assert not any("<CACHE_PATH>" in alias[1] for alias in aliases)


def test_apply_rewrites(array_types):
    """Test that apply_rewrites works correctly for different array types."""
    rewrites = [("/path/to", "<CACHE_PATH>")]

    for type_name, array, col_names in array_types:
        result = apply_rewrites(col_names, array, rewrites)

        # Verify type is preserved
        assert isinstance(result, type(array))

        # For struct arrays, we need to extract the path field for comparison
        paths = result.field("path") if type_name == "struct" else result

        # Convert to list for easier comparison
        if isinstance(paths, pa.ChunkedArray):
            paths = paths.combine_chunks()

        # Verify the rewrite was applied correctly
        expected = [
            "<DATA_PATH>/images/001.jpg",  # Unchanged
            "<MODEL_PATH>/weights.pt",  # Unchanged
            "<CACHE_PATH>/file.txt",  # Rewritten
            "/other/path/003.jpg",  # Unchanged
        ]
        assert paths.to_pylist() == expected, f"Failed for {type_name} array"


def test_handle_pa_table_selected_columns(tmp_parquet):
    """Test that handle_pa_table respects column selection."""
    # Read the table
    table = pq.read_table(tmp_parquet)

    # Process only image_path column
    handle_pa_table([tmp_parquet], table, ["image_path"], [("/data/images", "<DATA_PATH>")])

    # Read back and verify
    result = pq.read_table(tmp_parquet)

    # image_path should be modified
    assert "<DATA_PATH>" in result.column("image_path")[0].as_py()

    # mask_path should be unchanged
    assert "/data/masks" in result.column("mask_path")[0].as_py()

    # label should be unchanged
    assert result.column("label").equals(table.column("label"))


def test_handle_pa_table_multiple_rewrites(tmp_parquet):
    """Test that handle_pa_table can apply multiple rewrites."""
    # Read the table
    table = pq.read_table(tmp_parquet)

    # Apply multiple rewrites
    handle_pa_table(
        [tmp_parquet],
        table,
        [],  # all columns
        [("/data/images", "<DATA_PATH>"), ("/data/masks", "<MASK_PATH>")],
    )

    # Read back and verify
    result = pq.read_table(tmp_parquet)

    # Check image_path rewrites
    assert all("<DATA_PATH>" in path.as_py() for path in result.column("image_path")[:2])

    # Check mask_path rewrites
    assert all("<MASK_PATH>" in path.as_py() for path in result.column("mask_path"))


def test_handle_pa_table_invalid_column():
    """Test that handle_pa_table raises error for invalid column names."""
    # Create a simple table
    table = pa.table({"col1": [1, 2, 3]})

    # Try to process non-existent column
    with pytest.raises(ValueError, match="not found in the input table"):
        handle_pa_table(["test.parquet"], table, ["invalid_column"], [])
