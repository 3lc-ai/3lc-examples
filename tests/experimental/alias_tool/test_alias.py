from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tlc.core import Table, Url

from tlc_tools.experimental.alias_tool.alias import (
    handle_pa_table,
    list_column_aliases,
    rewrite_column_paths,
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


@pytest.fixture(
    params=[
        ("basic", ["col"], lambda x: x),
        ("chunked", ["col"], lambda x: x.combine_chunks()),
        ("struct", ["metadata"], lambda x: x.field("path")),
    ],
    ids=["basic_array", "chunked_array", "struct_array"],
)
def array_types(request):
    """Fixture providing different array types with the same test data."""
    array_type, col_names, get_paths = request.param

    if array_type == "basic":
        array = pa.array(
            [
                "<DATA_PATH>/images/001.jpg",  # Valid alias at start
                "<MODEL_PATH>/weights.pt",  # Valid alias at start
                "/path/to/<CACHE_PATH>/file.txt",  # Invalid - alias in middle
                "/other/path/003.jpg",  # No alias
            ]
        )
    elif array_type == "chunked":
        array = pa.chunked_array(
            [
                pa.array(
                    [
                        "<DATA_PATH>/images/001.jpg",
                        "<MODEL_PATH>/weights.pt",
                    ]
                ),
                pa.array(["/path/to/<CACHE_PATH>/file.txt", "/other/path/003.jpg"]),
            ]
        )
    else:  # struct
        struct_type = pa.struct([("id", pa.int32()), ("path", pa.string())])
        array = pa.array(
            [
                {"id": 1, "path": "<DATA_PATH>/images/001.jpg"},
                {"id": 2, "path": "<MODEL_PATH>/weights.pt"},
                {"id": 3, "path": "/path/to/<CACHE_PATH>/file.txt"},
                {"id": 4, "path": "/other/path/003.jpg"},
            ],
            type=struct_type,
        )

    return array_type, array, col_names, get_paths


def test_list_column_aliases(array_types):
    """Test that list_column_aliases works correctly for different array types."""
    array_type, array, col_names, _ = array_types

    # For struct arrays, we expect the path field to be appended to col_names
    expected_col_path = ".".join(col_names + (["path"] if array_type == "struct" else []))

    aliases = list_column_aliases(col_names, array)

    # Should only find aliases that appear at the start of paths
    assert len(aliases) == 2, f"Failed for {array_type} array"
    assert (expected_col_path, "<DATA_PATH>") in aliases
    assert (expected_col_path, "<MODEL_PATH>") in aliases
    # <CACHE_PATH> should not be detected since it's in the middle of the path
    assert not any("<CACHE_PATH>" in alias[1] for alias in aliases)


def test_rewrite_column_paths(array_types):
    """Test that rewrite_column_paths works correctly for different array types."""
    array_type, array, col_names, get_paths = array_types

    rewrites = [("/path/to", "<CACHE_PATH>")]
    result = rewrite_column_paths(col_names, array, rewrites)

    # Verify type is preserved
    assert isinstance(result, type(array))

    # Get the paths for comparison using the provided accessor
    paths = get_paths(result)

    # Verify the rewrite was applied correctly
    expected = [
        "<DATA_PATH>/images/001.jpg",  # Unchanged
        "<MODEL_PATH>/weights.pt",  # Unchanged
        "<CACHE_PATH>/<CACHE_PATH>/file.txt",  # Rewritten
        "/other/path/003.jpg",  # Unchanged
    ]
    assert paths.to_pylist() == expected, f"Failed for {array_type} array"


def test_handle_pa_table_selected_columns(tmp_parquet):
    """Test that handle_pa_table respects column selection."""
    # Read the table
    table = pq.read_table(tmp_parquet)

    # Process only image_path column
    handle_pa_table([Url(tmp_parquet)], table, ["image_path"], [("/data/images", "<DATA_PATH>")])

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
        [Url(tmp_parquet)],
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
        handle_pa_table([Url("test.parquet")], table, ["invalid_column"], [])
