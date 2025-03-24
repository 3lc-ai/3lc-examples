from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tlc.core import Table, Url

from tlc_tools.experimental.alias_tool.alias import (
    handle_pa_table,
    handle_parquet_column,
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


def test_handle_parquet_column_find_aliases():
    """Test that handle_parquet_column correctly identifies existing aliases at the start of paths."""
    # Create a column with some paths containing aliases
    paths = pa.array(
        [
            "<DATA_PATH>/images/001.jpg",  # Valid alias at start
            "<MODEL_PATH>/weights.pt",  # Valid alias at start
            "/data/path/to/<CACHE_PATH>/file.txt",  # Invalid - alias in middle
            "/other/path/003.jpg",  # No alias
        ]
    )

    # List mode (no rewrites)
    aliases, new_col = handle_parquet_column(["image_path"], paths, [])

    # Should only find aliases that appear at the start of paths
    assert len(aliases) == 2
    assert (".".join(["image_path"]), "<DATA_PATH>") in aliases
    assert (".".join(["image_path"]), "<MODEL_PATH>") in aliases
    # <CACHE_PATH> should not be detected since it's in the middle of the path
    assert not any("<CACHE_PATH>" in alias[1] for alias in aliases)

    # Column should be unchanged in list mode
    assert new_col.equals(paths)


def test_handle_parquet_column_find_aliases_nested():
    """Test that handle_parquet_column correctly identifies aliases in nested struct columns."""
    # Create a struct array with a string field that contains aliases
    struct_type = pa.struct([("id", pa.int32()), ("path", pa.string())])
    data = [
        {"id": 1, "path": "<DATA_PATH>/images/001.jpg"},  # Valid alias at start
        {"id": 2, "path": "<MODEL_PATH>/weights.pt"},  # Valid alias at start
        {"id": 3, "path": "/path/to/<CACHE_PATH>/file.txt"},  # Invalid - alias in middle
        {"id": 4, "path": "/other/path/003.jpg"},  # No alias
    ]
    struct_array = pa.array(data, type=struct_type)

    # List mode (no rewrites)
    aliases, new_col = handle_parquet_column(["metadata"], struct_array, [])

    # Should only find aliases that appear at the start of paths
    assert len(aliases) == 2
    assert (".".join(["metadata", "path"]), "<DATA_PATH>") in aliases
    assert (".".join(["metadata", "path"]), "<MODEL_PATH>") in aliases
    # <CACHE_PATH> should not be detected since it's in the middle of the path
    assert not any("<CACHE_PATH>" in alias[1] for alias in aliases)

    # Column should be unchanged in list mode
    assert new_col.equals(struct_array)


def test_handle_parquet_column_find_aliases_chunked():
    """Test that handle_parquet_column correctly identifies aliases in chunked arrays."""
    # Create a chunked array with paths containing aliases
    chunks = [
        pa.array(
            [
                "<DATA_PATH>/images/001.jpg",  # Valid alias at start
                "<MODEL_PATH>/weights.pt",  # Valid alias at start
            ]
        ),
        pa.array(
            [
                "/path/to/<CACHE_PATH>/file.txt",  # Invalid - alias in middle
                "/other/path/003.jpg",  # No alias
            ]
        ),
    ]
    chunked_array = pa.chunked_array(chunks)

    # List mode (no rewrites)
    aliases, new_col = handle_parquet_column(["image_path"], chunked_array, [])

    # Should only find aliases that appear at the start of paths
    assert len(aliases) == 2
    assert (".".join(["image_path"]), "<DATA_PATH>") in aliases
    assert (".".join(["image_path"]), "<MODEL_PATH>") in aliases
    # <CACHE_PATH> should not be detected since it's in the middle of the path
    assert not any("<CACHE_PATH>" in alias[1] for alias in aliases)

    # Column should be unchanged in list mode
    assert new_col.equals(chunked_array)
    # Verify it's still a chunked array with the same number of chunks
    assert isinstance(new_col, pa.ChunkedArray)
    assert new_col.num_chunks == 2
    # Verify each chunk is unchanged
    assert new_col.chunk(0).equals(chunks[0])
    assert new_col.chunk(1).equals(chunks[1])


def test_handle_parquet_column_apply_rewrite():
    """Test that handle_parquet_column correctly applies path rewrites."""
    # Create a column with some paths
    paths = pa.array(["/data/images/001.jpg", "/data/images/002.jpg", "/other/path/003.jpg"])

    # Apply a rewrite
    aliases, new_col = handle_parquet_column(["image_path"], paths, [("/data/images", "<DATA_PATH>")])

    # Check the rewritten values
    assert new_col.to_pylist() == ["<DATA_PATH>/001.jpg", "<DATA_PATH>/002.jpg", "/other/path/003.jpg"]


def test_handle_pa_table_list_mode(tmp_parquet, caplog):
    """Test that handle_pa_table correctly lists aliases in list mode."""
    # Read the table
    table = pq.read_table(tmp_parquet)

    # List mode (no rewrites)
    handle_pa_table([tmp_parquet], table, [], [])

    # Check log output for found aliases
    assert any("/data/images" in record.message for record in caplog.records)
    assert any("/data/masks" in record.message for record in caplog.records)


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


def test_handle_parquet_column_no_changes():
    """Test that handle_parquet_column preserves data when no changes are needed."""
    # Create a column with paths that won't be modified
    paths = pa.array(["/other/path/001.jpg", "/other/path/002.jpg"])

    # Apply a rewrite that won't match anything
    aliases, new_col = handle_parquet_column(["image_path"], paths, [("/data/images", "<DATA_PATH>")])

    # Column should be unchanged
    assert new_col.equals(paths)
