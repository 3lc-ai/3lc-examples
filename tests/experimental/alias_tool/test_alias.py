from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tlc.core import EditedTable, ObjectRegistry, Table, TableFromParquet, TableFromPydict, Url

from tlc_tools.experimental.alias_tool.common import get_input_object
from tlc_tools.experimental.alias_tool.list_aliases import (
    find_aliases_in_column,
    list_aliases,
    list_aliases_in_tlc_table,
)
from tlc_tools.experimental.alias_tool.replace_aliases import (
    replace_aliases_in_pa_table,
    replace_aliases_in_tlc_table,
    rewrite_column_values,
)

TEST_ALIAS_PROJECT_NAME = "test_alias"
TEST_ALIAS_DATASET_NAME = "test_alias_dataset"


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
            "<DATA_PATH>/images/003.jpg",
            "s3://bucket/images/004.jpg",
        ],
        "mask_path": [
            "/data/project/masks/001.png",
            "/data/project/masks/002.png",
            "<MASK_PATH>/003.png",
            "s3://bucket/masks/004.png",
        ],
        "metadata": pa.array(
            [
                {"id": 1, "path": "/data/project/meta/001.json", "type": "annotation"},
                {"id": 2, "path": "<META_PATH>/002.json", "type": "annotation"},
                {"id": 3, "path": "/path/to/<CACHE_PATH>/003.json", "type": "cache"},
                {"id": 4, "path": "s3://bucket/meta/004.json", "type": "annotation"},
            ],
            type=pa.struct([("id", pa.int32()), ("path", pa.string()), ("type", pa.string())]),
        ),
        "label": [1, 2, 3, 4],  # Non-string column to ignore
    }
    # Create the underlying pa.Table
    pa_table = pa.Table.from_pydict(data)
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa_table, parquet_path)

    # Create a tlc.Table object that wraps the parquet file
    table = Table.from_parquet(
        Url(str(parquet_path)),
        if_exists="raise",
        project_name=TEST_ALIAS_PROJECT_NAME,
        dataset_name=TEST_ALIAS_DATASET_NAME,
        add_weight_column=False,
    )
    table.ensure_fully_defined()
    yield table

    # Delete the parquet file
    parquet_path.unlink()

    # Delete the tlc.Table object
    table.url.delete()


@pytest.fixture
def sample_table_with_parent() -> Generator[Table, None, None]:
    """Create a sample table with a parent table."""
    # Create a parent table
    parent_table = Table.from_dict(
        {
            "image_path": ["/data/images/001.jpg", "/data/images/002.jpg", "/other/path/003.jpg"],
            "mask_path": ["/data/masks/001.png", "/data/masks/002.png", "/data/masks/003.png"],
            "label": [1, 2, 3],  # Non-string column should be ignored
        },
        project_name=TEST_ALIAS_PROJECT_NAME,
        dataset_name=TEST_ALIAS_DATASET_NAME,
        table_name="parent_table",
        if_exists="raise",
    )
    parent_table.ensure_fully_defined()

    # Create a child table with a reference to the parent table
    child_table_url = Url.create_table_url(
        "child_table",
        dataset_name=TEST_ALIAS_DATASET_NAME,
        project_name=TEST_ALIAS_PROJECT_NAME,
    )
    assert not child_table_url.exists()

    child_table = EditedTable(
        url=child_table_url,
        input_table_url=parent_table.url,
        edits={
            "image_path": {
                "runs_and_values": [[0], "/data/images/007.jpg"],
            },
        },
    )
    child_table.ensure_fully_defined()

    yield child_table

    # Delete the child table
    child_table.url.delete()

    # Delete the parent table
    parent_table.url.delete()


@pytest.fixture
def sample_table_with_pseudo_parent() -> Generator[Table, None, None]:
    """Create a sample table with a pseudo parent table."""
    # Create a pseudo parent table
    pseudo_parent_table = Table.from_dict(
        {
            "image_path": ["/data/images/001.jpg", "/data/images/002.jpg", "/other/path/003.jpg"],
            "mask_path": ["/data/masks/001.png", "/data/masks/002.png", "/data/masks/003.png"],
            "label": [1, 2, 3],  # Non-string column should be ignored
        },
        project_name=TEST_ALIAS_PROJECT_NAME,
        dataset_name=TEST_ALIAS_DATASET_NAME,
        table_name="pseudo_parent_table",
        if_exists="raise",
    )
    pseudo_parent_table.ensure_fully_defined()

    child_table = Table.from_dict(
        {
            "other_path": ["/other/path/001.jpg", "/data/images/002.jpg"],
        },
        project_name=TEST_ALIAS_PROJECT_NAME,
        dataset_name=TEST_ALIAS_DATASET_NAME,
        table_name="child_table",
        if_exists="raise",
        input_tables=[pseudo_parent_table.url],
    )
    child_table.ensure_fully_defined()

    yield child_table

    # Delete the child table
    child_table.url.delete()

    # Delete the pseudo parent table
    pseudo_parent_table.url.delete()


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


def test_list_column_basic(array_types):
    """Test listing aliases in a column for different array types (basic, chunked, struct)."""
    array_type, array, col_names, _ = array_types

    expected_col_path = ".".join(col_names + (["path"] if array_type == "struct" else []))
    aliases = find_aliases_in_column(expected_col_path, array)

    assert len(aliases) == 2, f"Failed for {array_type} array"
    assert any(alias[1] == "<DATA_PATH>" for alias in aliases)
    assert any(alias[1] == "<MODEL_PATH>" for alias in aliases)
    assert not any("<CACHE_PATH>" in alias[1] for alias in aliases)


def test_list_pa_table_basic(tmp_parquet):
    """Test listing aliases in a parquet table with basic string columns."""
    table = pa.table(
        {
            "col1": [
                "<DATA_PATH>/file.txt",
                "<MODEL_PATH>/model.pt",
                "<DATA_PATH>/other.txt",  # Added second instance of DATA_PATH
            ],
            "col2": [
                "path/to/<CACHE>/file.txt",  # Invalid - alias in middle
                "normal/path.txt",
                "another/path.txt",
            ],
            "metadata": pa.array(
                [
                    {"id": 1, "path": "<META_PATH>/meta.json"},
                    {"id": 2, "path": "<META_PATH>/other.json"},  # Added second instance of META_PATH
                    {"id": 3, "path": "/another/path.json"},
                ],
                type=pa.struct([("id", pa.int32()), ("path", pa.string())]),
            ),
        }
    )

    with patch("tlc_tools.experimental.alias_tool.list_aliases.logger") as mock_logger:
        list_aliases([Url(tmp_parquet)], table, [])

        # Verify correct aliases were found and logged
        found_aliases = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Found alias '<DATA_PATH>' in column 'col1'" in msg for msg in found_aliases)
        assert any("Found alias '<MODEL_PATH>' in column 'col1'" in msg for msg in found_aliases)
        assert any("Found alias '<META_PATH>' in column 'metadata.path'" in msg for msg in found_aliases)
        assert not any("<CACHE>" in msg for msg in found_aliases)


def test_list_pa_table_selected_columns(tmp_parquet):
    """Test listing aliases in a parquet table with column selection."""
    table = pa.table(
        {
            "col1": ["<DATA_PATH>/file.txt"],
            "col2": ["<MODEL_PATH>/model.pt"],
        }
    )

    with patch("tlc_tools.experimental.alias_tool.list_aliases.logger") as mock_logger:
        list_aliases([Url(tmp_parquet)], table, ["col1"])

        # Verify only aliases from selected column were found
        found_aliases = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("<DATA_PATH>" in msg for msg in found_aliases)
        assert not any("<MODEL_PATH>" in msg for msg in found_aliases)


def test_list_pa_table_no_aliases(tmp_parquet):
    """Test listing aliases in a parquet table that contains no aliases."""
    table = pa.table(
        {
            "col1": ["path/to/file.txt"],
            "col2": ["other/path/file.txt"],
        }
    )

    with patch("tlc_tools.experimental.alias_tool.list_aliases.logger") as mock_logger:
        list_aliases([Url(tmp_parquet)], table, [])

        # Verify "no aliases found" message was logged
        mock_logger.info.assert_called_once_with(f"No aliases found in file '{Path(tmp_parquet).as_posix()}'")


def test_list_tlc_table_basic(sample_table: Table) -> None:
    """Test listing aliases in a TLC table."""
    with patch("tlc_tools.experimental.alias_tool.list_aliases.logger") as mock_logger:
        list_aliases_in_tlc_table([sample_table.url], sample_table, [])

        # Verify correct aliases were found and logged
        found_aliases = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Found alias '<DATA_PATH>' in column 'image_path'" in msg for msg in found_aliases)
        assert any("Found alias '<MASK_PATH>' in column 'mask_path'" in msg for msg in found_aliases)
        assert any("Found alias '<META_PATH>' in column 'metadata.path'" in msg for msg in found_aliases)


def test_replace_pa_table_selected_columns(tmp_parquet):
    """Test replacing aliases in a parquet table with column selection."""
    table = pq.read_table(tmp_parquet)

    replace_aliases_in_pa_table([Url(tmp_parquet)], table, ["image_path"], [("/data/images", "<DATA_PATH>")])

    # Read back and verify
    result = pq.read_table(tmp_parquet)

    # image_path should be modified
    assert "<DATA_PATH>" in result.column("image_path")[0].as_py()

    # mask_path should be unchanged
    assert "/data/masks" in result.column("mask_path")[0].as_py()

    # label should be unchanged
    assert result.column("label").equals(table.column("label"))


def test_replace_column_basic(array_types):
    """Test replacing aliases in a column for different array types (basic, chunked, struct)."""
    array_type, array, col_names, get_paths = array_types

    rewrites = [("/path/to", "<CACHE_PATH>")]
    result, _ = rewrite_column_values(".".join(col_names), array, rewrites)

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


def test_replace_pa_table_multiple_rewrites(tmp_parquet):
    """Test replacing multiple aliases in a parquet table."""
    table = pq.read_table(tmp_parquet)

    replace_aliases_in_pa_table(
        [Url(tmp_parquet)], table, [], [("/data/images", "<DATA_PATH>"), ("/data/masks", "<MASK_PATH>")]
    )

    # Read back and verify
    result = pq.read_table(tmp_parquet)

    # Check image_path rewrites
    assert all("<DATA_PATH>" in path.as_py() for path in result.column("image_path")[:2])

    # Check mask_path rewrites
    assert all("<MASK_PATH>" in path.as_py() for path in result.column("mask_path"))


def test_replace_pa_table_invalid_column():
    """Test error handling when replacing aliases in non-existent column."""
    table = pa.table({"col1": [1, 2, 3]})

    with pytest.raises(ValueError, match=re.escape("not found in columns: ['col1']")):
        replace_aliases_in_pa_table([Url("test.parquet")], table, ["invalid_column"], [])


def test_replace_pa_table_backup_none_changed(mocker):
    """Test that no backup is created when no changes are made during replacement."""
    table = pa.table(
        {
            "col1": ["path/to/file.txt", "other/path/file.txt"],
        }
    )

    mock_backup = mocker.patch("tlc_tools.experimental.alias_tool.replace_aliases.backup_parquet")
    mock_write = mocker.patch("tlc.core.UrlAdapterRegistry.write_binary_content_to_url")

    # Process with rewrites that won't affect anything
    replace_aliases_in_pa_table([Url("test.parquet")], table, [], [("/nonexistent", "<ALIAS>")])

    # Verify no backup was created and no write attempted
    mock_backup.assert_not_called()
    mock_write.assert_not_called()


def test_replace_pa_table_backup_on_error(mocker):
    """Test backup creation and restoration when write fails during replacement."""
    table = pa.table(
        {
            "col1": ["/data/file.txt", "/data/other.txt"],
        }
    )

    mock_backup = mocker.patch("tlc_tools.experimental.alias_tool.replace_aliases.backup_parquet")
    mock_backup.return_value = Url("test.parquet.backup")

    mock_restore = mocker.patch("tlc_tools.experimental.alias_tool.replace_aliases.restore_from_backup")

    # Make write fail
    mock_write = mocker.patch("tlc.core.UrlAdapterRegistry.write_binary_content_to_url")
    mock_write.side_effect = OSError("Write failed")

    with pytest.raises(OSError, match="Write failed"):
        replace_aliases_in_pa_table(
            [Url("test.parquet")],
            table,
            [],
            [("/data", "<DATA>")],  # This will cause changes
        )

    # Verify backup was created and restore was attempted
    mock_backup.assert_called_once()
    mock_restore.assert_called_once_with(Url("test.parquet.backup"), Url("test.parquet"))


def test_replace_pa_table_backup_restore_integration(tmp_path, mocker):
    """Test actual backup and restore functionality when write fails.
    This test verifies that:
    1. Original file is backed up correctly
    2. Backup file contains the original content
    3. After write failure, original file is restored from backup
    4. Backup file is cleaned up
    """
    # Create original file with test data
    original_path = tmp_path / "test.parquet"
    original_table = pa.table({"col1": ["/data/file.txt", "/data/other.txt"]})
    pq.write_table(original_table, original_path)

    # Mock only the write to fail
    mock_write = mocker.patch("tlc.core.UrlAdapterRegistry.write_binary_content_to_url")
    mock_write.side_effect = OSError("Write failed")

    # Attempt the replacement which should fail
    with pytest.raises(OSError, match="Write failed"):
        replace_aliases_in_pa_table(
            [Url(str(original_path))],
            original_table,
            [],
            [("/data", "<DATA>")],
        )

    # Verify original file still exists and contains original content
    assert original_path.exists()
    restored_table = pq.read_table(original_path)
    assert restored_table.equals(original_table)

    # Verify backup file was cleaned up
    backup_path = original_path.with_suffix(".parquet.backup")
    assert not backup_path.exists()


def test_replace_tlc_table_basic(sample_table: Table) -> None:
    """Test basic alias replacement in a TLC table including all column types."""
    rewrites = [
        ("/data/project", "<PROJECT_PATH>"),  # Affects basic paths in image_path and mask_path
        ("s3://bucket", "<BUCKET_PATH>"),  # Affects cloud storage paths in all columns
        ("<META_PATH>", "/data/metadata"),  # Affects existing alias in metadata.path
    ]

    replace_aliases_in_tlc_table(
        [sample_table.url],
        sample_table,
        [],
        rewrites,
    )
    ObjectRegistry.drop_cache()
    reloaded_table = Table.from_url(sample_table.url)

    # Check that the table was modified
    image_column = reloaded_table.get_column("image_path")
    assert image_column.to_pylist() == [
        "<PROJECT_PATH>/images/001.jpg",
        "<PROJECT_PATH>/images/002.jpg",
        "<DATA_PATH>/images/003.jpg",
        "<BUCKET_PATH>/images/004.jpg",
    ]

    mask_column = reloaded_table.get_column("mask_path")
    assert mask_column.to_pylist() == [
        "<PROJECT_PATH>/masks/001.png",
        "<PROJECT_PATH>/masks/002.png",
        "<MASK_PATH>/003.png",
        "<BUCKET_PATH>/masks/004.png",
    ]

    metadata_column = reloaded_table.get_column("metadata")
    assert metadata_column.to_pylist() == [
        {"id": 1, "path": "<PROJECT_PATH>/meta/001.json", "type": "annotation"},
        {"id": 2, "path": "/data/metadata/002.json", "type": "annotation"},
        {"id": 3, "path": "/path/to/<CACHE_PATH>/003.json", "type": "cache"},
        {"id": 4, "path": "<BUCKET_PATH>/meta/004.json", "type": "annotation"},
    ]

    label_column = reloaded_table.get_column("label")
    assert label_column.to_pylist() == [1, 2, 3, 4]


def test_replace_tlc_table_selected_columns(sample_table: Table) -> None:
    """Test alias replacement in a TLC table with column selection."""
    rewrites = [("/data/project", "<PROJECT_PATH>")]

    initial_mask_path = sample_table.get_column("mask_path")
    initial_metadata = sample_table.get_column("metadata")
    initial_label = sample_table.get_column("label")

    replace_aliases_in_tlc_table(
        [sample_table.url],
        sample_table,
        ["image_path"],
        rewrites,
    )
    ObjectRegistry.drop_cache()
    reloaded_table = Table.from_url(sample_table.url)

    # Check that only image_path was modified
    assert reloaded_table.get_column("image_path").to_pylist() == [
        "<PROJECT_PATH>/images/001.jpg",
        "<PROJECT_PATH>/images/002.jpg",
        "<DATA_PATH>/images/003.jpg",
        "s3://bucket/images/004.jpg",
    ]

    # Check that mask_path, metadata, and label remain unchanged
    assert reloaded_table.get_column("mask_path").equals(initial_mask_path)
    assert reloaded_table.get_column("metadata").equals(initial_metadata)
    assert reloaded_table.get_column("label").equals(initial_label)


def test_replace_tlc_table_parent_basic(sample_table_with_parent: Table) -> None:
    """Test alias replacement in a TLC table with parent table processing."""
    replace_aliases_in_tlc_table(
        [sample_table_with_parent.url],
        sample_table_with_parent,
        [],
        [("/data/images", "<DATA_PATH>"), ("/data/masks", "<MASK_PATH>")],
        process_parents=True,  # the default
    )

    ObjectRegistry.drop_cache()

    # Check that the child table was modified
    reloaded_table = Table.from_url(sample_table_with_parent.url)
    assert reloaded_table[0]["image_path"] == "/data/images/007.jpg"
    assert reloaded_table[0]["mask_path"] == "<MASK_PATH>/001.png"

    assert reloaded_table[1]["image_path"] == "<DATA_PATH>/002.jpg"
    assert reloaded_table[1]["mask_path"] == "<MASK_PATH>/002.png"

    # Check that the parent table was modified
    parent_table = reloaded_table.input_table_url.object
    assert isinstance(parent_table, TableFromPydict)
    assert parent_table.get_column("image_path").to_pylist() == [
        "<DATA_PATH>/001.jpg",
        "<DATA_PATH>/002.jpg",
        "/other/path/003.jpg",
    ]
    assert parent_table.get_column("mask_path").to_pylist() == [
        "<MASK_PATH>/001.png",
        "<MASK_PATH>/002.png",
        "<MASK_PATH>/003.png",
    ]


def test_replace_tlc_table_parent_disabled(sample_table_with_parent: Table) -> None:
    """Test alias replacement in a TLC table with parent processing disabled."""
    replace_aliases_in_tlc_table(
        [sample_table_with_parent.url],
        sample_table_with_parent,
        [],
        [("/data/images", "<DATA_PATH>"), ("/data/masks", "<MASK_PATH>")],
        process_parents=False,
    )

    ObjectRegistry.drop_cache()

    # Check that the child table was not modified
    reloaded_table = Table.from_url(sample_table_with_parent.url)
    assert reloaded_table[0]["image_path"] == "/data/images/007.jpg"
    assert reloaded_table[0]["mask_path"] == "/data/masks/001.png"

    # Check that the parent table was not modified
    parent_table = reloaded_table.input_table_url.object
    assert isinstance(parent_table, TableFromPydict)
    assert parent_table.get_column("image_path").to_pylist() == [
        "/data/images/001.jpg",
        "/data/images/002.jpg",
        "/other/path/003.jpg",
    ]
    assert parent_table.get_column("mask_path").to_pylist() == [
        "/data/masks/001.png",
        "/data/masks/002.png",
        "/data/masks/003.png",
    ]


def test_replace_tlc_table_pseudo_parent(sample_table_with_pseudo_parent: Table) -> None:
    """Test alias replacement in a TLC table with pseudo parent table processing."""
    replace_aliases_in_tlc_table(
        [sample_table_with_pseudo_parent.url],
        sample_table_with_pseudo_parent,
        [],
        [("/other/path", "<OTHER_PATH>"), ("/data/images", "<DATA_PATH>")],
        process_parents=True,
    )

    ObjectRegistry.drop_cache()

    # Check that the child table was modified
    reloaded_table = Table.from_url(sample_table_with_pseudo_parent.url)

    assert reloaded_table.get_column("other_path").to_pylist() == ["<OTHER_PATH>/001.jpg", "<DATA_PATH>/002.jpg"]

    # Check that the pseudo parent table was modified
    pseudo_parent_table = Table.from_url(reloaded_table.input_tables[0].to_absolute(reloaded_table.url))

    assert pseudo_parent_table.get_column("image_path").to_pylist() == [
        "<DATA_PATH>/001.jpg",
        "<DATA_PATH>/002.jpg",
        "<OTHER_PATH>/003.jpg",
    ]

    assert pseudo_parent_table.get_column("mask_path").to_pylist() == [
        "/data/masks/001.png",
        "/data/masks/002.png",
        "/data/masks/003.png",
    ]

    assert pseudo_parent_table.get_column("label").to_pylist() == [1, 2, 3]


def test_get_input_object(sample_table: Table, tmp_path: Path) -> None:
    """Test getting input object from a parquet file."""

    assert isinstance(sample_table, TableFromParquet)
    existing_table_url = sample_table.url
    existing_parquet_path = sample_table.input_url

    assert existing_table_url.exists()
    assert existing_parquet_path.exists()

    # Test getting the object from the parquet file
    pa_table = get_input_object(existing_parquet_path)
    assert isinstance(pa_table, pa.Table)
    assert pa_table.equals(sample_table._to_pyarrow_table())

    # Test getting the object from the table URL
    tlc_table = get_input_object(existing_table_url)
    assert isinstance(tlc_table, TableFromParquet)
    assert tlc_table.url == existing_table_url
    assert tlc_table.input_url == existing_parquet_path

    # Test getting non-existent objects
    with pytest.raises(FileNotFoundError):
        get_input_object(Url("non_existent_file.parquet"))

    # Write a invalid parquet file
    invalid_parquet_path = tmp_path / "invalid.parquet"
    with invalid_parquet_path.open("wb") as f:
        f.write(b"invalid")

    with pytest.raises(ValueError, match="Input file '.*invalid.parquet' is not a valid 3LC object or Parquet file."):
        get_input_object(Url(invalid_parquet_path))
