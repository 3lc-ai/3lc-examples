from __future__ import annotations

from pathlib import Path
from typing import Any
import pyarrow.parquet as pq
import pytest
import tlc
from tlc.core import Table, Url

from tlc_tools.experimental.alias_tool.alias import (
    handle_object,
    handle_pa_table,
    handle_table,
    parse_alias_pair_string,
    validate_alias_name,
    validate_path_exists,
    handle_missing_alias,
)


@pytest.fixture
def sample_table(tmp_path: Path) -> Table:
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
    return table


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
    """Test basic parquet table handling.
    
    This test works directly with the pa.Table level, which is the underlying
    data structure. We access it via sample_table._pa_table.
    
    Note: When working with pa.Table directly, changes are immediate and don't
    require object recreation.
    """
    # Test listing aliases
    handle_pa_table(
        input_path=[sample_table.url],
        pa_table=sample_table._pa_table,  # Work directly with the pa.Table
        selected_column_name="",
        rewrite=[],
        inplace=False,
        output_url=None,
        create_alias=False,
        apply_alias=False,
        persist_config=False,
        config_scope="project",
    )

    # Test creating alias
    output_path = tmp_path / "output.parquet"
    handle_pa_table(
        input_path=[sample_table.url],
        pa_table=sample_table._pa_table,  # Work directly with the pa.Table
        selected_column_name="",
        rewrite=[("/data/project", "<DATA_PATH>")],
        inplace=False,
        output_url=output_path,
        create_alias=True,
        apply_alias=False,
        persist_config=False,
        config_scope="project",
    )

    # Verify output - need to read the new parquet file directly
    assert output_path.exists()
    output_pa_table = pq.read_table(output_path)
    assert "<DATA_PATH>/images/001.jpg" in output_pa_table.column("image_path").to_pylist()


def test_handle_table_basic(sample_table, tmp_path: Path):
    """Test basic table handling at the tlc.Table level.
    
    This test works with the tlc.Table object, which wraps the underlying pa.Table.
    When modifying the parquet file in-place, we need to recreate the tlc.Table
    object to see the changes.
    """
    # Test listing aliases
    handle_table(
        input_path=[sample_table.url],
        table=sample_table,
        column="",
        rewrite=[],
        inplace=False,
        output_url=None,
        create_alias=False,
        apply_alias=False,
        persist_config=False,
        config_scope="project",
    )

    # Test creating alias
    output_path = tmp_path / "output.parquet"
    handle_table(
        input_path=[sample_table.url],
        table=sample_table,
        column="",
        rewrite=[("/data/project", "<DATA_PATH>")],
        inplace=False,
        output_url=output_path,
        create_alias=True,
        apply_alias=False,
        persist_config=False,
        config_scope="project",
    )

    # Verify output - need to create a new tlc.Table object to see changes
    assert output_path.exists()
    output_table = Table.from_parquet(Url(str(output_path)), if_exists="rename")
    output_table.ensure_fully_defined()
    assert "<DATA_PATH>/images/001.jpg" in output_table._pa_table.column("image_path").to_pylist()


def test_handle_object_basic(sample_table, tmp_path: Path):
    """Test basic object handling."""
    # Test listing aliases
    handle_object(
        input_path=[sample_table.url],
        obj=sample_table,
        column="",
        rewrite=[],
        inplace=False,
        output_url=None,
        create_alias=False,
        apply_alias=False,
        persist_config=False,
        config_scope="project",
    )

    # Test creating alias
    output_path = tmp_path / "output.parquet"
    handle_object(
        input_path=[sample_table.url],
        obj=sample_table,
        column="",
        rewrite=[("/data/project", "<DATA_PATH>")],
        inplace=False,
        output_url=output_path,
        create_alias=True,
        apply_alias=False,
        persist_config=False,
        config_scope="project",
    )

    # Verify output
    output_table = Table.from_parquet(Url(str(output_path)), if_exists="rename")
    assert "<DATA_PATH>/images/001.jpg" in output_table.to_pa_table().column("image_path").to_pylist()


def test_column_specific_handling(sample_table, tmp_path: Path):
    """Test handling specific columns."""
    output_path = tmp_path / "output.parquet"
    handle_object(
        input_path=[sample_table.url],
        obj=sample_table,
        column="image_path",  # Only process image_path column
        rewrite=[("/data/project", "<DATA_PATH>")],
        inplace=False,
        output_url=output_path,
        create_alias=True,
        apply_alias=False,
        persist_config=False,
        config_scope="project",
    )

    # Verify output
    output_table = Table.from_parquet(Url(str(output_path)))
    assert "<DATA_PATH>/images/001.jpg" in output_table.to_pa_table().column("image_path").to_pylist()
    assert "/data/project/masks/001.png" in output_table.to_pa_table().column("mask_path").to_pylist()


def test_inplace_modification(sample_table):
    """Test inplace modification of tables.
    
    When modifying in-place, we need to be careful about the tlc.Table object's
    state. The underlying parquet file is modified, but the tlc.Table object
    needs to be recreated to see the changes.
    """
    # Store original paths for verification
    original_paths = sample_table._pa_table.column("image_path").to_pylist()
    
    # Modify in-place
    handle_object(
        input_path=[sample_table.url],
        obj=sample_table,
        column="",
        rewrite=[("/data/project", "<DATA_PATH>")],
        inplace=True,
        output_url=sample_table.url,
        create_alias=True,
        apply_alias=False,
        persist_config=False,
        config_scope="project",
    )

    # Need to recreate the tlc.Table object to see changes
    del sample_table
    sample_table = Table.from_url(sample_table.url)
    
    # Verify output
    assert "<DATA_PATH>/images/001.jpg" in sample_table._pa_table.column("image_path").to_pylist()


def test_persist_config(sample_table, tmp_path: Path):
    """Test persisting aliases to config."""
    # Register a test alias
    tlc.register_project_url_alias("TEST_DATA", "/data/project", project="test_project")

    # Test applying existing alias
    output_path = tmp_path / "output.parquet"
    handle_object(
        input_path=[sample_table.url],
        obj=sample_table,
        column="",
        rewrite=[("TEST_DATA", "")],  # Empty string as placeholder
        inplace=False,
        output_url=output_path,
        create_alias=False,
        apply_alias=True,
        persist_config=False,
        config_scope="project",
    )

    # Verify output
    output_table = Table.from_parquet(Url(str(output_path)))
    assert "<TEST_DATA>/images/001.jpg" in output_table.to_pa_table().column("image_path").to_pylist()


def test_validate_alias_name():
    """Test alias name validation."""
    # Valid cases
    assert validate_alias_name("<DATA_PATH>")
    assert validate_alias_name("<PROJECT_123>")
    assert validate_alias_name("<MY_CUSTOM_ALIAS>")
    
    # Invalid cases
    assert not validate_alias_name("DATA_PATH")  # Missing <>
    assert not validate_alias_name("<data_path>")  # Lowercase
    assert not validate_alias_name("<123_PATH>")  # Starts with number
    assert not validate_alias_name("<DATA-PATH>")  # Invalid character
    assert not validate_alias_name("DATA_PATH>")  # Missing opening <
    assert not validate_alias_name("<DATA_PATH")  # Missing closing >


def test_validate_path_exists(tmp_path: Path):
    """Test path existence validation."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.touch()
    
    # Valid cases
    assert validate_path_exists(str(test_file))
    assert validate_path_exists("http://example.com/path/to/file")
    assert validate_path_exists("s3://bucket/path/to/file")
    
    # Invalid cases
    assert not validate_path_exists(str(tmp_path / "nonexistent.txt"))
    assert not validate_path_exists("invalid://example.com")
    assert not validate_path_exists("not_a_url")


def test_handle_missing_alias():
    """Test handling of missing aliases."""
    # Register some test aliases
    tlc.register_project_url_alias("TEST_DATA", "/data/test", project="test_project")
    tlc.register_project_url_alias("TEST_IMAGES", "/data/images", project="test_project")
    
    # Test with non-existent alias
    with pytest.raises(ValueError) as exc_info:
        handle_missing_alias("NONEXISTENT")
    error_msg = str(exc_info.value)
    assert "not found in registered aliases" in error_msg
    assert "Did you mean one of these?" not in error_msg
    
    # Test with similar alias
    with pytest.raises(ValueError) as exc_info:
        handle_missing_alias("TEST_VIDEO")
    error_msg = str(exc_info.value)
    assert "Did you mean one of these?" in error_msg
    assert "TEST_DATA" in error_msg
    assert "TEST_IMAGES" in error_msg


def test_parse_alias_pair_string_validation():
    """Test alias pair string parsing with validation."""
    # Valid cases
    assert parse_alias_pair_string("/data/project::<DATA_PATH>", "::") == ("/data/project", "<DATA_PATH>")
    assert parse_alias_pair_string("<DATA_PATH>::/data/project", "::") == ("<DATA_PATH>", "/data/project")
    
    # Invalid cases
    with pytest.raises(ValueError) as exc_info:
        parse_alias_pair_string("<invalid_alias>::/data/project", "::")
    assert "Invalid alias name format" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        parse_alias_pair_string("<123_PATH>::/data/project", "::")
    assert "Invalid alias name format" in str(exc_info.value)


def create_table_with_lineage(tmp_path: Path) -> tuple[Table, Table, Table]:
    """Create a sample table with lineage for testing."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Create parent table 1
    parent1_data = {
        "image_path": ["/data/project/raw/001.jpg", "/data/project/raw/002.jpg"]
    }
    parent1_table = pa.Table.from_pydict(parent1_data)
    parent1_path = tmp_path / "parent1.parquet"
    pq.write_table(parent1_table, parent1_path)
    parent1 = Table.from_parquet(Url(str(parent1_path)))
    
    # Create parent table 2
    parent2_data = {
        "mask_path": ["/data/project/masks/001.png", "/data/project/masks/002.png"]
    }
    parent2_table = pa.Table.from_pydict(parent2_data)
    parent2_path = tmp_path / "parent2.parquet"
    pq.write_table(parent2_table, parent2_path)
    parent2 = Table.from_parquet(Url(str(parent2_path)))
    
    # Create child table that references both parents
    child_data = {
        "image_path": parent1_data["image_path"],
        "mask_path": parent2_data["mask_path"],
        "output_path": ["/data/project/output/001.jpg", "/data/project/output/002.jpg"]
    }
    child_table = pa.Table.from_pydict(child_data)
    child_path = tmp_path / "child.parquet"
    pq.write_table(child_table, child_path)
    child = Table.from_parquet(Url(str(child_path)))
    
    return child, parent1, parent2


def test_handle_table_deep_lineage(tmp_path: Path):
    """Test handling of tables with deep lineage."""
    # Create test tables with lineage
    child, parent1, parent2 = create_table_with_lineage(tmp_path)
    
    # Process the child table
    output_path = tmp_path / "output.parquet"
    handle_object(
        input_path=[child.url],
        obj=child,
        column="",
        rewrite=[("/data/project", "<DATA_PATH>")],
        inplace=False,
        output_url=output_path,
        create_alias=True,
        apply_alias=False,
        persist_config=False,
        config_scope="project",
    )
    
    # Verify child table output
    output_table = Table.from_parquet(Url(str(output_path)))
    assert "<DATA_PATH>/output/001.jpg" in output_table.to_pa_table().column("output_path").to_pylist()
    
    # Verify parent tables were processed
    parent1_output = Table.from_parquet(Url(str(tmp_path / "parent1.parquet")))
    assert "<DATA_PATH>/raw/001.jpg" in parent1_output.to_pa_table().column("image_path").to_pylist()
    
    parent2_output = Table.from_parquet(Url(str(tmp_path / "parent2.parquet")))
    assert "<DATA_PATH>/masks/001.png" in parent2_output.to_pa_table().column("mask_path").to_pylist()


def test_handle_table_cyclic_lineage(tmp_path: Path):
    """Test handling of tables with cyclic lineage references."""
    # Create test tables with a cycle
    child, parent1, parent2 = create_table_with_lineage(tmp_path)
    
    # Create a cycle by making parent1 reference the child table
    parent1_data = {
        "image_path": ["/data/project/raw/001.jpg"],
        "child_ref": [str(child.url)]
    }
    parent1_table = pa.Table.from_pydict(parent1_data)
    pq.write_table(parent1_table, tmp_path / "parent1.parquet")
    
    # Process the child table - should handle the cycle gracefully
    output_path = tmp_path / "output.parquet"
    handle_object(
        input_path=[child.url],
        obj=child,
        column="",
        rewrite=[("/data/project", "<DATA_PATH>")],
        inplace=False,
        output_url=output_path,
        create_alias=True,
        apply_alias=False,
        persist_config=False,
        config_scope="project",
    )
    
    # Verify the output was created without infinite recursion
    assert output_path.exists()
    output_table = Table.from_parquet(Url(str(output_path)))
    assert "<DATA_PATH>/output/001.jpg" in output_table.to_pa_table().column("output_path").to_pylist()
