from unittest.mock import patch

import pytest
from tlc.core import Url

from tlc_tools.experimental.alias_tool.alias import main


@patch("tlc_tools.experimental.alias_tool.alias.get_input_object")
@patch("tlc_tools.experimental.alias_tool.alias.handle_object")
def test_main_basic_replace(mock_handle_object, mock_get_input):
    """Test basic replace command with no additional arguments."""
    main(["replace", "table.parquet"])

    # Should try to get input object
    mock_get_input.assert_called_once()
    # Should call handle_object with empty rewrites
    mock_handle_object.assert_called_once()
    args = mock_handle_object.call_args[0]
    assert len(args) == 4  # [input_path], object, columns, rewrites
    assert args[2] == []  # no columns specified
    assert args[3] == []  # no rewrites


@patch("tlc_tools.experimental.alias_tool.alias.get_input_object")
@patch("tlc_tools.experimental.alias_tool.alias.handle_object")
def test_main_apply_single_alias(mock_handle_object, mock_get_input):
    """Test applying a single alias."""
    with patch("tlc.get_registered_url_aliases") as mock_get_aliases:
        mock_get_aliases.return_value = {"<DATA_PATH>": "/data/path"}
        main(["replace", "table.parquet", "--apply", "DATA_PATH"])

        # Should call handle_object with the rewrite
        mock_handle_object.assert_called_once()
        args = mock_handle_object.call_args[0]
        assert args[3] == [("/data/path", "<DATA_PATH>")]


@patch("tlc_tools.experimental.alias_tool.alias.get_input_object")
@patch("tlc_tools.experimental.alias_tool.alias.handle_object")
def test_main_apply_multiple_aliases(mock_handle_object, mock_get_input):
    """Test applying multiple aliases."""
    with patch("tlc.get_registered_url_aliases") as mock_get_aliases:
        mock_get_aliases.return_value = {"<DATA_PATH>": "/data/path", "<CACHE>": "/cache/path"}
        main(["replace", "table.parquet", "--apply", "DATA_PATH,CACHE"])

        # Should call handle_object with both rewrites
        mock_handle_object.assert_called_once()
        args = mock_handle_object.call_args[0]
        assert len(args[3]) == 2
        assert ("/data/path", "<DATA_PATH>") in args[3]
        assert ("/cache/path", "<CACHE>") in args[3]


@patch("tlc_tools.experimental.alias_tool.alias.get_input_object")
@patch("tlc_tools.experimental.alias_tool.alias.handle_object")
def test_main_from_to_single(mock_handle_object, mock_get_input):
    """Test replacing a single path."""
    main(["replace", "table.parquet", "--from", "/old/path", "--to", "/new/path"])

    # Should call handle_object with the rewrite
    mock_handle_object.assert_called_once()
    args = mock_handle_object.call_args[0]
    assert args[3] == [("/old/path", "/new/path")]


@patch("tlc_tools.experimental.alias_tool.alias.get_input_object")
@patch("tlc_tools.experimental.alias_tool.alias.handle_object")
def test_main_from_to_multiple(mock_handle_object, mock_get_input):
    """Test replacing multiple paths."""
    main(
        [
            "replace",
            "table.parquet",
            "--from",
            "/old/path1",
            "--to",
            "/new/path1",
            "--from",
            "/old/path2",
            "--to",
            "/new/path2",
        ]
    )

    # Should call handle_object with both rewrites
    mock_handle_object.assert_called_once()
    args = mock_handle_object.call_args[0]
    assert len(args[3]) == 2
    assert ("/old/path1", "/new/path1") in args[3]
    assert ("/old/path2", "/new/path2") in args[3]


@patch("tlc_tools.experimental.alias_tool.alias.get_input_object")
@patch("tlc_tools.experimental.alias_tool.alias.handle_object")
def test_main_columns(mock_handle_object, mock_get_input):
    """Test processing specific columns."""
    main(["replace", "table.parquet", "--columns", "col1,col2"])

    # Should call handle_object with the specified columns
    mock_handle_object.assert_called_once()
    args = mock_handle_object.call_args[0]
    assert args[2] == ["col1", "col2"]


def test_main_missing_to():
    """Test that --from without --to raises an error."""
    with pytest.raises(ValueError, match="--to PATH is required when using --from"):
        main(["replace", "table.parquet", "--from", "/old/path"])


def test_main_mismatched_from_to():
    """Test that unequal numbers of --from and --to raise an error."""
    with pytest.raises(ValueError, match="Number of --from and --to arguments must match"):
        main(["replace", "table.parquet", "--from", "/old/path1", "--to", "/new/path1", "--from", "/old/path2"])


@patch("tlc_tools.experimental.alias_tool.alias.get_input_object")
@patch("tlc_tools.experimental.alias_tool.alias.handle_object")
def test_main_no_process_parents(mock_handle_object, mock_get_input):
    """Test that --no-process-parents flag is respected."""
    main(["replace", "table.parquet", "--no-process-parents"])

    # Should call handle_object with process_parents=False
    mock_handle_object.assert_called_once()
    _, kwargs = mock_handle_object.call_args
    assert kwargs.get("process_parents") is False


def test_list_command_basic(mocker):
    """Test basic list command functionality."""
    # Create a mock table with some aliases
    mock_table = mocker.MagicMock()
    mock_get_input = mocker.patch("tlc_tools.experimental.alias_tool.alias.get_input_object")
    mock_get_input.return_value = mock_table

    mock_handle_list = mocker.patch("tlc_tools.experimental.alias_tool.alias.handle_list_command")

    # Run the list command
    main(["list", "table.parquet"])

    # Verify correct functions were called
    mock_get_input.assert_called_once()
    mock_handle_list.assert_called_once_with([Url("table.parquet")], mock_table, [])


def test_list_command_with_columns(mocker):
    """Test list command with specific columns."""
    mock_table = mocker.MagicMock()
    mock_get_input = mocker.patch("tlc_tools.experimental.alias_tool.alias.get_input_object")
    mock_get_input.return_value = mock_table

    mock_handle_list = mocker.patch("tlc_tools.experimental.alias_tool.alias.handle_list_command")

    # Run the list command with columns
    main(["list", "table.parquet", "--columns", "col1,col2"])

    # Verify columns were parsed correctly
    mock_handle_list.assert_called_once_with([Url("table.parquet")], mock_table, ["col1", "col2"])


def test_list_command_error_handling(mocker):
    """Test that list command properly handles and reports errors."""
    mock_get_input = mocker.patch("tlc_tools.experimental.alias_tool.alias.get_input_object")
    mock_get_input.side_effect = ValueError("Test error")

    # Run the list command and expect it to raise the error
    with pytest.raises(ValueError, match="Test error"):
        main(["list", "table.parquet"])
