import pytest
from tlc.core import Url

from tlc_tools.experimental.alias_tool.main import main


def test_main_basic_replace(mocker):
    """Test basic replace command with minimal arguments."""
    mock_replace_aliases = mocker.patch("tlc_tools.experimental.alias_tool.main.replace_aliases")
    mock_get_input = mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")
    mock_get_aliases = mocker.patch("tlc.get_registered_url_aliases")
    mock_get_aliases.return_value = {"<DATA_PATH>": "/data/path"}

    main(["replace", "table.parquet", "--apply", "DATA_PATH"])

    mock_get_input.assert_called_once()
    mock_replace_aliases.assert_called_once()
    args = mock_replace_aliases.call_args[0]
    assert len(args) == 4
    assert args[2] == []  # no columns specified
    assert args[3] == [("/data/path", "<DATA_PATH>")]  # rewrite from apply


def test_main_apply_single_alias(mocker):
    """Test applying a single alias."""
    mock_replace_aliases = mocker.patch("tlc_tools.experimental.alias_tool.main.replace_aliases")
    mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")
    mock_get_aliases = mocker.patch("tlc.get_registered_url_aliases")
    mock_get_aliases.return_value = {"<DATA_PATH>": "/data/path"}

    main(["replace", "table.parquet", "--apply", "DATA_PATH"])

    # Should call replace_aliases with the rewrite
    mock_replace_aliases.assert_called_once()
    args = mock_replace_aliases.call_args[0]
    assert args[3] == [("/data/path", "<DATA_PATH>")]


def test_main_apply_multiple_aliases(mocker):
    """Test applying multiple aliases."""
    mock_replace_aliases = mocker.patch("tlc_tools.experimental.alias_tool.main.replace_aliases")
    mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")
    mock_get_aliases = mocker.patch("tlc.get_registered_url_aliases")
    mock_get_aliases.return_value = {"<DATA_PATH>": "/data/path", "<CACHE>": "/cache/path"}

    main(["replace", "table.parquet", "--apply", "DATA_PATH,CACHE"])

    # Should call replace_aliases with both rewrites
    mock_replace_aliases.assert_called_once()
    args = mock_replace_aliases.call_args[0]
    assert len(args[3]) == 2
    assert ("/data/path", "<DATA_PATH>") in args[3]
    assert ("/cache/path", "<CACHE>") in args[3]


def test_main_from_to_single(mocker):
    """Test replacing a single path."""
    mock_replace_aliases = mocker.patch("tlc_tools.experimental.alias_tool.main.replace_aliases")
    mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")

    main(["replace", "table.parquet", "--from", "/old/path", "--to", "/new/path"])

    # Should call replace_aliases with the rewrite
    mock_replace_aliases.assert_called_once()
    args = mock_replace_aliases.call_args[0]
    assert args[3] == [("/old/path", "/new/path")]


def test_main_from_to_multiple(mocker):
    """Test replacing multiple paths."""
    mock_replace_aliases = mocker.patch("tlc_tools.experimental.alias_tool.main.replace_aliases")
    mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")

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

    # Should call replace_aliases with both rewrites
    mock_replace_aliases.assert_called_once()
    args = mock_replace_aliases.call_args[0]
    assert len(args[3]) == 2
    assert ("/old/path1", "/new/path1") in args[3]
    assert ("/old/path2", "/new/path2") in args[3]


def test_main_columns(mocker):
    """Test processing specific columns."""
    mock_replace_aliases = mocker.patch("tlc_tools.experimental.alias_tool.main.replace_aliases")
    mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")
    mock_get_aliases = mocker.patch("tlc.get_registered_url_aliases")
    mock_get_aliases.return_value = {"<DATA_PATH>": "/data/path"}

    main(["replace", "table.parquet", "--columns", "col1,col2", "--apply", "DATA_PATH"])

    mock_replace_aliases.assert_called_once()
    args = mock_replace_aliases.call_args[0]
    assert args[2] == ["col1", "col2"]


def test_main_missing_to(mocker):
    """Test that --from without --to raises an error."""
    mock_get_input = mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")

    # The error should be raised before get_input_object is called
    with pytest.raises(ValueError, match="--to PATH is required when using --from"):
        main(["replace", "table.parquet", "--from", "/old/path"])

    # Verify get_input_object was never called
    mock_get_input.assert_not_called()


def test_main_mismatched_from_to(mocker):
    """Test that unequal numbers of --from and --to raise an error."""
    mock_get_input = mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")

    # The error should be raised before get_input_object is called
    with pytest.raises(ValueError, match="Number of --from and --to arguments must match"):
        main(["replace", "table.parquet", "--from", "/old/path1", "--to", "/new/path1", "--from", "/old/path2"])

    # Verify get_input_object was never called
    mock_get_input.assert_not_called()


def test_main_no_process_parents(mocker):
    """Test that --no-process-parents flag is respected."""
    mock_replace_aliases = mocker.patch("tlc_tools.experimental.alias_tool.main.replace_aliases")
    mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")
    mock_get_aliases = mocker.patch("tlc.get_registered_url_aliases")
    mock_get_aliases.return_value = {"<DATA_PATH>": "/data/path"}

    main(["replace", "table.parquet", "--no-process-parents", "--apply", "DATA_PATH"])

    mock_replace_aliases.assert_called_once()
    _, kwargs = mock_replace_aliases.call_args
    assert kwargs.get("process_parents") is False


def test_list_command_basic(mocker):
    """Test basic list command functionality."""
    mock_table = mocker.MagicMock()
    mock_get_input = mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")
    mock_get_input.return_value = mock_table

    mock_list_aliases = mocker.patch("tlc_tools.experimental.alias_tool.main.list_aliases")

    # Run the list command
    main(["list", "table.parquet"])

    # Verify correct functions were called
    mock_get_input.assert_called_once()
    mock_list_aliases.assert_called_once_with([Url("table.parquet")], mock_table, [])


def test_list_command_with_columns(mocker):
    """Test list command with specific columns."""
    mock_table = mocker.MagicMock()
    mock_get_input = mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")
    mock_get_input.return_value = mock_table

    mock_list_aliases = mocker.patch("tlc_tools.experimental.alias_tool.main.list_aliases")

    # Run the list command with columns
    main(["list", "table.parquet", "--columns", "col1,col2"])

    # Verify columns were parsed correctly
    mock_list_aliases.assert_called_once_with([Url("table.parquet")], mock_table, ["col1", "col2"])


def test_list_command_error_handling(mocker):
    """Test that list command properly handles and reports errors."""
    mock_get_input = mocker.patch("tlc_tools.experimental.alias_tool.main.get_input_object")
    mock_get_input.side_effect = ValueError("Expected test error - this is normal test behavior")

    # Run the list command and expect it to raise the error
    with pytest.raises(ValueError, match="Expected test error - this is normal test behavior"):
        main(["list", "table.parquet"])
