from unittest.mock import patch

import pytest

from tlc_tools.cli.main import main
from tlc_tools.cli.registry import _TOOLS, ToolInfo


def dummy_tool(args=None, prog=None):
    """A dummy tool that just returns success"""
    return 0


def experimental_dummy_tool(args=None, prog=None):
    """A dummy experimental tool that just returns success"""
    return 0


@pytest.fixture(autouse=True)
def setup_tools():
    """Fixture to set up dummy tools in the registry before each test"""
    # Store original tools
    original_tools = _TOOLS.copy()

    # Add our test tools directly to the registry
    _TOOLS["dummy_tool"] = ToolInfo(
        callable=dummy_tool, is_experimental=False, module_path="test_cli", description="A dummy tool for testing"
    )

    _TOOLS["experimental_dummy_tool"] = ToolInfo(
        callable=experimental_dummy_tool,
        is_experimental=True,
        module_path="test_cli",
        description="An experimental dummy tool for testing",
    )

    yield

    # Restore original tools after test
    _TOOLS.clear()
    _TOOLS.update(original_tools)


def test_cli_shows_help(capsys):
    """Test that CLI displays help message when called with --help"""
    with pytest.raises(SystemExit) as exc_info, patch("sys.argv", ["3lc-tools", "--help"]):
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "usage:" in captured.out.lower()


def test_cli_version(capsys):
    """Test that CLI can display version information"""
    with patch("sys.argv", ["3lc-tools", "--version"]):
        assert main() == 0

    captured = capsys.readouterr()
    output = captured.out.strip()
    assert output.startswith("3lc-tools:")
    assert "3lc:" in output


def test_cli_invalid_command(capsys):
    """Test that CLI handles invalid commands appropriately"""
    with pytest.raises(SystemExit) as exc_info, patch("sys.argv", ["3lc-tools", "nonexistent-command"]):
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code != 0
    assert "error" in captured.err.lower() or "invalid" in captured.err.lower()


def test_list_command_help(capsys):
    """Test the list command help output"""
    with pytest.raises(SystemExit) as exc_info, patch("sys.argv", ["3lc-tools", "list", "--help"]):
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "usage:" in captured.out.lower()


def test_run_command_help(capsys):
    """Test the run command help output"""
    with pytest.raises(SystemExit) as exc_info, patch("sys.argv", ["3lc-tools", "run", "--help"]):
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "usage:" in captured.out.lower()


def test_run_dummy_tool(capsys):
    """Test running a registered dummy tool"""
    with patch("sys.argv", ["3lc-tools", "run", "dummy-tool"]):
        assert main() == 0

    captured = capsys.readouterr()
    assert not captured.out


def test_run_experimental_tool_without_flag(capsys):
    """Test that running an experimental tool without --exp flag fails"""
    with pytest.raises(SystemExit) as exc_info, patch("sys.argv", ["3lc-tools", "run", "experimental-dummy-tool"]):
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code != 0
    assert "error:" in captured.out.lower()


def test_run_experimental_tool_with_flag(capsys):
    """Test running an experimental tool with --exp flag"""
    with patch("sys.argv", ["3lc-tools", "run", "--exp", "experimental-dummy-tool"]):
        assert main() == 0

    captured = capsys.readouterr()
    assert not captured.out


def test_list_shows_tools(capsys):
    """Test that list command shows available tools"""
    with patch("sys.argv", ["3lc-tools", "list"]):
        main()

    captured = capsys.readouterr()
    assert "dummy-tool" in captured.out
    assert "experimental-dummy-tool" in captured.out
