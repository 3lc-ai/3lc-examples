import argparse
from unittest.mock import patch

import pytest

from tlc_tools.cli.main import main
from tlc_tools.cli.registry import _TOOLS, ToolInfo


def tool_with_parser(args=None, prog=None):
    if args and args[0] == "--help":
        print("--help was passed to the tool")
        return 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--arg1", type=int)
    parser.add_argument("--arg2", type=int)
    args = parser.parse_args(args)

    assert args.arg1 == 1
    assert args.arg2 == 2
    return 0


def official_dummy_tool(args=None, prog=None):
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
    _TOOLS["official_dummy_tool"] = ToolInfo(
        callable=official_dummy_tool,
        is_experimental=False,
        module_path="test_cli",
        description="A dummy tool for testing",
    )

    _TOOLS["experimental_dummy_tool"] = ToolInfo(
        callable=experimental_dummy_tool,
        is_experimental=True,
        module_path="test_cli",
        description="An experimental dummy tool for testing",
    )

    _TOOLS["tool_with_parser"] = ToolInfo(
        callable=tool_with_parser,
        is_experimental=False,
        module_path="test_cli",
        description="A tool with a custom parser",
    )

    yield

    # Restore original tools after test
    _TOOLS.clear()
    _TOOLS.update(original_tools)


@pytest.mark.parametrize(
    "args, expected_output, expected_code",
    [
        # Basic commands
        (["--help"], "usage:", 0),
        (["list", "--help"], "usage:", 0),
        (["run", "--help"], "usage:", 0),
        (["--version"], "3lc-tools:", 0),
        (["nonexistent-command"], "invalid choice", 2),
        ([], "usage:", 1),
        # Tool operations
        (["run", "official-dummy-tool"], "", 0),
        (["run", "experimental-dummy-tool"], "error:", 1),
        (["run", "--exp", "experimental-dummy-tool"], "", 0),
        (["run", "--experimental", "experimental-dummy-tool"], "", 0),
        (["list"], "official-dummy-tool", 0),
        # Tool argument parsing
        (["run", "tool_with_parser", "--arg1", "1", "--arg2", "2"], "", 0),
        (["run", "tool_with_parser", "--help"], "--help was passed to the tool", 0),
        (["run", "tool_with_parser", "--nonexisting-arg", "1"], "unrecognized arguments:", 2),
    ],
)
def test_cli_commands(capsys, args, expected_output, expected_code):
    """Test CLI commands including basic commands and tool operations"""
    test_args = ["3lc-tools"] + args

    with pytest.raises(SystemExit) as exc_info, patch("sys.argv", test_args):
        main()
    assert exc_info.value.code == expected_code

    captured = capsys.readouterr()

    # Argparse returns 2 for invalid arguments, 1 for errors, 0 for success, and outputs to stderr for 2
    output = captured.err.lower() if expected_code == 2 else captured.out.lower()
    assert (expected_output.lower() in output) or (not expected_output and not output)
