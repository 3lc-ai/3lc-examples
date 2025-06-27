# 3LC Tools & Tutorials Tests

We have tests in tools, cli and notebooks. 

Notebooks that are long running or dependent on external data are marked with `slow` or `dependent` in the notebook metadata.

## Quick Reference

To run all tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=src/tlc_tools --cov-report=html
```

## Notebook Tests

Notebook tests execute tutorial notebooks using papermill to ensure they run without errors.

To list all slow notebooks run:

```bash
pytest tests/notebooks/ --collect-only -m slow
```

To list all dependent notebooks run:

```bash
pytest tests/notebooks/ --collect-only -m dependent
```

To run all notebooks that are not marked with `slow` or `dependent` run:

```bash
pytest tests/notebooks/ -m "not slow and not dependent"
```

To run only slow notebooks:

```bash
pytest tests/notebooks/ -m slow
```

To run only dependent notebooks:

```bash
pytest tests/notebooks/ -m dependent
```

To run a specific notebook:

```bash
pytest tests/notebooks/test_notebooks.py::test_notebook_execution[1-create-tables/create-bb-table] -v
```

## CLI Tests

To run the cli tests run:

```bash
pytest tests/cli/
```

The CLI tests verify the command-line interface functionality including:

- Basic command parsing and help output
- Tool listing and execution
- Argument validation and error handling

## Tools Tests

To run the tools tests run:

```bash
pytest tests/tools/
```

Tools tests are organized by functionality:

- `alias_tool/` - Tests for alias management functionality
- `instance_metrics_tool/` - Tests for instance segmentation metrics
- `metric_jumps/` - Tests for metric jump detection

To run tests for a specific tool:

```bash
pytest tests/tools/alias_tool/
pytest tests/tools/instance_metrics_tool/
pytest tests/tools/metric_jumps/
```

## Test Markers

The following pytest markers are available:

- `slow` - Marks tests as slow running (use `-m "not slow"` to skip)
- `dependent` - Marks tests that depend on external resources (use `-m "not dependent"` to skip)

## Development Workflow

For development, you typically want to run fast tests first:

```bash
# Run all tests except slow and dependent ones
pytest -m "not slow and not dependent"

# Run specific test categories
pytest tests/cli/ tests/tools/

# Run with verbose output for debugging
pytest -v tests/tools/alias_tool/test_alias.py::test_list_column_basic
```

