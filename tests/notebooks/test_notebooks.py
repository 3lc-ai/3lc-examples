from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import papermill as pm
import pytest


def get_ordered_notebook_paths() -> list[Any]:
    """Get an ordered list of paths for notebook files, maintaining folder hierarchy and ordering."""

    notebooks_folder = Path("tutorials")
    ordered_notebooks: list[Path] = []

    def sort_func(item):
        """Extract numerical prefix for sorting or fallback to name."""
        try:
            return int(item.stem.split("-")[0] if item.is_file() else item.name.split("-")[0])
        except ValueError:
            return 1000  # Fallback to large number for non-numeric names

    def gather_notebooks(folder: Path):
        """Recursively gather notebooks in the given folder in sorted order."""
        sorted_items = sorted(folder.iterdir(), key=sort_func)
        for item in sorted_items:
            if item.is_dir():
                gather_notebooks(item)
            elif item.suffix == ".ipynb":
                ordered_notebooks.append(item)

    # Start by processing the top-level `tutorials` folder
    gather_notebooks(notebooks_folder)

    # Wrap notebooks as pytest parameters for testing
    ids = [nb.relative_to(notebooks_folder).as_posix().replace(".ipynb", "") for nb in ordered_notebooks]

    # Create pytest parameters with marks based on notebook metadata
    params = []
    for nb, id in zip(ordered_notebooks, ids):
        # Read notebook metadata
        with open(nb, encoding="utf-8") as f:
            notebook = json.load(f)

        # Get test marks from notebook metadata
        marks = []
        if "metadata" in notebook and "test_marks" in notebook["metadata"]:
            for mark in notebook["metadata"]["test_marks"]:
                marks.append(getattr(pytest.mark, mark))

        # Create parameter with marks if any exist
        if marks:
            params.append(pytest.param(nb, id=id, marks=marks))
        else:
            params.append(pytest.param(nb, id=id))

    return params


@pytest.mark.parametrize("notebook_path", get_ordered_notebook_paths())
def test_notebook_execution(notebook_path: Path, tmp_path: Path) -> None:
    output_path = tmp_path / f"executed_{notebook_path.name}"
    notebook_parent_dir = notebook_path.parent
    original_cwd = Path.cwd()

    try:
        print(f"Executing notebook {notebook_path}\nOutput will be saved to {output_path}")
        os.chdir(notebook_parent_dir)  # Change CWD to the notebook's parent directory
        pm.execute_notebook(
            input_path=notebook_path.name,
            output_path=str(output_path),
            kernel_name="python3",
        )
    except pm.PapermillExecutionError as e:
        pytest.fail(f"Notebook {notebook_path} failed during execution: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error during notebook execution: {str(e)}")
    finally:
        os.chdir(original_cwd)
