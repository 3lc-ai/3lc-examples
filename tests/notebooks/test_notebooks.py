from pathlib import Path
from typing import Any
import papermill as pm
import pytest


def get_ordered_notebook_paths() -> list[Any]:
    """Get an ordered list of paths for notebook files"""

    notebooks_folder = Path("tutorials")
    ordered_notebooks: list[Path] = []

    # Gather standalone notebooks at the root level
    root_notebooks = sorted(notebooks_folder.glob("*.ipynb"))
    ordered_notebooks.extend(root_notebooks)

    def sort_func(x):
        try:
            return int(x.stem.split("_")[0])
        except ValueError:
            return x.stem

    # Gather and sort notebooks inside directories
    for subfolder in sorted(notebooks_folder.iterdir()):
        if subfolder.is_dir():
            notebooks_in_subfolder = sorted(subfolder.glob("*.ipynb"), key=sort_func)
            ordered_notebooks.extend(notebooks_in_subfolder)

    return [pytest.param(nb, id=nb.stem) for nb in ordered_notebooks]


@pytest.mark.parametrize("notebook_path", get_ordered_notebook_paths())
def test_notebook_execution(notebook_path: Path, tmp_path: Path) -> None:
    output_path = tmp_path / f"executed_{notebook_path.name}"
    try:
        print(f"Executing notebook {notebook_path}\nOutput will be saved to {output_path}")
        pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_path),
            kernel_name="python3",
        )
    except pm.PapermillExecutionError as e:
        pytest.fail(f"Notebook {notebook_path} failed during execution: {str(e)}")
