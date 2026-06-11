"""Guard that tutorials/notebooks.yaml stays in sync with notebook content.

This eliminates the "forgot to regenerate metadata" failure mode: if a notebook's
title/blurb/tags/thumbnail/parameters change without running
``python -m utils.notebooks_metadata sync``, this test (and CI) fails.
"""

from pathlib import Path

from utils import notebooks_metadata as nm

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS_DIR = REPO_ROOT / "tutorials"
NOTEBOOKS_YAML = TUTORIALS_DIR / "notebooks.yaml"


def test_notebooks_yaml_in_sync_with_notebook_content() -> None:
    problems = nm.check(TUTORIALS_DIR, NOTEBOOKS_YAML)
    assert not problems, (
        "tutorials/notebooks.yaml is out of date with notebook content. "
        "Run `python -m utils.notebooks_metadata sync`.\n\n" + "\n".join(problems)
    )


def test_every_notebook_has_an_entry() -> None:
    entries = nm.load_yaml(NOTEBOOKS_YAML)
    yaml_paths = {e["path"] for e in entries}
    actual_paths = {p.relative_to(TUTORIALS_DIR).as_posix() for p in nm.iter_notebook_paths(TUTORIALS_DIR)}
    assert yaml_paths == actual_paths
