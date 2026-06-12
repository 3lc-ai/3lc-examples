# CLAUDE.md

## Project Overview

This repository contains examples, tutorials, and CLI tools for the **3LC** data platform. It provides:

- ~61 Jupyter notebooks demonstrating 3LC usage across ML workflows
- A Python package (`tlc_tools`) with CLI tools extending 3LC functionality
- Sample datasets for tutorials
- Integrations with PyTorch, HuggingFace, Detectron2, Ultralytics YOLO, PyTorch Lightning, and Super-Gradients

**Package:** `3lc_tools` v2.19 | **License:** Apache 2.0 | **Python:** 3.9 - 3.12

## Repository Structure

```
├── src/tlc_tools/           # Main Python package
│   ├── cli/                 # CLI framework (entry point: main.py, registry.py)
│   │   └── commands/        # Individual CLI commands
│   ├── alias/               # Find/replace alias tool
│   ├── augment_bbs/         # Bounding box augmentation and instance metrics
│   ├── export/              # Export to YOLO format
│   ├── common.py            # Shared utilities (device inference, validation)
│   ├── embeddings.py        # Embedding generation/collection
│   ├── metrics.py           # Metric calculations (diversity, clustering)
│   ├── split.py             # Data splitting strategies
│   └── ...
├── tutorials/               # Jupyter notebooks organized by category
│   ├── 1-create-tables/     # 21 notebooks: table creation from various sources
│   ├── 2-modify-tables/     # 13 notebooks: adding metrics, embeddings, splitting
│   ├── 3-training-and-metrics/ # 25 notebooks: training with various frameworks
│   └── 4-end-to-end-examples/ # 2 notebooks: complete workflows
├── examples/                # Additional example notebooks
├── tests/                   # Pytest test suite
│   ├── cli/                 # CLI command tests
│   └── tools/               # Tool-specific tests (alias, instance metrics, metric jumps)
├── data/                    # Sample datasets (COCO, balloons, cats-and-dogs, etc.)
└── utils/                   # Image normalization utilities for tutorial images
```

## Development Setup

```bash
# Install base package (editable)
pip install -e .

# Install with specific extras
pip install -e .[huggingface,umap]

# Install all optional dependencies
pip install -e .[all]

# Install dev dependencies (using uv)
uv sync --all-extras --dev
```

The project uses **uv** as its package manager (lock file: `uv.lock`).

## Commands

### Linting & Formatting

```bash
ruff check .                    # Lint (rules: B, E, F, UP, SIM, I)
ruff check --fix .              # Lint with auto-fix
ruff format .                   # Format code
```

### Type Checking

```bash
mypy .
```

### Testing

```bash
pytest tests --durations=0 -v   # Run all tests
pytest tests -m "not slow"      # Skip slow tests
pytest tests -m "not dependent" # Skip tests needing external resources
```

Tests require the `TLC_API_KEY` environment variable for some tests.

### Pre-commit

```bash
pre-commit run --all-files      # Run all hooks manually
```

Hooks configured:
- **ruff** (v0.11.4): lint with `--fix` + format
- **nbstripout** (v0.8.1): strip notebook outputs (excludes `example-notebooks/`)

### CLI Tool

```bash
3lc-tools                       # Main CLI entry point
```

## Code Style & Conventions

- **Line length:** 120 characters
- **Indent:** 4 spaces
- **Target Python:** 3.9
- **Linting rules:** B (bugbear), E (pycodestyle), F (pyflakes), UP (pyupgrade), SIM (simplify), I (isort)
- **Import sorting:** `tlc` is classified as known third-party
- **Ruff scope:** `src/tlc_tools/**/*.py`, `tests/**/*.py`, `pyproject.toml` (excludes `example-notebooks/`)
- **Type hints:** Used in source code; avoided in notebooks unless using 3LC-specific types

### 3LC Code Annotations

When showcasing 3LC-specific code in training notebooks, wrap it with:

```python
################## 3LC ##################
# 3LC-specific code here
#########################################
```

## Notebook Conventions

Notebooks must follow this structure:

1. **Title** with description and prerequisites
2. **Imports** (H2) - all imports in a single cell at the top
3. **Project Setup** (H2) - configuration parameters (tagged with `parameters` for papermill)
4. Additional **H2 sections** as needed
5. Optional: **Suggested Next Steps** section (bullet list)

Key rules:
- **No outputs on commit** - nbstripout pre-commit hook enforces this
- **Sanity checks** after loading data
- **Liberal comments** - explain code purpose
- **Info boxes** using HTML divs (blue=info, green=success, yellow=warning)
- **Data attribution** required for any new datasets in `/data`

## CI/CD

GitHub Actions workflow (`.github/workflows/lint.yml`) runs on PRs to `main`:

1. Pre-commit hooks (ruff lint/format + nbstripout)
2. MyPy type checking
3. Pytest test suite

Environment: Ubuntu latest, Python 3.9 (from `.python-version-ci`), uv 0.7.13.

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| 3lc | >=2.19.0, <3.0.0 | Core 3LC platform |
| opencv-python | >=4.10, <5.0 | Computer vision ops |
| scikit-learn | >=1.5.2, <2.0 | ML utilities |
| fpsample | >=0.3.3, <1.0 | Furthest point sampling |
| ruff | >=0.9.0, <1.0 | Linting & formatting |
| mypy | >=1.13, <2.0 | Type checking |
| pytest | >=8.3.3, <9.0 | Testing |
| papermill | >=2.6.0, <3.0 | Notebook execution |

## Testing Patterns

- Tests use **pytest** with fixtures and parametrized tests
- **Markers:** `slow` (long-running), `dependent` (needs external resources)
- **Mocking** via `pytest-mock` for external dependencies
- **Coverage** configured in `.coveragerc` (source: `src/tlc_tools`)
- Test structure mirrors source: `tests/cli/`, `tests/tools/`

## Important Notes

- The `ultralytics` dependency points to a custom 3LC fork: `git+https://github.com/3lc-ai/ultralytics`
- The `pandaset` dependency uses a specific git source: `git+https://github.com/scaleapi/pandaset-devkit.git`
- Build system uses **Hatchling** with wheel packages from `src/tlc_tools`
- The `example-notebooks/` directory is excluded from both ruff linting and nbstripout
