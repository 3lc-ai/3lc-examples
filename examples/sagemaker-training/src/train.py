"""Training entry point executed inside the SageMaker container.

Three labeled sections:

  1. SageMaker plumbing — boilerplate (arg parsing, env var reads,
     paths). Reusable as-is in any SageMaker training script.

  2. 3LC integration — the ~10 lines that make this template
     3LC-aware. Marked with `# 3LC ↓` / `# ↑ 3LC` fences.

  3. Task dispatch — import the task module named by the `task`
     hyperparameter and call its `build_tables` / `train` functions.

Tasks live in src/tasks/<name>/ as Python packages. Each one exposes,
via its __init__.py:

    build_tables(args) -> (train_table, val_table)
    train(args, train_table, val_table) -> None

Out-of-job (Mode B) table creators live inside the task package as
src/tasks/<name>/create_tables.py. They run on the laptop, not in the
container.

Two table-acquisition modes:

  A. Build fresh from raw data each run (default — calls the task's
     build_tables()).
  B. Load existing tables by URL — set --train_table_url and
     --val_table_url; build_tables() is skipped.

See README §"Tasks" for the conventions and §"Adding a task" for
how to plug in your own.

launch.py ships config.3lc.yaml into the container and sets
TLC_CONFIG_FILE, so `tlc` auto-discovers it on import.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import tlc  # 3LC

# ============================================================
# 1. SageMaker plumbing
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Hyperparameters — SageMaker passes these as --<key> <value>. Names
    # must match the keys under `hyperparameters:` in config.yaml.
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Module name under src/tasks/, e.g. 'balloons_yolo'.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, required=True)
    # "cpu" for local mode on Apple Silicon, "0" or "cuda" for remote GPU instances.
    parser.add_argument("--device", type=str, default="cpu")
    # SageMaker passes hyperparameters as strings, so bool-like flags need
    # a string→bool coercer rather than action="store_true".
    parser.add_argument(
        "--use_latest",
        type=lambda s: str(s).lower() in ("true", "1", "yes"),
        default=False,
    )
    # Optional: skip task.build_tables() and load existing tables by URL.
    # Empty string = unset (SageMaker stringifies all hyperparameters).
    parser.add_argument("--train_table_url", type=str, default="")
    parser.add_argument("--val_table_url", type=str, default="")

    # SageMaker-provided paths. Defaults let this script run standalone
    # outside a SageMaker container too.
    parser.add_argument(
        "--train",
        type=Path,
        default=Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")),
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path(os.environ.get("SM_CHANNEL_VAL", "/opt/ml/input/data/val")),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model")),
    )
    return parser.parse_args()


# ============================================================
# 2. 3LC integration
# ============================================================
# Two responsibilities, intentionally separate:
#   setup_project_aliases() — project-level setup, idempotent, persisted on S3
#   load_or_build_tables()  — per-job table acquisition (build fresh or load by URL)


def setup_project_aliases(project: str) -> None:
    """Persist project-scoped URL aliases on S3.

    Forward-looking: gives future readers of the project (the 3LC UI,
    downstream jobs, the Object Service) a default resolution target
    for whichever alias tokens our serialized tables embed.

    Static aliases in config.3lc.yaml drive what *this* job embeds
    when serializing tables. Project-persisted aliases (these) drive
    how those tokens *resolve* later — same token names, opposite
    directions.

    `force=True` is required precisely *because* the names match: the
    static alias from config.3lc.yaml already binds these tokens (to the
    container mount paths) in this session, so registering them at the
    S3 targets is a deliberate change of an existing binding, not a
    fresh claim. Still idempotent across runs — re-registering the same
    S3 target is a no-op write.

    Call this LAST, after training. Registering also flips this
    session's alias binding to the S3 targets, and the trainer needs the
    tokens to keep resolving to the LOCAL mounted channels while it reads
    image files (the tlc-ultralytics loader only accepts local paths).
    """
    tlc.helpers.ProjectHelper.register_project_url_alias(
        "SM_TRAIN_INPUT_DATA", os.environ["TRAIN_S3_URI"], project_name=project, force=True
    )
    tlc.helpers.ProjectHelper.register_project_url_alias(
        "SM_VAL_INPUT_DATA", os.environ["VAL_S3_URI"], project_name=project, force=True
    )

    print("Registered URL aliases:")
    for alias, value in tlc.url.get_registered_url_aliases().items():
        print(f"  {alias}: {value}")


def load_or_build_tables(args: argparse.Namespace, build_tables_fn) -> tuple[tlc.Table, tlc.Table]:
    """Mode A: call task's build_tables(). Mode B: load existing by URL."""
    if args.train_table_url and args.val_table_url:
        print(f"Loading existing tables: {args.train_table_url} / {args.val_table_url}")
        train_table = tlc.Table.from_url(args.train_table_url)
        val_table = tlc.Table.from_url(args.val_table_url)
    else:
        print("Building fresh tables from mounted channels.")
        train_table, val_table = build_tables_fn(args)

    # Follow each table to its latest revision so training picks up edits
    # made in the 3LC UI between runs (re-labeling, filter changes, etc.).
    if args.use_latest:
        train_table = train_table.latest()
        val_table = val_table.latest()
    return train_table, val_table


# ============================================================
# 3. Task dispatch
# ============================================================


def main() -> None:
    args = parse_args()
    print(f"task={args.task} project={args.project} device={args.device}")
    print(f"train channel: {args.train}")
    print(f"val channel:   {args.val}")
    print(f"model dir:     {args.model_dir}")

    task = importlib.import_module(f"tasks.{args.task}")

    # 3LC ↓ — build (or load) tables. The static aliases from
    # config.3lc.yaml mask the container mount paths, so the serialized
    # tables embed alias tokens (<SM_TRAIN_INPUT_DATA>/...) instead of
    # ephemeral /opt/ml/... paths.
    train_table, val_table = load_or_build_tables(args, task.build_tables)
    print(f"train table: {train_table}")
    print(f"val table:   {val_table}")
    # ↑ 3LC

    # Training reads the tables, resolving those tokens back to the LOCAL
    # mounted channels (still the active binding) so the trainer sees
    # local image files.
    task.train(args, train_table, val_table)

    # 3LC ↓ — only now persist the S3 resolution targets for future
    # readers (the 3LC UI, downstream jobs). This is deliberately last:
    # registering also flips this session's alias binding to S3, which
    # must not happen until the trainer is done reading local images.
    setup_project_aliases(args.project)
    # ↑ 3LC


if __name__ == "__main__":
    main()
