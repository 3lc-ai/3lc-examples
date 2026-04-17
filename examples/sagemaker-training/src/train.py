"""Training entry point executed inside the SageMaker container.

Two things happening here:

1. SageMaker plumbing
   - Hyperparameters arrive as CLI flags (keys must match config.yaml).
   - Input channels mounted under /opt/ml/input/data/<channel>/ and exposed
     as SM_CHANNEL_<CHANNEL> env vars.
   - Anything under SM_MODEL_DIR is tar.gz'd and uploaded to output_path
     automatically; ultralytics is pointed there via `project=args.model_dir`.

2. 3LC wiring
   - launch.py ships config.3lc.yaml into the container and sets
     TLC_CONFIG_FILE, so `tlc` auto-discovers it on import.
   - The static `aliases:` block in config.3lc.yaml drives what *this* job
     embeds into its serialized tables (typical use: masking SageMaker
     mount points like /opt/ml/input/data and /opt/ml/model).
   - create_tables() *persists* project-scoped aliases (SM_TRAIN_INPUT_DATA
     / SM_VAL_INPUT_DATA) onto the project on S3. This is forward-looking:
     it gives future readers (the 3LC UI, downstream jobs, the Object
     Service) a default resolution target for whatever alias tokens appear
     in the tables. It does not drive *this* job's table creation.
   - Tables and runs materialize on S3 via `indexing.project_root_url` in
     the 3LC config, not by SageMaker.

CUSTOMIZE when cloning: everything in create_tables() and the YOLO block in
main() is task-specific. The rest (arg parsing, env var reading, model_dir
handling) is reusable plumbing.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import tlc
import tlc_ultralytics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Hyperparameters — SageMaker passes these as --<key> <value>. Names
    # must match the keys under `hyperparameters:` in config.yaml.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
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

def create_tables(args: argparse.Namespace) -> tuple[tlc.Table, tlc.Table]:
    # Persist project-scoped URL aliases on S3. This is forward-looking:
    # it gives *future* readers of the project (the 3LC UI, downstream
    # jobs) a default resolution target for whichever alias tokens our
    # serialized tables end up using. The aliases that actually drive
    # *this* job's table creation come from the local config.3lc.yaml
    # (shipped into the container by launch.py) — those take precedence
    # over anything persisted on the project.
    train_s3_uri = os.environ["TRAIN_S3_URI"]
    val_s3_uri = os.environ["VAL_S3_URI"]
    tlc.register_project_url_alias(
        "SM_TRAIN_INPUT_DATA", train_s3_uri, project=args.project, force=False
    )
    tlc.register_project_url_alias(
        "SM_VAL_INPUT_DATA", val_s3_uri, project=args.project, force=False
    )

    print("Registered URL aliases:")
    for alias, value in tlc.get_registered_url_aliases().items():
        print(f"  {alias}: {value}")

    # CUSTOMIZE: dataset_name and annotations filenames are task-specific.
    train_table = tlc.Table.from_coco(
        annotations_file=args.train / "train-annotations.json",
        image_folder=args.train,
        table_name="train",
        dataset_name="balloons",
        project_name=args.project,
    )
    val_table = tlc.Table.from_coco(
        annotations_file=args.val / "val-annotations.json",
        image_folder=args.val,
        table_name="val",
        dataset_name="balloons",
        project_name=args.project,
    )
    # Follow each table to its latest revision so training picks up edits
    # made in the 3LC UI between runs (re-labeling, filter changes, etc.).
    if args.use_latest:
        train_table = train_table.latest()
        val_table = val_table.latest()
    return train_table, val_table

def main() -> None:
    args = parse_args()

    # Secrets stay as env vars; training knobs (project, device, ...) are
    # hyperparameters so they show up in the SageMaker console.
    tlc_api_key_set = bool(os.environ.get("TLC_API_KEY"))
    outputs_bucket = os.environ["OUTPUTS_BUCKET"]
    outputs_prefix = os.environ.get("OUTPUTS_PREFIX", "")
    print(f"project={args.project} device={args.device} TLC_API_KEY_set={tlc_api_key_set}")
    print(f"outputs: s3://{outputs_bucket}/{outputs_prefix}")

    print(f"train channel: {args.train}")
    print(f"val channel:   {args.val}")
    print(f"model dir:     {args.model_dir}")

    train_table, val_table = create_tables(args)
    print(f"train table: {train_table}")
    print(f"val table:   {val_table}")

    # CUSTOMIZE: the entire YOLO block is task-specific. Swap in your
    # model/framework here. Keep `project=args.model_dir` so artifacts
    # land under SM_MODEL_DIR and get uploaded to S3 by SageMaker.
    yolo = tlc_ultralytics.YOLO("yolo11s.pt")
    settings = tlc_ultralytics.Settings(project_name=args.project)
    yolo.train(
        tables={"train": train_table, "val": val_table},
        settings=settings,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=640,
        device=args.device,
        workers=4,
        verbose=True,
        project=args.model_dir,
    )


if __name__ == "__main__":
    main()
