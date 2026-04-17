"""Training entry point executed inside the SageMaker container.

SageMaker contract used here:
- Hyperparameters arrive as CLI flags (keys must match config.yaml).
- Input channels are mounted under /opt/ml/input/data/<channel>/ and the
  paths are exposed as SM_CHANNEL_<CHANNEL> env vars.
- Final model artifacts written under SM_MODEL_DIR are tar.gz'd and
  uploaded to output_path automatically (YOLO's `project=args.model_dir`
  routes ultralytics output there).
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

def create_tables(args: argparse.Namespace, outputs_bucket: str) -> tuple[tlc.Table, tlc.Table]:
    print("Registered URL aliases:")
    for alias, value in tlc.get_registered_url_aliases().items():
        print(f"{alias}: {value}")
    
    TRAIN_S3_URI = os.environ.get("TRAIN_S3_URI")
    VAL_S3_URI = os.environ.get("VAL_S3_URI")
    if not TRAIN_S3_URI or not VAL_S3_URI:
        raise ValueError("TRAIN_S3_URI and VAL_S3_URI must be set")
    
    tlc.register_project_url_alias(
        "SM_TRAIN_INPUT_DATA",
        TRAIN_S3_URI,
        project=args.project,
        force=False,
    )
    tlc.register_project_url_alias(
        "SM_VAL_INPUT_DATA",
        VAL_S3_URI,
        project=args.project,
        force=False,
    )

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

    # Shipped by launch.py as a `dependencies` file; lands next to this
    # script inside the container if config.3lc.yaml exists on the host.
    tlc_config = Path(__file__).parent / "config.3lc.yaml"
    if tlc_config.exists():
        print(f"3LC config available at: {tlc_config}")

    train_table, val_table = create_tables(args, outputs_bucket)
    print(f"train table: {train_table}")
    print(f"val table:   {val_table}")

    yolo = tlc_ultralytics.YOLO("yolo11n.pt")
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
