"""Training entry point executed inside the SageMaker container.

This file is intentionally a skeleton: it wires up the SageMaker-specific
plumbing so you can focus on the actual model + training loop.

SageMaker contract used here:
- Hyperparameters arrive as CLI flags (keys must match config.yaml).
- Input channels are mounted under /opt/ml/input/data/<channel>/ and the
  paths are exposed as SM_CHANNEL_<CHANNEL> env vars.
- Final model artifacts written under SM_MODEL_DIR are tar.gz'd and
  uploaded to output_path automatically.
- We additionally write a small run-metrics JSON directly to S3 with
  boto3, so observability doesn't depend on the model tar upload.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import tlc_ultralytics

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Hyperparameters — SageMaker passes these as --<key> <value>. Names
    # must match the keys under `hyperparameters:` in config.yaml.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.005)

    # SageMaker-provided paths. Defaults let this script run standalone
    # outside a SageMaker container too.
    parser.add_argument(
        "--train",
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--val",
        default=os.environ.get("SM_CHANNEL_VAL", "/opt/ml/input/data/val"),
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    return parser.parse_args()


def write_metrics(metrics: dict) -> None:
    """Upload run metrics to s3://<OUTPUTS_BUCKET>/<OUTPUTS_PREFIX>/runs/<job>/metrics.json."""
    bucket = os.environ.get("OUTPUTS_BUCKET")
    if not bucket:
        print("OUTPUTS_BUCKET not set; skipping metrics upload.")
        return
    prefix = os.environ.get("OUTPUTS_PREFIX", "").strip("/")
    job = os.environ.get("TRAINING_JOB_NAME", "local")
    key = "/".join(p for p in [prefix, "runs", job, "metrics.json"] if p)
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(metrics, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"Wrote metrics to s3://{bucket}/{key}")


def main() -> None:
    args = parse_args()
    started = time.time()

    # Custom env vars set via the estimator's `environment` kwarg in
    # launch.py. Treat API keys as secrets — don't log the value itself.
    project_name = os.environ.get("PROJECT_NAME", "<unset>")
    tlc_api_key_set = bool(os.environ.get("TLC_API_KEY"))
    print(f"PROJECT_NAME={project_name} TLC_API_KEY_set={tlc_api_key_set}")

    print(f"train channel: {args.train}")
    print(f"val channel:   {args.val}")
    print(f"model dir:     {args.model_dir}")

    # Shipped by launch.py as a `dependencies` file; lands next to this
    # script inside the container if config.3lc.yaml exists on the host.
    tlc_config = Path(__file__).parent / "config.3lc.yaml"
    if tlc_config.exists():
        print(f"3LC config available at: {tlc_config}")

    # ------------------------------------------------------------------
    # TODO: real training code goes here.
    # Read COCO annotations from args.train/annotations/instances_train.json,
    # images from args.train/images/train/, train your model for args.epochs,
    # save final weights under args.model_dir. Push per-epoch loss into
    # `loss_per_epoch` and populate `final_metrics` so they flow through
    # to metrics.json below.
    # ------------------------------------------------------------------
    loss_per_epoch: list[float] = []
    final_metrics: dict = {}

    # Make sure SM_MODEL_DIR isn't empty — SageMaker tars this directory,
    # and an empty tar is fine but a placeholder makes the upload visible
    # during the end-to-end plumbing test.
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.model_dir) / "placeholder.txt").write_text(
        "Replace with real model artifacts from train.py.\n"
    )

    metrics = {
        "job_name": os.environ.get("TRAINING_JOB_NAME", "local"),
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - started, 2),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "env": {"project_name": project_name},
        "loss_per_epoch": loss_per_epoch,
        "final": final_metrics,
    }
    write_metrics(metrics)


if __name__ == "__main__":
    main()
