#!/usr/bin/env python
"""Launch a SageMaker training job.

Usage:
    python launch.py            # remote job on the instance type in config.yaml
    python launch.py --local    # same code, run in Docker on this machine

Reads config.yaml for all user-specific settings. Re-tars src/ on every
call, so edits to train.py are picked up without any rebuild step.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import boto3
import sagemaker
import yaml
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
TLC_CONFIG_PATH = ROOT / "config.3lc.yaml"
SOURCE_DIR = ROOT / "src"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"Missing {CONFIG_PATH}. Copy config.example.yaml to config.yaml and fill it in."
        )
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def s3_uri(bucket: str, *parts: str) -> str:
    path = "/".join(p.strip("/") for p in parts if p)
    return f"s3://{bucket}/{path}" if path else f"s3://{bucket}"


def build_estimator(cfg: dict, local: bool) -> PyTorch:
    aws = cfg["aws"]
    s3 = cfg["s3"]
    training = cfg["training"]

    # Local mode: "local" = CPU, "local_gpu" = host NVIDIA GPU. On Apple
    # Silicon only "local" works (amd64 emulation, CPU-only — slow, for
    # smoke tests and debugging only).
    instance_type = "local" if local else training["instance_type"]

    # Bind the session to the configured region so the SDK doesn't depend
    # on AWS_DEFAULT_REGION / ~/.aws/config being set. Local mode still
    # needs a region to look up the DLC image URI from ECR. `profile`
    # is optional — falls back to the default credential chain if unset.
    boto_session = boto3.Session(
        region_name=aws["region"],
        profile_name=aws.get("profile"),
    )
    if local:
        session = sagemaker.local.LocalSession(boto_session=boto_session)
    else:
        session = sagemaker.Session(boto_session=boto_session)

    # Env vars forwarded into the container. OUTPUTS_* lets train.py write
    # run metrics to a predictable S3 path using boto3. The rest are the
    # user-defined vars (TLC_API_KEY, PROJECT_NAME, ...) from config.yaml.
    env = {
        "OUTPUTS_BUCKET": s3["outputs_bucket"],
        "OUTPUTS_PREFIX": s3.get("outputs_prefix", ""),
        **{k: str(v) for k, v in (cfg.get("env") or {}).items()},
    }

    # Ship config.3lc.yaml into the container alongside train.py if present.
    # `dependencies` files land in the same dir as the entry_point script.
    dependencies = [str(TLC_CONFIG_PATH)] if TLC_CONFIG_PATH.exists() else []

    return PyTorch(
        entry_point="train.py",
        source_dir=str(SOURCE_DIR),
        dependencies=dependencies,
        role=aws["role_arn"],
        instance_type=instance_type,
        instance_count=1 if local else training.get("instance_count", 1),
        framework_version=training["framework_version"],
        py_version=training["py_version"],
        hyperparameters=cfg.get("hyperparameters") or {},
        environment=env,
        base_job_name=training.get("job_base_name", "training"),
        max_run=training.get("max_run_seconds", 3600),
        output_path=s3_uri(s3["outputs_bucket"], s3.get("outputs_prefix", ""), "models"),
        # Keep the source tarball under the same bucket/prefix as outputs.
        # Without this, SageMaker SDK uses Session.default_bucket() which
        # can resolve unexpectedly (sagemaker.config.yaml, auto-created
        # sagemaker-<region>-<account> bucket, etc.).
        code_location=s3_uri(s3["outputs_bucket"], s3.get("outputs_prefix", ""), "code"),
        sagemaker_session=session,
    )


def build_inputs(cfg: dict, local: bool) -> dict[str, TrainingInput]:
    s3 = cfg["s3"]
    dataset_root = s3_uri(s3["inputs_bucket"], s3.get("inputs_prefix", "")) + "/"

    # Both channels mount the dataset root so that annotations files
    # (which reference images/train/... and images/val/...) resolve
    # correctly. train.py reads SM_CHANNEL_TRAIN and SM_CHANNEL_VAL and
    # picks the right annotations + image subdir for each split.
    #
    # FastFile streams objects on demand; avoids a full download and is
    # the right default for image datasets. Local mode can't use FastFile,
    # so fall back to File (downloads into the container).
    mode = "File" if local else "FastFile"
    return {
        "train": TrainingInput(s3_data=dataset_root, input_mode=mode),
        "val": TrainingInput(s3_data=dataset_root, input_mode=mode),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run the job in SageMaker local mode (Docker on this machine).",
    )
    args = parser.parse_args()

    cfg = load_config()
    estimator = build_estimator(cfg, local=args.local)
    inputs = build_inputs(cfg, local=args.local)

    estimator.fit(inputs)

    if not args.local:
        job_name = estimator.latest_training_job.name
        s3cfg = cfg["s3"]
        print()
        print(f"Job name:        {job_name}")
        print(f"Model artifact:  {estimator.model_data}")
        print(
            "Metrics (if train.py wrote them): "
            + s3_uri(
                s3cfg["outputs_bucket"],
                s3cfg.get("outputs_prefix", ""),
                "runs",
                job_name,
                "metrics.json",
            )
        )


if __name__ == "__main__":
    main()
