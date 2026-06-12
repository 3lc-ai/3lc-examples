#!/usr/bin/env python
"""Launch a 3LC-instrumented SageMaker training job.

Usage:
    uv run launch.py            # remote job on the instance type in config.yaml
    uv run launch.py --local    # same code, run in Docker on this machine

Reads config.yaml for all user-specific settings. Re-tars src/ on every call,
so edits to train.py are picked up without any rebuild step.

The non-obvious bits (each has a comment at the relevant line):

- TLC_CONFIG_FILE is set to the *in-container* path of config.3lc.yaml so
  the `tlc` library finds it on import (URL aliases, etc.).
- code_location and output_path are set explicitly so the source tarball,
  model artifacts, and 3LC outputs all live under the same prefix instead
  of SageMaker's default bucket.
- TRAIN_S3_URI / VAL_S3_URI env vars carry the original S3 locations of
  the mounted channels, so train.py can register URL aliases that make
  3LC tables resolvable after the container exits.
- channel_s3_uris() is the single source of truth for what each channel
  maps to; both TrainingInput (SageMaker) and env vars (3LC) read from it.
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
        raise SystemExit(f"Missing {CONFIG_PATH}. Copy config.example.yaml to config.yaml and fill it in.")
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)  # type: ignore


def s3_uri(bucket: str, *parts: str) -> str:
    path = "/".join(p.strip("/") for p in parts if p)
    return f"s3://{bucket}/{path}" if path else f"s3://{bucket}"


def channel_s3_uris(cfg: dict) -> dict[str, str]:
    """Per-channel remote S3 URIs — single source of truth for what each
    channel mounts. Used both for TrainingInput and for env var forwarding."""
    s3 = cfg["s3"]
    base = s3_uri(s3["inputs_bucket"], s3.get("inputs_prefix", ""))
    return {
        "train": f"{base}/train/",
        "val": f"{base}/val/",
    }


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
    session: sagemaker.Session
    if local:
        session = sagemaker.local.LocalSession(boto_session=boto_session)
    else:
        session = sagemaker.Session(boto_session=boto_session)

    # Env vars forwarded into the container. OUTPUTS_* lets train.py write
    # run metrics to a predictable S3 path using boto3. TRAIN_S3_URI /
    # VAL_S3_URI carry the remote S3 source of each mounted channel —
    # useful when training code needs to reference original S3 paths
    # (e.g. for 3LC tables). The rest are user-defined vars from config.yaml.
    channels = channel_s3_uris(cfg)
    env = {
        "OUTPUTS_BUCKET": s3["outputs_bucket"],
        "OUTPUTS_PREFIX": s3.get("outputs_prefix", ""),
        "TRAIN_S3_URI": channels["train"],
        "VAL_S3_URI": channels["val"],
        **{k: str(v) for k, v in (cfg.get("env") or {}).items()},
    }

    # Ship config.3lc.yaml into the container alongside train.py if present.
    # `dependencies` files land in the same dir as the entry_point script
    # (/opt/ml/code/ under SageMaker script mode). Point TLC_CONFIG_FILE at
    # it so the `tlc` library picks it up automatically on import.
    dependencies = []
    if TLC_CONFIG_PATH.exists():
        dependencies.append(str(TLC_CONFIG_PATH))
        env["TLC_CONFIG_FILE"] = f"/opt/ml/code/{TLC_CONFIG_PATH.name}"

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
    # Each channel mounts its own split's S3 prefix; train.py sees them
    # at SM_CHANNEL_TRAIN / SM_CHANNEL_VAL.
    #
    # FastFile streams objects on demand; avoids a full download and is
    # the right default for image datasets. Local mode can't use FastFile,
    # so fall back to File (downloads into the container).
    mode = "File" if local else "FastFile"
    return {name: TrainingInput(s3_data=uri, input_mode=mode) for name, uri in channel_s3_uris(cfg).items()}


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
        assert estimator.latest_training_job is not None
        job_name = estimator.latest_training_job.name
        print()
        print(f"Job name:        {job_name}")
        print(f"Model artifact:  {estimator.model_data}")


if __name__ == "__main__":
    main()
