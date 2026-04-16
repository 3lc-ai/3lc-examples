# SageMaker Training Workflow Bootstrap

A minimal, end-to-end SageMaker training setup. Edit `src/train.py`, run
one command for a local Docker smoke test, run the same command with one
flag flipped to launch a real remote job.

```
launch.py                   # run locally: orchestrates the job
config.example.yaml         # template → copy to config.yaml (gitignored)
config.3lc.example.yaml     # template → copy to config.3lc.yaml (gitignored)
pyproject.toml              # laptop deps for launch.py (managed by uv)
src/
├── train.py                # runs inside the training container
└── requirements.txt        # container deps installed before train.py runs
```

---

## 1. One-time AWS setup

### IAM role for SageMaker

Create an IAM role that SageMaker assumes to run jobs. You'll paste its
ARN into `config.yaml`.

1. In the AWS console, go to **IAM → Roles → Create role**.
2. Select trusted entity **AWS service → SageMaker**. Click through; AWS
   attaches `AmazonSageMakerFullAccess` by default. That's fine as a
   starting point.
3. Attach a second policy granting access to your buckets. Inline is
   fine:

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": ["s3:GetObject", "s3:ListBucket"],
          "Resource": [
            "arn:aws:s3:::<your-inputs-bucket>",
            "arn:aws:s3:::<your-inputs-bucket>/*"
          ]
        },
        {
          "Effect": "Allow",
          "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
          "Resource": [
            "arn:aws:s3:::<your-outputs-bucket>",
            "arn:aws:s3:::<your-outputs-bucket>/*"
          ]
        }
      ]
    }
    ```

   If inputs and outputs are the same bucket, collapse into a single
   statement with `Get*`, `Put*`, `ListBucket`.
4. Name the role something like `SageMakerExecutionRole`.
5. Copy the role ARN — `IAM → Roles → <your role> → ARN` at the top.

### Local tooling

```bash
uv sync
```

AWS credentials must be resolvable by boto3 (any of: `~/.aws/credentials`,
`aws sso login`, `AWS_PROFILE` env var, instance profile). Local mode
*also* runs the container with these credentials, so `train.py`'s boto3
call to write metrics works locally too.

For `--local`, install **Docker Desktop** (or Colima / OrbStack) and make
sure `docker info` works before running.

---

## 2. Configure

```bash
cp config.example.yaml config.yaml
cp config.3lc.example.yaml config.3lc.yaml   # optional; leave empty if unused
```

Fill in `config.yaml`:
- `aws.region`, `aws.role_arn`
- `s3.inputs_bucket` + `inputs_prefix`
- `s3.outputs_bucket` + `outputs_prefix` (can be the same bucket)
- Adjust `training.instance_type`, hyperparameters, and env vars as needed

Both `config.yaml` and `config.3lc.yaml` are gitignored.

---

## 3. Upload a dataset

`launch.py` expects this layout under `s3://<inputs_bucket>/<inputs_prefix>/`:

```
annotations/
  instances_train.json
  instances_val.json
images/
  train/
    000000000001.jpg
    ...
  val/
    000000000042.jpg
    ...
```

Quick upload:

```bash
aws s3 sync ./local-dataset/ s3://<inputs_bucket>/<inputs_prefix>/
```

Both `train` and `val` channels mount the dataset **root** inside the
container (so relative paths inside the annotations JSON files resolve
correctly). Your training code reads `SM_CHANNEL_TRAIN` and
`SM_CHANNEL_VAL` and picks the appropriate annotations file + image
subdir for each split.

---

## 4. Run

### Local smoke test (Docker)

```bash
uv run launch.py --local
```

This runs the exact same training image locally in Docker. Channel data
is downloaded to a container volume (FastFile isn't available locally).
Use this to iterate on `train.py` without waiting on SageMaker provisioning.

### Remote job

```bash
uv run launch.py
```

The SDK re-tars `src/` on every call, so `train.py` edits are picked up
with no rebuild step. Watch the job in the SageMaker console → Training
jobs, or tail logs from the CLI:

```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

---

## 5. Outputs

After a remote run, `launch.py` prints the two paths you care about:

- **Model artifact**: `s3://<outputs_bucket>/<outputs_prefix>/models/<job-name>/output/model.tar.gz`
  (SageMaker tars everything under `SM_MODEL_DIR` and uploads it.)
- **Run metrics**: `s3://<outputs_bucket>/<outputs_prefix>/runs/<job-name>/metrics.json`
  (Written by `train.py` directly with boto3 — see `write_metrics()`.)

---

## Apple Silicon note

SageMaker's local mode runs x86_64 Linux containers. On Apple Silicon
Macs these run under Rosetta 2 / QEMU emulation — functional but slow
(often 10×+ slower than a real x86 box). Good enough to prove the data
path and metric upload end-to-end; **not** good for actual training
iterations. Do real training on a remote instance.

Make sure Docker Desktop has **"Use Rosetta for x86_64/amd64 emulation
on Apple Silicon"** enabled (Settings → General).

---

## Extending

- **Hyperparameters**: add a key under `hyperparameters:` in `config.yaml`
  and a matching `--<key>` argparse flag in `train.py`. SageMaker wires
  the two together.
- **Env vars**: add under `env:` in `config.yaml`. They land in the
  container environment; read with `os.environ`. The existing
  `PROJECT_NAME` / `TLC_API_KEY` entries show the pattern.
- **3LC config**: put your 3LC settings in `config.3lc.yaml`. `launch.py`
  ships the file into the container next to `train.py` when it's present.
- **Spot instances, checkpointing, multi-GPU**: not wired up; add
  `use_spot_instances`, `max_wait`, `checkpoint_s3_uri` to the `PyTorch(...)`
  call in `launch.py` when needed.
