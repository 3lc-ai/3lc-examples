"""Task: YOLO on the pyronear/pyro-sdis HuggingFace dataset.

Mode B native — tables are seeded out-of-job by
`create_tables.py` (sibling file), then loaded by URL via the
`train_table_url` / `val_table_url` hyperparameters.

Activate via the `task: pyronear_yolo` hyperparameter in config.yaml.
"""
from __future__ import annotations

import argparse

import tlc
import tlc_ultralytics


def build_tables(args: argparse.Namespace) -> tuple[tlc.Table, tlc.Table]:
    raise NotImplementedError(
        "pyronear_yolo is Mode-B only. Run `python src/tasks/pyronear_yolo/create_tables.py` "
        "to seed tables, then set train_table_url / val_table_url in config.yaml."
    )


def train(args: argparse.Namespace, train_table: tlc.Table, val_table: tlc.Table) -> None:
    yolo = tlc_ultralytics.YOLO("yolo11s.pt")
    settings = tlc_ultralytics.Settings(project_name=args.project)
    yolo.train(
        tables={"train": train_table, "val": val_table},
        settings=settings,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=640,
        device=args.device,
        workers=args.workers,
        verbose=True,
        project=args.model_dir,
    )

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.project = "hf-pyronear"
            self.epochs = 10
            self.batch_size = 8
            self.workers = 8
            self.device = "cuda"
            self.model_dir = None
            self.train_table_url = "s3://3lc-projects/hf-pyronear/datasets/pyro-sdis/tables/train"
            self.val_table_url = "s3://3lc-projects/hf-pyronear/datasets/pyro-sdis/tables/val"
    
    args = Args()
    # train_table, val_table = build_tables(args)
    train(args, args.train_table_url, args.val_table_url)