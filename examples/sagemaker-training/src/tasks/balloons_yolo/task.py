"""Task: YOLO on a COCO-style balloons dataset.

Mode A native — `build_tables()` materializes 3LC tables from raw
data mounted at the SageMaker channels each run.

Activate via the `task: balloons_yolo` hyperparameter in config.yaml.
"""
from __future__ import annotations

import argparse

import tlc
import tlc_ultralytics


def build_tables(args: argparse.Namespace) -> tuple[tlc.Table, tlc.Table]:
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
    return train_table, val_table


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
        workers=4,
        verbose=True,
        project=args.model_dir,
    )
