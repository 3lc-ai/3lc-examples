"""Offline (Mode B) table seeding for the pyronear_yolo task.

Pulls pyronear/pyro-sdis from HuggingFace, transforms annotations,
writes 3LC tables, and persists a project-scoped URL alias so the
path tokens embedded in the tables resolve later.

Run on your laptop (not in the training container):

    uv run python src/tasks/pyronear_yolo/create_tables.py

Then copy the printed URLs into config.yaml:

    hyperparameters:
      task: pyronear_yolo
      train_table_url: <printed train url>
      val_table_url:   <printed val url>

Resolution caveat: the SageMaker container needs to resolve the
`<PYRONEAR>` tokens that the tables embed. Set BULK_DATA_ROOT to
an S3 URI (and ensure write credentials), or upload the data to S3
afterwards and re-register the alias with `force=True`.
"""
from __future__ import annotations

import datasets
import tlc

# CUSTOMIZE: project/dataset names and where bulk image data lives.
# BULK_DATA_ROOT can be local (laptop iteration) or s3:// (so a
# SageMaker training job can read the same tables).
PROJECT_NAME = "hf-pyronear"
DATASET_NAME = "pyro-sdis"
BULK_DATA_ROOT = "/Users/gudbrand/Data/pyro-sdis"  # or "s3://my-bucket/pyronear"
ALIAS_NAME = "PYRONEAR"


def anns_to_3lc_bbs(anns: str, image_w: int, image_h: int) -> tlc.data_types.BoundingBoxes2D:
    """Parse a pyro-sdis annotation string (label cx cy w h, normalized) into 3LC bboxes."""
    values = anns.split()
    boxes: list[list[float]] = []
    labels: list[int] = []
    for i in range(0, len(values), 5):
        labels.append(int(values[i]) - 1)  # 0-indexed labels
        cx, cy, w, h = (float(v) for v in values[i + 1 : i + 5])
        boxes.append([cx, cy, w, h])
    return tlc.data_types.BoundingBoxes2D(
        image_width=image_w,
        image_height=image_h,
        bboxes=boxes,
        labels=labels,
        bbox_format="cxywh",
        normalized=True,
    )


def main() -> None:
    # Persist the alias *into the project* so future readers (the 3LC UI,
    # the SageMaker training job, downstream consumers) can resolve the
    # `<PYRONEAR>` tokens that end up in serialized tables.
    tlc.register_project_url_alias(ALIAS_NAME, BULK_DATA_ROOT, project=PROJECT_NAME, force=False)

    ds_dict = datasets.load_dataset("pyronear/pyro-sdis")
    for split in ["train", "val"]:
        bulk_data_location = f"{BULK_DATA_ROOT}/{split}"
        writer = tlc.TableWriter(
            table_name=split,
            project_name=PROJECT_NAME,
            dataset_name=DATASET_NAME,
            if_exists="overwrite",
            schema={
                "image": tlc.schemas.ImageSchema(format="jpeg", bulk_data_location=bulk_data_location),
                "bbs": tlc.schemas.BoundingBoxes2DSchema(["fire"]),
                "weight": tlc.schemas.SampleWeightSchema(),
            },
        )
        ds = ds_dict[split]
        for i in range(len(ds)):
            row = ds[i]
            image = row["image"]
            bbs = anns_to_3lc_bbs(row["annotations"], image.width, image.height)
            writer.add_row({"image": image, "bbs": bbs})
        table = writer.finalize()
        print(f"{split}: {len(table.table_rows)} rows -> {table.url}")


if __name__ == "__main__":
    main()
