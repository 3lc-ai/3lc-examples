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
import os
from tqdm import tqdm
os.environ["AWS_PROFILE"] = "dev"
# CUSTOMIZE: project/dataset names and where bulk image data lives.
# BULK_DATA_ROOT can be local (laptop iteration) or s3:// (so a
# SageMaker training job can read the same tables).
PROJECT_NAME = "hf-pyronear"
DATASET_NAME = "pyro-sdis"
PROJECT_ROOT_URL = "s3://3lc-projects"
BULK_DATA_ROOT = "s3://3lc-projects/data/pyronear"
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
        bounding_boxes=boxes,
        labels=labels,
        bounding_box_format="cxywh",
        normalized=True,
    )

def get_camera_and_partner_maps(ds_dict: datasets.DatasetDict) -> tuple[dict[str, int], dict[str, int]]:
    cameras = set()
    partners = set()

    for split in ds_dict.values():
        cameras.update(split.unique("camera"))
        partners.update(split.unique("partner"))

    camera_map = {v: i for i, v in enumerate(sorted(cameras))}
    partner_map = {v: i for i, v in enumerate(sorted(partners))}

    return camera_map, partner_map

def main() -> None:
    # Persist the alias *into the project* so future readers (the 3LC UI,
    # the SageMaker training job, downstream consumers) can resolve the
    # `<PYRONEAR>` tokens that end up in serialized tables.
    tlc.helpers.ProjectHelper.register_project_url_alias(
        ALIAS_NAME,
        BULK_DATA_ROOT,
        project_name=PROJECT_NAME,
        root_url=PROJECT_ROOT_URL,
        force=False,
    )

    ds_dict = datasets.load_dataset("pyronear/pyro-sdis")
    camera_map, partner_map = get_camera_and_partner_maps(ds_dict)

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
                "date": tlc.schemas.DatetimeStringSchema(),
                "camera": tlc.schemas.CategoricalLabelSchema(classes=list(camera_map.keys())),
                "partner": tlc.schemas.CategoricalLabelSchema(classes=list(partner_map.keys())),
            },
            root_url=PROJECT_ROOT_URL,
        )
        ds = ds_dict[split]

        for i in tqdm(range(len(ds)), desc=f"Writing {split} table"):
            row = ds[i]
            camera = camera_map[row["camera"]]
            partner = partner_map[row["partner"]]
            image = row["image"]
            bbs = anns_to_3lc_bbs(row["annotations"], image.width, image.height)
            writer.add_row({"image": image, "bbs": bbs, "date": row["date"], "camera": camera, "partner": partner})
        table = writer.finalize()
        print(f"{split}: {len(table)} rows -> {table.url}")


if __name__ == "__main__":
    main()
