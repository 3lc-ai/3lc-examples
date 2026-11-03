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

Two consumers resolve the `<PYRONEAR>` tokens these tables embed, and
they have different needs:

- The **dashboard / Object Service** streams images, so an S3
  BULK_DATA_ROOT (the project-persisted default below) is fine.
- The **tlc-ultralytics trainer** reads image bytes from a *local*
  path only — it rejects `s3://` image URLs. So to actually train on
  SageMaker, mount the pyronear images into the container as a channel
  and add a static alias in `config.3lc.yaml` overriding `PYRONEAR` to
  that local mount (e.g. `PYRONEAR: /opt/ml/input/data/train`). The S3
  default persisted here still serves the dashboard. See the README
  "Tables vs. image bytes" callout.
"""

from __future__ import annotations

import datasets
import numpy as np
import tlc
from tqdm import tqdm

# AWS credentials are resolved from the default boto3/AWS chain
# (~/.aws/credentials, AWS_PROFILE, SSO, env vars). Set AWS_PROFILE in
# your shell if you need a specific named profile for the write to S3.

# CUSTOMIZE: project/dataset names and where bulk image data lives.
# BULK_DATA_ROOT is the project-persisted default the dashboard uses to
# stream images. S3 here is fine for viewing; training needs a local
# alias override in the container (see module docstring).
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
        bounding_boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        labels=np.array(labels, dtype=np.int32),
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
                "image": tlc.schemas.ImageSchema(sample_type="pil_jpeg", bulk_data_location=bulk_data_location),
                "bbs": tlc.data_types.BoundingBoxes2D.schema(classes=["fire"]),
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
