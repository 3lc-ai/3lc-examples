# Copyright 2026 3LC Inc. All rights reserved.

"""Create a demo 3LC Table from synthetic ``supervision`` detections.

Generates a handful of synthetic images (colored rectangles on a background) and a
matching :class:`supervision.Detections` per image, then writes them to a Table using
the custom ``supervision_detections`` sample type. Run with::

    create-supervision-demo-table

or::

    python -m supervision_sample_type.create_demo_table

"""

from __future__ import annotations

import numpy as np
import supervision as sv
import tlc
from PIL import Image, ImageDraw

from supervision_sample_type import SupervisionDetectionsSampleType
from supervision_sample_type.sample_type import IMAGE_HEIGHT_KEY, IMAGE_WIDTH_KEY

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 192
CLASSES = ["circle", "square", "triangle"]
# Per-class fill color for drawing the synthetic objects.
CLASS_COLORS = [(220, 70, 70), (70, 160, 220), (90, 200, 120)]
NUM_IMAGES = 6
RNG_SEED = 0


def _make_sample(rng: np.random.Generator, image_index: int) -> tuple[Image.Image, sv.Detections]:
    """Build one synthetic image and its matching supervision detections."""
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(30, 30, 40))
    draw = ImageDraw.Draw(image)

    num_objects = int(rng.integers(1, 4))  # 1-3 objects per image
    boxes: list[list[float]] = []
    class_ids: list[int] = []
    confidences: list[float] = []

    for _ in range(num_objects):
        box_w = int(rng.integers(30, 80))
        box_h = int(rng.integers(30, 70))
        x_min = int(rng.integers(0, IMAGE_WIDTH - box_w))
        y_min = int(rng.integers(0, IMAGE_HEIGHT - box_h))
        x_max, y_max = x_min + box_w, y_min + box_h
        class_id = int(rng.integers(0, len(CLASSES)))

        draw.rectangle([x_min, y_min, x_max, y_max], fill=CLASS_COLORS[class_id])
        boxes.append([x_min, y_min, x_max, y_max])
        class_ids.append(class_id)
        confidences.append(round(float(rng.uniform(0.5, 1.0)), 3))

    detections = sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        confidence=np.array(confidences, dtype=np.float32),
        # Stable tracker ids so the same object id can recur across the "video".
        tracker_id=np.arange(image_index * 10, image_index * 10 + num_objects, dtype=int),
    )
    # Carry the image dimensions so 3LC knows the coordinate frame for the boxes.
    detections.metadata[IMAGE_WIDTH_KEY] = IMAGE_WIDTH
    detections.metadata[IMAGE_HEIGHT_KEY] = IMAGE_HEIGHT

    return image, detections


def main() -> None:
    """Generate synthetic detections and write them to a 3LC Table."""
    rng = np.random.default_rng(RNG_SEED)

    writer = tlc.TableWriter(
        project_name="3LC Examples - Supervision Sample Type",
        dataset_name="synthetic-detections",
        table_name="initial",
        schema={
            "image": tlc.schemas.ImageSchema(sample_type="pil_png"),
            "detections": SupervisionDetectionsSampleType.schema(
                classes=CLASSES,
                include_confidence=True,
                include_tracker_id=True,
                image_width=IMAGE_WIDTH,
                image_height=IMAGE_HEIGHT,
                display_name="Detections",
            ),
        },
        if_exists="overwrite",
    )

    for image_index in range(NUM_IMAGES):
        image, detections = _make_sample(rng, image_index)
        writer.add_row({"image": image, "detections": detections})

    table = writer.finalize()

    print(f"Created table with {len(table)} rows")
    print(f"Table URL: {table.url}")
    print()

    # Round-trip check: sample view returns supervision Detections again.
    sample = table[0]
    detections = sample["detections"]
    print(f"Sample view type: {type(detections).__module__}.{type(detections).__name__}")
    print(f"  num detections : {len(detections)}")
    print(f"  xyxy[0]        : {detections.xyxy[0].tolist()}")
    print(f"  class_id       : {detections.class_id.tolist()}")
    print(f"  confidence     : {np.round(detections.confidence, 3).tolist()}")
    print(f"  tracker_id     : {detections.tracker_id.tolist()}")
    print(f"  metadata       : {detections.metadata}")


if __name__ == "__main__":
    main()
