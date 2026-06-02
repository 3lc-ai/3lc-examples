# Supervision Sample Type Example

Demonstrates how to interface 3LC with [Roboflow's `supervision`](https://supervision.roboflow.com/latest/detection/core/)
library by building a **custom sample type** that accepts `supervision.Detections` objects directly.

Columns using this sample type accept `sv.Detections` on write and return `sv.Detections` in sample view — while
storing the data in 3LC's standard 2D bounding-box format, so the Dashboard renders the boxes like any other detection
column.

The key idea: **don't invent a new wire format**. The sample type converts `sv.Detections` to the builtin
`BoundingBoxes2D` annotation dataclass and reuses its `to_row()` / `from_row()` / `schema()`. The only custom part is that the schema's `sample_type` points at us instead of `"bounding_boxes_2d"`, so
reads come back as `sv.Detections` instead of `BoundingBoxes2D`. This is the recommended pattern whenever you want a
third-party object to round-trip through a builtin 3LC annotation type.

## What it demonstrates

1. **Custom inline sample type** (`SupervisionDetectionsSampleType`): A `SampleType` subclass whose `to_row()` /
   `from_row()` delegate to the builtin `BoundingBoxes2D` annotation dataclass
2. **Accepting a third-party object**: `accepts()` recognizes live `sv.Detections` instances on write
3. **Reusing an annotation dataclass + its schema**: `schema()` builds on `BoundingBoxes2D.schema()` and just retargets
   `sample_type`, so the Dashboard renders boxes with zero extra work
4. **Entry-point registration**: `pip install` makes the type discoverable — a Table can be reopened in a fresh process
   (no explicit import) and still resolve to `sv.Detections`
5. **Round-trip**: boxes (`xyxy`), `class_id`, `confidence`, `tracker_id`, and image dimensions all survive

## Quick start

```bash
pip install -e .
create-supervision-demo-table
```

This creates a Table with 6 synthetic images and their detections. Open the 3LC Dashboard to browse it — the
`detections` column renders as ordinary bounding boxes, colored by class.

## What round-trips

| `supervision.Detections` field | 3LC representation                     | Notes                                              |
| ------------------------------ | -------------------------------------- | -------------------------------------------------- |
| `xyxy` (absolute pixels)       | `BoundingBoxes2D.bounding_boxes`       | xyxy is 3LC's native storage format — no conversion |
| `class_id`                     | per-instance label (against class map) | pass `classes=[...]` to `schema()` for names        |
| `confidence`                   | per-instance confidence                | enabled by `include_confidence=True` (default)      |
| `tracker_id`                   | per-instance extra `"tracker_id"`      | opt in with `include_tracker_id=True`               |
| image width / height           | scene bounds (`x_max` / `y_max`)       | see note below                                      |

### Image dimensions

`sv.Detections` does not carry the dimensions of the image it came from, but 3LC needs them as scene bounds so the
Dashboard knows the coordinate frame for the boxes. This example uses supervision's collection-level `metadata` dict to
carry them:

```python
detections.metadata["image_width"] = width
detections.metadata["image_height"] = height
```

If they are absent, the `image_width` / `image_height` defaults passed to `schema()` are used instead.

## Dashboard behavior

Because the stored wire format is byte-identical to the builtin `bounding_boxes_2d` type, the Dashboard treats this
column as a normal bounding-box column: boxes are drawn on the image, colored and filterable by class and confidence,
and editable. `tracker_id` is stored as a per-instance attribute.

## Structure

```
src/supervision_sample_type/
├── __init__.py           # Package exports
├── sample_type.py        # SupervisionDetectionsSampleType + sv.Detections <-> BoundingBoxes2D converters + schema()
├── create_demo_table.py  # Demo script generating synthetic images + detections
└── compat_check.py       # Compatibility report: supervision converters vs 3LC ops
```

## Using in your own project

```python
import supervision as sv
import tlc
from supervision_sample_type import SupervisionDetectionsSampleType

writer = tlc.TableWriter(
    project_name="My Detection Project",
    schema={
        "image": tlc.schemas.ImageSchema(sample_type="url"),
        "detections": SupervisionDetectionsSampleType.schema(
            classes=["person", "car", "dog"],
            include_confidence=True,
            include_tracker_id=True,
        ),
    },
)

# A typical supervision flow: run a model, wrap the result in sv.Detections
for image_path in image_paths:
    result = model(image_path)[0]
    detections = sv.Detections.from_ultralytics(result)
    width, height = ...  # from the image you ran inference on
    detections.metadata["image_width"] = width
    detections.metadata["image_height"] = height
    writer.add_row({"image": image_path, "detections": detections})

table = writer.finalize()

# Sample view: the detections column returns supervision Detections again
sample = table[0]
sample["detections"]                 # supervision.Detections
sample["detections"].xyxy            # numpy.ndarray, shape=(num_detections, 4)
sample["detections"].class_id        # numpy.ndarray of ints
```

## Helpers

The conversion functions are exported for direct use (e.g. converting an existing `BoundingBoxes2D` column to
`sv.Detections` without going through a Table):

```python
from supervision_sample_type import detections_to_bounding_boxes_2d, bounding_boxes_2d_to_detections
```

## Converter compatibility with 3LC

Supervision ships a large set of geometry [converter helpers](https://supervision.roboflow.com/latest/detection/utils/).
3LC has its own conversion ops but with a **narrower scope** — it covers *data representation* (box formats, mask /
polygon / RLE), not *analysis* (IoU, NMS, image transforms). Run the check yourself:

```bash
check-supervision-compatibility
```

### Where they correspond

| supervision                    | 3LC                                              | Result                                            |
| ------------------------------ | ------------------------------------------------ | ------------------------------------------------- |
| `xyxy_to_xywh`                 | `xyxy_to_xywh` / `BoundingBoxes2D.bounding_boxes_xywh` | ✅ exact match                              |
| `xywh_to_xyxy`                 | `BoundingBoxes2D(bounding_box_format="xywh")`    | ✅ exact match                                    |
| `xcycwh_to_xyxy`               | `BoundingBoxes2D(bounding_box_format="cxywh")`   | ✅ exact match (3LC calls center-xywh `cxywh`)    |
| `denormalize_boxes`            | `denormalize_bbs_2d` / `BoundingBoxes2D(normalized=True)` | ✅ exact match (3LC takes `width, height`) |
| `mask_to_polygons`             | `SegmentationHelper.polygons_from_mask`          | ✅ same contour points (different return shape)   |
| `mask_to_rle` / `rle_to_mask`  | `SegmentationHelper.rles_from_masks` / `masks_from_rles` | ⚠️ different RLE encodings; sv can decode 3LC's `pycocotools` bytes |
| `polygon_to_mask`              | `SegmentationHelper.mask_from_polygons`          | ⚠️ ~1px boundary difference (see below)           |
| `mask_to_xyxy`                 | `SegmentationHelper.bounding_box_from_rle`       | ⚠️ different format **and** boundary (see below)  |

### The one thing to watch: pixel-boundary convention

supervision treats the far box/polygon edge as **inclusive** (closed intervals); 3LC, via `pycocotools`, treats it as
**exclusive** (half-open). A filled mask `[10:40, 15:55]` becomes:

- `sv.mask_to_xyxy` → `[15, 10, 54, 39]` — **xyxy**, max = last filled pixel index
- `SegmentationHelper.bounding_box_from_rle` → `[15, 10, 40, 30]` — **xywh**, width = pixel count

These differ both in *format* (xyxy vs xywh) and by *one pixel* on the max edge. Likewise `polygon_to_mask` produces a
mask that is a strict subset of supervision's (it omits the far edge row/column). For round-tripping detections through
3LC this is irrelevant — boxes stay xyxy and never touch these helpers — but if you mix the two libraries' raster ops
directly, account for it.

### No 3LC equivalent (not full parity)

These supervision converters have **no 3LC counterpart** — use supervision (or your training framework) directly:

- **IoU**: `box_iou`, `box_iou_batch`, `mask_iou_batch`, `oriented_box_iou_batch`
- **NMS / merge**: `box_non_max_suppression`, `box_non_max_merge`, `mask_non_max_suppression`, `mask_non_max_merge`
- **Box arithmetic**: `clip_boxes`, `pad_boxes`, `scale_boxes`, `move_boxes`, `xyxy_to_xcycarh`
- **Polygon↔box**: `xyxy_to_polygons`, `polygon_to_xyxy`
- **Image transforms**: `crop_image`, `resize_image`, `scale_image`, `letterbox_image`, `grayscale_image`
- **Other mask ops**: `move_masks`, `calculate_masks_centroids`, `approximate_polygon`, `filter_polygons_by_area`

> Note: `SegmentationHelper.bounding_box_from_rle` is documented as returning `[x1, y1, x2, y2]` but actually returns
> `pycocotools` `[x, y, w, h]` — a docstring bug in 3LC (filed separately, not specific to this example).
