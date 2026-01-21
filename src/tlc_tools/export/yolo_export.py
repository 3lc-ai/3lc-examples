from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import tlc
import yaml
from tlc.client.sample_type import InstanceSegmentationPolygons
from tlc.core.data_formats.bounding_boxes import CenteredXYWHBoundingBox
from tqdm import tqdm

from tlc_tools.common import is_windows

# Task type constants
TASK_DETECT = "detect"
TASK_SEGMENT = "segment"


def _infer_task_type(table: tlc.Table, label_column: str) -> str:
    """Infer the task type (detection or segmentation) from the table schema.

    :param table: The table to infer the task type from.
    :param label_column: The name of the label column (top-level, e.g. "bbs" or "segmentations").
    :returns: Either TASK_DETECT or TASK_SEGMENT.
    """
    table.ensure_complete_schema()
    column_schema = table.schema.values["rows"].values.get(label_column)

    if column_schema is None:
        msg = f"Column '{label_column}' not found in table schema."
        raise ValueError(msg)

    sample_type = getattr(column_schema, "sample_type", None)

    if sample_type and sample_type.startswith("instance_segmentation"):
        return TASK_SEGMENT

    # Default to detection (includes bounding box schemas)
    return TASK_DETECT


def _parse_label_column(label_column: str | None, task_type: str) -> tuple[str, str, str]:
    """Parse a label column path into its three components.

    For detection, the path is: "bb_column.bb_list_key.label_key"
    For segmentation, the path is: "seg_column.instance_properties.label_key"

    If fewer than 3 parts are provided, defaults are filled in based on task type:
    - Detection defaults: ("bbs", "bb_list", "label")
    - Segmentation defaults: ("segmentations", "instance_properties", "label")

    :param label_column: The dot-separated label column path, or None for defaults.
    :param task_type: The task type (TASK_DETECT or TASK_SEGMENT).
    :returns: A tuple of (column, list_key, label_key).
    """
    if task_type == TASK_SEGMENT:
        default_column = tlc.SEGMENTATIONS  # "segmentations"
        default_list_key = tlc.INSTANCE_PROPERTIES  # "instance_properties"
    else:
        default_column = tlc.BOUNDING_BOXES  # "bbs"
        default_list_key = tlc.BOUNDING_BOX_LIST  # "bb_list"

    default_label_key = tlc.LABEL  # "label"

    if label_column is None:
        return (default_column, default_list_key, default_label_key)

    parts = label_column.split(".")

    if len(parts) == 1:
        return (parts[0], default_list_key, default_label_key)
    elif len(parts) == 2:
        return (parts[0], parts[1], default_label_key)
    elif len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    else:
        msg = f"label_column must have at most 3 dot-separated parts, got {len(parts)}: '{label_column}'"
        raise ValueError(msg)


def export_to_yolo(
    tables: Mapping[str, str | Path | tlc.Url | tlc.Table],
    output_url: tlc.Url | Path | str,
    dataset_name: str = "dataset",
    image_strategy: Literal["ignore", "copy", "symlink", "move"] | None = None,
    image_column: str | None = None,
    label_column: str | None = None,
) -> None:
    """Export bounding boxes or instance segmentations from a set of tables to a YOLO dataset.

    The task type (detection or segmentation) is automatically inferred from the table schema.
    For detection, writes bounding boxes in YOLO format: <class> <x_center> <y_center> <width> <height>
    For segmentation, writes polygons in YOLO format: <class> <x1> <y1> <x2> <y2> ... <xn> <yn>

    The function writes out all of the labels, a YOLO dataset yaml file, and optionally copies
    all the images to the output directory, to the following structure:

    |- <output url>
    | |- <dataset_name>.yaml
    | |- images
    | | |- image1.jpg
    | | |- image2.jpg
    | | |- ...
    | |- labels
    | | |- image1.txt
    | | |- image2.txt
    | | |- ...

    Note therefore that any original folder structure is not preserved, and all images are placed
    in the same directory. Resulting file name conflicts are resolved by adding a number to the end of the name,
    which means that even original file names may not be preserved.

    :param tables: The mapping from split names to tables to export as a YOLO dataset.
    :param output_url: The location to export the dataset to.
    :param dataset_name: The name of the dataset, used to name the yaml file.
    :param image_strategy: The strategy to use for handling images. Options are:
        - "ignore": Do not copy or symlink images.
        - "copy": Copy images to the output directory, keeping the original images. This is done by default.
        - "symlink": Create symlinks to the images in the output directory.
        - "move": Move images to the output directory, removing the original images.
    :param image_column: The name of the column containing image paths. Defaults to "image".
    :param label_column: Dot-separated path to the label field within the label structure.
        For detection: "bb_column.bb_list_key.label_key" (defaults: "bbs.bb_list.label")
        For segmentation: "seg_column.instance_properties.label_key"
        (defaults: "segmentations.instance_properties.label").
        Can specify 1-3 parts, missing parts are filled with defaults based on the inferred task type.
    """
    print(f"Exporting {len(tables)} tables to YOLO dataset at {output_url}...")

    # TODO: support writing to s3 as well (use tlc.Url everywhere)
    output_url = tlc.Url(output_url)

    if output_url.scheme == tlc.Scheme.RELATIVE:
        output_url = output_url.absolute_from_relative(tlc.Url(Path.cwd()))
    elif output_url.scheme != tlc.Scheme.FILE:
        msg = (
            f"Output URL must be a local file or relative path, got: {output_url}. Can not export to remote locations."
        )
        raise ValueError(msg)

    # TODO: consider supply categories? (subset e.g.)

    if image_column is None:
        image_column = tlc.IMAGE

    if is_windows() and image_strategy == "symlink":
        msg = "Symlinking images is not supported on Windows, choose a different 'image_strategy'."
        raise ValueError(msg)

    if image_strategy is None:
        print(
            "WARNING: No image strategy provided, defaulting to 'copy'. "
            " If your dataset is large, consider using 'symlink', 'move' or 'ignore'."
        )
        image_strategy = "copy"

    # Load all tables
    tables_dict: dict[str, tlc.Table] = {}
    for split in tables:
        if isinstance(tables[split], (str, Path, tlc.Url)):
            tables_dict[split] = tlc.Table.from_url(str(tables[split]))
        else:
            tables_dict[split] = tables[split]

    # Infer task type from the first table
    first_table = next(iter(tables_dict.values()))

    # Determine the top-level column name for task type inference
    if label_column is not None:
        top_level_column = label_column.split(".")[0]
    else:
        # Check which default column exists in the table
        first_table.ensure_complete_schema()
        row_schema = first_table.schema.values["rows"].values
        top_level_column = tlc.SEGMENTATIONS if tlc.SEGMENTATIONS in row_schema else tlc.BOUNDING_BOXES

    task_type = _infer_task_type(first_table, top_level_column)
    print(f"Inferred task type: {task_type}")

    # Parse column names with task-specific defaults
    data_column, list_key, label_key = _parse_label_column(label_column, task_type)
    label_column_path = f"{data_column}.{list_key}.{label_key}"

    # Verify all tables have the expected schema
    for table in tables_dict.values():
        _verify_table_schema(table, image_column, data_column, list_key, label_key, task_type)

    output_url = tlc.Url(output_url)

    output_path = Path(output_url.to_str())
    if output_path.exists():
        msg = f"Output url {output_url} already exists, can only export to a new location."
        raise ValueError(msg)

    if output_path.suffix != "":
        msg = f"Output url must be a directory, not a file, got: {output_url}."
        raise ValueError(msg)

    output_path.mkdir(parents=True)

    images_path = output_path / "images"

    # Iterate over the dataset and write labels to the output url, based on the filenames
    for split, table in tables_dict.items():
        table.ensure_complete_schema()

        # Create the appropriate row converter for this task type
        if task_type == TASK_DETECT:
            row_to_lines = _create_detection_row_converter(table, data_column, list_key, label_key)
        else:
            row_to_lines = _create_segmentation_row_converter(table, data_column, list_key, label_key, image_column)

        _export_split(table, split, images_path, image_column, row_to_lines, image_strategy)

    # Write dataset yaml file
    categories = tlc.SchemaHelper.to_simple_value_map(table.get_value_map(label_column_path))

    yaml_content = {
        "path": output_path.absolute().as_posix(),
        **{split: (images_path / split).relative_to(output_path).as_posix() for split in tables_dict},
        "names": categories,
    }

    dataset_yaml_file = output_path / f"{dataset_name}.yaml"
    with open(dataset_yaml_file, "w", encoding="utf-8") as f:
        f.write("# YOLO Dataset YAML file - Created from 3LC Tables:\n")
        for split, table in tables_dict.items():
            f.write(f"# {split}: {table.url}\n")
        f.write("\n")

        if image_strategy == "ignore":
            helper_str = 'image_strategy="ignore"'
            f.write(
                f"# NOTE! Images were not copied to the output directory because {helper_str},"
                "the dataset can therefore not be used as-is. Copy or move the images manually.\n"
            )
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False, line_break="\n\n")

    print("Finished exporting to YOLO dataset. Dataset yaml file written to:", dataset_yaml_file)


def _export_split(
    table: tlc.Table,
    split: str,
    images_path: Path,
    image_column: str,
    row_to_lines: Callable[[dict], list[str]],
    image_strategy: Literal["ignore", "copy", "symlink", "move"],
) -> None:
    """Export a single split to YOLO format.

    :param table: The table to export.
    :param split: The name of the split (e.g. "train", "val").
    :param images_path: The base path for images output.
    :param image_column: The name of the image column.
    :param row_to_lines: A callable that converts a table row to YOLO label lines.
    :param image_strategy: The strategy for handling images.
    """
    for row in tqdm(table.table_rows, desc=f"Exporting {split} split", total=len(table)):
        image_path = Path(tlc.Url(row[image_column]).to_absolute().to_str())  # Resolve aliases
        lines = row_to_lines(row)

        output_image_path = images_path / split / image_path.name

        # Handle image according to the strategy
        if image_strategy != "ignore":
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            output_image_path = _handle_image(image_path, output_image_path, image_strategy)

        # Write the label file if there are any labels
        output_label_path = _image_path_to_label_path(output_image_path)
        if lines:
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_label_path, "w") as f:
                f.write("\n".join(lines))


def _create_detection_row_converter(
    table: tlc.Table,
    bb_column: str,
    bb_list_key: str,
    label_key: str,
) -> Callable[[dict], list[str]]:
    """Create a row-to-lines converter for detection data.

    :returns: A callable that takes a row dict and returns YOLO detection label lines.
    """
    bb_schema = table.schema.values["rows"].values[bb_column].values[bb_list_key]
    bb_type = tlc.BoundingBox.from_schema(bb_schema)

    def convert(row: dict) -> list[str]:
        image_width = row.get(tlc.WIDTH, row[bb_column][tlc.IMAGE_WIDTH])
        image_height = row.get(tlc.HEIGHT, row[bb_column][tlc.IMAGE_HEIGHT])
        bounding_box_dicts = row[bb_column][bb_list_key]

        lines = []
        for bounding_box_dict in bounding_box_dicts:
            bounding_box = (
                bb_type(
                    [
                        bounding_box_dict["x0"],
                        bounding_box_dict["y0"],
                        bounding_box_dict["x1"],
                        bounding_box_dict["y1"],
                    ]
                )
                .normalize(image_width, image_height)
                .to_top_left_xywh()
            )
            bounding_box_xywh = CenteredXYWHBoundingBox.from_top_left_xywh(bounding_box)
            line = f"{bounding_box_dict[label_key]} {' '.join(str(c) for c in bounding_box_xywh)}"
            lines.append(line)

        return lines

    return convert


def _create_segmentation_row_converter(
    table: tlc.Table,
    seg_column: str,
    instance_props_key: str,
    label_key: str,
    image_column: str,
) -> Callable[[dict], list[str]]:
    """Create a row-to-lines converter for segmentation data.

    YOLO segmentation format: <class> <x1> <y1> <x2> <y2> ... <xn> <yn>
    where coordinates are normalized to [0, 1].

    :returns: A callable that takes a row dict and returns YOLO segmentation label lines.
    """
    from tlc.client.sample_type import SampleType

    seg_schema = table.schema.values["rows"].values[seg_column]

    # Create sample type with relative=True to ensure normalized coordinates for YOLO,
    # regardless of what the original schema specifies.
    sample_type = InstanceSegmentationPolygons(
        name=seg_column,
        instance_properties_structure={
            k: SampleType.from_schema(v) for k, v in seg_schema.values[tlc.INSTANCE_PROPERTIES].values.items()
        },
        relative=True,  # YOLO always needs normalized coordinates
    )

    # Get label value map for logging class names
    label_column_path = f"{seg_column}.{instance_props_key}.{label_key}"
    label_value_map = table.get_value_map(label_column_path)
    label_value_map = tlc.SchemaHelper.to_simple_value_map(label_value_map) if label_value_map else {}

    # Minimum 3 points (6 coordinates) required for a valid polygon
    MIN_POLYGON_COORDS = 6

    def convert(row: dict) -> list[str]:
        seg_data = row[seg_column]
        image_url = row[image_column]

        # Convert internal RLE format to polygons using the sample type
        polygons_data = sample_type.sample_from_row(seg_data)
        polygons = polygons_data[tlc.POLYGONS]
        labels = polygons_data[instance_props_key][label_key]

        lines = []
        for idx, (label, polygon) in enumerate(zip(labels, polygons)):
            # Skip empty polygons
            if not polygon:
                continue

            # Skip degenerate polygons (fewer than 3 points)
            # This can happen due to lossy mask-to-polygon conversion
            if len(polygon) < MIN_POLYGON_COORDS:
                class_name = label_value_map.get(label, f"class_{label}")
                print(
                    f"WARNING: Skipping degenerate polygon (instance {idx}, class '{class_name}') "
                    f"with {len(polygon) // 2} points in image: {image_url}"
                )
                continue

            coords_str = " ".join(str(coord) for coord in polygon)
            line = f"{label} {coords_str}"
            lines.append(line)

        return lines

    return convert


def _image_path_to_label_path(full_path: Path) -> Path:
    # Replace the last occurrence of 'images' with 'labels', if there is one
    parts = list(full_path.parts)

    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break

    return Path(*parts).with_suffix(".txt")


def _handle_image(
    original_image_path: Path,
    output_image_path: Path,
    image_strategy: Literal["copy", "symlink", "move"],
) -> Path:
    """Handle copying/symlinking/moving an image to the output path.

    Note: This function is only called when image_strategy is not "ignore".
    """
    # Handle conflicting image name where output image already exists
    if output_image_path.exists():
        output_image_path = _handle_conflicting_image_name(output_image_path)

    if image_strategy == "copy":
        shutil.copy(original_image_path, output_image_path)
    elif image_strategy == "symlink":
        output_image_path.symlink_to(original_image_path)
    elif image_strategy == "move":
        shutil.move(original_image_path, output_image_path)

    return output_image_path


def _handle_conflicting_image_name(output_image_path: Path) -> Path:
    # Find the next available filename by adding a number to the end
    i = 1
    while output_image_path.exists() and i < 10000:
        output_image_path = output_image_path.with_name(f"{output_image_path.stem}_{i:04d}{output_image_path.suffix}")
        i += 1

    return output_image_path


def _verify_table_schema(
    table: tlc.Table,
    image_column: str,
    data_column: str,
    list_key: str,
    label_key: str,
    task_type: str,
) -> None:
    """Verify that the table has the expected schema for the given task type.

    :param table: The table to verify.
    :param image_column: The name of the image column.
    :param data_column: The name of the data column (bbs for detection, segmentations for segmentation).
    :param list_key: The key for the list within the data column.
    :param label_key: The key for labels.
    :param task_type: The task type (TASK_DETECT or TASK_SEGMENT).
    """
    # Get the first row of the table and use to check if the table is in the correct format
    row = table.table_rows[0]

    # Check if row has all the required top-level keys
    required_keys = [image_column, data_column]
    for key in required_keys:
        if key not in row:
            msg = f"Table does not have the required key: {key}"
            raise ValueError(msg)

    data = row[data_column]
    if not isinstance(data, dict):
        msg = f"Column '{data_column}' must be a dict, got: {type(data)}"
        raise ValueError(msg)

    if task_type == TASK_DETECT:
        _verify_detection_schema(data, data_column, list_key, label_key)
    else:
        _verify_segmentation_schema(data, data_column, list_key, label_key)

    # Check if the table has a value-mapping for its labels.
    label_column_path = f"{data_column}.{list_key}.{label_key}"
    if table.get_value_map(label_column_path) is None:
        msg = f"Table does not have a value-mapping for its labels at '{label_column_path}'."
        raise ValueError(msg)


def _verify_detection_schema(data: dict, data_column: str, list_key: str, label_key: str) -> None:
    """Verify detection-specific schema requirements."""
    if list_key not in data:
        msg = f"Bounding boxes column '{data_column}' must have a '{list_key}' key."
        raise ValueError(msg)
    bb_list = data[list_key]

    # Check if bounding_boxes is a list of dicts with the required keys
    required_bounding_box_keys = ["x0", "y0", "x1", "y1", label_key]
    for bounding_box in bb_list:
        for key in required_bounding_box_keys:
            if key not in bounding_box:
                msg = f"Bounding box does not have the required key: {key}"
                raise ValueError(msg)


def _verify_segmentation_schema(data: dict, data_column: str, list_key: str, label_key: str) -> None:
    """Verify segmentation-specific schema requirements."""
    # Segmentation data has: image_height, image_width, instance_properties, rles
    required_keys = [tlc.IMAGE_HEIGHT, tlc.IMAGE_WIDTH, tlc.RLES]
    for key in required_keys:
        if key not in data:
            msg = f"Segmentation column '{data_column}' must have a '{key}' key."
            raise ValueError(msg)

    # Check instance_properties has the label key
    if list_key not in data:
        msg = f"Segmentation column '{data_column}' must have a '{list_key}' key."
        raise ValueError(msg)

    instance_props = data[list_key]
    if not isinstance(instance_props, dict):
        msg = f"'{list_key}' in column '{data_column}' must be a dict, got: {type(instance_props)}"
        raise ValueError(msg)

    if label_key not in instance_props:
        msg = f"'{list_key}' must have a '{label_key}' key for labels."
        raise ValueError(msg)


def main(args):
    table = tlc.Table.from_url(args.table_url)
    export_to_yolo(table, args.output_url)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Export bounding boxes to YOLO labels.")
    # parser.add_argument("--table-url", type=str, help="The url of the table to export labels from.")
    # parser.add_argument("--output-url", type=str, help="The url to write the labels to.")

    # args = parser.parse_args()

    # main(args)

    yolo_dataset_path = Path(__file__).parent.parent.parent.parent / "data" / "yolo"

    # table = tlc.TableFromYolo(
    #     input_url=tlc.Url(yolo_dataset_path / "simple.yaml"),
    #     split="train",
    # )
    table = tlc.TableFromCoco(
        input_url=tlc.Url(yolo_dataset_path.parent / "coco128" / "annotations.json"),
        image_folder_url=tlc.Url(yolo_dataset_path.parent / "coco128" / "images"),
    )
    output_url = yolo_dataset_path.parent / "yolo_exported"

    export_to_yolo(
        tables={"train": table, "val": table},
        output_url=output_url,
        dataset_name="simple",
        image_strategy="symlink",
    )

    # Segmentation example

    table = tlc.TableFromCoco(
        input_url=tlc.Url(yolo_dataset_path.parent / "coco128" / "annotations.json"),
        image_folder_url=tlc.Url(yolo_dataset_path.parent / "coco128" / "images"),
        task="segment",
    )

    output_url = yolo_dataset_path.parent / "yolo_exported_segmentation"
    export_to_yolo(
        tables={"train": table, "val": table},
        output_url=output_url,
        dataset_name="simple",
        image_strategy="symlink",
    )
