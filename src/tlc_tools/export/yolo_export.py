from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import tlc
import yaml
from tlc.core.data_formats.bounding_boxes import CenteredXYWHBoundingBox
from tqdm import tqdm

from tlc_tools.common import is_windows


def _parse_label_column(label_column: str | None) -> tuple[str, str, str]:
    """Parse a label column path into its three components.

    The label column path is dot-separated: "bb_column.bb_list_key.label_key"
    If fewer than 3 parts are provided, defaults are filled in:
    - 1 part: "my_bb_column" -> ("my_bb_column", "bb_list", "label")
    - 2 parts: "my_bb_column.my_list" -> ("my_bb_column", "my_list", "label")
    - 3 parts: "my_bb_column.my_list.my_label" -> ("my_bb_column", "my_list", "my_label")
    - None: uses defaults ("bbs", "bb_list", "label")

    :param label_column: The dot-separated label column path, or None for defaults.
    :returns: A tuple of (bb_column, bb_list_key, label_key).
    """
    default_bb_column = tlc.BOUNDING_BOXES  # "bbs"
    default_bb_list_key = tlc.BOUNDING_BOX_LIST  # "bb_list"
    default_label_key = tlc.LABEL  # "label"

    if label_column is None:
        return (default_bb_column, default_bb_list_key, default_label_key)

    parts = label_column.split(".")

    if len(parts) == 1:
        return (parts[0], default_bb_list_key, default_label_key)
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
    """Export the bounding boxes from a set of tables to a YOLO dataset.

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
    :param label_column: Dot-separated path to the label field within the bounding box structure.
        Can specify 1-3 parts: "bb_column", "bb_column.bb_list_key", or "bb_column.bb_list_key.label_key".
        Missing parts are filled with defaults: "bbs", "bb_list", "label".
        For example, "my_boxes" becomes "my_boxes.bb_list.label".
    """

    print(f"Exporting {len(tables)} tables to YOLO dataset at {output_url}...")

    # TODO: support writing to s3 as well (use tlc.Url everywhere)
    # TODO: consider supply categories? (subset e.g.)

    # Parse column names with defaults
    if image_column is None:
        image_column = tlc.IMAGE
    bb_column, bb_list_key, label_key = _parse_label_column(label_column)
    label_column_path = f"{bb_column}.{bb_list_key}.{label_key}"

    if is_windows() and image_strategy == "symlink":
        msg = "Symlinking images is not supported on Windows, choose a different 'image_strategy'."
        raise ValueError(msg)

    if image_strategy is None:
        print(
            "WARNING: No image strategy provided, defaulting to 'copy'. "
            " If your dataset is large, consider using 'symlink', 'move' or 'ignore'."
        )
        image_strategy = "copy"

    tables_dict: dict[str, tlc.Table] = {}
    for split in tables:
        if isinstance(tables[split], (str, Path, tlc.Url)):
            tables_dict[split] = tlc.Table.from_url(str(tables[split]))
        else:
            tables_dict[split] = tables[split]

        _verify_table_schema(tables_dict[split], image_column, bb_column, bb_list_key, label_key)

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
        bb_schema = table.schema.values["rows"].values[bb_column].values[bb_list_key]
        bb_type = tlc.BoundingBox.from_schema(bb_schema)

        for row in tqdm(table.table_rows, desc=f"Exporting {split} split", total=len(table)):
            image_width = row.get(tlc.WIDTH, row[bb_column][tlc.IMAGE_WIDTH])
            image_height = row.get(tlc.HEIGHT, row[bb_column][tlc.IMAGE_HEIGHT])

            image_path = Path(tlc.Url(row[image_column]).to_absolute().to_str())  # Resolve aliases
            bounding_box_dicts = row[bb_column][bb_list_key]

            # Read out the bounding boxes
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

                line = f"{bounding_box_dict[label_key]} {' '.join(str(coordinate) for coordinate in bounding_box_xywh)}"
                lines.append(line)

            # <output path>/images/train/0000.jpg
            # <output path>/labels/train/0000.txt
            output_image_path = images_path / split / image_path.name

            # Handle image according to the strategy
            if image_strategy != "ignore":
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                output_image_path = _handle_image(image_path, output_image_path, image_strategy)

            # Write the label file if there are any bounding boxes
            output_label_path = _image_path_to_label_path(output_image_path)
            if lines:
                output_label_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_label_path, "w") as f:
                    f.write("\n".join(lines))

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
    image_strategy: Literal["ignore", "copy", "symlink", "move"],
) -> Path:
    # Handle conflicting image name where output image already exists
    if output_image_path.exists():
        output_image_path = _handle_conflicting_image_name(output_image_path)

    if image_strategy == "copy":
        shutil.copy(original_image_path, output_image_path)
    elif image_strategy == "symlink":
        output_image_path.symlink_to(original_image_path)
    elif image_strategy == "move":
        shutil.move(original_image_path, output_image_path)
    elif image_strategy == "ignore":
        pass
    else:
        msg = f"Invalid image_strategy: {image_strategy}"
        raise ValueError(msg)

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
    bb_column: str,
    bb_list_key: str,
    label_key: str,
) -> None:
    """Verify that the table has the expected schema.

    :param table: The table to verify.
    :param image_column: The name of the image column.
    :param bb_column: The name of the bounding box column.
    :param bb_list_key: The key for the bounding box list within the bb column.
    :param label_key: The key for labels within each bounding box.
    """
    # Get the first row of the table and use to check if the table is in the correct format
    row = table.table_rows[0]

    # Check if row has all the required top-level keys
    required_keys = [image_column, bb_column]
    for key in required_keys:
        if key not in row:
            msg = f"Table does not have the required key: {key}"
            raise ValueError(msg)

    # Check if bb column is a Mapping with the bb_list key
    bbs = row[bb_column]
    if not isinstance(bbs, dict):
        msg = f"Bounding boxes column '{bb_column}' must be a dict, got: {type(bbs)}"
        raise ValueError(msg)
    if bb_list_key not in bbs:
        msg = f"Bounding boxes column '{bb_column}' must have a '{bb_list_key}' key."
        raise ValueError(msg)
    bb_list = bbs[bb_list_key]

    # Check if bounding_boxes is a list of dicts with the required keys
    required_bounding_box_keys = ["x0", "y0", "x1", "y1", label_key]
    for bounding_box in bb_list:
        for key in required_bounding_box_keys:
            if key not in bounding_box:
                msg = f"Bounding box does not have the required key: {key}"
                raise ValueError(msg)

    # Check if the table has a value-mapping for its labels.
    label_column_path = f"{bb_column}.{bb_list_key}.{label_key}"
    if table.get_value_map(label_column_path) is None:
        msg = f"Table does not have a value-mapping for its bounding box labels at '{label_column_path}'."
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

    yolo_dataset_path = Path(__file__).parent.parent / "data" / "yolo"

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
