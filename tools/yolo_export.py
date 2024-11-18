import argparse
import shutil
from pathlib import Path
from typing import Literal, Mapping

import tlc
import yaml
from tlc.core.builtins.types.bounding_box import CenteredXYWHBoundingBox
from tqdm import tqdm

from tools.common import is_windows


def export_to_yolo(
    tables: Mapping[str, str | Path | tlc.Url | tlc.Table],
    output_url: tlc.Url | Path | str,
    dataset_name: str = "dataset",
    image_strategy: Literal["ignore", "copy", "symlink", "move"] | None = None,
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

    :param table: The mapping from split names to tables to export as a YOLO dataset.
    :param output_url: The location to export the dataset to.
    :param dataset_name: The name of the dataset, used to name the yaml file.
    :param image_strategy: The strategy to use for handling images. Options are:
        - "ignore": Do not copy or symlink images.
        - "copy": Copy images to the output directory, keeping the original images. This is done by default.
        - "symlink": Create symlinks to the images in the output directory.
        - "move": Move images to the output directory, removing the original images.
    """

    print(f"Exporting {len(tables)} tables to YOLO dataset at {output_url}...")

    # TODO: support writing to s3 as well (use tlc.Url everywhere)
    # TODO: consider supply categories? (subset e.g.)
    # TODO: consider overriding names of various fields (image, bbs, etc.)

    if is_windows() and image_strategy == "symlink":
        raise ValueError("Symlinking images is not supported on Windows, choose a different 'image_strategy'.")

    if image_strategy is None:
        print(
            "WARNING: No image strategy provided, defaulting to 'copy'. "
            " If your dataset is large, consider using 'symlink' or 'move'."
        )
        image_strategy = "copy"

    for split in tables:
        if isinstance(tables[split], (str, Path)):
            tables[split] = tlc.Table.from_url(str(tables[split]))

        _verify_table_schema(tables[split])

    output_url = tlc.Url(output_url)

    output_path = Path(output_url.to_str())
    if output_path.exists():
        msg = f"output url {output_url} already exists, can only export to a new location."
        raise ValueError(msg)

    if output_path.suffix != "":
        msg = f"output url must be a directory, not a file, got: {output_url}."
        raise ValueError(msg)

    output_path.mkdir(parents=True)

    images_path = output_path / "images"

    # Iterate over the dataset and write labels to the output url, based on the filenames
    for split, table in tables.items():
        table.ensure_complete_schema()
        bb_schema = table.schema.values["rows"].values[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST]
        bb_type = tlc.BoundingBox.from_schema(bb_schema)

        for row in tqdm(table.table_rows, desc=f"Exporting {split} split", total=len(table)):
            image_width = row.get(tlc.WIDTH, row[tlc.BOUNDING_BOXES][tlc.IMAGE_WIDTH])
            image_height = row.get(tlc.HEIGHT, row[tlc.BOUNDING_BOXES][tlc.IMAGE_HEIGHT])

            image_path = Path(tlc.Url(row["image"]).to_absolute().to_str())  # Resolve aliases
            bounding_box_dicts = row["bbs"]["bb_list"]

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

                line = f"{bounding_box_dict['label']} {' '.join(str(coordinate) for coordinate in bounding_box_xywh)}"
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
    categories = tlc.SchemaHelper.to_simple_value_map(table.get_value_map("bbs.bb_list.label"))

    yaml_content = {
        "path": output_path.absolute().as_posix(),
        **{split: (images_path / split).relative_to(output_path).as_posix() for split in tables},
        "names": categories,
    }

    dataset_yaml_file = output_path / f"{dataset_name}.yaml"
    with open(dataset_yaml_file, "w", encoding="utf-8") as f:
        f.write("# YOLO Dataset YAML file - Created from 3LC Tables:\n")
        for split, table in tables.items():
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
        raise ValueError(f"Invalid image_strategy: {image_strategy}")

    return output_image_path


def _handle_conflicting_image_name(output_image_path: Path) -> Path:
    # Find the next available filename by adding a number to the end
    i = 1
    while output_image_path.exists() and i < 10000:
        output_image_path = output_image_path.with_name(f"{output_image_path.stem}_{i:04d}{output_image_path.suffix}")
        i += 1

    return output_image_path


def _verify_table_schema(table: tlc.Table) -> None:
    # Verify that the table has the expected schema

    # Get the first row of the table and use to check if the table is in the correct format
    row = table.table_rows[0]

    # Check if row has all the required top-level keys
    required_keys = [tlc.IMAGE, tlc.BOUNDING_BOXES]
    for key in required_keys:
        if key not in row:
            raise ValueError(f"Table does not have the required key: {key}")

    # Check if bboxes is a Mapping with a bounding_boxes key
    if tlc.BOUNDING_BOXES not in row:
        raise ValueError(f"Table does not have a '{tlc.BOUNDING_BOXES}' key in its rows.")
    bbs = row[tlc.BOUNDING_BOXES]
    if not isinstance(bbs, dict):
        raise ValueError(f"Bounding boxes must be a dict, got: {type(bbs)}")
    if tlc.BOUNDING_BOX_LIST not in bbs:
        raise ValueError(f"Bounding boxes must have a '{tlc.BOUNDING_BOX_LIST}' key.")
    bb_list = bbs[tlc.BOUNDING_BOX_LIST]

    # Check if bounding_boxes is a list of dicts with the required keys
    required_bounding_box_keys = ["x0", "y0", "x1", "y1", "label"]
    for bounding_box in bb_list:
        for key in required_bounding_box_keys:
            if key not in bounding_box:
                raise ValueError(f"Bounding box does not have the required key: {key}")

    # Check if the table has a value-mapping for its labels.
    if table.get_value_map("bbs.bb_list.label") is None:
        raise ValueError("Table does not have a value-mapping for its bounding box labels.")


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
