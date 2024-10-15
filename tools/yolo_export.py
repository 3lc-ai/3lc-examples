from __future__ import annotations

import argparse
from pathlib import Path

import tlc
import yaml
from tqdm import tqdm


def export_yolo_labels(
    table: tlc.Table | str | Path | tlc.Url,
    output_url: tlc.Url | Path | str,
    overwrite: bool = False,
) -> None:
    """Export a the bounding boxes originating from TableFromYolo to YOLO formatted labels in output_url.

    :param table: The table to export bounding boxes from.
    :param output_url: The url to write the bounding box data to.
    :param overwrite: Whether to overwrite files at the output_url.
    """

    if isinstance(table, (str, Path, tlc.Url)):
        table = tlc.Table.from_url(table)

    output_url = tlc.Url(output_url)

    output_path = Path(output_url.to_str())
    assert output_path.suffix == "", f"output_url must be a directory, got {output_url}"
    output_path.mkdir(parents=True, exist_ok=overwrite)

    # Check that the output_url is a directory
    if not output_path.is_dir():
        raise ValueError(f"output_url must be a directory, got {output_url}")

    if output_path.exists() and not overwrite:
        raise ValueError("output_url already exists, use --overwrite to overwrite")

    label_path = None

    # Iterate over the dataset and write labels to the output url, based on the filenames
    for row in tqdm(table.table_rows):
        image_path = row["image"]
        bounding_boxes = row["bbs"]["bb_list"]

        # Read out the bouinding boxes
        lines = []
        for bounding_box in bounding_boxes:
            values = (bounding_box["x0"], bounding_box["y0"], bounding_box["x1"], bounding_box["y1"])
            line = f"{bounding_box['label']} {' '.join(map(str, values))}"
            lines.append(line)

        # Get the part of the image path after the last occurence of 'images'
        subpath = _get_subpath_after_images(image_path)

        label_path = output_path / "labels" / subpath.with_suffix(".txt")
        if lines:
            label_path.parent.mkdir(parents=True, exist_ok=overwrite)
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

    # Write a draft dataset yaml file
    categories = tlc.SchemaHelper.to_simple_value_map(table.get_value_map("bbs.bb_list.label"))

    yaml_content = {
        "path": str(output_path.absolute().as_posix()),
        "train": "",
        "val": "",
        "categories": categories,
    }

    with open(output_path / "dataset.yaml", "w", encoding="utf-8") as f:
        # Add a comment at the top of the file, then dump the yaml
        f.write("# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license\n")
        f.write("# This is a draft dataset file, please update the paths. The path is set to the chosen output_dir.\n")
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)


def _get_subpath_after_images(full_path):
    path = Path(full_path)
    parts = path.parts
    try:
        index = parts[::-1].index("images")  # Find last occurrence of 'images'
    except ValueError:
        return None  # 'images' not found in path
    subpath = Path(*parts[-(index):])
    return subpath


def main(args):
    table = tlc.Table.from_url(args.table_url)
    export_yolo_labels(table, args.output_url, overwrite=args.overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export bounding boxes to YOLO labels.")
    parser.add_argument("--table-url", type=tlc.Url, help="The url of the table to export labels from.")
    parser.add_argument("--output-url", type=tlc.Url, help="The url to write the labels to.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite files at the output_url.")

    args = parser.parse_args()

    main(args)

    table_url = "C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/coco8-YOLOv8/datasets/coco8-train/tables/initial"
    output_url = "./output"
    overwrite = True

    export_yolo_labels(table_url, output_url, overwrite)

    # Move images?
    # supply split?
    # handle different input schemas?
    # supply column names? bbs, bb_list, label, x0, y0, x1, y1
    # supply categories? (subset e.g.)
