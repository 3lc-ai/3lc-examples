from __future__ import annotations

import typing
from collections import defaultdict
from typing import Any, Callable

import tlc
from PIL import Image
from .common import check_package_version

_SampleTypeStructure = typing.Union[
    tlc.SampleType, typing.Type[tlc.SampleType], list, tuple, dict, tlc.Schema, tlc.ScalarValue, Callable
]


def _check_columns_and_schemas(columns: dict[str, list], schemas: dict[str, _SampleTypeStructure]) -> None:
    assert isinstance(columns, dict), f"columns must be a dictionary, got {type(columns)}"
    assert columns, "columns must not be empty"
    first_column = next(iter(columns.values()))
    assert all(len(first_column) == len(column) for column in columns.values()), "columns must have the same length"
    if schemas:
        assert all(key in columns for key in schemas), "schemas must have the same keys as columns"


def _infer_schemas(
    columns: dict[str, Any],
    schemas: dict[str, _SampleTypeStructure] | None,
) -> dict[str, _SampleTypeStructure]:
    inferred_schemas = schemas if schemas is not None else {}

    missing_schemas = set(columns.keys()) - set(inferred_schemas.keys())
    for column_name in missing_schemas:
        column_values = columns[column_name]
        inferred_schema = tlc.SampleType.from_sample(column_values[0], name=column_name).schema
        inferred_schema.sample_type = tlc.Hidden.sample_type
        inferred_schema.writable = False
        inferred_schemas[column_name] = inferred_schema

    return inferred_schemas

def cast_bbs(bbs):
    for bb in bbs["bb_list"]:
        for key in ["x0", "y0", "x1", "y1"]:
            bb[key] = float(bb[key])
    bbs["segmentation"] = [[float(x) for x in sublist] for sublist in bbs["segmentation"]]
    return bbs

def add_columns_to_table(
    table: tlc.Table,
    columns: dict[str, list],
    schemas: dict[str, _SampleTypeStructure] | None = None,
    output_table_name: str = "added_columns",
    description: str = "Table with added columns",
) -> tlc.Table:
    """"""
    check_package_version("tlc", "2.9")

    schemas = _infer_schemas(columns, schemas)
    _check_columns_and_schemas(columns, schemas)

    input_schemas = table.row_schema.values
    schemas.update(input_schemas)

    table_writer = tlc.TableWriter(
        project_name=table.project_name,
        dataset_name=table.dataset_name,
        description=description,
        table_name=output_table_name,
        column_schemas=schemas,
        if_exists="rename",
        input_tables=[table.url],
    )

    # TableWriter accepts data as a dictionary of column names to lists
    data = defaultdict(list)

    ### 
    # # Copy over all rows from the input table
    # for row in table.table_rows:
    #     for column_name, column_value in row.items():
    #         if column_name == "bbs":
    #             data[column_name].append(cast_bbs(column_value))
    #             continue
    #         if input_schemas[column_name].sample_type == tlc.PILImage.sample_type:
    #             image_url = tlc.Url(column_value).to_absolute(table.url)
    #             # if not isinstance(image_url, tlc.Url)
    #             column_value = Image.open(image_url.to_str())
    #         data[column_name].append(column_value)

    # # Add the new columns
    # for column_name, column_values in columns.items():
    #     data[column_name] = column_values

    # assert len({len(data[column_name]) for column_name in data}) == 1, "All columns must have the same length"

    # table_writer.add_batch(data)
    ###

    # Alternate implementation
    for i, row in enumerate(table.table_rows):
        
        output_row = {}
        for column_name, column_value in row.items():
            if column_name == "bbs":
                data[column_name].append(cast_bbs(column_value))
                continue
            if input_schemas[column_name].sample_type == tlc.PILImage.sample_type:
                image_url = tlc.Url(column_value).to_absolute(table.url)
                # if not isinstance(image_url, tlc.Url)
                column_value = Image.open(image_url.to_str())
            
            output_row[column_name] = column_value

        # Add the new columns
        for column_name in columns:
            output_row[column_name] = columns[column_name][i]

        table_writer.add_row(output_row)

    new_table = table_writer.finalize()
    return new_table
