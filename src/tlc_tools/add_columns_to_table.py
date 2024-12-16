from __future__ import annotations

import typing
from collections import defaultdict
from typing import Any, Callable

import tlc
from PIL import Image
from tqdm.auto import tqdm

from tlc_tools.metrics import IMAGE_METRICS, compute_image_metrics

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


def add_columns_to_table(
    table: tlc.Table,
    columns: dict[str, list],
    schemas: dict[str, _SampleTypeStructure] | None = None,
    output_table_name: str = "added_columns",
    description: str = "Table with added columns",
) -> tlc.Table:
    """"""

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

    for i, row in tqdm(enumerate(table.table_rows), desc="Adding columns to table", total=len(table)):
        output_row = {}
        for column_name, column_value in row.items():
            if input_schemas[column_name].sample_type == tlc.PILImage.sample_type:
                image_url = tlc.Url(column_value).to_absolute(table.url)
                column_value = Image.open(image_url.to_str())

            output_row[column_name] = column_value

        # Add the new columns
        for column_name in columns:
            output_row[column_name] = columns[column_name][i]

        table_writer.add_row(output_row)

    new_table = table_writer.finalize()
    return new_table


def add_image_metrics_to_table(
    table: tlc.Table,
    image_metrics: list[IMAGE_METRICS] | None = None,
    image_column_name: str = "image",
    output_table_name: str = "added_image_metrics",
    description: str = "Table with added image metrics",
) -> tlc.Table:
    """"""
    new_columns = defaultdict(list)
    for row in table.table_rows:
        image_path = tlc.Url(row[image_column_name]).to_absolute().to_str()
        metrics = compute_image_metrics(image_path, image_metrics)
        for column_name, value in metrics.items():
            new_columns[column_name].append(value)

    extended_table = add_columns_to_table(
        table=table,
        columns=new_columns,
        output_table_name=output_table_name,
        description=description,
    )
    return extended_table
