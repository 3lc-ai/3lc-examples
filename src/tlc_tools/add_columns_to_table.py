from __future__ import annotations

from collections import defaultdict
from typing import Any

import tlc
from tqdm.auto import tqdm

from tlc_tools.metrics import IMAGE_METRICS, compute_image_metrics


def _check_columns_and_schemas(columns: dict[str, list], schemas: dict[str, Any]) -> None:
    assert isinstance(columns, dict), f"columns must be a dictionary, got {type(columns)}"
    assert columns, "columns must not be empty"
    first_column = next(iter(columns.values()))
    assert all(len(first_column) == len(column) for column in columns.values()), "columns must have the same length"
    if schemas:
        assert all(key in columns for key in schemas), "schemas must have the same keys as columns"


def _infer_schemas(
    columns: dict[str, Any],
    schemas: dict[str, Any] | None,
) -> dict[str, Any]:
    inferred_schemas = schemas if schemas is not None else {}

    missing_schemas = set(columns.keys()) - set(inferred_schemas.keys())
    for column_name in missing_schemas:
        column_values = columns[column_name]
        inferred_schema = tlc.Schema.from_sample(column_values[0])
        inferred_schema.sample_type_config = "hidden"
        inferred_schema.writable = False
        inferred_schemas[column_name] = inferred_schema

    return inferred_schemas


def add_columns_to_table(
    table: tlc.Table,
    columns: dict[str, list],
    schemas: dict[str, Any] | None = None,
    output_table_name: str = "added_columns",
    description: str = "Table with added columns",
) -> tlc.Table:
    """"""

    schemas = _infer_schemas(columns, schemas)
    _check_columns_and_schemas(columns, schemas)

    input_schemas = table.row_schema.values
    schemas.update(input_schemas)

    # Existing columns from table_rows are already in row form; new columns may be in sample form.
    input_mode = {col: "row" for col in input_schemas}
    input_mode.update({col: "auto" for col in columns})

    table_writer = tlc.TableWriter(
        project_name=table.project_name,
        dataset_name=table.dataset_name,
        description=description,
        table_name=output_table_name,
        schema=schemas,
        if_exists="rename",
        input_tables=[table.url],
        input_mode=input_mode,
    )

    for i, row in tqdm(enumerate(table.table_rows), desc="Adding columns to table", total=len(table)):
        output_row = dict(row)

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
    for row in tqdm(table.table_rows, desc="Computing image metrics", total=len(table)):
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
