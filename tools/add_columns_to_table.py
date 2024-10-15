"""Add columns to a table in a project."""

from __future__ import annotations

import typing
from collections import defaultdict
from typing import Callable

import tlc
from common import check_package_version

_SampleTypeStructure = typing.Union[
    tlc.SampleType, typing.Type[tlc.SampleType], list, tuple, dict, tlc.Schema, tlc.ScalarValue, Callable
]


def check_columns_and_schemas(columns: dict[str, list], schemas: dict[str, _SampleTypeStructure]) -> None:
    assert isinstance(columns, dict), f"columns must be a dictionary, got {type(columns)}"
    assert columns, "columns must not be empty"
    first_column = next(iter(columns.values()))
    assert all(len(first_column) == len(column) for column in columns.values()), "columns must have the same length"
    assert all(key in columns for key in schemas), "schemas must have the same keys as columns"


def add_columns_to_table(
    table: tlc.Table,
    columns: dict[str, list],
    schemas: dict[str, _SampleTypeStructure],
    output_table_name: str = "added_columns",
    description: str = "Table with added columns",
) -> tlc.Table:
    """ """
    check_package_version("tlc", "2.9")
    check_columns_and_schemas(columns, schemas)

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

    # Copy over all rows from the input table
    for row in table.table_rows:
        for column_name, column_value in row.items():
            data[column_name].append(column_value)

    # Add the new columns
    for column_name, column_values in columns.items():
        data[column_name] = column_values

    assert set(len(data[column_name]) for column_name in data) == 1, "All columns must have the same length"

    table_writer.add_batch(data)
    new_table = table_writer.finalize()
    return new_table


if __name__ == "__main__":
    table = tlc.Table.from_dict({"col": [1, 2, 3]}, project_name="add_columns", table_name="initial", if_exists="reuse")
    columns = {"new_col": [[4], [5], [6]]}
    schemas: dict[str, tlc.Schema] = {}

    new_table = add_columns_to_table(table, columns, schemas)
    assert True
