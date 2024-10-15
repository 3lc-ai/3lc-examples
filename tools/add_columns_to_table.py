import tlc

import typing
from typing import Callable

_SampleTypeStructure = typing.Union[
    tlc.SampleType, typing.Type[tlc.SampleType], list, tuple, dict, tlc.Schema, tlc.ScalarValue, Callable
]


def add_columns_to_table(
    table: tlc.Table,
    columns: dict[str, list],
    schemas: dict[str, _SampleTypeStructure],
) -> tlc.Table:
    pass
