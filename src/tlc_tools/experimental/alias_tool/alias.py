# =============================================================================
# <copyright>
# Copyright (c) 2023 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tlc.core import Run, SchemaHelper, Table, Url, UrlAdapterRegistry

from tlc_tools.cli import register_tool

if TYPE_CHECKING:
    from tlc.core import Run, Table, Url

_DEFAULT_SEPARATOR = "::"
_ALIAS_PATTERN = r"<[A-Z][A-Z0-9_]*>"


def _format_columns(columns: list[str] | None = None) -> str:
    if not columns:
        return ""
    return "".join([f"['{col}']" for col in columns if col])


def _format_paths(paths: list[Url], columns: list[str] | None = None) -> str:
    if len(paths) == 1:
        return str(paths[0])

    return str(paths[0]) + " => " + " => ".join([str(url.to_relative(paths[0])) for url in paths[1:]])


def _format_paths_and_columns(paths: list[Url], columns: list[str] | None = None) -> str:
    return _format_paths(paths, columns) + _format_columns(columns)


def parse_alias_pair_string(alias_pair: str, separator: str) -> tuple[str, str]:
    if separator not in alias_pair:
        raise ValueError(f"Invalid alias pair ({alias_pair}) missing separator: {separator}")
    cur, replace = tuple(alias_pair.split(separator, 1))
    return cur, replace


def get_input_table(input_url: Url) -> Table:
    table = Table.from_url(input_url)
    return table


def get_input_run(input_url: Url) -> Run:
    run = Run.from_url(input_url)
    return run


def get_input_parquet(input_url: Url) -> pa.Table:
    import io

    parquet_data = UrlAdapterRegistry.read_binary_content_from_url(input_url)
    if parquet_data[:4] != b"PAR1":
        raise ValueError(f"Input file '{input_url}' is not a Parquet. Header is {(parquet_data[:4]).decode()}")

    with io.BytesIO(parquet_data) as buffer, pq.ParquetFile(buffer) as pq_file:
        input_table = pq_file.read()

    return input_table


def get_input_object(input_url: Url) -> pa.Table | Table | Run:
    try:
        return get_input_run(input_url)
    except ValueError:
        pass

    try:
        return get_input_table(input_url)
    except ValueError:
        pass

    try:
        return get_input_parquet(input_url)
    except ValueError:
        pass

    raise ValueError(f"Input file '{input_url}' is not a valid 3LC object.")


def handle_parquet_column(
    column_names: list[str],
    column: pa.Array,
    rewrite: list[tuple[str, str]],
) -> tuple[set[tuple[str, str]], pa.Array]:
    import re

    alias_pattern = re.compile(_ALIAS_PATTERN)
    col_aliases: set[tuple[str, str]] = set()
    if isinstance(column, pa.ChunkedArray):
        chunks = []
        for chunk in column.iterchunks():
            chunk_aliases, chunk_col = handle_parquet_column(column_names, chunk, rewrite)
            col_aliases.update(chunk_aliases)
            chunks.append(chunk_col)
        return col_aliases, pa.chunked_array(chunks, column.type)

    if pa.types.is_struct(column.type):
        sub_cols: list[pa.Array] = []

        for field in column.type:
            sub_aliases, sub_col = handle_parquet_column(column_names + [field.name], column.field(field.name), rewrite)
            sub_cols.append(sub_col)
            col_aliases.update(sub_aliases)

        return col_aliases, pa.StructArray.from_arrays(sub_cols, fields=column.type)
    elif pa.types.is_string(column.type):
        for r in column:
            matches = alias_pattern.findall(r.as_py())
            for match in matches:
                col_aliases.add((".".join(column_names), match))

        if rewrite and col_aliases:
            for alias, new_alias in rewrite:
                # Perform search and replace
                modified_column = pc.replace_substring(column, alias, new_alias)  # Replace the column
                num_modified_rows = pc.sum(pc.invert(pc.equal(modified_column, column))).as_py()
                if num_modified_rows > 0:
                    column = modified_column
                    print(
                        f"Rewrote {num_modified_rows} occurrences of '{alias}' to '{new_alias}'"
                        f" in column '{column_names}'"
                    )

    return col_aliases, column


def handle_pa_table(
    input_path: list[Url],
    pa_table: pa.Table,
    selected_column_name: str,
    rewrite: list[tuple[str, str]],
    inplace: bool,
    output_url: Url | None,
    create_alias: bool = False,
    apply_alias: bool = False,
    persist_config: bool = False,
    config_scope: str = "project",
) -> None:
    import io

    import tlc

    if selected_column_name:
        if selected_column_name not in pa_table.column_names:
            raise ValueError(
                f"Selected column '{selected_column_name}' not found in the input table's columns"
                f" ({pa_table.column_names})."
            )
        # select a single column to work on
        pa_table = pa_table.select([selected_column_name])

    found_aliases: set[tuple[str, str]] = set()
    new_columns: dict[str, pa.Array] = {}

    # Handle alias application
    if apply_alias and rewrite:
        alias_name = rewrite[0][0]
        # Look up the alias value
        registered_aliases = tlc.get_registered_url_aliases()
        if alias_name not in registered_aliases:
            raise ValueError(f"Alias '{alias_name}' not found in registered aliases")
        alias_value = registered_aliases[alias_name]
        rewrite = [(alias_value, f"<{alias_name}>")]

    for col_name in pa_table.column_names:
        col_alias, col = handle_parquet_column(
            [col_name],
            pa_table.column(col_name),
            rewrite,
        )
        found_aliases.update(col_alias)
        if col_alias and (rewrite or create_alias):
            new_columns[col_name] = col

    if found_aliases and not (rewrite or create_alias):
        print(
            f"{_format_paths_and_columns(input_path, columns=[selected_column_name])}"
            f" contains the following aliases: {found_aliases}"
        )

    if not (rewrite or create_alias):
        return

    if not output_url:
        raise RuntimeError("Expected output URL")

    if not new_columns:
        print("No aliases to rewrite.")
        return

    # Handle alias creation and persistence
    if create_alias and persist_config:
        alias_name = rewrite[0][1].strip("<>")  # Remove < > from alias name
        alias_value = rewrite[0][0]
        if config_scope == "project":
            tlc.register_project_url_alias(alias_name, alias_value)
            print(f"Registered project alias '{alias_name}' with value '{alias_value}'")
        else:
            tlc.register_url_alias(alias_name, alias_value)
            print(f"Registered global alias '{alias_name}' with value '{alias_value}'")

    if not selected_column_name:
        output_pa_table = pa.table(new_columns)
    else:
        # insert the selected rewritten column back into the original table
        columns = {name: pa_table.column(name) for name in pa_table.column_names if name != selected_column_name}
        columns[selected_column_name] = new_columns
        output_pa_table = pa.table(columns)

    # write the output table to the output path
    with io.BytesIO() as buffer, pq.ParquetWriter(buffer, output_pa_table.schema) as pq_writer:
        pq_writer.write_table(output_pa_table)
        pq_writer.close()
        buffer.seek(0)
        UrlAdapterRegistry.write_binary_content_to_url(output_url, buffer.read())

    print(f"Aliases written to '{output_url}'")


def handle_run(
    input_path: list[Url],
    run: Run,
    column: str,
    rewrite: list[tuple[str, str]],
    inplace: bool,
    output_url: Url | None,
    create_alias: bool = False,
    apply_alias: bool = False,
    persist_config: bool = False,
    config_scope: str = "project",
) -> None:
    if not inplace and rewrite:
        raise ValueError("Runs can only be modified inplace.")

    # iterate over input tables and rewrite aliases
    for input_table_info in run.constants.get("inputs") or []:
        try:
            input_table_url = Url(input_table_info["input_table_url"]).to_absolute(run.url)
            input_table = Table.from_url(input_table_url)

            handle_object(
                input_path + [input_table_url],
                input_table,
                column,
                rewrite,
                inplace,
                output_url,
                create_alias,
                apply_alias,
                persist_config,
                config_scope,
            )
        except Exception as e:
            print(f"Error: {e}")


def handle_table(
    input_path: list[Url],
    table: Table,
    column: str,
    rewrite: list[tuple[str, str]],
    inplace: bool,
    output_url: Url | None,
    create_alias: bool = False,
    apply_alias: bool = False,
    persist_config: bool = False,
    config_scope: str = "project",
) -> None:
    if table.row_cache_populated and table.row_cache_url:
        pq_url = table.row_cache_url.to_absolute(table.url)
        object = get_input_object(pq_url)

        if not inplace and rewrite:
            raise ValueError("Tables can only be modified inplace.")

        try:
            handle_object(
                input_path + [pq_url],
                object,
                column,
                rewrite,
                inplace,
                pq_url if inplace else output_url,
                create_alias,
                apply_alias,
                persist_config,
                config_scope,
            )
        except Exception as e:
            print(f"Error: {e}")

    def parents(_table: Table) -> list[Table]:
        """Return a list of parent tables of the given table."""
        input_tables = [
            Table.from_url(input_url.to_absolute(owner=_table.url))
            for input_url in SchemaHelper.object_input_urls(_table, table.schema)
        ]
        return input_tables

    for input_table in parents(table):
        try:
            handle_object(
                input_path + [input_table.url],
                input_table,
                column,
                rewrite,
                inplace,
                output_url,
                create_alias,
                apply_alias,
                persist_config,
                config_scope,
            )
        except Exception as e:
            print(f"Error: {e}")


def handle_object(
    input_path: list[Url],
    obj: pa.Table | Table | Run,
    column: str,
    rewrite: list[tuple[str, str]],
    inplace: bool,
    output_url: Url | None,
    create_alias: bool = False,
    apply_alias: bool = False,
    persist_config: bool = False,
    config_scope: str = "project",
) -> None:
    if isinstance(obj, Table):
        return handle_table(
            input_path,
            obj,
            column,
            rewrite,
            inplace,
            output_url,
            create_alias,
            apply_alias,
            persist_config,
            config_scope,
        )
    elif isinstance(obj, Run):
        return handle_run(
            input_path,
            obj,
            column,
            rewrite,
            inplace,
            output_url,
            create_alias,
            apply_alias,
            persist_config,
            config_scope,
        )
    elif isinstance(obj, pa.Table):
        return handle_pa_table(
            input_path,
            obj,
            column,
            rewrite,
            inplace,
            output_url,
            create_alias,
            apply_alias,
            persist_config,
            config_scope,
        )
    else:
        raise ValueError("Input is not a valid 3LC object.")


@register_tool(experimental=True, description="List, rewrite, and create URL aliases in 3LC objects")
def main(tool_args: list[str] | None = None, prog: str | None = None) -> None:
    """
    Main function to process aliases in 3LC objects

    :param tool_args: List of arguments. If None, will parse from command line.
    :param prog: Program name. If None, will use the tool name.
    """
    parser = argparse.ArgumentParser(prog=prog, description="List, rewrite, and create URL aliases in 3LC objects")

    # Positional argument for input-file
    parser.add_argument("input-path", help="The input object to investigate", type=str)

    # Operation selection group
    operation_group = parser.add_mutually_exclusive_group()
    operation_group.add_argument(
        "--list", action="store_true", default=True, help="List all aliases in the input (default behavior)"
    )
    operation_group.add_argument(
        "--rewrite",
        action="append",
        help="Rewrite aliases in input",
        metavar="<ALIAS>" + _DEFAULT_SEPARATOR + "<NEW_ALIAS>",
    )
    operation_group.add_argument(
        "--create-alias",
        type=str,
        help="Create new alias and replace matching paths (format: <ALIAS_NAME>::/path/to/value)",
        metavar="<ALIAS_NAME>" + _DEFAULT_SEPARATOR + "/path/to/value",
    )
    operation_group.add_argument(
        "--apply-alias",
        type=str,
        help="Apply existing alias to matching paths",
        metavar="ALIAS_NAME",
    )

    # Common arguments
    parser.add_argument(
        "--separator",
        action="store",
        type=str,
        help="Separator to use for alias pairs",
        default=_DEFAULT_SEPARATOR,
        metavar="<ALIAS>" + _DEFAULT_SEPARATOR + "<NEW_ALIAS>",
    )
    parser.add_argument("--column", type=str, default="", help="Select a specific column of input to work on")
    parser.add_argument(
        "--persist-config",
        action="store_true",
        help="Persist new aliases in project/global config",
    )
    parser.add_argument(
        "--config-scope",
        choices=["project", "global"],
        default="project",
        help="Scope for persisting alias config",
    )

    # Output behavior group
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o",
        "--output-path",
        dest="output-path",
        type=str,
        help="The path to which the modified object will be written",
    )
    output_group.add_argument(
        "-i",
        "--inplace",
        action="store_true",
        help="Overwrite the input file with the new changes",
    )

    args = parser.parse_args(tool_args)

    input_path = args.input_path
    input_url = Url(input_path)
    if not input_url.is_absolute():
        input_url = input_url.to_absolute(os.getcwd())

    # Parse operation-specific arguments
    separator = args.separator
    if args.rewrite:
        aliases = [parse_alias_pair_string(alias_pair, separator=separator) for alias_pair in args.rewrite]
    elif args.create_alias:
        alias_name, alias_value = parse_alias_pair_string(args.create_alias, separator=separator)
        aliases = [(alias_value, f"<{alias_name}>")]  # We'll replace the value with the alias
    elif args.apply_alias:
        alias_name = args.apply_alias
        # We'll look up the alias value later
        aliases = [(alias_name, "")]  # Empty string as placeholder, will be replaced with actual value
    else:
        aliases = []

    # Handle output path
    inplace = args.inplace
    output_path = getattr(args, "output-path", "")
    output_url = Url(output_path) if output_path else None
    if output_url and not output_url.is_absolute():
        output_url = output_url.to_absolute(os.getcwd())

    if (args.rewrite or args.create_alias or args.apply_alias) and not output_url:
        raise ValueError("Output path (--output/-o) or inplace (--inplace/-i) is required for modifying aliases.")

    object = get_input_object(input_url)

    try:
        handle_object(
            [input_url],
            object,
            column=args.column,
            rewrite=aliases,
            inplace=inplace,
            output_url=output_url,
            create_alias=bool(args.create_alias),
            apply_alias=bool(args.apply_alias),
            persist_config=args.persist_config,
            config_scope=args.config_scope,
        )
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
