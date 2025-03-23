from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
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


def validate_alias_name(alias_name: str) -> bool:
    """
    Validate that an alias name follows the required format:
    - Must be wrapped in < >
    - Must start with a capital letter
    - Can only contain capital letters, numbers, and underscores
    """
    pattern = re.compile(_ALIAS_PATTERN)
    return bool(pattern.match(alias_name))


def validate_path_exists(path: str) -> bool:
    """
    Validate that a path exists on the filesystem.
    For URLs, we only validate the scheme and basic format.
    """
    if path.startswith(("http://", "https://", "s3://")):
        # For URLs, just validate basic format
        return bool(re.match(r"^[a-zA-Z]+://[^/\s]+(/[^/\s]*)*$", path))
    return Path(path).exists()


def parse_alias_pair_string(alias_pair: str, separator: str) -> tuple[str, str]:
    """Parse and validate an alias pair string."""
    if separator not in alias_pair:
        raise ValueError(f"Invalid alias pair ({alias_pair}) missing separator: {separator}")

    cur, replace = tuple(alias_pair.split(separator, 1))

    # Validate alias format when creating new aliases
    if cur.startswith("<") and not validate_alias_name(cur):
        raise ValueError(
            f"Invalid alias name format: {cur}. "
            "Aliases must be wrapped in <>, start with a capital letter, "
            "and contain only capital letters, numbers, and underscores."
        )

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


def handle_missing_alias(alias_name: str) -> None:
    """Handle case where an alias is not found in registry."""
    import tlc

    registered_aliases = tlc.get_registered_url_aliases()
    similar_aliases = [a for a in registered_aliases if a.startswith(alias_name[:2])]

    msg = f"Alias '{alias_name}' not found in registered aliases."
    if similar_aliases:
        msg += f"\nDid you mean one of these? {', '.join(similar_aliases)}"
    msg += "\nUse tlc.register_project_url_alias() to register new aliases."

    raise ValueError(msg)


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
    """Handle operations on a pyarrow Table directly.

    This function works at the pa.Table level, which is the underlying data structure.
    Changes to the pa.Table are immediate and don't require object recreation.

    Args:
        input_path: List of URLs representing the path to this table
        pa_table: The pyarrow Table to process
        selected_column_name: Optional column to process
        rewrite: List of (old, new) pairs to rewrite
        inplace: Whether to modify the input file
        output_url: Where to write the output (required if modifying)
        create_alias: Whether to create new aliases
        apply_alias: Whether to apply existing aliases
        persist_config: Whether to persist aliases to config
        config_scope: Scope for config persistence ("project" or "global")
    """
    import io

    import tlc

    # Validate selected column exists if specified
    if selected_column_name and selected_column_name not in pa_table.column_names:
        raise ValueError(
            f"Selected column '{selected_column_name}' not found in the input table's columns"
            f" ({pa_table.column_names})."
        )

    # Handle alias application
    if apply_alias and rewrite:
        alias_name = rewrite[0][0]
        # Look up the alias value
        registered_aliases = tlc.get_registered_url_aliases()
        if alias_name not in registered_aliases:
            handle_missing_alias(alias_name)
        alias_value = registered_aliases[alias_name]

        # Validate the path exists when applying aliases
        if not validate_path_exists(alias_value):
            raise ValueError(
                f"Path '{alias_value}' referenced by alias '{alias_name}' does not exist. "
                "Please ensure the path is valid before applying the alias."
            )

        rewrite = [(alias_value, f"<{alias_name}>")]

    new_columns: dict[str, pa.Array] = {}

    # Process each column
    for col_name in pa_table.column_names:
        # Skip columns that aren't selected (if a column is selected)
        if selected_column_name and col_name != selected_column_name:
            new_columns[col_name] = pa_table.column(col_name)
            continue

        column = pa_table.column(col_name)
        if pa.types.is_string(column.type):
            # For string columns, apply the rewrites
            modified_column = column
            for old, new in rewrite:
                modified_column = pc.replace_substring(modified_column, old, new)

            # Only store if changes were made
            # For ChunkedArrays, we need to check each chunk
            if isinstance(modified_column, pa.ChunkedArray):
                has_changes = False
                chunks = []
                for i, chunk in enumerate(modified_column.iterchunks()):
                    # Get the corresponding chunk from the original column
                    orig_chunk = column.chunk(i) if isinstance(column, pa.ChunkedArray) else column
                    if not pc.all(pc.equal(chunk, orig_chunk)).as_py():
                        has_changes = True
                    chunks.append(chunk)
                if has_changes:
                    new_columns[col_name] = pa.chunked_array(chunks, column.type)
                else:
                    new_columns[col_name] = column
            else:
                if not pc.all(pc.equal(modified_column, column)).as_py():
                    new_columns[col_name] = modified_column
                else:
                    new_columns[col_name] = column
        elif pa.types.is_struct(column.type):
            # For struct columns, process each field
            sub_cols: list[pa.Array] = []
            for field in column.type:
                sub_col = column.field(field.name)
                if pa.types.is_string(sub_col.type):
                    modified_sub_col = sub_col
                    for old, new in rewrite:
                        modified_sub_col = pc.replace_substring(modified_sub_col, old, new)
                    # Handle ChunkedArrays in struct fields
                    if isinstance(modified_sub_col, pa.ChunkedArray):
                        has_changes = False
                        chunks = []
                        for i, chunk in enumerate(modified_sub_col.iterchunks()):
                            # Get the corresponding chunk from the original column
                            orig_chunk = sub_col.chunk(i) if isinstance(sub_col, pa.ChunkedArray) else sub_col
                            if not pc.all(pc.equal(chunk, orig_chunk)).as_py():
                                has_changes = True
                            chunks.append(chunk)
                        if has_changes:
                            sub_cols.append(pa.chunked_array(chunks, sub_col.type))
                        else:
                            sub_cols.append(sub_col)
                    else:
                        if not pc.all(pc.equal(modified_sub_col, sub_col)).as_py():
                            sub_cols.append(modified_sub_col)
                        else:
                            sub_cols.append(sub_col)
                else:
                    sub_cols.append(sub_col)
            new_columns[col_name] = pa.StructArray.from_arrays(sub_cols, fields=column.type)
        else:
            new_columns[col_name] = column

    if not rewrite:
        return

    if not output_url:
        raise RuntimeError("Expected output URL")

    if not new_columns:
        print("No changes to apply.")
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

    # Create output table with all columns
    output_pa_table = pa.table(new_columns)

    # write the output table to the output path
    with io.BytesIO() as buffer, pq.ParquetWriter(buffer, output_pa_table.schema) as pq_writer:
        pq_writer.write_table(output_pa_table)
        pq_writer.close()
        buffer.seek(0)
        UrlAdapterRegistry.write_binary_content_to_url(Url(output_url), buffer.read())

    print(f"Changes written to '{output_url}'")


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
    """Handle operations on a tlc.Table object.

    This function works at the tlc.Table level, which wraps the underlying pa.Table.
    When modifying the parquet file in-place, the tlc.Table object needs to be
    recreated to see the changes.

    Args:
        input_path: List of URLs representing the path to this table
        table: The tlc.Table object to process
        column: Optional column to process
        rewrite: List of (old, new) pairs to rewrite
        inplace: Whether to modify the input file
        output_url: Where to write the output (required if modifying)
        create_alias: Whether to create new aliases
        apply_alias: Whether to apply existing aliases
        persist_config: Whether to persist aliases to config
        config_scope: Scope for config persistence ("project" or "global")
    """
    processed_tables = set()  # Track processed tables to avoid cycles

    def process_table_recursive(current_table: Table, current_path: list[Url]) -> None:
        """Recursively process a table and its lineage."""
        if current_table.url in processed_tables:
            return
        processed_tables.add(current_table.url)

        # Process the current table's parquet cache if it exists
        if (current_table.row_cache_populated and current_table.row_cache_url) or (hasattr(current_table, "input_url") and current_table.input_url.exists()):
            pq_url = (
                current_table.row_cache_url.to_absolute(current_table.url)
                if current_table.row_cache_populated and current_table.row_cache_url
                else current_table.input_url
            )
            try:
                object = get_input_object(pq_url)
                handle_object(
                    current_path + [pq_url],
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
                print(f"Warning: Failed to process cache for table {current_table.url}: {e}")

        # Process parent tables recursively
        for parent_url in SchemaHelper.object_input_urls(current_table, current_table.schema):
            try:
                parent_table = Table.from_url(parent_url.to_absolute(owner=current_table.url))
                process_table_recursive(parent_table, current_path + [parent_url])
            except Exception as e:
                print(f"Warning: Failed to process parent table {parent_url}: {e}")

    # Start recursive processing from the input table
    process_table_recursive(table, input_path)


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
