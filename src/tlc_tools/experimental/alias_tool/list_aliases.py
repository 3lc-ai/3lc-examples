from __future__ import annotations

import pyarrow as pa
from tlc.core import Run, SchemaHelper, Table, TableFromParquet, Url

from .common import get_input_parquet, logger


def list_aliases_in_column(column_path: str, column: pa.Array) -> list[tuple[str, str, str]]:
    """List aliases in a single column.

    Args:
        column_path: Path to the column (dot-separated for nested columns)
        column: The column to process

    Returns:
        List of (column_path, alias, example) tuples found in the column
    """
    found_aliases = []

    if isinstance(column, pa.ChunkedArray):
        for chunk in column.iterchunks():
            found_aliases.extend(list_aliases_in_column(column_path, chunk))
        return found_aliases

    if pa.types.is_struct(column.type):
        for field in column.type:
            # Maintain the original behavior by appending .path to struct fields
            nested_path = f"{column_path}.{field.name}"
            sub_aliases = list_aliases_in_column(nested_path, column.field(field.name))
            found_aliases.extend(sub_aliases)
        return found_aliases

    if pa.types.is_string(column.type):
        seen_aliases = set()
        for value in column:
            if value is not None:
                str_val = value.as_py()
                if str_val and str_val.startswith("<"):
                    end = str_val.find(">")
                    if end > 0:  # Must be at least one character between < and >
                        potential_alias = str_val[: end + 1]
                        if (
                            potential_alias.upper() == potential_alias
                            and potential_alias[1:-1].replace("_", "").isalnum()
                            and potential_alias[1].isupper()  # First character after < must be uppercase
                            and potential_alias not in seen_aliases
                        ):
                            seen_aliases.add(potential_alias)
                            found_aliases.append((column_path, potential_alias, str_val))

    return found_aliases


def list_aliases_in_pa_table(input_path: list[Url], pa_table: pa.Table, columns: list[str]) -> None:
    """List aliases in a PyArrow table.

    Args:
        input_path: List of URLs representing the path to this table
        pa_table: The table to process
        columns: List of columns to process. If empty, process all columns.
    """
    target_url = input_path[-1]
    found_any = False

    # Validate selected columns exist if specified
    if columns:
        for col_name in columns:
            if col_name not in pa_table.column_names:
                raise ValueError(
                    f"Selected column '{col_name}' not found in the input table's columns ({pa_table.column_names})."
                )

    # Process each column
    for col_name in pa_table.column_names:
        if columns and col_name not in columns:
            continue

        aliases = list_aliases_in_column(col_name, pa_table[col_name])
        if aliases:
            found_any = True
            for found_column, alias, example in aliases:
                logger.info(f"Found alias '{alias}' in column '{found_column}' in file '{target_url}'")
                logger.debug(f"  Example: {example}")

    if not found_any:
        logger.info(f"No aliases found in file '{target_url}'")


def list_aliases_in_table(input_path: list[Url], table: Table, columns: list[str]) -> None:
    """List aliases in a TLC Table and its lineage.

    Args:
        input_path: List of URLs representing the path to this table
        table: The Table object to process
        columns: List of columns to process. If empty, process all columns.
    """
    processed_tables = set()  # Track processed tables to avoid cycles

    def process_table_recursive(current_table: Table, current_path: list[Url]) -> None:
        if current_table.url in processed_tables:
            return
        current_table.ensure_fully_defined()
        processed_tables.add(current_table.url)

        logger.debug(f"Processing table: {current_table.url}")

        # Process the current table's parquet cache if it exists
        has_cache = current_table.row_cache_populated and current_table.row_cache_url
        is_table_from_parquet = isinstance(current_table, TableFromParquet)

        if has_cache or is_table_from_parquet:
            # Get URL of file to process - prefer cache if available
            if has_cache:
                pq_url = current_table.row_cache_url.to_absolute(current_table.url)
                logger.debug(f"  Using cache URL: {pq_url}")
            else:
                pq_url = current_table.input_url.to_absolute(current_table.url)
                logger.debug(f"  Using input URL: {pq_url}")

            try:
                pa_table = get_input_parquet(pq_url)
                list_aliases_in_pa_table(current_path + [pq_url], pa_table, columns)
            except Exception as e:
                logger.warning(f"Failed to process cache for table {current_table.url}: {e}")

        # Process parent tables recursively
        parent_urls = list(SchemaHelper.object_input_urls(current_table, current_table.schema))
        logger.debug(f"  Parent tables: {parent_urls}")
        for parent_url in parent_urls:
            try:
                parent_table = Table.from_url(parent_url.to_absolute(owner=current_table.url))
                process_table_recursive(parent_table, current_path + [parent_url])
            except Exception as e:
                logger.warning(f"Failed to process parent table {parent_url}: {e}")

    # Start recursive processing from the input table
    process_table_recursive(table, input_path)


def list_aliases(input_path: list[Url], obj: pa.Table | Table | Run, columns: list[str]) -> None:
    """List all aliases found in a 3LC object.

    Args:
        input_path: List of URLs representing the path to this object
        obj: The object to process (Table, Run, or pa.Table)
        columns: List of columns to process. If empty, process all columns.
    """
    if isinstance(obj, Table):
        list_aliases_in_table(input_path, obj, columns)
    elif isinstance(obj, Run):
        raise NotImplementedError("Listing aliases in Runs is not yet supported.")
    elif isinstance(obj, pa.Table):
        list_aliases_in_pa_table(input_path, obj, columns)
    else:
        raise ValueError("Input is not a valid 3LC object.")
