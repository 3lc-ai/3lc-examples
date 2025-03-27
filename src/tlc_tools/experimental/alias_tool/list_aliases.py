from __future__ import annotations

import logging
import re

import pyarrow as pa
from tlc.core import Run, SchemaHelper, Table, TableFromParquet, Url

from .common import get_input_parquet

logger = logging.getLogger(__name__)

ALIAS_PATTERN = re.compile(r"^<[A-Z][A-Z0-9_]*>$")


def find_aliases_in_column(column_path: str, column: pa.Array) -> list[tuple[str, str, str]]:
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
            found_aliases.extend(find_aliases_in_column(column_path, chunk))
        return found_aliases

    if pa.types.is_struct(column.type):
        for field in column.type:
            # Maintain the original behavior by appending .path to struct fields
            nested_path = f"{column_path}.{field.name}"
            sub_aliases = find_aliases_in_column(nested_path, column.field(field.name))
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
                        # Use regex for a single validation check instead of multiple string operations
                        if ALIAS_PATTERN.match(potential_alias) and potential_alias not in seen_aliases:
                            seen_aliases.add(potential_alias)
                            found_aliases.append((column_path, potential_alias, str_val))

    return found_aliases


def list_aliases_in_pa_table(pa_table: pa.Table, columns: list[str], input_url: Url | None = None) -> None:
    """List aliases in a PyArrow table.

    Args:
        pa_table: The PyArrow table to process
        columns: List of columns to process. If empty, process all columns.
        input_url: Optional URL of the parquet file (for logging purposes)
    """
    # Validate selected columns exist if specified
    if columns:
        for col_name in columns:
            if col_name not in pa_table.column_names:
                cols = pa_table.column_names
                raise ValueError(f"Selected column '{col_name}' not found in columns: {cols}")

    # Process each column
    for col_name in pa_table.column_names:
        if columns and col_name not in columns:
            continue

        found_aliases = find_aliases_in_column(col_name, pa_table[col_name])
        if found_aliases:
            logger.info(f"Found aliases in column '{col_name}' of table '{input_url or 'unknown'}'")
            for _column_path, alias, value in found_aliases:
                logger.info(f"  {alias} ({value})")


def list_aliases_in_tlc_table(table: Table, columns: list[str], process_parents: bool = True) -> None:
    """List aliases in a TLC Table and its lineage.

    Args:
        table: The Table object to process
        columns: List of columns to process. If empty, process all columns.
        process_parents: Whether to process parent tables recursively
    """
    processed_tables = set()  # Track processed tables to avoid cycles

    def process_table_recursive(current_table: Table) -> None:
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
                list_aliases_in_pa_table(pa_table, columns, pq_url)
            except Exception as e:
                logger.warning(f"Failed to process cache for table {current_table.url}: {e}")

        # Process parent tables recursively if enabled
        if process_parents:
            parent_urls = list(SchemaHelper.object_input_urls(current_table, current_table.schema))
            logger.debug(f"  Parent tables: {parent_urls}")
            for parent_url in parent_urls:
                try:
                    parent_table = Table.from_url(parent_url.to_absolute(owner=current_table.url))
                    process_table_recursive(parent_table)
                except Exception as e:
                    logger.warning(f"Failed to process parent table {parent_url}: {e}")

    # Start recursive processing from the input table
    process_table_recursive(table)


def list_aliases(
    obj: pa.Table | Table | Run,
    columns: list[str],
    process_parents: bool = True,
    input_url: Url | None = None,
) -> None:
    """List aliases in a 3LC object.

    Args:
        obj: The object to process (Table, Run, or pa.Table)
        columns: List of columns to process. If empty, process all columns.
        process_parents: Whether to process parent tables recursively
        input_url: Optional URL of the parquet file (for PyArrow tables)
    """
    if isinstance(obj, Table):
        list_aliases_in_tlc_table(obj, columns, process_parents)
    elif isinstance(obj, Run):
        raise NotImplementedError("Listing aliases in Runs is not yet supported.")
    elif isinstance(obj, pa.Table):
        list_aliases_in_pa_table(obj, columns, input_url)
    else:
        raise ValueError("Input is not a valid 3LC object.")
