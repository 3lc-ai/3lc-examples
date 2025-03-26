from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tlc.core import Run, SchemaHelper, Table, TableFromParquet, Url, UrlAdapterRegistry

from .common import get_input_parquet, logger


def rewrite_column_values(column_path: str, column: pa.Array, rewrites: list[tuple[str, str]]) -> tuple[pa.Array, bool]:
    """Replace aliases in a single column.

    Args:
        column_path: Path to the column (dot-separated for nested columns)
        column: The column to process
        rewrites: List of (old_path, new_path) pairs to rewrite

    Returns:
        Tuple of (modified_column, was_modified)
    """
    if isinstance(column, pa.ChunkedArray):
        chunks = []
        was_modified = False
        for chunk in column.iterchunks():
            modified_chunk, chunk_modified = rewrite_column_values(column_path, chunk, rewrites)
            chunks.append(modified_chunk)
            was_modified = was_modified or chunk_modified
        return pa.chunked_array(chunks, column.type), was_modified

    if pa.types.is_struct(column.type):
        sub_cols = []
        was_modified = False
        for field in column.type:
            nested_path = f"{column_path}.{field.name}" if column_path else field.name
            modified_col, col_modified = rewrite_column_values(nested_path, column.field(field.name), rewrites)
            sub_cols.append(modified_col)
            was_modified = was_modified or col_modified
        return pa.StructArray.from_arrays(sub_cols, fields=column.type), was_modified

    if pa.types.is_string(column.type):
        modified_column = column
        was_modified = False
        for old, new in rewrites:
            new_column = pc.replace_substring(modified_column, old, new)
            # Check if any changes were made
            num_modified_rows = pc.sum(pc.invert(pc.equal(new_column, modified_column))).as_py()
            if num_modified_rows > 0:
                logger.info(f"Rewrote {num_modified_rows} occurrences of '{old}' to '{new}' in column '{column_path}'")
                was_modified = True
            modified_column = new_column
        return modified_column, was_modified

    return column, False


def backup_parquet(input_path: Url) -> Url:
    """Create a backup of a parquet file before modification.

    Args:
        input_path: URL of the parquet file to backup

    Returns:
        URL of the backup file
    """
    backup_url = Url(str(input_path) + ".backup")
    UrlAdapterRegistry.copy_url(input_path, backup_url)
    logger.debug(f"Created backup at {backup_url}")
    return backup_url


def restore_from_backup(backup_url: Url, original_url: Url) -> None:
    """Restore a parquet file from its backup.

    Args:
        backup_url: URL of the backup file
        original_url: URL of the file to restore
    """
    UrlAdapterRegistry.copy_url(backup_url, original_url)
    logger.debug(f"Restored {original_url} from backup {backup_url}")


def replace_aliases_in_pa_table(
    input_path: list[Url], pa_table: pa.Table, columns: list[str], rewrites: list[tuple[str, str]]
) -> None:
    """Replace aliases in a PyArrow table."""
    target_url = input_path[-1]
    backup_url = None

    try:
        new_columns: dict[str, pa.Array] = {}
        changes_made = False

        # Validate selected columns exist if specified
        if columns:
            for col_name in columns:
                if col_name not in pa_table.column_names:
                    cols = pa_table.column_names
                    raise ValueError(f"Selected column '{col_name}' not found in columns: {cols}")

        # Process each column
        for col_name in pa_table.column_names:
            if columns and col_name not in columns:
                new_columns[col_name] = pa_table[col_name]
                continue

            modified_col, was_modified = rewrite_column_values(col_name, pa_table[col_name], rewrites)
            new_columns[col_name] = modified_col
            changes_made = changes_made or was_modified

        if not changes_made:
            logger.info("No changes to apply.")
            return

        # Create backup before any modifications
        backup_url = backup_parquet(target_url)

        # Create output table with all columns
        output_pa_table = pa.table(new_columns)

        # Write the output table back to the input path
        try:
            with io.BytesIO() as buffer, pq.ParquetWriter(buffer, output_pa_table.schema) as pq_writer:
                pq_writer.write_table(output_pa_table)
                pq_writer.close()
                buffer.seek(0)
                UrlAdapterRegistry.write_binary_content_to_url(target_url, buffer.read())
            logger.info(f"Changes written to '{target_url}'")
        except Exception as e:
            if backup_url:
                logger.warning(f"Failed to write changes: {e}. Restoring from backup...")
                restore_from_backup(backup_url, target_url)
            raise

    finally:
        if backup_url:
            backup_url.delete()


def replace_aliases_in_tlc_table(
    input_path: list[Url],
    table: Table,
    columns: list[str],
    rewrites: list[tuple[str, str]],
    process_parents: bool = True,
) -> None:
    """Replace aliases in a TLC Table and its lineage.

    Args:
        input_path: List of URLs representing the path to this table
        table: The Table object to process
        columns: List of columns to process. If empty, process all columns.
        rewrites: List of (old_path, new_path) pairs to rewrite
        process_parents: Whether to process parent tables recursively
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
                replace_aliases_in_pa_table(current_path + [pq_url], pa_table, columns, rewrites)
            except Exception as e:
                logger.warning(f"Failed to process cache for table {current_table.url}: {e}")

        # Process parent tables recursively if enabled
        if process_parents:
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


def replace_aliases(
    input_path: list[Url],
    obj: pa.Table | Table | Run,
    columns: list[str],
    rewrites: list[tuple[str, str]],
    process_parents: bool = True,
) -> None:
    """Replace paths with aliases in a 3LC object.

    Args:
        input_path: List of URLs representing the path to this object
        obj: The object to process (Table, Run, or pa.Table)
        columns: List of columns to process. If empty, process all columns.
        rewrites: List of (old_path, new_path) pairs to rewrite
        process_parents: Whether to process parent tables recursively
    """
    if isinstance(obj, Table):
        replace_aliases_in_tlc_table(input_path, obj, columns, rewrites, process_parents)
    elif isinstance(obj, Run):
        raise NotImplementedError("Replacing aliases in Runs is not yet supported.")
    elif isinstance(obj, pa.Table):
        replace_aliases_in_pa_table(input_path, obj, columns, rewrites)
    else:
        raise ValueError("Input is not a valid 3LC object.")
