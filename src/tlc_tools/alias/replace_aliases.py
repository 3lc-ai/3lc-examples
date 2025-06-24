from __future__ import annotations

import io
import logging
import re
from copy import deepcopy
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tlc.core import EditedTable, Run, SchemaHelper, Table, TableFromParquet, Url, UrlAdapterRegistry

from .common import get_input_parquet

logger = logging.getLogger(__name__)

# Number of sample rewrites to show in debug logs
SAMPLE_REWRITES_COUNT = 3


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
        # Skip processing if no rewrites to apply
        if not rewrites:
            return column, False

        # Process each rewrite sequentially
        modified_column = column
        was_modified = False
        affected_rows = 0
        sample_rewrites: list[tuple[str, str]] = []  # Only store first N changes for logging

        for old, new in rewrites:
            # Escape special characters in the old path
            old_pattern = re.escape(old)
            new_column = pc.replace_substring_regex(modified_column, old_pattern, new)

            # Count affected rows
            changed_mask = pc.not_equal(new_column, modified_column)
            changed_count = pc.sum(changed_mask).as_py()
            if changed_count > 0:
                was_modified = True
                affected_rows += changed_count
                # Only store first N changes for logging
                if len(sample_rewrites) < SAMPLE_REWRITES_COUNT:
                    sample_rewrites.append((old, new))

            modified_column = new_column

        if was_modified:
            if sample_rewrites:
                sample_text = ", ".join(f"'{old}'→'{new}'" for old, new in sample_rewrites)
                if len(rewrites) > len(sample_rewrites):
                    sample_text += f" (and {len(rewrites) - len(sample_rewrites)} more patterns)"
                logger.debug(f"Column '{column_path}': {sample_text}")
            logger.info(f"Changed {affected_rows} rows in column '{column_path}'")

        return modified_column, was_modified

    return column, False


def backup_file(input_path: Url) -> Url:
    """Create a backup of a file before modification.

    Args:
        input_path: URL of the file to backup

    Returns:
        URL of the backup file
    """
    backup_url = Url(str(input_path) + ".backup")
    UrlAdapterRegistry.copy_url(input_path, backup_url)
    logger.debug(f"Created backup at {backup_url}")
    return backup_url


def restore_from_backup(backup_url: Url, original_url: Url) -> None:
    """Restore a file from its backup.

    Args:
        backup_url: URL of the backup file
        original_url: URL of the file to restore
    """
    UrlAdapterRegistry.copy_url(backup_url, original_url)
    logger.debug(f"Restored {original_url} from backup {backup_url}")


def replace_aliases_in_pa_table(
    target_url: Url, pa_table: pa.Table, columns: list[str], rewrites: list[tuple[str, str]], dry_run: bool = False
) -> None:
    """Replace aliases in a PyArrow table.

    Args:
        target_url: URL where the modified table should be written
        pa_table: The PyArrow table to process
        columns: List of columns to process. If empty, process all columns.
        rewrites: List of (old_path, new_path) pairs to rewrite
        dry_run: If True, show changes without making them
    """
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

        if dry_run:
            logger.info(f"[DRY RUN] Would write changes to '{target_url}'")
            return

        # Create backup only if changes were made
        backup_url = backup_file(target_url)

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


def replace_aliases_in_value(value: Any, rewrites: list[tuple[str, str]]) -> Any:
    """Replace aliases in a value.

    Args:
        value: The value to process
        rewrites: List of (old_path, new_path) pairs to rewrite
    """
    if isinstance(value, dict):
        return {k: replace_aliases_in_value(v, rewrites) for k, v in value.items()}
    elif isinstance(value, list):
        return [replace_aliases_in_value(v, rewrites) for v in value]
    elif isinstance(value, str):
        for old, new in rewrites:
            if old in value:
                value = value.replace(old, new)
    return value


def replace_aliases_in_edited_table(
    table: EditedTable,
    columns: list[str],
    rewrites: list[tuple[str, str]],
    dry_run: bool = False,
) -> None:
    """Replace aliases in an EditedTable's edits attribute.

    Args:
        table: The EditedTable to process
        columns: List of columns to process. If empty, process all columns.
        rewrites: List of (old_path, new_path) pairs to rewrite
        dry_run: If True, show changes without making them
    """
    edits = table.edits

    # First, make a copy of the edits
    edits_copy = deepcopy(edits)
    was_modified = False

    # Then, replace any "new_value" that matches an old path with a new path (from rewrites)
    for column_name, column_data in edits_copy.items():
        if columns and column_name not in columns:
            continue

        runs_and_values = column_data["runs_and_values"]
        for i in range(1, len(runs_and_values), 2):
            rewritten_value = replace_aliases_in_value(runs_and_values[i], rewrites)
            if rewritten_value != runs_and_values[i]:
                runs_and_values[i] = rewritten_value
                was_modified = True

    if was_modified:
        if dry_run:
            logger.info(f"[DRY RUN] Would write 'edits' attribute changes to '{table.url}'")
            return

        # Write the new edits back to the table json file
        table.edits = edits_copy
        backup_url = backup_file(table.url)
        try:
            table.write_to_url(force=True)
        except Exception as e:
            if backup_url:
                logger.warning(f"Failed to write changes: {e}. Restoring from backup...")
                restore_from_backup(backup_url, table.url)
            raise
        finally:
            if backup_url:
                backup_url.delete()


def replace_aliases_in_tlc_table(
    table: Table,
    columns: list[str],
    rewrites: list[tuple[str, str]],
    process_parents: bool = True,
    dry_run: bool = False,
) -> None:
    """Replace aliases in a TLC Table and its lineage.

    Args:
        table: The Table object to process
        columns: List of columns to process. If empty, process all columns.
        rewrites: List of (old_path, new_path) pairs to rewrite
        process_parents: Whether to process parent tables recursively
        dry_run: If True, show changes without making them
    """
    processed_tables = set()  # Track processed tables to avoid cycles

    def process_table_recursive(current_table: Table) -> None:
        if current_table.url in processed_tables:
            return
        current_table.ensure_fully_defined()
        processed_tables.add(current_table.url)

        logger.info(f"├─ {current_table.url.name}")

        if isinstance(current_table, EditedTable):
            replace_aliases_in_edited_table(current_table, columns, rewrites, dry_run)
            # Continue processing, since the EditedTable might still have a row cache

        # Process the current table's parquet cache if it exists
        has_cache = current_table.row_cache_populated and current_table.row_cache_url
        is_table_from_parquet = isinstance(current_table, TableFromParquet)

        if has_cache or is_table_from_parquet:
            # Get URL of file to process - prefer cache if available
            if has_cache:
                pq_url = current_table.row_cache_url.to_absolute(current_table.url)
            else:
                pq_url = current_table.input_url.to_absolute(current_table.url)

            try:
                pa_table = get_input_parquet(pq_url)
                replace_aliases_in_pa_table(pq_url, pa_table, columns, rewrites, dry_run)
            except Exception as e:
                logger.warning(f"Failed to process cache for table {current_table.url}: {e}")

        # Process parent tables recursively if enabled
        if process_parents:
            parent_urls = list(SchemaHelper.object_input_urls(current_table, current_table.schema))
            for parent_url in parent_urls:
                try:
                    parent_table = Table.from_url(parent_url.to_absolute(owner=current_table.url))
                    process_table_recursive(parent_table)
                except Exception as e:
                    logger.warning(f"Failed to process parent table {parent_url}: {e}")

    # Start recursive processing from the input table
    process_table_recursive(table)


def replace_aliases(
    obj: pa.Table | Table | Run,
    columns: list[str],
    rewrites: list[tuple[str, str]],
    process_parents: bool = True,
    input_url: Url | None = None,
    dry_run: bool = False,
) -> None:
    """Replace paths with aliases in a 3LC object.

    Args:
        obj: The object to process (Table, Run, or pa.Table)
        columns: List of columns to process. If empty, process all columns.
        rewrites: List of (old_path, new_path) pairs to rewrite
        process_parents: Whether to process parent tables recursively
        input_url: Optional URL of the input object
        dry_run: If True, show changes without making them
    """
    if isinstance(obj, Table):
        replace_aliases_in_tlc_table(obj, columns, rewrites, process_parents, dry_run)
    elif isinstance(obj, Run):
        raise NotImplementedError("Replacing aliases in Runs is not yet supported.")
    elif isinstance(obj, pa.Table):
        replace_aliases_in_pa_table(input_url, obj, columns, rewrites, dry_run)
    else:
        raise ValueError("Input is not a valid 3LC object.")
