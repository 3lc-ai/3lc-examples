from __future__ import annotations

import argparse
import io
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tlc.core import Run, SchemaHelper, Table, TableFromParquet, Url, UrlAdapterRegistry

from tlc_tools.cli import register_tool

if TYPE_CHECKING:
    from tlc.core import Run, Table, Url

# Configure logger
logger = logging.getLogger(__name__)


def setup_logging(verbosity: int = 0, quiet: bool = False) -> None:
    """Configure logging for the alias tool.

    Args:
        verbosity: 0 for default (INFO), 1 for DEBUG
        quiet: If True, only show WARNING and above
    """
    # Create console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")  # Simple format, just the message
    console_handler.setFormatter(formatter)

    # Set log level based on verbosity/quiet
    if quiet:
        level = logging.WARNING
    elif verbosity > 0:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Configure root logger for the alias tool
    logger.setLevel(level)
    logger.addHandler(console_handler)

    # Prevent duplicate logging
    logger.propagate = False


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


def handle_pa_table(
    input_path: list[Url],
    pa_table: pa.Table,
    selected_columns: list[str],
    rewrite: list[tuple[str, str]],
) -> None:
    """Process a PyArrow table, either listing aliases or applying rewrites."""
    if not rewrite:
        list_aliases_in_pa_table(input_path, pa_table, selected_columns)
    else:
        replace_aliases_in_pa_table(input_path, pa_table, selected_columns, rewrite)


def handle_run(
    input_path: list[Url],
    run: Run,
    columns: list[str],
    rewrite: list[tuple[str, str]],
    inplace: bool,
    output_url: Url | None,
    create_alias: bool = False,
    apply_alias: bool = False,
    persist_config: bool = False,
    config_scope: str = "project",
) -> None:
    """Should apply aliases to the following:
    - Input table URLs
    - Input metric tables
    - Parameters
    """
    if not inplace and rewrite:
        raise ValueError("Runs can only be modified inplace.")

    raise NotImplementedError("Runs are not yet supported.")


def handle_tlc_table(
    input_path: list[Url],
    table: Table,
    columns: list[str],
    rewrite: list[tuple[str, str]],
    process_parents: bool = True,
) -> None:
    """Process a Table object, handling both the table and its lineage.

    Args:
        input_path: List of URLs representing the path to this table
        table: The Table object to process
        columns: List of columns to process. If empty, process all columns.
        rewrite: List of (old, new) pairs to rewrite
        process_parents: Whether to process parent tables recursively
    """
    processed_tables = set()  # Track processed tables to avoid cycles

    def process_table_recursive(current_table: Table, current_path: list[Url]) -> None:
        """Recursively process a table and its lineage."""
        if current_table.url in processed_tables:
            return
        current_table.ensure_fully_defined()

        processed_tables.add(current_table.url)

        logger.debug(f"Processing table: {current_table.url}")

        # Process the current table's parquet cache if it exists
        has_cache = current_table.row_cache_populated and current_table.row_cache_url
        is_table_from_parquet = isinstance(current_table, TableFromParquet)
        logger.debug(f"  Has cache: {has_cache}")
        logger.debug(f"  Is TableFromParquet: {is_table_from_parquet}")

        if has_cache or is_table_from_parquet:
            # Get URL of file to process - prefer cache if available
            if has_cache:
                pq_url = current_table.row_cache_url.to_absolute(current_table.url)
                logger.debug(f"  Using cache URL: {pq_url}")
            else:
                pq_url = current_table.input_url.to_absolute(current_table.url)
                logger.debug(f"  Using input URL: {pq_url}")

            try:
                # Process the parquet file
                pa_table = get_input_parquet(pq_url)
                logger.debug(f"  Processing parquet with columns: {pa_table.column_names}")
                handle_pa_table(
                    current_path + [pq_url],
                    pa_table,
                    columns,
                    rewrite,
                )
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


def handle_object(
    input_path: list[Url],
    obj: pa.Table | Table | Run,
    columns: list[str],
    rewrite: list[tuple[str, str]],
    process_parents: bool = True,
) -> None:
    """Process any 3LC object, applying the specified rewrites.

    Args:
        input_path: List of URLs representing the path to this object
        obj: The object to process (Table, Run, or pa.Table)
        columns: List of columns to process. If empty, process all columns.
        rewrite: List of (old, new) pairs to rewrite
        process_parents: Whether to process parent tables recursively
    """
    if isinstance(obj, Table):
        return handle_tlc_table(
            input_path,
            obj,
            columns,
            rewrite,
            process_parents,
        )
    elif isinstance(obj, Run):
        raise NotImplementedError("Runs are not yet supported.")
    elif isinstance(obj, pa.Table):
        return handle_pa_table(
            input_path,
            obj,
            columns,
            rewrite,
        )
    else:
        raise ValueError("Input is not a valid 3LC object.")


def handle_list_command(input_path: list[Url], obj: pa.Table | Table | Run, columns: list[str]) -> None:
    """Process any 3LC object, listing all aliases found.

    Args:
        input_path: List of URLs representing the path to this object
        obj: The object to process (Table, Run, or pa.Table)
        columns: List of columns to process. If empty, process all columns.
    """
    if isinstance(obj, Table):
        list_aliases_in_table(input_path, obj, columns)
    elif isinstance(obj, Run):
        raise NotImplementedError("Runs are not yet supported.")
    elif isinstance(obj, pa.Table):
        list_aliases_in_pa_table(input_path, obj, columns)
    else:
        raise ValueError("Input is not a valid 3LC object.")


def replace_aliases_in_column(
    column_path: str, column: pa.Array, rewrites: list[tuple[str, str]]
) -> tuple[pa.Array, bool]:
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
            modified_chunk, chunk_modified = replace_aliases_in_column(column_path, chunk, rewrites)
            chunks.append(modified_chunk)
            was_modified = was_modified or chunk_modified
        return pa.chunked_array(chunks, column.type), was_modified

    if pa.types.is_struct(column.type):
        sub_cols = []
        was_modified = False
        for field in column.type:
            nested_path = f"{column_path}.{field.name}" if column_path else field.name
            modified_col, col_modified = replace_aliases_in_column(nested_path, column.field(field.name), rewrites)
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

            modified_col, was_modified = replace_aliases_in_column(col_name, pa_table[col_name], rewrites)
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


def replace_aliases_in_table(
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
        replace_aliases_in_table(input_path, obj, columns, rewrites, process_parents)
    elif isinstance(obj, Run):
        raise NotImplementedError("Replacing aliases in Runs is not yet supported.")
    elif isinstance(obj, pa.Table):
        replace_aliases_in_pa_table(input_path, obj, columns, rewrites)
    else:
        raise ValueError("Input is not a valid 3LC object.")


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


@register_tool(experimental=True, description="List, rewrite, and create URL aliases in 3LC objects")
def main(tool_args: list[str] | None = None, prog: str | None = None) -> None:
    """Main function to process aliases in 3LC objects.

    Args:
        tool_args: List of arguments. If None, will parse from command line.
        prog: Program name. If None, will use the tool name.
    """
    parser = argparse.ArgumentParser(prog=prog, description="List, rewrite, and create URL aliases in 3LC objects")

    # Verbosity control for all commands
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (use -v for debug output)",
    )
    verbosity_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output except for warnings and errors",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command - for finding aliases in files
    list_parser = subparsers.add_parser("list", help="List all aliases found in files")
    list_parser.add_argument("input_path", help="The input object to process")
    list_parser.add_argument("--columns", help="Comma-separated list of columns to process")

    # Replace command - for applying aliases and path rewrites
    replace_parser = subparsers.add_parser("replace", help="Replace paths with aliases in files")
    replace_parser.add_argument("input_path", help="The input object to process")
    replace_parser.add_argument("--columns", help="Comma-separated list of columns to process")
    replace_parser.add_argument(
        "--no-process-parents",
        action="store_true",
        help="Do not process parent tables when handling Table objects",
    )

    # Replacement specification options
    replace_mode = replace_parser.add_mutually_exclusive_group(required=True)
    replace_mode.add_argument(
        "--apply",
        metavar="ALIAS[,ALIAS,...]",
        help="Apply existing aliases to matching paths (comma-separated list)",
    )
    replace_mode.add_argument(
        "--from",
        dest="from_paths",
        metavar="PATH",
        action="append",
        help="Replace occurrences of this path (can be specified multiple times)",
    )

    # Required when using --from
    replace_parser.add_argument(
        "--to",
        dest="to_paths",
        metavar="PATH",
        action="append",
        help="Replace with this path (must match number of --from arguments)",
    )

    args = parser.parse_args(tool_args)

    # Setup logging based on verbosity flags
    setup_logging(verbosity=args.verbose, quiet=args.quiet)
    logger.debug("Debug logging enabled")

    # Parse columns if specified (shared between commands)
    columns = [col.strip() for col in args.columns.split(",")] if args.columns else []

    try:
        if args.command == "replace":
            # Validate arguments first
            rewrites = []
            if args.apply:
                # Parse comma-separated aliases
                alias_names = [name.strip() for name in args.apply.split(",")]
                import tlc

                registered_aliases = tlc.get_registered_url_aliases()
                for alias_name in alias_names:
                    bracketed_name = f"<{alias_name}>" if not alias_name.startswith("<") else alias_name
                    if bracketed_name not in registered_aliases:
                        raise ValueError(f"Alias '{alias_name}' not found in registered aliases")
                    alias_value = registered_aliases[bracketed_name]
                    rewrites.append((alias_value, bracketed_name))
            elif args.from_paths:
                if not args.to_paths:
                    raise ValueError("--to PATH is required when using --from")
                if len(args.from_paths) != len(args.to_paths):
                    raise ValueError("Number of --from and --to arguments must match")
                rewrites = list(zip(args.from_paths, args.to_paths))

        # Only load input object after argument validation
        input_url = Url(args.input_path)
        try:
            obj = get_input_object(input_url)
        except Exception as e:
            logger.error(f"Failed to load input object: {e}")
            raise

        if args.command == "list":
            list_aliases([input_url], obj, columns)
        elif args.command == "replace":
            replace_aliases(
                [input_url],
                obj,
                columns,
                rewrites,
                process_parents=not args.no_process_parents,
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
