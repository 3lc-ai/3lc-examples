from __future__ import annotations

import io
import logging

import pyarrow as pa
import pyarrow.parquet as pq
from tlc.core import Run, Table, Url, UrlAdapterRegistry

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
    root_logger = logging.getLogger("tlc_tools.experimental.alias_tool")
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    # Prevent duplicate logging
    root_logger.propagate = False


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

    raise ValueError(f"Input file '{input_url}' is not a valid 3LC object or Parquet file.")
