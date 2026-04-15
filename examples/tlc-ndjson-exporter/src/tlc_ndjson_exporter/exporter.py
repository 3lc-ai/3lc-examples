"""Newline-delimited JSON (NDJSON / JSON Lines) exporter for 3LC tables.

Each row of the table becomes one JSON object on its own line. This format is handy
for streaming pipelines, log-like append workflows, and tools that process one record
at a time (jq, DuckDB's ``read_ndjson``, BigQuery load jobs, etc.).
"""

from __future__ import annotations

import json
from typing import Any

import tlc


class NdjsonExporter(tlc.RowExporter):
    """Export a 3LC table as newline-delimited JSON.

    The exporter is discovered automatically via the ``tlc.exporters`` entry point — no
    manual registration is required once this package is installed.

    Example:
        >>> import tlc
        >>> table = tlc.Table.from_url("path/to/table")
        >>> table.export("rows.ndjson")  # format inferred from extension
        >>> table.export("rows.jsonl", format="ndjson")  # or specify explicitly

    """

    supported_format = "ndjson"
    file_extensions = frozenset({".ndjson", ".jsonl"})
    separator = "\n"

    def export_row(self, row: dict[str, Any], ensure_ascii: bool = False, **_: Any) -> str:
        """Convert a single row to a single JSON object on one line.

        Args:
            row: The table row as a mapping of column name to value.
            ensure_ascii: If True, escape non-ASCII characters in the output.
                Exposed to the CLI as ``--ndjson-ensure-ascii``. Defaults to False,
                which leaves Unicode characters as-is (smaller, human-readable output).

        Returns:
            A single-line JSON string for this row.

        """
        return json.dumps(row, ensure_ascii=ensure_ascii, default=str)
