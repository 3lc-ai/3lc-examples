# tlc-ndjson-exporter

Minimal example showing how to write and distribute a **3LC table exporter plugin**.

This exporter writes a 3LC {py:class}`~tlc.Table` as [newline-delimited JSON](https://jsonlines.org/) — one JSON object per row, one row per line. That format works well with streaming pipelines and tools like `jq`, DuckDB's `read_ndjson`, or BigQuery load jobs.

The point of this example is the **entry-point mechanism**: once installed, 3LC picks up the `ndjson` format automatically, and it works with both the Python API (`table.export("out.ndjson")`) and the CLI (`3lc export ... out.ndjson`). No code changes to 3LC itself are required.

## Quick start

```bash
# Install the plugin into the same environment where `3lc` is installed
pip install -e .

# Confirm 3LC has discovered it
3lc exporters list
#   FORMAT  CLASS           MODULE                         SOURCE
#   csv     CsvExporter     tlc.core.export.exporters.csv  builtin
#   json    DefaultJsonExporter …                          builtin
#   coco    CocoExporter    …                              builtin
#   yolo    YoloExporter    …                              builtin
#   ndjson  NdjsonExporter  tlc_ndjson_exporter.exporter   entrypoint   ← here
```

## Usage

### Python

```python
import tlc

table = tlc.Table.from_url("path/to/table")

# Format inferred from the .ndjson extension
table.export("rows.ndjson")

# Or specify the format explicitly
table.export("rows.jsonl", format="ndjson")

# Weight-filter before writing
table.export("cleaned.ndjson", weight_threshold=0.5)
```

### CLI

```bash
3lc export path/to/table rows.ndjson
3lc export path/to/table rows.ndjson --ndjson-ensure-ascii
3lc export path/to/table rows.ndjson --weight-threshold 0.5
```

The `--ndjson-ensure-ascii` flag is derived automatically from the `ensure_ascii` parameter on `NdjsonExporter.export_row`. Any kwarg you expose on your exporter's main method is wired up to the CLI the same way (see [the 3LC export docs](https://docs.3lc.ai/user-guide/tables/export.html#cli-option-naming)).

## How it works

### The exporter class

See [`src/tlc_ndjson_exporter/exporter.py`](src/tlc_ndjson_exporter/exporter.py). The whole implementation fits in one method:

```python
import json
import tlc

class NdjsonExporter(tlc.RowExporter):
    supported_format = "ndjson"
    file_extensions = frozenset({".ndjson", ".jsonl"})
    separator = "\n"

    def export_row(self, row, ensure_ascii: bool = False, **_) -> str:
        return json.dumps(row, ensure_ascii=ensure_ascii, default=str)
```

Subclassing {py:class}`tlc.RowExporter` means the framework handles iteration, weight filtering, progress reporting, and writing to the output URL. You only implement the per-row conversion.

Note: we do **not** apply the `@tlc.register_exporter` decorator here — registration happens via the entry point (see below). Using the decorator for an entry-point plugin would cause the exporter to be registered twice (once at import, once at discovery) and show up with `source=runtime` instead of `source=entrypoint`. Use the decorator only for exporters that live inside your own application code and aren't distributed as a plugin.

### Entry-point registration

The `pyproject.toml` declares:

```toml
[project.entry-points."tlc.exporters"]
ndjson = "tlc_ndjson_exporter.exporter:NdjsonExporter"
```

When 3LC needs to find an exporter (the first time `table.export(...)` is called, or when the CLI builds its options), it scans all installed packages for entry points in the `tlc.exporters` group, instantiates the class with no arguments, and registers it. That's all that's needed — no manual import, no config file entry.

## Building your own exporter

Three patterns are available, from simplest to most flexible:

1. {py:class}`tlc.RowExporter` — one method, `export_row`, converts a single row to a string. Used here.
2. {py:class}`tlc.SerializingExporter` — `serialize` returns the whole output as a string (useful when you need headers or cross-row state).
3. {py:class}`tlc.Exporter` — full control; override `_do_export` (for directory output, multi-file formats, etc.).

See the [Custom Exporters section of the 3LC docs](https://docs.3lc.ai/user-guide/tables/export.html#custom-exporters) for details on each pattern.
