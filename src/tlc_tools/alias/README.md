# Alias Tool

A tool for managing and modifying paths in 3LC objects. It provides two main functions:

1. Listing existing aliases in tables
2. Replacing paths with aliases or new paths

## Basic Usage

The tool has two subcommands: `list` and `replace`.

### Listing Aliases

```bash
# List all aliases in a table
3lc-tools run list path/to/table

# List aliases in specific columns
3lc-tools run list path/to/table --columns "image_path,mask_path"
```

Example output:

```bash
$ 3lc-tools run list data/images
Found alias '<DATA_PATH>' in column 'image_path'
Found alias '<CACHE_PATH>' in column 'mask_path'
```

### Replacing Paths

```bash
# Apply registered aliases
3lc-tools run replace path/to/table --apply DATA_PATH          # Single alias
3lc-tools run replace path/to/table --apply "DATA_PATH,CACHE"  # Multiple aliases

# Replace specific paths
3lc-tools run replace path/to/table --from /old/path --to /new/path
3lc-tools run replace path/to/table \
    --from /old/path1 --to /new/path1 \
    --from /old/path2 --to /new/path2

# Process specific columns
3lc-tools run replace path/to/table --columns "image_path,mask_path" --apply DATA_PATH

# Skip processing parent tables
3lc-tools run replace path/to/table --no-process-parents --apply DATA_PATH
```

Example output:

```bash
$ 3lc-tools run replace data/images --apply DATA_PATH
Rewrote 5 occurrences of '/data/images' to '<DATA_PATH>' in column 'image_path'
```

## Advanced Usage

### Verbosity Control

```bash
# Debug output
3lc-tools run list path/to/table -v

# Quiet mode (warnings/errors only)
3lc-tools run replace path/to/table -q
```

### Parent Table Processing

By default, the tool processes parent tables when handling Table objects. You can disable this with `--no-process-parents`:

```bash
# Skip parent table processing
3lc-tools run replace path/to/table --no-process-parents --apply DATA_PATH
```

## Notes

- All modifications are performed in-place
- Aliases are only detected at the start of paths
- Parent table processing is enabled by default
- Changes are automatically backed up before modification

## Limitations

- The tool currently only supports 3LC Tables and Parquet files. Support for 3LC Runs is planned.
- The tool currently only handles pyarrow columns contains strings or structs.

## TODO

- [ ] Add support for 3LC Runs
- [ ] Add support for list-like column values (e.g. lists of URLs)
- [ ] Add subcommand for setting aliases (3lc-tools alias set ... [--scope=project|global])
- [ ] Add subcommand for removing aliases (3lc-tools alias remove ... [--scope=project|global])
- [ ] Add subcommand for listing aliases (3lc-tools alias list ... [--scope=project|global])
- [ ] Allow supplying empty --apply to apply any relevant registered aliases
- [ ] Add tooling for inspecting "inputs" to 3LC objects (relative URLs, aliased URLs, bulk data, metric tables, etc.)
- [ ] Add back full column name when listing/replacing in nested structs
