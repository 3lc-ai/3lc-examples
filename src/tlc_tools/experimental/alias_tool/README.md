# Alias Tool

A tool for managing and modifying paths in 3LC objects. It is intended to apply
and rewrite aliases, but is basically a flexible string-replacement wrapper. The
tool supports listing existing aliases, applying registered aliases, and
performing path rewrites.

## Usage

The tool provides a `replace` subcommand with various options for handling aliases and path rewrites:

```bash
# List all aliases in a table (default behavior)
3lc alias replace path/to/table

# List aliases in specific columns
3lc alias replace path/to/table --columns "image_path,mask_path"

# Apply registered aliases
3lc alias replace path/to/table --apply DATA_PATH          # Single alias
3lc alias replace path/to/table --apply "DATA_PATH,CACHE"  # Multiple aliases

# Replace specific paths with new paths
3lc alias replace path/to/table --from /data/path --to /new/path
3lc alias replace path/to/table \
    --from /data/path1 --to /new/path1 \
    --from /data/path2 --to /new/path2

# Control parent table processing
3lc alias replace path/to/table --no-process-parents  # Skip parent tables

# Control output verbosity
3lc alias replace path/to/table -v   # Debug output
3lc alias replace path/to/table -q   # Quiet mode (warnings/errors only)
```

## Examples

### Listing Aliases

List all aliases in a table:
```bash
$ 3lc alias replace data/images
Found alias '<DATA_PATH>' in column 'image_path'
Found alias '<CACHE_PATH>' in column 'mask_path'
```

List aliases in specific columns:
```bash
$ 3lc alias replace data/images --columns "image_path"
Found alias '<DATA_PATH>' in column 'image_path'
```

### Applying Aliases

Apply a single registered alias:
```bash
$ 3lc alias replace data/images --apply DATA_PATH
Rewrote 5 occurrences of '/data/images' to '<DATA_PATH>' in column 'image_path'
```

Apply multiple aliases:
```bash
$ 3lc alias replace data/images --apply "DATA_PATH,CACHE_PATH"
Rewrote 5 occurrences of '/data/images' to '<DATA_PATH>' in column 'image_path'
Rewrote 3 occurrences of '/cache/masks' to '<CACHE_PATH>' in column 'mask_path'
```

### Path Rewrites

Replace a single path:
```bash
$ 3lc alias replace data/images --from /old/path --to /new/path
Rewrote 2 occurrences of '/old/path' to '/new/path' in column 'image_path'
```

Replace multiple paths:
```bash
$ 3lc alias replace data/images \
    --from /old/path1 --to /new/path1 \
    --from /old/path2 --to /new/path2
Rewrote 2 occurrences of '/old/path1' to '/new/path1' in column 'image_path'
Rewrote 3 occurrences of '/old/path2' to '/new/path2' in column 'mask_path'
```

### Advanced Usage

Skip processing parent tables:
```bash
$ 3lc alias replace data/images --no-process-parents --apply DATA_PATH
Rewrote 5 occurrences of '/data/images' to '<DATA_PATH>' in column 'image_path'
```

Debug output with verbose mode:
```bash
$ 3lc alias replace data/images -v
Debug: Processing table: data/images
Debug: Has cache: True
Debug: Using cache URL: data/.cache/images
Found alias '<DATA_PATH>' in column 'image_path'
```

## Notes

- All modifications are performed in-place
- Aliases are only detected at the start of paths
- Parent table processing is enabled by default
- The tool supports basic arrays, chunked arrays, and struct arrays
