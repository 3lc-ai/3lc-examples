# Alias Tool

```bash
# Main subcommands
3lc alias manage  # For managing aliases (register/list/delete)
3lc alias replace # For string replacement operations in tables/runs

# Replace subcommand examples:

# List Mode (default) - Shows aliases without modifying
# Default output (INFO level) shows found aliases and basic processing info
3lc alias replace path/to/table

# Replace a path with an alias
3lc alias replace path/to/table \
    --from "/data/project/images" \
    --to "<PROJECT_IMAGES>"

# Replace multiple patterns (can mix paths and aliases)
3lc alias replace path/to/table \
    --from "/data/project/images" --to "<PROJECT_IMAGES>" \
    --from "<OLD_ALIAS>" --to "<NEW_ALIAS>"

# Column-specific replacement (comma-separated list)
3lc alias replace path/to/table \
    --from "/data/project/images" \
    --to "<PROJECT_IMAGES>" \
    --columns "image_path,mask_path"

# Apply registered aliases (uses built-in precedence)
3lc alias replace path/to/table \
    --apply-registered

# Apply specific registered alias
3lc alias replace path/to/table \
    --apply-alias "PROJECT_IMAGES"

# Skip processing parent tables
3lc alias replace path/to/table \
    --from "/data/project/images" \
    --to "<PROJECT_IMAGES>" \
    --no-process-parents

# Verbosity control
3lc alias replace path/to/table            # Default (INFO level): Shows found aliases, changes made
3lc alias replace path/to/table -v         # DEBUG level: Detailed processing info
3lc alias replace path/to/table --quiet    # WARNING level: Only shows errors/warnings

# Manage subcommand examples (for completeness):

# List all registered aliases
3lc alias manage list

# Register a new alias
3lc alias manage register PROJECT_IMAGES "/data/project/images" --scope project

# Delete an alias
3lc alias manage delete PROJECT_IMAGES --scope project

# Show alias details
3lc alias manage show PROJECT_IMAGES
```
