# Introduction

This module contains experimental scripts and notebooks, which might be moved into the top level `tlc_tools` package when they are ready.

Please note that code in this directory is subject to change without notice. We recommend not using it in production code, but rather to use it for one-off use cases and inspiration.

## Adding a new experimental tool

When adding a new experimental tool, please add a `README.md` file in the directory to describe the tool and how to use it. If it has a command line interface, please decorate it with the `@register_tool` decorator
with `experimental=True`:

```python
from tlc_tools.cli import register_tool

@register_tool(experimental=True, description="A dummy tool for demonstration")
def cli_main(args=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description="...")
    ...
```
