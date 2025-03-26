"""Tool for managing URL aliases in 3LC objects."""

from .alias import main
from .list_aliases import list_aliases
from .replace_aliases import replace_aliases

__all__ = ["main", "list_aliases", "replace_aliases"]
