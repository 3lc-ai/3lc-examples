"""Tool for managing URL aliases in 3LC objects."""

from .list_aliases import list_aliases
from .main import main
from .replace_aliases import replace_aliases

__all__ = ["main", "list_aliases", "replace_aliases"]
