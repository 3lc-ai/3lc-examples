"""Tool for managing URL aliases in 3LC objects."""

from .find_aliases import list_aliases
from .replace import replace_aliases

__all__ = ["list_aliases", "replace_aliases"]
