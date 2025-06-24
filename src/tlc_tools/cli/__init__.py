"""CLI package for tlc-tools."""

from .registry import register_tool  # noqa: I001
from .commands.alias import main as alias_main  # noqa: F401, F403, I001
from .commands.augment_instance_table import main as augment_instance_table_main  # noqa: F401, F403, I001
from .commands.metric_jumps_cli import main as metric_jumps_main  # noqa: F401, F403, I001

__all__ = ["register_tool"]
