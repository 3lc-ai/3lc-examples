"""Common logging setup for CLI tools."""

from __future__ import annotations

import logging


def setup_logging(verbosity: int = 0, quiet: bool = False, logger_name: str = "tlc_tools") -> None:
    """Configure logging for CLI tools.

    Args:
        verbosity: 0 for default (INFO), 1 for DEBUG
        quiet: If True, only show WARNING and above
        logger_name: Optional logger name to configure.
    """
    # Create console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")  # Simple format, just the message
    console_handler.setFormatter(formatter)

    # Set log level based on verbosity/quiet
    if quiet:
        level = logging.WARNING
    elif verbosity > 0:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Configure the specified logger or root logger
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)

    # Prevent duplicate logging
    logger.propagate = False
