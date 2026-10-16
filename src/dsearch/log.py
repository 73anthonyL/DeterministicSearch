"""Logging: silent by default, detailed on request.

`DSEARCH_LOG=DEBUG` (or `dsearch -v`) prints what the pipeline is doing to
stderr: model load times, corpus sizes, embedding throughput, per-page
extraction notes. Nothing is logged to stdout, so `--json` output stays clean.
"""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

ENV_VAR = "DSEARCH_LOG"
DEFAULT_LEVEL = "WARNING"

_HANDLER_FLAG = "_dsearch_handler"


def level_from_env(explicit: str | None = None) -> int:
    """The log level: `explicit`, else `$DSEARCH_LOG`, else WARNING. Unknown names fall back."""
    name = (explicit or os.environ.get(ENV_VAR) or DEFAULT_LEVEL).strip().upper()
    numeric = logging.getLevelName(name)
    return numeric if isinstance(numeric, int) else logging.WARNING


def configure(level: str | None = None, *, console: Console | None = None) -> int:
    """Install one Rich handler on the `dsearch` logger and return the level set.

    Calling it again replaces the handler rather than stacking a second one,
    so tests and the Streamlit app can reconfigure freely.
    """
    numeric = level_from_env(level)
    logger = logging.getLogger("dsearch")
    logger.setLevel(numeric)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_FLAG, False):
            logger.removeHandler(handler)
    handler = RichHandler(
        console=console or Console(stderr=True),
        show_path=False,
        show_time=False,
        markup=False,
        rich_tracebacks=False,
    )
    setattr(handler, _HANDLER_FLAG, True)
    logger.addHandler(handler)
    logger.propagate = False
    return numeric
