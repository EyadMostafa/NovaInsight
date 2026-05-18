from __future__ import annotations

import copy
import logging

_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",     # cyan
    logging.INFO: "\033[32m",      # green
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[35m",  # magenta
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)
        color = _LEVEL_COLORS.get(record.levelno, "")
        record.levelname = f"{_BOLD}{color}{record.levelname:<8}{_RESET}"
        record.name = f"\033[34m{record.name}{_RESET}"
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
