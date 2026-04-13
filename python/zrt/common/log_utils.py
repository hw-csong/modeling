from __future__ import annotations

import logging
import os
import sys

_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


def _get_log_level() -> int:
    env = os.environ.get("LOG_LEVEL", "INFO").upper()
    return _LOG_LEVEL_MAP.get(env, logging.INFO)


class Logger:
    _instances: dict[str, "Logger"] = {}

    def __init__(self, name: str | None = None):
        self._logger = logging.getLogger(name or "zrt")
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
            self._logger.addHandler(handler)
        self._logger.setLevel(_get_log_level())

    @classmethod
    def get(cls, name: str | None = None) -> "Logger":
        key = name or "zrt"
        if key not in cls._instances:
            cls._instances[key] = cls(name)
        return cls._instances[key]

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    def set_level(self, level: str):
        self._logger.setLevel(_LOG_LEVEL_MAP.get(level.upper(), logging.INFO))


logger = Logger.get()
