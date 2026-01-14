from __future__ import annotations

import os
import sys
from typing import Optional

import loguru
from loguru import logger


def create_logger(
    name: str = "loggy",
    path: str = "./",
    filename: str = "logfile.log",
    format: Optional[str] = None,
    level: str = "DEBUG",
    *file_args,
    **file_kwargs,
) -> loguru.Logger:
    """Instantiates a logging object with loguru and returns that.
    - Time format: hh:mm AM/PM
    - Date/time and logger name: white dim
    - Level colors:
        TRACE    -> white dim
        DEBUG    -> white
        INFO     -> blue
        SUCCESS  -> green bold
        WARNING  -> yellow
        ERROR    -> red bold
        CRITICAL -> red bold underline
    - Extra file sink arguments are forwarded via *file_args/**file_kwargs.

    Args:
        name (str, optional): the name of the logging object. Defaults to "loggy".
        path (str, optional): the path where the log file will be saved. Defaults to "./".
        filename (str, optional): the filename of the log file. Defaults to "logfile.log".
        format (str, optional): the format of the logging messages. Defaults to None.
        level (str, optional): the level of logging. Defaults to "DEBUG"
        *file_args, **file_kwargs: more optional arguments to be passed to the file sink

    Returns:
        logger: logging object
    """
    logger.remove()

    fmt = format or (
        "<white><dim>{time:%I:%M %p}</dim></white> | "
        "<white><dim>{name:<8}</dim></white> | "
        "<level>{level:<9}</level> | <level>{message}</level>"
    )

    styles = {
        "TRACE": "<white><dim>",
        "DEBUG": "<white><normal>",
        "INFO": "<blue><normal>",
        "SUCCESS": "<green><bold>",
        "WARNING": "<yellow><normal>",
        "ERROR": "<red><bold>",
        "CRITICAL": "<bg red><fg black><bold>",
    }
    for lvl, style in styles.items():
        logger.level(lvl, color=style)

    logger.add(sys.stderr, format=fmt, level=level, colorize=True, backtrace=False)
    logger.add(
        os.path.join(path, filename),
        format=fmt,
        level=level,
        backtrace=False,
        *file_args,
        **file_kwargs,
    )

    return logger
