"""Structured logging setup using loguru."""

import sys

from loguru import logger

from rag_agent.config import settings


def setup_logging() -> None:
    """Configure loguru logger for the application.

    Removes the default handler and adds a customized one based on
    the application settings.
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.debug("Logging configured at level {}", settings.log_level)


__all__ = ["logger", "setup_logging"]
