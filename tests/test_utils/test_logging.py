"""Tests for structured logging setup."""

import sys
from unittest.mock import MagicMock, patch

from rag_agent.utils.logging import setup_logging


def test_setup_logging_removes_default_handler():
    mock_settings = MagicMock(log_level="INFO")
    with (
        patch("rag_agent.utils.logging.logger.remove") as mock_remove,
        patch("rag_agent.utils.logging.logger.add") as mock_add,
        patch("rag_agent.utils.logging.logger.debug") as mock_debug,
        patch("rag_agent.utils.logging.settings", mock_settings),
    ):
        setup_logging()

    mock_remove.assert_called_once()


def test_setup_logging_adds_stderr_handler_with_level_from_settings():
    mock_settings = MagicMock(log_level="WARNING")
    with (
        patch("rag_agent.utils.logging.logger.remove"),
        patch("rag_agent.utils.logging.logger.add") as mock_add,
        patch("rag_agent.utils.logging.logger.debug"),
        patch("rag_agent.utils.logging.settings", mock_settings),
    ):
        setup_logging()

    mock_add.assert_called_once()
    args, kwargs = mock_add.call_args
    assert args[0] is sys.stderr
    assert kwargs["level"] == "WARNING"


def test_setup_logging_calls_debug_after_configuration():
    mock_settings = MagicMock(log_level="DEBUG")
    with (
        patch("rag_agent.utils.logging.logger.remove"),
        patch("rag_agent.utils.logging.logger.add"),
        patch("rag_agent.utils.logging.logger.debug") as mock_debug,
        patch("rag_agent.utils.logging.settings", mock_settings),
    ):
        setup_logging()

    mock_debug.assert_called_once()
    assert mock_debug.call_args[0][0] == "Logging configured at level {}"
    assert mock_debug.call_args[0][1] == "DEBUG"
