"""Logging setup for the TPA Analyzer."""

from __future__ import annotations

import logging

from tpa_analyzer.config.settings import AppSettings


def configure_logging(settings: AppSettings) -> None:
    """Configure package logging once at application startup."""
    level_name = settings.log_level.upper().strip() or "INFO"
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the requested module."""
    return logging.getLogger(name)
