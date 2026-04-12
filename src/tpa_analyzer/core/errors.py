"""Domain-specific exceptions used across the TPA Analyzer."""

from __future__ import annotations


class TPAAnalyzerError(Exception):
    """Base exception for package-specific failures."""


class DataParseError(TPAAnalyzerError):
    """Raised when an input file cannot be parsed into required columns."""


class AnalysisError(TPAAnalyzerError):
    """Raised when TPA analysis cannot produce valid cycle metrics."""


class PlotSpecError(TPAAnalyzerError):
    """Raised when a custom plot specification is invalid."""


class SessionError(TPAAnalyzerError):
    """Raised when session data cannot be loaded or saved safely."""
