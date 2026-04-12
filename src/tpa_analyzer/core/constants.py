"""Shared constants for the TPA Analyzer."""

from __future__ import annotations

from typing import Final

SESSION_FILE_NAME: Final[str] = ".tpa_analyzer_session.json"
SESSION_SCHEMA_VERSION: Final[int] = 2

COMPUTED_METRICS: Final[tuple[str, ...]] = (
    "Hardness (N)",
    "Cohesiveness",
    "Springiness",
    "Resilience",
    "Chewiness",
    "Adhesiveness",
    "Modulus (kPa)",
)

DEFAULT_TRACE_Y_VARIABLES: Final[tuple[str, ...]] = (
    "Force (N)",
    "Deformation (mm)",
)

DEFAULT_METRIC_Y_VARIABLES: Final[tuple[str, ...]] = (
    "Hardness (N)",
    "Modulus (kPa)",
)

SUPPORTED_DATA_EXTENSIONS: Final[frozenset[str]] = frozenset({".csv", ".tra"})

LAYOUT_WIDE_MIN: Final[int] = 160
LAYOUT_MEDIUM_MIN: Final[int] = 120
