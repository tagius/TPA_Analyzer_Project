"""Compatibility wrapper for the packaged TPA analysis engine."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tpa_analyzer.analysis.tpa import (
    TPAConfig,
    build_metrics_row,
    calculate_tpa,
    generate_plots,
    infer_group_from_filename,
    parse_zwick_data,
)

__all__ = [
    "TPAConfig",
    "build_metrics_row",
    "calculate_tpa",
    "generate_plots",
    "infer_group_from_filename",
    "parse_zwick_data",
]
