"""Compatibility wrapper for the packaged statistics engine."""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tpa_analyzer.stats.engine import StatsDecision, run_statistics

__all__ = ["StatsDecision", "run_statistics"]
