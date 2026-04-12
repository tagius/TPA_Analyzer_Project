"""Compatibility wrapper for the packaged plotting engine."""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tpa_analyzer.core.models import FigureConfig, GraphSpec, PlotStyleConfig
from tpa_analyzer.plotting.engine import (
    build_mean_band,
    export_qc_report,
    expand_graph_spec_jobs,
    normalize_graph_spec,
    plot_custom_graphs,
    plot_grouped_metrics,
    plot_overlay_traces,
    plot_trace_stack,
    serialize_graph_specs,
    validate_graph_spec,
)
from tpa_analyzer.plotting.registry import VARIABLE_REGISTRY

__all__ = [
    "FigureConfig",
    "GraphSpec",
    "PlotStyleConfig",
    "VARIABLE_REGISTRY",
    "build_mean_band",
    "export_qc_report",
    "expand_graph_spec_jobs",
    "normalize_graph_spec",
    "plot_custom_graphs",
    "plot_grouped_metrics",
    "plot_overlay_traces",
    "plot_trace_stack",
    "serialize_graph_specs",
    "validate_graph_spec",
]
