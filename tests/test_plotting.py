"""Tests for plot-spec validation and expansion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tpa_analyzer.core.errors import PlotSpecError
from tpa_analyzer.core.models import (
    CustomGraphAxisLayer,
    CustomGraphOverlay,
    CustomGraphSpec,
    FigureConfig,
    GraphSpec,
    PlotStyleConfig,
)
from tpa_analyzer.core.exporting import export_plot_bundle
from tpa_analyzer.plotting.engine import (
    expand_graph_spec_jobs,
    normalize_graph_spec,
    plot_custom_graphs,
    validate_graph_spec,
)
from tpa_analyzer.ui.app import filter_assigned_plot_export_payload


def test_expand_graph_spec_jobs_creates_one_job_per_x_axis() -> None:
    """Multi-select x variables should expand into one render job per x value."""
    spec = GraphSpec(
        title="Trace Job",
        plot_type="trace",
        x_cols=["Time (s)", "Aligned Time (s)"],
        y_cols=["Force (N)", "Deformation (mm)"],
    )
    jobs = expand_graph_spec_jobs(spec)
    assert [job.x_label for job in jobs] == ["Time (s)", "Aligned Time (s)"]


def test_validate_graph_spec_rejects_mixed_sources() -> None:
    """Trace plots must not accept metric y variables."""
    spec = GraphSpec(
        title="Mixed",
        plot_type="trace",
        x_cols=["Time (s)"],
        y_cols=["Hardness (N)"],
    )
    with pytest.raises(PlotSpecError):
        validate_graph_spec(spec)


def test_normalize_graph_spec_migrates_legacy_payload() -> None:
    """Legacy graph-spec payloads should migrate into the typed model."""
    migrated = normalize_graph_spec(
        {
            "title": "Legacy",
            "x_col": "Time (s)",
            "y_vars": "Force (N), Deformation (mm)",
            "mode": "panel",
        }
    )
    assert migrated.plot_type == "trace"
    assert migrated.x_cols == ["Time (s)"]
    assert migrated.y_cols == ["Force (N)", "Deformation (mm)"]


def test_filter_assigned_plot_export_payload_excludes_blank_groups() -> None:
    trace_df = pd.DataFrame(
        [
            {"Filename": "a.csv", "Group": "", "Aligned Time (s)": 0.0, "Force (N)": 1.0},
            {"Filename": "b.csv", "Group": "Control", "Aligned Time (s)": 0.0, "Force (N)": 2.0},
        ]
    )
    metrics_df = pd.DataFrame(
        [
            {"Filename": "a.csv", "Group": "", "Hardness (N)": 1.0},
            {"Filename": "b.csv", "Group": "Control", "Hardness (N)": 2.0},
        ]
    )
    qc_df = pd.DataFrame(
        [
            {"Filename": "a.csv", "Group": ""},
            {"Filename": "b.csv", "Group": "Control"},
        ]
    )
    stats_results = {
        "Hardness (N)": {
            "summary_df": pd.DataFrame(
                [
                    {"Group": "", "Mean": 1.0, "SD": 0.0, "Significance": ""},
                    {"Group": "Control", "Mean": 2.0, "SD": 0.0, "Significance": "a"},
                ]
            ),
            "pairwise_df": pd.DataFrame(
                [
                    {"Group1": "", "Group2": "Control", "P Value": 0.5},
                    {"Group1": "Control", "Group2": "", "P Value": 0.5},
                ]
            ),
            "test_info": {"group_col": "Group", "group_order": ["", "Control"]},
        }
    }

    filtered_trace, filtered_metrics, filtered_qc, filtered_stats, filtered_order = filter_assigned_plot_export_payload(
        trace_df=trace_df,
        metrics_df=metrics_df,
        qc_df=qc_df,
        stats_results=stats_results,
        group_order=["", "Control"],
    )

    assert filtered_trace["Group"].tolist() == ["Control"]
    assert filtered_metrics["Group"].tolist() == ["Control"]
    assert filtered_qc["Group"].tolist() == ["Control"]
    assert filtered_stats["Hardness (N)"]["summary_df"]["Group"].tolist() == ["Control"]
    assert filtered_stats["Hardness (N)"]["pairwise_df"].empty
    assert filtered_stats["Hardness (N)"]["test_info"]["group_order"] == ["Control"]
    assert filtered_order == ["Control"]


def test_plot_custom_graphs_renders_one_dual_axis_composed_trace_plot(tmp_path) -> None:
    """A composed trace recipe should render once with left and right axes."""
    trace_df = pd.DataFrame(
        [
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 0.0, "Force (N)": 1.0, "Deformation (mm)": 0.20},
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 0.5, "Force (N)": 2.0, "Deformation (mm)": 0.35},
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 1.0, "Force (N)": 1.6, "Deformation (mm)": 0.40},
            {"File": "b.csv", "Filename": "b.csv", "Group": "Control", "Time (s)": 0.0, "Force (N)": 1.2, "Deformation (mm)": 0.18},
            {"File": "b.csv", "Filename": "b.csv", "Group": "Control", "Time (s)": 0.5, "Force (N)": 2.1, "Deformation (mm)": 0.33},
            {"File": "b.csv", "Filename": "b.csv", "Group": "Control", "Time (s)": 1.0, "Force (N)": 1.7, "Deformation (mm)": 0.38},
        ]
    )
    spec = CustomGraphSpec(
        title="Dual Axis Graph",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force (N)", role="left")],
        right_axis=CustomGraphAxisLayer(variable="Deformation (mm)", role="right"),
    )

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control"],
    )

    assert len(payload["paths"]) == 1
    assert payload["warnings"] == []
    assert Path(payload["paths"][0]).exists()


def test_plot_custom_graphs_warns_when_composed_overlay_prereqs_are_missing(tmp_path) -> None:
    """Overlay recipes should warn narrowly when required QC data is unavailable."""
    trace_df = pd.DataFrame(
        [
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 0.0, "Force Corrected (N)": 1.0},
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 0.5, "Force Corrected (N)": 2.0},
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 1.0, "Force Corrected (N)": 1.5},
        ]
    )
    spec = CustomGraphSpec(
        title="Overlay Missing",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        overlay=CustomGraphOverlay(kind="segment", key="b1_start_to_peak1"),
    )

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control"],
    )

    assert len(payload["paths"]) == 1
    assert Path(payload["paths"][0]).exists()
    assert len(payload["warnings"]) == 1
    assert "Overlay Missing" in payload["warnings"][0]
    assert "overlay" in payload["warnings"][0].lower()
    assert "missing" in payload["warnings"][0].lower()


def test_plot_custom_graphs_overlay_warning_does_not_abort_other_specs(tmp_path) -> None:
    """Missing overlay prereqs should warn for that spec and still export other specs."""
    trace_df = pd.DataFrame(
        [
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 0.0, "Force Corrected (N)": 1.0, "Force (N)": 1.0},
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 0.5, "Force Corrected (N)": 2.0, "Force (N)": 2.0},
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 1.0, "Force Corrected (N)": 1.5, "Force (N)": 1.5},
        ]
    )
    overlay_spec = CustomGraphSpec(
        title="Overlay Missing",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        overlay=CustomGraphOverlay(kind="segment", key="b1_start_to_peak1"),
    )
    valid_spec = CustomGraphSpec(
        title="Plain Trace",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force (N)", role="left")],
    )

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        graph_specs=[overlay_spec, valid_spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control"],
    )

    assert len(payload["paths"]) == 2
    assert all(Path(path).exists() for path in payload["paths"])
    assert len(payload["warnings"]) == 1
    assert "Overlay Missing" in payload["warnings"][0]
    assert "overlay" in payload["warnings"][0].lower()


def test_plot_custom_graphs_overlay_render_failure_warns_and_still_saves_plot(tmp_path, monkeypatch) -> None:
    """Overlay render failures should roll back partial overlay artists before saving."""
    trace_df = pd.DataFrame(
        [
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Force Corrected (N)": 1.0,
                "Bite1 Start Index": 0,
                "Peak1 Index": 0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Force Corrected (N)": 2.0,
                "Bite1 Start Index": 0,
                "Peak1 Index": 1,
            },
        ]
    )
    spec = CustomGraphSpec(
        title="Overlay Explodes",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        overlay=CustomGraphOverlay(kind="segment", key="b1_start_to_peak1"),
    )
    captured_axes: list[object] = []

    def boom(ax, *args, **kwargs):
        captured_axes.append(ax)
        ax.plot([0.0, 0.5], [1.0, 1.5], color="#ff00ff", label="Partial Overlay Artifact")
        raise PlotSpecError("overlay exploded")

    monkeypatch.setattr("tpa_analyzer.plotting.engine._render_composed_overlay", boom)

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control"],
    )

    assert len(payload["paths"]) == 1
    assert Path(payload["paths"][0]).exists()
    assert len(payload["warnings"]) == 1
    assert "Overlay Explodes" in payload["warnings"][0]
    assert "overlay exploded" in payload["warnings"][0].lower()
    assert len(captured_axes) == 1
    assert all(line.get_label() != "Partial Overlay Artifact" for line in captured_axes[0].lines)


def test_export_plot_bundle_warns_instead_of_aborting_when_trace_exports_are_unavailable(tmp_path) -> None:
    """Batch plot export should degrade to warnings when trace plots cannot render."""
    warnings = export_plot_bundle(
        root=tmp_path,
        trace_df=pd.DataFrame(),
        metrics_df=pd.DataFrame(),
        qc_df=pd.DataFrame(),
        stats_results={},
        graph_specs=[],
        style=PlotStyleConfig(),
        fig_cfg=FigureConfig(dpi=72),
        overlay_mode="mean_band",
        band_mode="sd",
        group_order=[],
        include_plots_dir=False,
    )

    assert warnings
    assert any("default" in warning.lower() or "stack" in warning.lower() for warning in warnings)
    assert any("overlay" in warning.lower() for warning in warnings)


def test_plot_custom_graphs_warns_for_none_payload_and_continues(tmp_path) -> None:
    """Malformed saved specs should warn per-spec and not abort later valid specs."""
    trace_df = pd.DataFrame(
        [
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 0.0, "Force (N)": 1.0},
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 0.5, "Force (N)": 2.0},
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 1.0, "Force (N)": 1.5},
        ]
    )
    valid_spec = GraphSpec(
        title="Still Renders",
        plot_type="trace",
        x_cols=["Time (s)"],
        y_cols=["Force (N)"],
    )

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        graph_specs=[None, valid_spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control"],
    )

    assert len(payload["paths"]) == 1
    assert Path(payload["paths"][0]).exists()
    assert len(payload["warnings"]) == 1
    assert "invalid" in payload["warnings"][0].lower()
