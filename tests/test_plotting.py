"""Tests for plot-spec validation and expansion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from tpa_analyzer.core.errors import PlotSpecError
from tpa_analyzer.core.exporting import export_plot_bundle
from tpa_analyzer.core.models import (
    CustomGraphAnnotation,
    CustomGraphAxisLayer,
    CustomGraphOverlay,
    CustomGraphSpec,
    FigureConfig,
    GraphSpec,
    PlotStyleConfig,
)
from tpa_analyzer.core.session import migrate_graph_specs
from tpa_analyzer.plotting import engine as plotting_engine
from tpa_analyzer.plotting.engine import (
    expand_composed_graph_spec,
    expand_graph_spec_jobs,
    normalize_graph_spec,
    plot_custom_graphs,
    plot_grouped_metrics,
    validate_graph_spec,
)
from tpa_analyzer.ui.app import filter_assigned_plot_export_payload


def _semantic_segment_trace_payload() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small trace/QC payload with two samples and one shared segment."""
    trace_df = pd.DataFrame(
        [
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Force Corrected (N)": 0.5,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Force Corrected (N)": 1.1,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 1.0,
                "Force Corrected (N)": 2.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 1.5,
                "Force Corrected (N)": 1.4,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 2.0,
                "Force Corrected (N)": 0.8,
            },
            {
                "File": "b.csv",
                "Filename": "b.csv",
                "Group": "Treatment",
                "Time (s)": 0.0,
                "Force Corrected (N)": 0.6,
            },
            {
                "File": "b.csv",
                "Filename": "b.csv",
                "Group": "Treatment",
                "Time (s)": 0.5,
                "Force Corrected (N)": 1.0,
            },
            {
                "File": "b.csv",
                "Filename": "b.csv",
                "Group": "Treatment",
                "Time (s)": 1.0,
                "Force Corrected (N)": 1.8,
            },
            {
                "File": "b.csv",
                "Filename": "b.csv",
                "Group": "Treatment",
                "Time (s)": 1.5,
                "Force Corrected (N)": 1.2,
            },
            {
                "File": "b.csv",
                "Filename": "b.csv",
                "Group": "Treatment",
                "Time (s)": 2.0,
                "Force Corrected (N)": 0.7,
            },
        ]
    )
    qc_df = pd.DataFrame(
        [
            {"Filename": "a.csv", "Group": "Control", "Bite1 Start Index": 1, "Peak1 Index": 3},
            {"Filename": "b.csv", "Group": "Treatment", "Bite1 Start Index": 1, "Peak1 Index": 3},
        ]
    )
    return trace_df, qc_df


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

    filtered_trace, filtered_metrics, filtered_qc, filtered_stats, filtered_order = (
        filter_assigned_plot_export_payload(
            trace_df=trace_df,
            metrics_df=metrics_df,
            qc_df=qc_df,
            stats_results=stats_results,
            group_order=["", "Control"],
        )
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
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Force (N)": 1.0,
                "Deformation (mm)": 0.20,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Force (N)": 2.0,
                "Deformation (mm)": 0.35,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 1.0,
                "Force (N)": 1.6,
                "Deformation (mm)": 0.40,
            },
            {
                "File": "b.csv",
                "Filename": "b.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Force (N)": 1.2,
                "Deformation (mm)": 0.18,
            },
            {
                "File": "b.csv",
                "Filename": "b.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Force (N)": 2.1,
                "Deformation (mm)": 0.33,
            },
            {
                "File": "b.csv",
                "Filename": "b.csv",
                "Group": "Control",
                "Time (s)": 1.0,
                "Force (N)": 1.7,
                "Deformation (mm)": 0.38,
            },
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
        qc_df=pd.DataFrame(),
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control"],
    )

    assert len(payload["paths"]) == 1
    assert payload["warnings"] == []
    assert Path(payload["paths"][0]).exists()


def test_expand_composed_graph_spec_rejects_incompatible_left_axis_units() -> None:
    """Composed graphs should reject left-axis variables with different units."""
    spec = CustomGraphSpec(
        title="Mixed Left Axis",
        x_domain="Time (s)",
        left_axis=[
            CustomGraphAxisLayer(variable="Force (N)", role="left"),
            CustomGraphAxisLayer(variable="Deformation (mm)", role="left"),
        ],
    )

    with pytest.raises(PlotSpecError):
        expand_composed_graph_spec(spec)


def test_plot_custom_graphs_generates_unique_filenames_for_duplicate_specs(tmp_path) -> None:
    """Repeated custom specs should save distinct files instead of overwriting."""
    trace_df = pd.DataFrame(
        [
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Force (N)": 1.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Force (N)": 2.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 1.0,
                "Force (N)": 1.5,
            },
        ]
    )
    repeated_specs = [
        CustomGraphSpec(
            title="Repeated Graph",
            x_domain="Time (s)",
            left_axis=[CustomGraphAxisLayer(variable="Force (N)", role="left")],
        ),
        CustomGraphSpec(
            title="Repeated Graph",
            x_domain="Time (s)",
            left_axis=[CustomGraphAxisLayer(variable="Force (N)", role="left")],
        ),
    ]

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=pd.DataFrame(),
        graph_specs=repeated_specs,
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control"],
    )

    assert len(payload["paths"]) == 2
    assert payload["warnings"] == []
    assert len({Path(path).name for path in payload["paths"]}) == 2
    assert all(Path(path).exists() for path in payload["paths"])


def test_plot_custom_graphs_renders_composed_overlay_from_qc_df(tmp_path) -> None:
    """Overlay recipes should use QC summary markers instead of trace columns."""
    trace_df = pd.DataFrame(
        [
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Force Corrected (N)": 1.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Force Corrected (N)": 2.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 1.0,
                "Force Corrected (N)": 1.5,
            },
        ]
    )
    qc_df = pd.DataFrame(
        [
            {
                "Filename": "a.csv",
                "Group": "Control",
                "Bite1 Start Index": 0,
                "Peak1 Index": 1,
            }
        ]
    )
    spec = CustomGraphSpec(
        title="Overlay From QC",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        overlay=CustomGraphOverlay(kind="segment", key="b1_start_to_peak1"),
    )

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=qc_df,
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control"],
    )

    assert len(payload["paths"]) == 1
    assert Path(payload["paths"][0]).exists()
    assert payload["warnings"] == []


def test_plot_custom_graphs_warns_when_composed_overlay_prereqs_are_missing(tmp_path) -> None:
    """Overlay recipes should warn narrowly when required QC data is unavailable."""
    trace_df = pd.DataFrame(
        [
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Force Corrected (N)": 1.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Force Corrected (N)": 2.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 1.0,
                "Force Corrected (N)": 1.5,
            },
        ]
    )
    qc_df = pd.DataFrame([{"Filename": "a.csv", "Group": "Control"}])
    spec = CustomGraphSpec(
        title="Overlay Missing",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        overlay=CustomGraphOverlay(kind="segment", key="b1_start_to_peak1"),
    )

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=qc_df,
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


def test_plot_custom_graphs_renders_migrated_legacy_overlays(tmp_path) -> None:
    """Migrated legacy segment and annotation overlays should still render."""
    trace_df = pd.DataFrame(
        [
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Force Corrected (N)": 1.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Force Corrected (N)": 2.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 1.0,
                "Force Corrected (N)": 1.5,
            },
        ]
    )
    qc_df = pd.DataFrame(
        [
            {
                "Filename": "a.csv",
                "Group": "Control",
                "Bite1 Start Index": 0,
                "Peak1 Index": 1,
            }
        ]
    )
    migrated_specs = migrate_graph_specs(
        [
            {
                "title": "Legacy Segment",
                "x_domain": "Time (s)",
                "left_axis": [{"variable": "Force Corrected (N)", "role": "left"}],
                "overlay": {"kind": "segment", "key": "b1_start_to_peak1"},
            },
            {
                "title": "Legacy Annotation",
                "x_domain": "Time (s)",
                "left_axis": [{"variable": "Force Corrected (N)", "role": "left"}],
                "overlay": {"kind": "annotation", "key": "hardness_peak1"},
            },
        ]
    )

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=qc_df,
        graph_specs=migrated_specs,
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control"],
    )

    assert len(payload["paths"]) == 2
    assert all(Path(path).exists() for path in payload["paths"])
    assert payload["warnings"] == []


def test_plot_custom_graphs_overlay_warning_does_not_abort_other_specs(tmp_path) -> None:
    """Missing overlay prereqs should warn for that spec and still export other specs."""
    trace_df = pd.DataFrame(
        [
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Force Corrected (N)": 1.0,
                "Force (N)": 1.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Force Corrected (N)": 2.0,
                "Force (N)": 2.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 1.0,
                "Force Corrected (N)": 1.5,
                "Force (N)": 1.5,
            },
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
        qc_df=pd.DataFrame([{"Filename": "a.csv", "Group": "Control"}]),
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


def test_plot_custom_graphs_overlay_render_failure_warns_and_still_saves_plot(
    tmp_path, monkeypatch
) -> None:
    """Overlay render failures should roll back partial overlay artists before saving."""
    trace_df = pd.DataFrame(
        [
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Force Corrected (N)": 1.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Force Corrected (N)": 2.0,
            },
        ]
    )
    qc_df = pd.DataFrame(
        [
            {
                "Filename": "a.csv",
                "Group": "Control",
                "Bite1 Start Index": 0,
                "Peak1 Index": 1,
            }
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
        qc_df=qc_df,
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


def test_export_plot_bundle_warns_instead_of_aborting_when_trace_exports_are_unavailable(
    tmp_path,
) -> None:
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


def test_plot_grouped_metrics_supports_single_metric_grid(tmp_path: Path) -> None:
    """A single metric should render without tripping subplot grid zip mismatches."""
    stats_by_metric = {
        "Hardness (N)": {
            "summary_df": pd.DataFrame(
                [
                    {"Group": "Control", "Mean": 2.0, "SD": 0.1, "Significance": "a"},
                    {"Group": "Treatment", "Mean": 3.0, "SD": 0.2, "Significance": "b"},
                ]
            ),
            "pairwise_df": pd.DataFrame(),
            "test_info": {"group_col": "Group", "group_order": ["Control", "Treatment"]},
        }
    }

    output_path = tmp_path / "grouped_metrics.png"
    saved_path = plot_grouped_metrics(
        stats_by_metric=stats_by_metric,
        style=PlotStyleConfig(),
        output_path=output_path,
        figure_config=FigureConfig(dpi=72),
    )

    assert Path(saved_path) == output_path
    assert output_path.exists()


def test_export_plot_bundle_renders_custom_overlay_from_qc_df(tmp_path) -> None:
    """Batch export should pass QC summary markers into custom overlay rendering."""
    trace_df = pd.DataFrame(
        [
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Aligned Time (s)": 0.0,
                "Force (N)": 0.5,
                "Force Corrected (N)": 0.4,
                "Deformation (mm)": 0.1,
                "True Stress (kPa)": 1.0,
                "True Strain (%)": 0.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Aligned Time (s)": 0.5,
                "Force (N)": 1.2,
                "Force Corrected (N)": 1.1,
                "Deformation (mm)": 0.2,
                "True Stress (kPa)": 2.0,
                "True Strain (%)": 2.5,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 1.0,
                "Aligned Time (s)": 1.0,
                "Force (N)": 0.8,
                "Force Corrected (N)": 0.7,
                "Deformation (mm)": 0.3,
                "True Stress (kPa)": 1.5,
                "True Strain (%)": 5.0,
            },
        ]
    )
    qc_df = pd.DataFrame(
        [
            {
                "Filename": "a.csv",
                "Group": "Control",
                "Trigger Force (N)": 0.2,
                "Modulus Strain Min (%)": 0.0,
                "Modulus Strain Max (%)": 5.0,
                "Peak1 Index": 1,
                "Peak2 Index": 2,
                "Bite1 Start Index": 0,
                "Bite1 End Index": 1,
                "Bite2 Start Index": 2,
                "Bite2 End Index": 2,
            }
        ]
    )
    graph_specs = [
        CustomGraphSpec(
            title="Overlay Export",
            x_domain="Time (s)",
            left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
            overlay=CustomGraphOverlay(kind="segment", key="b1_start_to_peak1"),
        )
    ]

    warnings = export_plot_bundle(
        root=tmp_path,
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=qc_df,
        stats_results={},
        graph_specs=graph_specs,
        style=PlotStyleConfig(),
        fig_cfg=FigureConfig(dpi=72),
        overlay_mode="mean_band",
        band_mode="sd",
        group_order=["Control"],
        include_plots_dir=False,
    )

    custom_paths = list((tmp_path / "custom").glob("*.png"))
    assert warnings == []
    assert len(custom_paths) == 1


def test_plot_custom_graphs_warns_for_none_payload_and_continues(tmp_path) -> None:
    """Malformed saved specs should warn per-spec and not abort later valid specs."""
    trace_df = pd.DataFrame(
        [
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.0,
                "Force (N)": 1.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 0.5,
                "Force (N)": 2.0,
            },
            {
                "File": "a.csv",
                "Filename": "a.csv",
                "Group": "Control",
                "Time (s)": 1.0,
                "Force (N)": 1.5,
            },
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
        qc_df=pd.DataFrame(),
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


def test_plot_custom_graphs_renders_grouped_semantic_segment_graph(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grouped semantic-segment exports should slice traces and keep marker annotations."""
    trace_df, qc_df = _semantic_segment_trace_payload()
    spec = CustomGraphSpec(
        title="Grouped Segment",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        view_domain="semantic_segment",
        segment_key="b1_start_to_peak1",
        rebase_x=True,
        annotations=[CustomGraphAnnotation(kind="annotation", key="hardness_peak1")],
        overlay=CustomGraphOverlay(kind="segment", key="b1_start_to_peak1"),
        data_scope="grouped",
    )
    original_slice = plotting_engine._slice_trace_to_segment
    slice_calls: list[str] = []
    saved_figures: list[Figure] = []

    def tracking_slice(frame, qc_row, segment_key, x_label, *, rebase_x):
        slice_calls.append(str(frame["Filename"].iloc[0]))
        return original_slice(frame, qc_row, segment_key, x_label, rebase_x=rebase_x)

    def capture_savefig(self, *args, **kwargs):
        saved_figures.append(self)

    monkeypatch.setattr(plotting_engine, "_slice_trace_to_segment", tracking_slice)
    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=qc_df,
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control", "Treatment"],
    )

    assert len(payload["paths"]) == 1
    assert payload["warnings"] == []
    assert slice_calls == ["a.csv", "b.csv"]
    assert len(saved_figures) == 1

    ax = saved_figures[0].axes[0]
    assert len(ax.patches) == 0
    assert any(text.get_text() == "Hardness at Peak1" for text in ax.texts)
    assert any(
        line.get_xdata()[0] == pytest.approx(0.0) for line in ax.lines if len(line.get_xdata())
    )


def test_plot_custom_graphs_renders_selected_sample_segment_graph_as_one_stacked_figure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selected-sample stacked exports should slice each sample and save one multi-panel figure."""
    trace_df, qc_df = _semantic_segment_trace_payload()
    spec = CustomGraphSpec(
        title="Stacked Segment",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        right_axis=CustomGraphAxisLayer(variable="Force Corrected (N)", role="right"),
        view_domain="semantic_segment",
        segment_key="b1_start_to_peak1",
        rebase_x=True,
        data_scope="selected_samples",
        selected_samples=["a.csv", "b.csv"],
        display_mode="stacked",
        annotations=[CustomGraphAnnotation(kind="annotation", key="hardness_peak1")],
        overlay=CustomGraphOverlay(kind="segment", key="b1_start_to_peak1"),
    )
    original_slice = plotting_engine._slice_trace_to_segment
    slice_calls: list[str] = []
    saved_figures: list[Figure] = []

    def tracking_slice(frame, qc_row, segment_key, x_label, *, rebase_x):
        slice_calls.append(str(frame["Filename"].iloc[0]))
        return original_slice(frame, qc_row, segment_key, x_label, rebase_x=rebase_x)

    def capture_savefig(self, *args, **kwargs):
        saved_figures.append(self)

    monkeypatch.setattr(plotting_engine, "_slice_trace_to_segment", tracking_slice)
    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=qc_df,
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control", "Treatment"],
    )

    assert len(payload["paths"]) == 1
    assert payload["warnings"] == []
    assert slice_calls == ["a.csv", "b.csv"]
    assert len(saved_figures) == 1
    assert len(saved_figures[0].axes) == 4

    x_limits = [tuple(axis.get_xlim()) for axis in saved_figures[0].axes]
    assert x_limits[0] == pytest.approx(x_limits[1])
    legend = saved_figures[0].axes[0].get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert "Hardness at Peak1" in legend_labels
    assert "B1 start -> Peak1" in legend_labels
    assert any(axis.get_ylabel() == "Force Corrected (N)" for axis in saved_figures[0].axes[1:])


def test_plot_custom_graphs_renders_selected_samples_as_one_overlay_figure_with_distinct_colors(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlay mode should render all selected samples on one axis with per-sample colors."""
    trace_df, qc_df = _semantic_segment_trace_payload()
    trace_df["Group"] = "Control"
    qc_df["Group"] = "Control"
    spec = CustomGraphSpec(
        title="Overlay Segment",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        view_domain="semantic_segment",
        segment_key="b1_start_to_peak1",
        rebase_x=True,
        data_scope="selected_samples",
        selected_samples=["a.csv", "b.csv"],
        display_mode="overlay",
    )
    saved_figures: list[Figure] = []

    def capture_savefig(self, *args, **kwargs):
        saved_figures.append(self)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=qc_df,
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control"],
    )

    assert len(payload["paths"]) == 1
    assert payload["warnings"] == []
    assert len(saved_figures) == 1
    assert len(saved_figures[0].axes) == 1

    ax = saved_figures[0].axes[0]
    colored_lines = [line for line in ax.lines if len(line.get_xdata())]
    assert len(colored_lines) == 2
    assert len({line.get_color() for line in colored_lines}) == 2
    assert all(line.get_xdata()[0] == pytest.approx(0.0) for line in colored_lines)
    legend = ax.get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert any("a.csv" in label for label in legend_labels)
    assert any("b.csv" in label for label in legend_labels)


def test_plot_custom_graphs_uses_distinct_colors_for_left_and_right_axes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Left and right trace axes should not reuse the same curve color family."""
    trace_df, qc_df = _semantic_segment_trace_payload()
    spec = CustomGraphSpec(
        title="Dual Axis Colors",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        right_axis=CustomGraphAxisLayer(variable="Force Corrected (N)", role="right"),
        data_scope="grouped",
    )
    saved_figures: list[Figure] = []

    def capture_savefig(self, *args, **kwargs):
        saved_figures.append(self)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=qc_df,
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control", "Treatment"],
    )

    assert len(payload["paths"]) == 1
    assert payload["warnings"] == []
    assert len(saved_figures) == 1
    assert len(saved_figures[0].axes) == 2

    ax_left, ax_right = saved_figures[0].axes
    assert ax_left.lines
    assert ax_right.lines
    assert ax_left.lines[0].get_color() != ax_right.lines[0].get_color()
    assert ax_left.yaxis.label.get_color() != ax_right.yaxis.label.get_color()


def test_plot_custom_graphs_renders_segment_regression_overlay_on_selected_sample_overlay(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Segment-focused overlay graphs should support a fitted segment regression line."""
    trace_df, qc_df = _semantic_segment_trace_payload()
    spec = CustomGraphSpec(
        title="Regression Segment",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        view_domain="semantic_segment",
        segment_key="b1_start_to_peak1",
        rebase_x=True,
        data_scope="selected_samples",
        selected_samples=["a.csv", "b.csv"],
        display_mode="overlay",
        overlay=CustomGraphOverlay(kind="regression", key="b1_start_to_peak1"),
    )
    saved_figures: list[Figure] = []

    def capture_savefig(self, *args, **kwargs):
        saved_figures.append(self)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=qc_df,
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control", "Treatment"],
    )

    assert len(payload["paths"]) == 1
    assert payload["warnings"] == []
    assert len(saved_figures) == 1

    ax = saved_figures[0].axes[0]
    raw_lines = [line for line in ax.lines if len(line.get_xdata()) and line.get_linestyle() == "-"]
    regression_lines = [
        line for line in ax.lines if len(line.get_xdata()) and line.get_linestyle() == "--"
    ]
    assert len(raw_lines) == 2
    assert len(regression_lines) == 2
    assert all(line.get_xdata()[0] == pytest.approx(0.0) for line in regression_lines)


def test_plot_custom_graphs_selected_sample_individual_skips_missing_marker_sample_with_warning(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Individual selected-sample exports should warn for invalid samples and save valid ones."""
    trace_df, qc_df = _semantic_segment_trace_payload()
    qc_df.loc[qc_df["Filename"] == "b.csv", "Bite1 Start Index"] = pd.NA
    spec = CustomGraphSpec(
        title="Individual Segment",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        view_domain="semantic_segment",
        segment_key="b1_start_to_peak1",
        rebase_x=True,
        data_scope="selected_samples",
        selected_samples=["a.csv", "b.csv"],
        display_mode="individual",
    )
    saved_figures: list[Figure] = []

    def capture_savefig(self, *args, **kwargs):
        saved_figures.append(self)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=qc_df,
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control", "Treatment"],
    )

    assert len(payload["paths"]) == 1
    assert len(saved_figures) == 1
    assert len(payload["warnings"]) == 1
    assert "Individual Segment" in payload["warnings"][0]
    assert "b.csv" in payload["warnings"][0]
    assert "Bite1 Start Index" in payload["warnings"][0]
    assert Path(payload["paths"][0]).stem == "individual_segment_a_csv_time_s"
    assert saved_figures[0].axes[0].get_title() == "Individual Segment [Time (s)]"


def test_plot_custom_graphs_preserves_grouped_segment_skip_warnings_when_all_samples_fail(
    tmp_path,
) -> None:
    """Grouped semantic-segment exports should retain warnings when every sample is skipped."""
    trace_df, qc_df = _semantic_segment_trace_payload()
    qc_df["Bite1 Start Index"] = pd.NA
    spec = CustomGraphSpec(
        title="Grouped All Missing",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        view_domain="semantic_segment",
        segment_key="b1_start_to_peak1",
        rebase_x=True,
        data_scope="grouped",
    )

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        qc_df=qc_df,
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        figure_config=FigureConfig(dpi=72),
        group_order=["Control", "Treatment"],
    )

    assert payload["paths"] == []
    assert len(payload["warnings"]) == 3
    assert any(
        "a.csv" in warning and "Bite1 Start Index" in warning for warning in payload["warnings"]
    )
    assert any(
        "b.csv" in warning and "Bite1 Start Index" in warning for warning in payload["warnings"]
    )
    assert any("unavailable for all files" in warning for warning in payload["warnings"])
