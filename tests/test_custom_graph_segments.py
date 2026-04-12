from pathlib import Path

import pytest
import pandas as pd

from tpa_analyzer.core.errors import PlotSpecError
from tpa_analyzer.core.models import CustomGraphAnnotation, CustomGraphAxisLayer, CustomGraphSpec, PlotStyleConfig
from tpa_analyzer.plotting.custom_graphs import eligible_annotation_keys, eligible_overlay_keys, semantic_segment_keys
from tpa_analyzer.plotting.engine import _slice_trace_to_segment, expand_composed_graph_spec, plot_custom_graphs


def test_segment_registry_lists_all_supported_semantic_segments() -> None:
    assert semantic_segment_keys() == [
        "b1_start_to_peak1",
        "peak1_to_b1_end",
        "b1_end_to_b2_start",
        "b2_start_to_peak2",
        "peak2_to_b2_end",
        "modulus_window",
    ]


def test_annotations_are_filtered_by_segment_meaning() -> None:
    assert eligible_annotation_keys("b1_start_to_peak1") == ["hardness_peak1"]
    assert eligible_annotation_keys("b1_end_to_b2_start") == ["adhesiveness"]
    assert eligible_annotation_keys("modulus_window") == ["modulus_window"]


def test_legacy_overlay_helper_stays_on_existing_segment_set() -> None:
    assert eligible_overlay_keys(
        x_domain="Time (s)",
        left_variables=["Force Corrected (N)"],
        analysis_ready=True,
    ) == [
        "b1_start_to_peak1",
        "peak1_to_b1_end",
        "b1_end_to_b2_start",
        "b2_start_to_peak2",
        "peak2_to_b2_end",
        "hardness_peak1",
        "adhesiveness",
    ]


def test_slice_trace_to_segment_rebases_x_to_zero() -> None:
    frame = pd.DataFrame(
        [
            {"Time (s)": 0.2, "Force Corrected (N)": 0.5},
            {"Time (s)": 0.5, "Force Corrected (N)": 1.2},
            {"Time (s)": 0.7, "Force Corrected (N)": 1.5},
        ]
    )
    qc_row = pd.Series({"Bite1 Start Index": 1, "Peak1 Index": 2})

    segment = _slice_trace_to_segment(frame, qc_row, "b1_start_to_peak1", "Time (s)", rebase_x=True)

    assert list(segment["Time (s)"]) == [0.0, 0.2]
    assert list(segment["Force Corrected (N)"]) == [1.2, 1.5]


def test_semantic_segment_expands_without_legacy_overlay_validation() -> None:
    spec = CustomGraphSpec(
        title="Segment graph",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        view_domain="semantic_segment",
        segment_key="peak2_to_b2_end",
        rebase_x=True,
    )

    job = expand_composed_graph_spec(spec)[0]

    assert job.segment_key == "peak2_to_b2_end"
    assert job.rebase_x is True


def test_semantic_segment_rejects_value_based_modulus_window() -> None:
    spec = CustomGraphSpec(
        title="Segment graph",
        x_domain="True Strain (%)",
        left_axis=[CustomGraphAxisLayer(variable="True Stress (kPa)", role="left")],
        view_domain="semantic_segment",
        segment_key="modulus_window",
        rebase_x=True,
    )

    with pytest.raises(PlotSpecError, match="value-based QC markers"):
        expand_composed_graph_spec(spec)


def test_semantic_annotation_must_match_selected_segment() -> None:
    spec = CustomGraphSpec(
        title="Segment graph",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        view_domain="semantic_segment",
        segment_key="b1_end_to_b2_start",
        rebase_x=True,
        annotations=[CustomGraphAnnotation(kind="annotation", key="hardness_peak1")],
    )

    with pytest.raises(PlotSpecError, match="not compatible with semantic segment"):
        expand_composed_graph_spec(spec)


def test_segment_render_warns_when_some_files_lack_qc_markers(tmp_path) -> None:
    trace_df = pd.DataFrame(
        [
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 0.2, "Force Corrected (N)": 0.5},
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 0.5, "Force Corrected (N)": 1.2},
            {"File": "a.csv", "Filename": "a.csv", "Group": "Control", "Time (s)": 0.7, "Force Corrected (N)": 1.5},
            {"File": "b.csv", "Filename": "b.csv", "Group": "Control", "Time (s)": 0.2, "Force Corrected (N)": 0.6},
            {"File": "b.csv", "Filename": "b.csv", "Group": "Control", "Time (s)": 0.5, "Force Corrected (N)": 1.1},
            {"File": "b.csv", "Filename": "b.csv", "Group": "Control", "Time (s)": 0.7, "Force Corrected (N)": 1.4},
        ]
    )
    qc_df = pd.DataFrame([{"Filename": "a.csv", "Bite1 Start Index": 1, "Peak1 Index": 2}])
    spec = CustomGraphSpec(
        title="Segment graph",
        x_domain="Time (s)",
        left_axis=[CustomGraphAxisLayer(variable="Force Corrected (N)", role="left")],
        view_domain="semantic_segment",
        segment_key="b1_start_to_peak1",
        rebase_x=True,
    )

    payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=pd.DataFrame(),
        graph_specs=[spec],
        style=PlotStyleConfig(),
        output_dir=tmp_path,
        qc_df=qc_df,
    )

    assert len(payload["paths"]) == 1
    assert Path(payload["paths"][0]).exists()
    assert len(payload["warnings"]) == 1
    assert "b.csv" in payload["warnings"][0]
    assert "missing QC summary row" in payload["warnings"][0]
