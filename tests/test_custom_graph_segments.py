import pandas as pd

from tpa_analyzer.plotting.custom_graphs import eligible_annotation_keys, eligible_overlay_keys, semantic_segment_keys
from tpa_analyzer.plotting.engine import _slice_trace_to_segment


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
