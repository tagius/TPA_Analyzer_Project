"""Tests for session persistence helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tpa_analyzer.core.models import (
    CustomGraphAnnotation,
    CustomGraphAxisLayer,
    CustomGraphOverlay,
    CustomGraphSpec,
    GraphSpec,
)
from tpa_analyzer.core.errors import SessionError
from tpa_analyzer.core import session as session_module
from tpa_analyzer.core.session import load_session_data, migrate_graph_specs, save_session_data


def test_migrate_graph_specs_converts_legacy_trace_payload() -> None:
    """Legacy trace graph specs should migrate into ``GraphSpec`` objects."""
    specs = migrate_graph_specs(
        [
            {
                "title": "Legacy",
                "x_col": "Time (s)",
                "y_vars": "Force (N), Deformation (mm)",
            }
        ]
    )
    assert len(specs) == 1
    assert isinstance(specs[0], GraphSpec)
    assert specs[0].title == "Legacy"
    assert specs[0].plot_type == "trace"
    assert specs[0].x_cols == ["Time (s)"]
    assert specs[0].y_cols == ["Force (N)", "Deformation (mm)"]


def test_migrate_graph_specs_preserves_composed_payload() -> None:
    """Composed graph specs should migrate into ``CustomGraphSpec`` objects."""
    specs = migrate_graph_specs(
        [
            {
                "title": "Composed",
                "x_domain": "Time (s)",
                "left_axis": [{"variable": "Force (N)", "role": "left", "curve_mode": "mean_band"}],
                "right_axis": {"variable": "Deformation (mm)", "role": "right", "curve_mode": "individual"},
                "overlay": {"kind": "segment", "key": "markers"},
                "band_mode": "ci95",
            }
        ]
    )
    assert len(specs) == 1
    assert isinstance(specs[0], CustomGraphSpec)
    assert specs[0].left_axis[0] == CustomGraphAxisLayer(variable="Force (N)", role="left", curve_mode="mean_band")
    assert specs[0].right_axis == CustomGraphAxisLayer(variable="Deformation (mm)", role="right", curve_mode="individual")
    assert specs[0].view_domain == "semantic_segment"
    assert specs[0].segment_key == "markers"
    assert specs[0].rebase_x is True
    assert specs[0].overlay is None


def test_migrate_graph_specs_promotes_overlay_recipe_to_segment_fields() -> None:
    payload = [
        {
            "title": "Legacy Overlay",
            "x_domain": "Time (s)",
            "left_axis": [{"variable": "Force Corrected (N)", "role": "left"}],
            "overlay": {"kind": "segment", "key": "b1_start_to_peak1"},
            "band_mode": "sd",
        }
    ]

    migrated = migrate_graph_specs(payload)

    spec = migrated[0]
    assert isinstance(spec, CustomGraphSpec)
    assert spec.view_domain == "semantic_segment"
    assert spec.segment_key == "b1_start_to_peak1"
    assert spec.rebase_x is True
    assert spec.annotations == []
    assert spec.data_scope == "grouped"


def test_migrate_custom_graph_specs_promotes_annotation_overlay_to_annotations() -> None:
    payload = [
        {
            "title": "Legacy Annotation",
            "x_domain": "Time (s)",
            "left_axis": [{"variable": "Force Corrected (N)", "role": "left"}],
            "overlay": {"kind": "annotation", "key": "hardness_peak1"},
        }
    ]

    migrated = migrate_graph_specs(payload)

    spec = migrated[0]
    assert isinstance(spec, CustomGraphSpec)
    assert spec.view_domain == "full_curve"
    assert spec.segment_key is None
    assert spec.rebase_x is False
    assert spec.annotations == [CustomGraphAnnotation(kind="annotation", key="hardness_peak1")]


def test_migrate_graph_specs_propagates_unexpected_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected migration failures should not be swallowed."""
    def boom(_: object) -> GraphSpec:
        raise RuntimeError("boom")

    monkeypatch.setattr(session_module, "_normalize_session_graph_spec", boom)

    with pytest.raises(RuntimeError, match="boom"):
        migrate_graph_specs([{"title": "Legacy"}])


def test_migrate_graph_specs_raises_on_invalid_dict_payload() -> None:
    """Invalid saved graph-spec dictionaries should fail session restore safely."""
    with pytest.raises(SessionError, match="invalid graph spec payload"):
        migrate_graph_specs([{"title": "Raw", "unexpected": {"nested": True}}])


def test_save_and_load_session_data_roundtrip(tmp_path: Path) -> None:
    """Saved session payloads should round-trip through JSON persistence."""
    path = tmp_path / ".tpa_analyzer_session.json"
    payload = {"graph_specs": [GraphSpec(title="Graph", plot_type="trace", x_cols=["Time (s)"], y_cols=["Force (N)"])]}
    save_session_data(path, payload)
    loaded = load_session_data(path)
    assert loaded["schema_version"] >= 1
    assert loaded["graph_specs"][0]["title"] == "Graph"


def test_load_session_data_wraps_invalid_schema_version(tmp_path: Path) -> None:
    """Malformed schema versions should raise ``SessionError``."""
    path = tmp_path / ".tpa_analyzer_session.json"
    path.write_text('{"schema_version": "oops", "graph_specs": []}', encoding="utf-8")

    with pytest.raises(SessionError, match="Session load failed"):
        load_session_data(path)


def test_save_session_data_serializes_composed_graph_specs_cleanly(tmp_path: Path) -> None:
    """Composed graph specs should persist as plain JSON dictionaries."""
    path = tmp_path / ".tpa_analyzer_session.json"
    payload = {
        "graph_specs": [
            CustomGraphSpec(
                title="Composed",
                x_domain="Time (s)",
                left_axis=[CustomGraphAxisLayer(variable="Force (N)", role="left", curve_mode="mean_band")],
                right_axis=CustomGraphAxisLayer(variable="Deformation (mm)", role="right", curve_mode="individual"),
                overlay=CustomGraphOverlay(kind="segment", key="markers"),
                band_mode="ci95",
            )
        ]
    }
    save_session_data(path, payload)
    loaded = load_session_data(path)
    graph_spec = loaded["graph_specs"][0]
    assert graph_spec["title"] == "Composed"
    assert graph_spec["x_domain"] == "Time (s)"
    assert graph_spec["left_axis"][0] == {
        "variable": "Force (N)",
        "role": "left",
        "curve_mode": "mean_band",
    }
    assert graph_spec["right_axis"] == {
        "variable": "Deformation (mm)",
        "role": "right",
        "curve_mode": "individual",
    }
    assert graph_spec["overlay"] == {"kind": "segment", "key": "markers"}


def test_save_session_data_preserves_unrecognized_graph_spec_dict(tmp_path: Path) -> None:
    """Unknown graph spec dictionaries should be preserved in session JSON."""
    path = tmp_path / ".tpa_analyzer_session.json"
    raw_spec = {"title": "Raw", "unexpected": {"nested": True}}

    save_session_data(path, {"graph_specs": [raw_spec]})

    loaded = load_session_data(path)
    assert loaded["graph_specs"] == [raw_spec]


def test_save_and_load_session_data_preserves_grouping_payload(tmp_path: Path) -> None:
    path = tmp_path / ".tpa_analyzer_session.json"
    payload = {
        "group_order": ["Control", "Treatment"],
        "active_group": "Treatment",
        "selected_file_index": 1,
        "file_records": [
            {"filename": "a.csv", "group": "Control"},
            {"filename": "b.csv", "group": ""},
        ],
    }
    save_session_data(path, payload)
    loaded = load_session_data(path)
    assert loaded["group_order"] == ["Control", "Treatment"]
    assert loaded["active_group"] == "Treatment"
    assert loaded["selected_file_index"] == 1
    assert loaded["file_records"][1]["group"] == ""
