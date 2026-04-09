"""Tests for composed custom graph registry models."""

from __future__ import annotations

import pytest

from tpa_analyzer.core.models import CustomGraphAxisLayer, CustomGraphOverlay, CustomGraphSpec
from tpa_analyzer.plotting.custom_graphs import (
    eligible_overlay_keys,
    eligible_right_axis_variables,
)


def test_custom_graph_spec_supports_axis_and_overlay_layers() -> None:
    """A custom graph spec should hold axis layers and an overlay layer."""
    spec = CustomGraphSpec(
        title="Composite Graph",
        x_domain="Time (s)",
        left_axis=[
            CustomGraphAxisLayer(variable="Force (N)", role="left"),
            CustomGraphAxisLayer(variable="Deformation (mm)", role="left", curve_mode="both"),
        ],
        right_axis=CustomGraphAxisLayer(variable="Temperature (C)", role="right"),
        overlay=CustomGraphOverlay(kind="window", key="highlight"),
    )

    assert spec.title == "Composite Graph"
    assert spec.x_domain == "Time (s)"
    assert [layer.variable for layer in spec.left_axis] == ["Force (N)", "Deformation (mm)"]
    assert spec.right_axis is not None
    assert spec.right_axis.role == "right"
    assert spec.overlay is not None
    assert spec.overlay.kind == "window"


def test_custom_graph_spec_rejects_mismatched_axis_roles() -> None:
    """Axis layers must match the side they are assigned to."""
    with pytest.raises(ValueError, match="left_axis layers must have role='left'"):
        CustomGraphSpec(
            title="Invalid Graph",
            x_domain="Time (s)",
            left_axis=[CustomGraphAxisLayer(variable="Force (N)", role="right")],
        )


def test_eligible_right_axis_variables_include_deformation_for_time_domain() -> None:
    """Time-domain trace graphs should allow deformation on the right axis."""
    eligible = eligible_right_axis_variables(
        x_domain="Time (s)",
        left_variables=["Force Corrected (N)"],
        analysis_ready=True,
    )

    assert "Deformation (mm)" in eligible


def test_eligible_overlay_keys_include_modulus_window_for_analysis_ready_strain_graph() -> None:
    """Analysis-ready strain graphs should expose the modulus window overlay."""
    eligible = eligible_overlay_keys(
        x_domain="True Strain (%)",
        left_variables=["True Stress (kPa)"],
        analysis_ready=True,
    )

    assert "modulus_window" in eligible
