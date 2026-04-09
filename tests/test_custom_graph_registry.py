"""Tests for composed custom graph registry models."""

from __future__ import annotations

import pytest

from tpa_analyzer.core.models import CustomGraphAxisLayer, CustomGraphOverlay, CustomGraphSpec


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
