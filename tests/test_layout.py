"""Tests for responsive layout helpers."""

from __future__ import annotations

from tpa_analyzer.ui.layout import resolve_layout_mode


def test_resolve_layout_mode_breakpoints() -> None:
    """Layout mode should change at the expected breakpoints."""
    assert resolve_layout_mode(180) == "wide"
    assert resolve_layout_mode(130) == "medium"
    assert resolve_layout_mode(90) == "narrow"
