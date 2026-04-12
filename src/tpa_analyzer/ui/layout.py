"""Responsive layout helpers for the Textual app."""

from __future__ import annotations

from tpa_analyzer.core.constants import LAYOUT_MEDIUM_MIN, LAYOUT_WIDE_MIN
from tpa_analyzer.core.models import LayoutMode


def resolve_layout_mode(width: int) -> LayoutMode:
    """Resolve a layout mode from the current terminal width."""
    if width >= LAYOUT_WIDE_MIN:
        return "wide"
    if width >= LAYOUT_MEDIUM_MIN:
        return "medium"
    return "narrow"
