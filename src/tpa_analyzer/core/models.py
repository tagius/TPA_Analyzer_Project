"""Shared typed models used by the TPA Analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

import matplotlib


PlotSource = Literal["trace", "metric"]
AxisKind = Literal["x", "y"]
AxisScale = Literal["numeric", "categorical"]
GraphMode = Literal["panel", "overlay", "auto"]
TraceCurveMode = Literal["individual", "mean_band", "both"]
TraceBandMode = Literal["sd", "ci95"]
MetricViewMode = Literal["raw", "summary", "both"]
PlotType = Literal["trace", "metric"]
LayoutMode = Literal["wide", "medium", "narrow"]


@dataclass(frozen=True)
class PlotVariable:
    """Metadata describing a selectable plotting variable."""

    label: str
    column: str
    axis: str
    unit: str
    kind: AxisKind
    source: PlotSource
    scale: AxisScale = "numeric"


@dataclass
class FigureConfig:
    """Figure sizing and resolution settings for exported plots."""

    ratio_preset: str = "4:3"
    width_in: float | None = None
    height_in: float | None = None
    dpi: int = 300

    def resolve_size(self, default: tuple[float, float] = (10.0, 7.5)) -> tuple[float, float]:
        """Resolve the effective figure size from preset and overrides."""
        ratio_map = {
            "1:1": (8.0, 8.0),
            "4:3": (10.0, 7.5),
            "16:9": (12.0, 6.75),
            "A4 portrait": (8.27, 11.69),
            "A4 landscape": (11.69, 8.27),
        }
        base_width, base_height = ratio_map.get(self.ratio_preset, default)
        width = self.width_in or base_width
        height = self.height_in or base_height

        if self.width_in and not self.height_in:
            height = width * (base_height / base_width)
        elif self.height_in and not self.width_in:
            width = height * (base_width / base_height)

        return (max(float(width), 2.0), max(float(height), 2.0))


def is_hex_color(value: str) -> bool:
    """Return ``True`` when ``value`` is a valid ``#RRGGBB`` color."""
    return bool(re.fullmatch(r"#[0-9a-fA-F]{6}", (value or "").strip()))


@dataclass
class PlotStyleConfig:
    """Group-aware style settings shared across all exported plots."""

    group_colors: dict[str, str] = field(default_factory=dict)
    palette_name: str = "nature_npg"
    replicate_alpha: float = 0.25
    replicate_linewidth: float = 1.0
    mean_linewidth: float = 2.2

    def _palette(self) -> list[str]:
        """Return the base palette used for stable group coloring."""
        return [
            "#E64B35",
            "#4DBBD5",
            "#00A087",
            "#3C5488",
            "#F39B7F",
            "#8491B4",
            "#91D1C2",
            "#DC0000",
            "#7E6148",
            "#B09C85",
        ]

    def ensure_group_colors(self, groups: list[str]) -> None:
        """Assign stable colors to all provided groups."""
        clean_groups = [str(group).strip() for group in groups if str(group).strip()]
        if not clean_groups:
            return

        used: set[str] = set()
        for group in clean_groups:
            existing = self.group_colors.get(group, "")
            if is_hex_color(existing):
                normalized = existing.upper()
                self.group_colors[group] = normalized
                used.add(normalized)

        palette = [color.upper() for color in self._palette()]
        palette_cursor = 0

        def next_color() -> str:
            """Return the next unused color from the palette or fallback generator."""
            nonlocal palette_cursor
            while palette_cursor < len(palette):
                candidate = palette[palette_cursor]
                palette_cursor += 1
                if candidate not in used:
                    return candidate

            extra_index = len(used) - len(palette)
            hue = (extra_index * 0.61803398875) % 1.0
            rgb = matplotlib.colors.hsv_to_rgb((hue, 0.55, 0.86))
            return matplotlib.colors.to_hex(rgb, keep_alpha=False).upper()

        for group in clean_groups:
            existing = self.group_colors.get(group, "")
            if is_hex_color(existing):
                continue
            candidate = next_color()
            while candidate in used:
                candidate = next_color()
            self.group_colors[group] = candidate
            used.add(candidate)

    def get_color(self, group: str) -> str:
        """Return the canonical plot color for a group."""
        if group in self.group_colors and is_hex_color(self.group_colors[group]):
            return self.group_colors[group].upper()
        self.ensure_group_colors([group])
        return self.group_colors[group].upper()


@dataclass
class CustomGraphAxisLayer:
    """A single graph layer anchored to one side of the plot."""

    variable: str
    role: Literal["left", "right"]
    curve_mode: TraceCurveMode = "mean_band"


@dataclass(frozen=True)
class CustomGraphOverlay:
    """An auxiliary overlay rendered on top of a custom graph."""

    kind: Literal["segment", "annotation", "inset_bar", "window"]
    key: str


@dataclass
class CustomGraphSpec:
    """Typed custom graph configuration saved by the plot builder."""

    title: str
    x_domain: str
    left_axis: list[CustomGraphAxisLayer] = field(default_factory=list)
    right_axis: CustomGraphAxisLayer | None = None
    overlay: CustomGraphOverlay | None = None
    enabled: bool = True
    band_mode: TraceBandMode = "sd"

    def __post_init__(self) -> None:
        """Validate that axis layers are assigned to the intended side."""
        for layer in self.left_axis:
            if layer.role != "left":
                raise ValueError("left_axis layers must have role='left'.")
        if self.right_axis is not None and self.right_axis.role != "right":
            raise ValueError("right_axis must have role='right'.")


@dataclass
class GraphSpec:
    """Typed custom graph configuration saved by the plot builder."""

    title: str
    plot_type: PlotType
    x_cols: list[str]
    y_cols: list[str]
    mode: GraphMode = "auto"
    enabled: bool = True
    curve_mode: TraceCurveMode = "mean_band"
    band_mode: TraceBandMode = "sd"
    metric_view: MetricViewMode = "both"
