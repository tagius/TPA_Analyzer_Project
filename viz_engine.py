from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIABLE_REGISTRY: dict[str, dict[str, str]] = {
    "Time (s)": {"column": "Time (s)", "axis": "Time (s)", "unit": "s", "kind": "x"},
    "Aligned Time (s)": {
        "column": "Aligned Time (s)",
        "axis": "Aligned Time (s)",
        "unit": "s",
        "kind": "x",
    },
    "True Strain (%)": {
        "column": "True Strain (%)",
        "axis": "True Strain (%)",
        "unit": "%",
        "kind": "x",
    },
    "Force (N)": {"column": "Force (N)", "axis": "Force (N)", "unit": "N", "kind": "y"},
    "Force Corrected (N)": {
        "column": "Force Corrected (N)",
        "axis": "Force Corrected (N)",
        "unit": "N",
        "kind": "y",
    },
    "Deformation (mm)": {
        "column": "Deformation (mm)",
        "axis": "Deformation (mm)",
        "unit": "mm",
        "kind": "y",
    },
    "True Stress (kPa)": {
        "column": "True Stress (kPa)",
        "axis": "True Stress (kPa)",
        "unit": "kPa",
        "kind": "y",
    },
}


@dataclass
class FigureConfig:
    ratio_preset: str = "4:3"
    width_in: float | None = None
    height_in: float | None = None
    dpi: int = 300

    def resolve_size(self, default: tuple[float, float] = (10.0, 7.5)) -> tuple[float, float]:
        ratio_map = {
            "1:1": (8.0, 8.0),
            "4:3": (10.0, 7.5),
            "16:9": (12.0, 6.75),
            "A4 portrait": (8.27, 11.69),
            "A4 landscape": (11.69, 8.27),
        }
        width, height = ratio_map.get(self.ratio_preset, default)

        if self.width_in and self.height_in:
            width, height = self.width_in, self.height_in
        elif self.width_in and not self.height_in:
            width = self.width_in
            height = width * (ratio_map.get(self.ratio_preset, default)[1] / ratio_map.get(self.ratio_preset, default)[0])
        elif self.height_in and not self.width_in:
            height = self.height_in
            width = height * (ratio_map.get(self.ratio_preset, default)[0] / ratio_map.get(self.ratio_preset, default)[1])

        return (max(float(width), 2.0), max(float(height), 2.0))


@dataclass
class PlotStyleConfig:
    group_force_colors: dict[str, str] = field(default_factory=dict)
    group_deformation_colors: dict[str, str] = field(default_factory=dict)
    group_stress_colors: dict[str, str] = field(default_factory=dict)
    palette_name: str = "nature_npg"
    replicate_alpha: float = 0.25
    replicate_linewidth: float = 1.0
    mean_linewidth: float = 2.2

    def _palette(self) -> list[str]:
        # Nature-style NPG palette (10 colors), tuned for publication readability.
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
        """Assign stable distinct colors per group using Nature palette, then fallback hues."""
        clean_groups = [str(group).strip() for group in groups if str(group).strip()]
        if not clean_groups:
            return

        used: set[str] = set()
        for group in clean_groups:
            existing = self.group_force_colors.get(group, "")
            if _is_hex(existing):
                normalized = existing.upper()
                self.group_force_colors[group] = normalized
                used.add(normalized)

        palette = [color.upper() for color in self._palette()]
        palette_cursor = 0

        def next_color() -> str:
            nonlocal palette_cursor
            while palette_cursor < len(palette):
                candidate = palette[palette_cursor]
                palette_cursor += 1
                if candidate not in used:
                    return candidate

            # If groups exceed palette length, generate additional distinct colors.
            extra_index = len(used) - len(palette)
            hue = (extra_index * 0.61803398875) % 1.0
            sat = 0.55
            val = 0.86
            rgb = matplotlib.colors.hsv_to_rgb((hue, sat, val))
            return matplotlib.colors.to_hex(rgb, keep_alpha=False).upper()

        for group in clean_groups:
            existing = self.group_force_colors.get(group, "")
            if _is_hex(existing):
                continue
            candidate = next_color()
            while candidate in used:
                candidate = next_color()
            self.group_force_colors[group] = candidate
            used.add(candidate)

    def get_color(self, group: str, trace_type: str) -> str:
        _ = trace_type.lower().strip()
        # Canonical group color shared across all panels.
        if group in self.group_force_colors and _is_hex(self.group_force_colors[group]):
            return self.group_force_colors[group].upper()

        self.ensure_group_colors([group])
        return self.group_force_colors[group].upper()


@dataclass
class GraphSpec:
    title: str
    x_col: str
    y_cols: list[str]
    mode: str = "panel"
    enabled: bool = True
    curve_mode: str = "individual"
    band_mode: str = "sd"


def _is_hex(value: str) -> bool:
    return bool(re.fullmatch(r"#[0-9a-fA-F]{6}", value or ""))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return slug.lower() or "plot"


def _normalize_graph_spec(spec: GraphSpec | dict[str, Any]) -> GraphSpec:
    if isinstance(spec, GraphSpec):
        return spec

    return GraphSpec(
        title=str(spec.get("title", "Custom Graph")),
        x_col=str(spec.get("x_col", "Time (s)")),
        y_cols=list(spec.get("y_cols", ["Force (N)"])),
        mode=str(spec.get("mode", "panel")),
        enabled=bool(spec.get("enabled", True)),
        curve_mode=str(spec.get("curve_mode", "individual")),
        band_mode=str(spec.get("band_mode", "sd")),
    )


def _require_column(trace_df: pd.DataFrame, label: str) -> str:
    if label in VARIABLE_REGISTRY:
        column = VARIABLE_REGISTRY[label]["column"]
    else:
        column = label

    if column not in trace_df.columns:
        raise ValueError(f"Missing required trace column: {column}")

    return column


def _are_y_units_compatible(labels: list[str]) -> bool:
    units = {
        VARIABLE_REGISTRY[label]["unit"]
        for label in labels
        if label in VARIABLE_REGISTRY and VARIABLE_REGISTRY[label]["kind"] == "y"
    }
    return len(units) <= 1


def _plot_individual(
    ax: Any,
    trace_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    style: PlotStyleConfig,
) -> None:
    for (group, file_name), frame in trace_df.groupby(["Group", "File"], sort=False):
        ordered = frame.sort_values(x_col)
        ax.plot(
            ordered[x_col],
            ordered[y_col],
            color=style.get_color(str(group), y_col),
            alpha=style.replicate_alpha,
            linewidth=style.replicate_linewidth,
        )


def build_mean_band(
    trace_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str = "Group",
    file_col: str = "File",
    band_mode: str = "sd",
) -> pd.DataFrame:
    """Interpolate replicate curves and compute mean +/- SD or mean +/- 95% CI."""
    rows: list[dict[str, Any]] = []

    for group_name, group_frame in trace_df.groupby(group_col, sort=False):
        replicate_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for _, rep_frame in group_frame.groupby(file_col, sort=False):
            ordered = rep_frame.sort_values(x_col)
            x_vals = ordered[x_col].to_numpy(dtype=float)
            y_vals = ordered[y_col].to_numpy(dtype=float)

            mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            x_vals = x_vals[mask]
            y_vals = y_vals[mask]

            if len(x_vals) < 3:
                continue

            unique_x, unique_idx = np.unique(x_vals, return_index=True)
            unique_y = y_vals[unique_idx]
            if len(unique_x) < 3:
                continue

            replicate_curves.append((unique_x, unique_y))

        if not replicate_curves:
            continue

        left = max(float(curve[0][0]) for curve in replicate_curves)
        right = min(float(curve[0][-1]) for curve in replicate_curves)

        if right <= left:
            # Fallback: use union range if overlap is too small.
            left = min(float(curve[0][0]) for curve in replicate_curves)
            right = max(float(curve[0][-1]) for curve in replicate_curves)

        if right <= left:
            continue

        point_count = int(np.clip(np.mean([len(curve[0]) for curve in replicate_curves]), 80, 400))
        grid = np.linspace(left, right, point_count)

        interpolated = []
        for x_vals, y_vals in replicate_curves:
            interp = np.interp(grid, x_vals, y_vals)
            interpolated.append(interp)

        stack = np.vstack(interpolated)
        mean_vals = np.nanmean(stack, axis=0)
        if stack.shape[0] > 1:
            sd_vals = np.nanstd(stack, axis=0, ddof=1)
        else:
            sd_vals = np.zeros_like(mean_vals)

        if band_mode.lower() == "ci95":
            spread = 1.96 * sd_vals / np.sqrt(stack.shape[0])
        else:
            spread = sd_vals

        lower = mean_vals - spread
        upper = mean_vals + spread

        for x_val, mean_val, lower_val, upper_val in zip(grid, mean_vals, lower, upper):
            rows.append(
                {
                    group_col: group_name,
                    x_col: x_val,
                    "Mean": mean_val,
                    "Lower": lower_val,
                    "Upper": upper_val,
                    "Replicates": stack.shape[0],
                    "Band": band_mode.lower(),
                }
            )

    return pd.DataFrame(rows)


def _plot_mean_band(
    ax: Any,
    trace_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    style: PlotStyleConfig,
    band_mode: str,
) -> None:
    band_df = build_mean_band(trace_df, x_col=x_col, y_col=y_col, band_mode=band_mode)
    if band_df.empty:
        return

    for group_name, frame in band_df.groupby("Group", sort=False):
        ordered = frame.sort_values(x_col)
        color = style.get_color(str(group_name), y_col)
        ax.fill_between(
            ordered[x_col],
            ordered["Lower"],
            ordered["Upper"],
            color=color,
            alpha=0.18,
        )
        ax.plot(
            ordered[x_col],
            ordered["Mean"],
            color=color,
            linewidth=style.mean_linewidth,
            label=str(group_name),
        )


def _apply_curve_mode(
    ax: Any,
    trace_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    style: PlotStyleConfig,
    curve_mode: str,
    band_mode: str,
) -> None:
    mode = curve_mode.lower().strip()
    if mode in {"individual", "both"}:
        _plot_individual(ax, trace_df, x_col, y_col, style)
    if mode in {"mean_band", "both"}:
        _plot_mean_band(ax, trace_df, x_col, y_col, style, band_mode=band_mode)


def _axis_label(label: str) -> str:
    return VARIABLE_REGISTRY.get(label, {}).get("axis", label)


def plot_trace_stack(
    trace_df: pd.DataFrame,
    spec: dict[str, Any] | None,
    style: PlotStyleConfig,
    output_path: str | Path,
    figure_config: FigureConfig | None = None,
) -> str:
    """Generate the default two-panel vertical stack plot: Time-Force and Time-Deformation."""
    figure_config = figure_config or FigureConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_label = "Time (s)"
    y_labels = ["Force (N)", "Deformation (mm)"]
    curve_mode = str((spec or {}).get("curve_mode", "individual"))
    band_mode = str((spec or {}).get("band_mode", "sd"))

    x_col = _require_column(trace_df, x_label)

    figsize = figure_config.resolve_size(default=(10.0, 8.0))
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for idx, y_label in enumerate(y_labels):
        y_col = _require_column(trace_df, y_label)
        ax = axes[idx]
        _apply_curve_mode(ax, trace_df, x_col, y_col, style, curve_mode=curve_mode, band_mode=band_mode)
        ax.set_ylabel(_axis_label(y_label))
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[-1].set_xlabel(_axis_label(x_label))
    axes[0].set_title("TPA Default Trace Stack")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=figure_config.dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_custom_graphs(
    trace_df: pd.DataFrame,
    graph_specs: list[GraphSpec | dict[str, Any]],
    style: PlotStyleConfig,
    output_dir: str | Path,
    figure_config: FigureConfig | None = None,
) -> dict[str, Any]:
    """Generate user-defined x/y plots in panel or overlay mode."""
    figure_config = figure_config or FigureConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    warnings: list[str] = []

    for idx, raw_spec in enumerate(graph_specs, start=1):
        spec = _normalize_graph_spec(raw_spec)
        if not spec.enabled:
            continue

        x_col = _require_column(trace_df, spec.x_col)
        y_cols = [_require_column(trace_df, label) for label in spec.y_cols]

        mode = spec.mode.lower().strip()
        if mode == "overlay" and len(y_cols) > 1 and not _are_y_units_compatible(spec.y_cols):
            warnings.append(
                f"Graph '{spec.title}' requested overlay with mixed units; switched to panel mode."
            )
            mode = "panel"

        figsize = figure_config.resolve_size()

        if mode == "overlay":
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            for y_col, y_label in zip(y_cols, spec.y_cols):
                _apply_curve_mode(
                    ax,
                    trace_df,
                    x_col,
                    y_col,
                    style,
                    curve_mode=spec.curve_mode,
                    band_mode=spec.band_mode,
                )
            ax.set_xlabel(_axis_label(spec.x_col))
            ax.set_ylabel(" / ".join(_axis_label(label) for label in spec.y_cols))
            ax.set_title(spec.title)
            ax.grid(True, linestyle="--", alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, frameon=False)
        else:
            fig, axes = plt.subplots(len(y_cols), 1, figsize=figsize, sharex=True)
            if len(y_cols) == 1:
                axes = [axes]

            for ax, y_col, y_label in zip(axes, y_cols, spec.y_cols):
                _apply_curve_mode(
                    ax,
                    trace_df,
                    x_col,
                    y_col,
                    style,
                    curve_mode=spec.curve_mode,
                    band_mode=spec.band_mode,
                )
                ax.set_ylabel(_axis_label(y_label))
                ax.grid(True, linestyle="--", alpha=0.25)

            axes[-1].set_xlabel(_axis_label(spec.x_col))
            axes[0].set_title(spec.title)
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(handles, labels, frameon=False)

        fig.tight_layout()
        filename = f"custom_{idx:02d}_{_slugify(spec.title)}.png"
        path = output_dir / filename
        fig.savefig(path, dpi=figure_config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(path))

    return {"paths": saved_paths, "warnings": warnings}


def plot_grouped_metrics(
    stats_by_metric: dict[str, dict[str, Any]],
    style: PlotStyleConfig,
    output_path: str | Path,
    figure_config: FigureConfig | None = None,
) -> str:
    """Plot grouped bar charts with SD error bars and significance letters."""
    figure_config = figure_config or FigureConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["Hardness (N)", "Cohesiveness", "Springiness", "Chewiness"]
    present_metrics = [metric for metric in metrics if metric in stats_by_metric]
    if not present_metrics:
        raise ValueError("No supported metrics found for grouped bar plot.")

    ordered_groups_union: list[str] = []
    for metric in present_metrics:
        summary = stats_by_metric[metric]["summary_df"]
        group_col = stats_by_metric[metric]["test_info"]["group_col"]
        for group in summary[group_col].astype(str).tolist():
            if group not in ordered_groups_union:
                ordered_groups_union.append(group)
    style.ensure_group_colors(ordered_groups_union)

    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=figure_config.resolve_size(default=(12.0, 8.0)))
    flat_axes = axes.flatten()

    for ax in flat_axes[len(present_metrics) :]:
        ax.axis("off")

    for ax, metric in zip(flat_axes, present_metrics):
        summary = stats_by_metric[metric]["summary_df"].copy()
        group_col = stats_by_metric[metric]["test_info"]["group_col"]
        groups = summary[group_col].astype(str).tolist()
        means = summary["Mean"].to_numpy(dtype=float)
        sds = summary["SD"].to_numpy(dtype=float)
        letters = summary["Significance"].astype(str).tolist()

        colors = [style.get_color(group, metric) for group in groups]
        bars = ax.bar(groups, means, yerr=sds, capsize=4, color=colors, alpha=0.9)

        ymax = float(np.nanmax(means + sds)) if len(means) else 1.0
        offset = max(ymax * 0.04, 0.02)

        for bar, letter, mean, sd in zip(bars, letters, means, sds):
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = mean + sd + offset
            ax.text(x_pos, y_pos, letter, ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=figure_config.dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_overlay_traces(
    trace_df: pd.DataFrame,
    overlay_spec: dict[str, Any],
    style: PlotStyleConfig,
    output_dir: str | Path,
    figure_config: FigureConfig | None = None,
) -> list[str]:
    """Plot per-group overlays as individual, mean-band, or both."""
    figure_config = figure_config or FigureConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = str(overlay_spec.get("mode", "mean_band"))
    band_mode = str(overlay_spec.get("band_mode", "sd"))
    x_label = str(overlay_spec.get("x_col", "Aligned Time (s)"))
    y_labels = list(overlay_spec.get("y_cols", ["Force (N)", "Deformation (mm)"]))

    x_col = _require_column(trace_df, x_label)
    y_cols = [_require_column(trace_df, label) for label in y_labels]

    output_paths: list[str] = []
    requested_order = [str(group).strip() for group in overlay_spec.get("group_order", []) if str(group).strip()]
    groups_present = trace_df["Group"].dropna().astype(str).tolist()
    unique_present: list[str] = []
    for group in groups_present:
        if group not in unique_present:
            unique_present.append(group)

    ordered_groups = [group for group in requested_order if group in unique_present]
    ordered_groups.extend([group for group in unique_present if group not in ordered_groups])

    for group_name in ordered_groups:
        group_frame = trace_df[trace_df["Group"].astype(str) == group_name]
        if group_frame.empty:
            continue
        figsize = figure_config.resolve_size(default=(10.0, 8.0))
        fig, axes = plt.subplots(len(y_cols), 1, figsize=figsize, sharex=True)
        if len(y_cols) == 1:
            axes = [axes]

        for ax, y_col, y_label in zip(axes, y_cols, y_labels):
            _apply_curve_mode(
                ax,
                group_frame,
                x_col,
                y_col,
                style,
                curve_mode=mode,
                band_mode=band_mode,
            )
            ax.set_ylabel(_axis_label(y_label))
            ax.grid(True, linestyle="--", alpha=0.25)

        axes[-1].set_xlabel(_axis_label(x_label))
        axes[0].set_title(f"Group Overlay: {group_name}")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, frameon=False)

        fig.tight_layout()
        save_path = output_dir / f"overlay_{_slugify(str(group_name))}.png"
        fig.savefig(save_path, dpi=figure_config.dpi, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(str(save_path))

    return output_paths
