"""Plot generation and export helpers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, is_dataclass
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tpa_analyzer.core.constants import COMPUTED_METRICS
from tpa_analyzer.core.errors import PlotSpecError
from tpa_analyzer.core.models import (
    CustomGraphAnnotation,
    CustomGraphAxisLayer,
    CustomGraphOverlay,
    CustomGraphSpec,
    FigureConfig,
    GraphSpec,
    PlotStyleConfig,
)
from tpa_analyzer.plotting.custom_graphs import (
    ANNOTATION_COMPATIBILITY,
    OVERLAY_COMPATIBILITY,
    SEMANTIC_SEGMENTS,
)
from tpa_analyzer.plotting.registry import VARIABLE_REGISTRY, axis_label, registry_entry

LEFT_AXIS_ACCENT = "#9A3412"
RIGHT_AXIS_ACCENT = "#1D4ED8"


@dataclass(frozen=True)
class ResolvedGraphJob:
    """Concrete render job expanded from a saved graph specification."""

    spec_title: str
    plot_type: str
    x_label: str
    y_labels: list[str]
    mode: str
    curve_mode: str
    band_mode: str
    metric_view: str


@dataclass(frozen=True)
class ResolvedComposedGraphJob:
    """Concrete render job for one composed custom-graph recipe."""

    spec_title: str
    x_label: str
    left_layers: list[CustomGraphAxisLayer]
    right_layer: CustomGraphAxisLayer | None
    overlays: list[CustomGraphOverlay]
    band_mode: str
    segment_key: str | None
    annotations: list[CustomGraphAnnotation]
    data_scope: str
    selected_samples: list[str]
    display_mode: str
    rebase_x: bool
    export_stem_suffix: str | None = None


def _graph_spec_title(spec: Any) -> str:
    """Return a best-effort title for warning messages."""
    if isinstance(spec, (GraphSpec, CustomGraphSpec)):
        return spec.title
    if isinstance(spec, dict):
        return str(spec.get("title", "Custom Graph"))
    return "Custom Graph"


def _is_composed_graph_payload(spec: Any) -> bool:
    """Return ``True`` when the payload matches the composed graph model."""
    if isinstance(spec, CustomGraphSpec):
        return True
    if not isinstance(spec, dict):
        return False
    return any(
        key in spec
        for key in (
            "x_domain",
            "left_axis",
            "right_axis",
            "overlay",
            "annotations",
            "view_domain",
            "segment_key",
            "data_scope",
            "selected_samples",
            "display_mode",
        )
    )


def _normalize_axis_layer(raw_layer: Any, *, default_role: str) -> CustomGraphAxisLayer:
    """Normalize one axis-layer payload into ``CustomGraphAxisLayer``."""
    if isinstance(raw_layer, CustomGraphAxisLayer):
        return raw_layer
    if not isinstance(raw_layer, dict):
        raise PlotSpecError(f"Invalid {default_role}-axis layer payload.")
    return CustomGraphAxisLayer(
        variable=str(raw_layer.get("variable", "")).strip(),
        role=str(raw_layer.get("role", default_role)).strip() or default_role,
        curve_mode=str(raw_layer.get("curve_mode", "mean_band")).strip() or "mean_band",
    )


def _normalize_overlay(raw_overlay: Any) -> CustomGraphOverlay:
    """Normalize one overlay payload into ``CustomGraphOverlay``."""
    if isinstance(raw_overlay, CustomGraphOverlay):
        return raw_overlay
    if not isinstance(raw_overlay, dict):
        raise PlotSpecError("Invalid overlay payload.")
    return CustomGraphOverlay(
        kind=str(raw_overlay.get("kind", "")).strip(),
        key=str(raw_overlay.get("key", "")).strip(),
    )


def _normalize_annotation(raw_annotation: Any) -> CustomGraphAnnotation:
    """Normalize one annotation payload into ``CustomGraphAnnotation``."""
    if isinstance(raw_annotation, CustomGraphAnnotation):
        return raw_annotation
    if not isinstance(raw_annotation, dict):
        raise PlotSpecError("Invalid annotation payload.")
    return CustomGraphAnnotation(
        kind=str(raw_annotation.get("kind", "annotation")).strip() or "annotation",
        key=str(raw_annotation.get("key", "")).strip(),
    )


def normalize_composed_graph_spec(spec: CustomGraphSpec | dict[str, Any]) -> CustomGraphSpec:
    """Normalize a composed custom-graph payload into ``CustomGraphSpec``."""
    if isinstance(spec, CustomGraphSpec):
        return spec
    if not isinstance(spec, dict):
        raise PlotSpecError("Invalid composed graph payload.")

    raw_left_axis = spec.get("left_axis", [])
    if not isinstance(raw_left_axis, list):
        raise PlotSpecError("Composed graph left_axis must be a list.")

    raw_right_axis = spec.get("right_axis")
    raw_overlay = spec.get("overlay")
    raw_annotations = spec.get("annotations", [])
    raw_selected_samples = spec.get("selected_samples", [])
    if not isinstance(raw_annotations, list):
        raise PlotSpecError("Composed graph annotations must be a list.")
    if not isinstance(raw_selected_samples, list):
        raise PlotSpecError("Composed graph selected_samples must be a list.")

    segment_key = spec.get("segment_key")
    normalized_segment_key = None
    if segment_key is not None:
        normalized_segment_key = str(segment_key).strip() or None

    try:
        return CustomGraphSpec(
            title=str(spec.get("title", "Custom Graph")),
            x_domain=str(spec.get("x_domain", "")).strip(),
            left_axis=[_normalize_axis_layer(item, default_role="left") for item in raw_left_axis],
            right_axis=_normalize_axis_layer(raw_right_axis, default_role="right")
            if raw_right_axis is not None
            else None,
            view_domain=str(spec.get("view_domain", "full_curve")).strip() or "full_curve",
            segment_key=normalized_segment_key,
            rebase_x=bool(spec.get("rebase_x", False)),
            annotations=[_normalize_annotation(item) for item in raw_annotations],
            data_scope=str(spec.get("data_scope", "grouped")).strip() or "grouped",
            selected_samples=[
                str(item).strip() for item in raw_selected_samples if str(item).strip()
            ],
            display_mode=str(spec.get("display_mode", "stacked")).strip() or "stacked",
            overlay=_normalize_overlay(raw_overlay) if raw_overlay is not None else None,
            enabled=bool(spec.get("enabled", True)),
            band_mode=str(spec.get("band_mode", "sd")).strip() or "sd",
        )
    except ValueError as exc:
        raise PlotSpecError(str(exc)) from exc


def _effective_composed_overlays(spec: CustomGraphSpec) -> list[CustomGraphOverlay]:
    """Return overlays implied by the persisted custom-graph recipe."""
    overlays: list[CustomGraphOverlay] = []

    if spec.overlay is not None:
        overlays.append(spec.overlay)

    deduped: list[CustomGraphOverlay] = []
    seen: set[tuple[str, str]] = set()
    for overlay in overlays:
        identity = (overlay.kind, overlay.key)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(overlay)
    return deduped


def _segment_is_index_based(segment_key: str) -> bool:
    """Return ``True`` when a semantic segment can be sliced by row indices."""
    segment = SEMANTIC_SEGMENTS.get(segment_key)
    return bool(segment and all(column.endswith("Index") for column in segment.qc_columns))


def validate_composed_graph_spec(spec: CustomGraphSpec) -> None:
    """Validate that a composed graph spec is self-consistent and trace-safe."""
    if not spec.title.strip():
        raise PlotSpecError("Graph title cannot be empty.")
    if not spec.x_domain.strip():
        raise PlotSpecError("Select an X variable.")
    if spec.x_domain not in VARIABLE_REGISTRY:
        raise PlotSpecError(f"Unknown X variable: {spec.x_domain}")

    x_meta = registry_entry(spec.x_domain)
    if x_meta.source != "trace":
        raise PlotSpecError(f"X variable '{spec.x_domain}' does not belong to trace plots.")
    if x_meta.kind != "x":
        raise PlotSpecError(f"X variable '{spec.x_domain}' is not selectable on the x-axis.")

    if not spec.left_axis:
        raise PlotSpecError("Select at least one left-axis variable.")
    left_units = {
        registry_entry(layer.variable).unit
        for layer in spec.left_axis
        if registry_entry(layer.variable).unit
    }
    if len(left_units) > 1:
        raise PlotSpecError("Left-axis variables must share the same unit.")

    if spec.view_domain == "semantic_segment":
        segment = SEMANTIC_SEGMENTS.get(spec.segment_key or "")
        if segment is None:
            raise PlotSpecError(f"Unknown semantic segment: {spec.segment_key}")
        if not segment.allowed_x_domains or spec.x_domain not in segment.allowed_x_domains:
            raise PlotSpecError(
                "Semantic segment "
                f"'{spec.segment_key}' does not support x-domain '{spec.x_domain}'."
            )
        if not _segment_is_index_based(spec.segment_key):
            raise PlotSpecError(
                "Semantic segment "
                f"'{spec.segment_key}' uses value-based QC markers "
                "and is not supported by the shared slice/rebase path."
            )
        if not spec.rebase_x:
            raise PlotSpecError("Semantic segment graphs must rebase the X axis.")

        left_variables = {layer.variable for layer in spec.left_axis}
        for annotation in spec.annotations:
            annotation_meta = ANNOTATION_COMPATIBILITY.get(annotation.key)
            if annotation_meta is None:
                raise PlotSpecError(f"Unknown annotation: {annotation.key}")
            if spec.segment_key not in annotation_meta.allowed_segments:
                raise PlotSpecError(
                    "Annotation "
                    f"'{annotation.key}' is not compatible with "
                    f"semantic segment '{spec.segment_key}'."
                )
            missing_left = [
                variable
                for variable in annotation_meta.required_left_variables
                if variable not in left_variables
            ]
            if missing_left:
                raise PlotSpecError(
                    f"Annotation '{annotation.key}' requires left-axis variables: "
                    f"{', '.join(missing_left)}"
                )

    for layer in [*spec.left_axis, *([spec.right_axis] if spec.right_axis is not None else [])]:
        if layer.variable not in VARIABLE_REGISTRY:
            raise PlotSpecError(f"Unknown Y variable: {layer.variable}")
        y_meta = registry_entry(layer.variable)
        if y_meta.source != "trace":
            raise PlotSpecError(f"Y variable '{layer.variable}' does not belong to trace plots.")
        if y_meta.kind != "y":
            raise PlotSpecError(f"Y variable '{layer.variable}' is not selectable on the y-axis.")

    effective_overlays = _effective_composed_overlays(spec)
    if not effective_overlays:
        return

    left_variables = {layer.variable for layer in spec.left_axis}
    for overlay in effective_overlays:
        if overlay.kind == "regression":
            if spec.view_domain != "semantic_segment":
                raise PlotSpecError("Regression overlays require a semantic segment graph.")
            if overlay.key not in SEMANTIC_SEGMENTS:
                raise PlotSpecError(f"Unknown regression segment: {overlay.key}")
            if overlay.key != (spec.segment_key or ""):
                raise PlotSpecError("Regression overlay must match the selected semantic segment.")
            continue
        overlay_meta = OVERLAY_COMPATIBILITY.get(overlay.key)
        if overlay_meta is None:
            raise PlotSpecError(f"Unknown overlay: {overlay.key}")
        if overlay_meta.item_type != overlay.kind:
            raise PlotSpecError(f"Overlay '{overlay.key}' does not match kind '{overlay.kind}'.")
        if spec.x_domain not in overlay_meta.allowed_x_domains:
            raise PlotSpecError(
                f"Overlay '{overlay.key}' does not support x-domain '{spec.x_domain}'."
            )

        missing_left = [
            variable
            for variable in overlay_meta.requires_left_variables
            if variable not in left_variables
        ]
        if missing_left:
            raise PlotSpecError(
                f"Overlay '{overlay.key}' requires left-axis variables: {', '.join(missing_left)}"
            )


def expand_composed_graph_spec(spec: CustomGraphSpec) -> list[ResolvedComposedGraphJob]:
    """Expand one composed spec into its single render job."""
    validate_composed_graph_spec(spec)
    return [
        ResolvedComposedGraphJob(
            spec_title=spec.title,
            x_label=spec.x_domain,
            left_layers=list(spec.left_axis),
            right_layer=spec.right_axis,
            overlays=_effective_composed_overlays(spec),
            band_mode=spec.band_mode,
            segment_key=spec.segment_key,
            annotations=list(spec.annotations),
            data_scope=spec.data_scope,
            selected_samples=list(spec.selected_samples),
            display_mode=spec.display_mode,
            rebase_x=spec.rebase_x,
        )
    ]


def _slugify(value: str) -> str:
    """Return a filename-safe slug for a plot title."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return slug.lower() or "plot"


def _allocate_plot_path(
    output_dir: Path,
    spec_title: str,
    x_label: str,
    allocated_stems: dict[str, int],
    *,
    export_stem_suffix: str | None = None,
) -> Path:
    """Return a unique output path for one rendered plot."""
    stem_parts = [_slugify(spec_title)]
    if export_stem_suffix:
        stem_parts.append(_slugify(export_stem_suffix))
    stem_parts.append(_slugify(x_label))
    stem = "_".join(stem_parts)
    next_index = allocated_stems.get(stem, 0) + 1
    candidate = output_dir / (f"{stem}.png" if next_index == 1 else f"{stem}_{next_index}.png")
    while candidate.exists():
        next_index += 1
        candidate = output_dir / f"{stem}_{next_index}.png"
    allocated_stems[stem] = next_index
    return candidate


def normalize_graph_spec(spec: GraphSpec | dict[str, Any]) -> GraphSpec:
    """Normalize modern and legacy graph spec payloads into ``GraphSpec``."""
    if isinstance(spec, GraphSpec):
        return spec
    if not isinstance(spec, dict):
        raise PlotSpecError("Invalid graph spec payload.")

    raw_y_cols = spec.get("y_cols")
    raw_y_vars = spec.get("y_vars")
    if isinstance(raw_y_cols, list):
        y_cols = [str(item) for item in raw_y_cols]
    elif isinstance(raw_y_vars, str):
        y_cols = [token.strip() for token in raw_y_vars.split(",") if token.strip()]
    else:
        y_cols = [str(item) for item in raw_y_cols or []]

    raw_x_cols = spec.get("x_cols")
    if isinstance(raw_x_cols, list):
        x_cols = [str(item) for item in raw_x_cols]
    else:
        legacy_x = spec.get("x_col", "Time (s)")
        x_cols = [str(legacy_x)] if str(legacy_x).strip() else []

    plot_type = str(spec.get("plot_type", "")).strip()
    if not plot_type:
        sources = {
            registry_entry(label).source
            for label in [*x_cols, *y_cols]
            if label in VARIABLE_REGISTRY
        }
        plot_type = "metric" if sources == {"metric"} else "trace"

    return GraphSpec(
        title=str(spec.get("title", "Custom Graph")),
        plot_type="metric" if plot_type == "metric" else "trace",
        x_cols=x_cols,
        y_cols=y_cols,
        mode=str(spec.get("mode", "auto")),
        enabled=bool(spec.get("enabled", True)),
        curve_mode=str(spec.get("curve_mode", "mean_band")),
        band_mode=str(spec.get("band_mode", "sd")),
        metric_view=str(spec.get("metric_view", "both")),
    )


def validate_graph_spec(spec: GraphSpec) -> None:
    """Validate that a graph spec is self-consistent and source-safe."""
    if not spec.title.strip():
        raise PlotSpecError("Graph title cannot be empty.")
    if not spec.x_cols:
        raise PlotSpecError("Select at least one X variable.")
    if not spec.y_cols:
        raise PlotSpecError("Select at least one Y variable.")

    allowed_source = spec.plot_type
    for x_label in spec.x_cols:
        if x_label not in VARIABLE_REGISTRY:
            raise PlotSpecError(f"Unknown X variable: {x_label}")
        x_meta = registry_entry(x_label)
        if x_meta.source != allowed_source:
            raise PlotSpecError(
                f"X variable '{x_label}' does not belong to {allowed_source} plots."
            )
        if x_meta.kind != "x" and not (allowed_source == "metric" and x_meta.scale == "numeric"):
            raise PlotSpecError(f"X variable '{x_label}' is not selectable on the x-axis.")

    for y_label in spec.y_cols:
        if y_label not in VARIABLE_REGISTRY:
            raise PlotSpecError(f"Unknown Y variable: {y_label}")
        y_meta = registry_entry(y_label)
        if y_meta.source != allowed_source:
            raise PlotSpecError(
                f"Y variable '{y_label}' does not belong to {allowed_source} plots."
            )
        if y_meta.kind != "y":
            raise PlotSpecError(f"Y variable '{y_label}' is not selectable on the y-axis.")


def expand_graph_spec_jobs(spec: GraphSpec) -> list[ResolvedGraphJob]:
    """Expand a single graph spec into one render job per selected x-axis."""
    validate_graph_spec(spec)
    jobs: list[ResolvedGraphJob] = []
    for x_label in spec.x_cols:
        jobs.append(
            ResolvedGraphJob(
                spec_title=spec.title,
                plot_type=spec.plot_type,
                x_label=x_label,
                y_labels=list(spec.y_cols),
                mode=spec.mode,
                curve_mode=spec.curve_mode,
                band_mode=spec.band_mode,
                metric_view=spec.metric_view,
            )
        )
    return jobs


def _require_column(frame: pd.DataFrame, label: str) -> str:
    """Resolve a registry label into a DataFrame column name and validate it exists."""
    column = registry_entry(label).column if label in VARIABLE_REGISTRY else label
    if column not in frame.columns:
        raise PlotSpecError(f"Missing required plot column: {column}")
    return column


def _blend_color(color: Any, accent: str, mix: float = 0.45) -> tuple[float, float, float, float]:
    """Blend one color toward an axis accent while preserving alpha."""
    rgba = np.array(matplotlib.colors.to_rgba(color), dtype=float)
    accent_rgba = np.array(matplotlib.colors.to_rgba(accent), dtype=float)
    blended = rgba.copy()
    blended[:3] = ((1.0 - mix) * rgba[:3]) + (mix * accent_rgba[:3])
    return tuple(float(component) for component in blended)


def _recolor_axis_artists(ax: Any, accent: str) -> None:
    """Shift the rendered line/band colors toward one axis accent."""
    for line in ax.lines:
        line.set_color(_blend_color(line.get_color(), accent))

    for collection in ax.collections:
        facecolors = collection.get_facecolors()
        if len(facecolors):
            collection.set_facecolors([_blend_color(color, accent) for color in facecolors])
        edgecolors = collection.get_edgecolors()
        if len(edgecolors):
            collection.set_edgecolors([_blend_color(color, accent) for color in edgecolors])

    for patch in ax.patches:
        facecolor = patch.get_facecolor()
        if facecolor is not None:
            patch.set_facecolor(_blend_color(facecolor, accent))
        edgecolor = patch.get_edgecolor()
        if edgecolor is not None:
            patch.set_edgecolor(_blend_color(edgecolor, accent))


def _style_trace_axis_side(ax: Any, accent: str, *, recolor_artists: bool) -> None:
    """Apply one accent to a y-axis and optionally recolor its rendered artists."""
    if recolor_artists:
        _recolor_axis_artists(ax, accent)
    ax.yaxis.label.set_color(accent)
    ax.tick_params(axis="y", colors=accent)
    if "left" in ax.spines:
        ax.spines["left"].set_color(accent)
    if "right" in ax.spines:
        ax.spines["right"].set_color(accent)


def _units_compatible(labels: list[str]) -> bool:
    """Return ``True`` when all requested variables share the same unit."""
    units = {registry_entry(label).unit for label in labels if label in VARIABLE_REGISTRY}
    return len(units) <= 1


def _plot_individual(
    ax: Any,
    trace_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    style: PlotStyleConfig,
    group_order: list[str] | None = None,
) -> None:
    """Plot individual replicate traces grouped by file and color."""
    requested_order = [str(group).strip() for group in (group_order or []) if str(group).strip()]
    groups_present = trace_df["Group"].dropna().astype(str).tolist()
    unique_present: list[str] = []
    for group in groups_present:
        if group not in unique_present:
            unique_present.append(group)
    ordered_groups = [group for group in requested_order if group in unique_present]
    ordered_groups.extend([group for group in unique_present if group not in ordered_groups])

    for group in ordered_groups:
        group_frame = trace_df[trace_df["Group"].astype(str) == group]
        for _, frame in group_frame.groupby("File", sort=False):
            ordered = frame.sort_values(x_col)
            ax.plot(
                ordered[x_col],
                ordered[y_col],
                color=style.get_color(str(group)),
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
    """Interpolate replicate curves and compute mean with spread."""
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
            left = min(float(curve[0][0]) for curve in replicate_curves)
            right = max(float(curve[0][-1]) for curve in replicate_curves)
        if right <= left:
            continue

        point_count = int(np.clip(np.mean([len(curve[0]) for curve in replicate_curves]), 80, 400))
        grid = np.linspace(left, right, point_count)
        interpolated = [np.interp(grid, x_vals, y_vals) for x_vals, y_vals in replicate_curves]
        stack = np.vstack(interpolated)
        mean_vals = np.nanmean(stack, axis=0)
        sd_vals = (
            np.nanstd(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean_vals)
        )
        spread = (
            1.96 * sd_vals / np.sqrt(stack.shape[0]) if band_mode.lower() == "ci95" else sd_vals
        )

        for x_val, mean_val, lower_val, upper_val in zip(
            grid,
            mean_vals,
            mean_vals - spread,
            mean_vals + spread,
            strict=True,
        ):
            rows.append(
                {
                    group_col: group_name,
                    x_col: x_val,
                    "Mean": mean_val,
                    "Lower": lower_val,
                    "Upper": upper_val,
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
    group_order: list[str] | None = None,
) -> None:
    """Plot group mean curves with SD or CI bands."""
    band_df = build_mean_band(trace_df, x_col=x_col, y_col=y_col, band_mode=band_mode)
    if band_df.empty:
        return

    requested_order = [str(group).strip() for group in (group_order or []) if str(group).strip()]
    groups_present = band_df["Group"].dropna().astype(str).tolist()
    unique_present: list[str] = []
    for group in groups_present:
        if group not in unique_present:
            unique_present.append(group)
    ordered_groups = [group for group in requested_order if group in unique_present]
    ordered_groups.extend([group for group in unique_present if group not in ordered_groups])

    for group_name in ordered_groups:
        frame = band_df[band_df["Group"].astype(str) == group_name]
        if frame.empty:
            continue
        ordered = frame.sort_values(x_col)
        color = style.get_color(str(group_name))
        ax.fill_between(ordered[x_col], ordered["Lower"], ordered["Upper"], color=color, alpha=0.18)
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
    group_order: list[str] | None = None,
) -> None:
    """Apply the selected trace rendering mode to an axis."""
    mode = curve_mode.lower().strip()
    if mode in {"individual", "both"}:
        _plot_individual(ax, trace_df, x_col, y_col, style, group_order=group_order)
    if mode in {"mean_band", "both"}:
        _plot_mean_band(
            ax, trace_df, x_col, y_col, style, band_mode=band_mode, group_order=group_order
        )


def _ordered_legend(
    handles: list[Any],
    labels: list[str],
    group_order: list[str] | None = None,
) -> tuple[list[Any], list[str]]:
    """Deduplicate and order legend entries using the requested group order."""
    dedup_handles: list[Any] = []
    dedup_labels: list[str] = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels, strict=True):
        label_clean = str(label).strip()
        if not label_clean or label_clean in seen:
            continue
        seen.add(label_clean)
        dedup_handles.append(handle)
        dedup_labels.append(label_clean)

    requested_order = [str(group).strip() for group in (group_order or []) if str(group).strip()]
    if not requested_order:
        return dedup_handles, dedup_labels

    order_map = {group: index for index, group in enumerate(requested_order)}
    ordered_items = sorted(
        list(zip(dedup_handles, dedup_labels, strict=True)),
        key=lambda item: (order_map.get(item[1], 10_000), dedup_labels.index(item[1])),
    )
    if not ordered_items:
        return dedup_handles, dedup_labels
    ordered_handles, ordered_labels = zip(*ordered_items, strict=True)
    return list(ordered_handles), list(ordered_labels)


def _apply_axis_legend(
    ax: Any, group_order: list[str] | None = None, extra_axes: list[Any] | None = None
) -> None:
    """Apply a deduplicated legend to one axis, optionally merging labels from sibling axes."""
    handles, labels = ax.get_legend_handles_labels()
    for extra_ax in extra_axes or []:
        extra_handles, extra_labels = extra_ax.get_legend_handles_labels()
        handles.extend(extra_handles)
        labels.extend(extra_labels)
    handles, labels = _ordered_legend(handles, labels, group_order=group_order)
    if handles:
        ax.legend(handles, labels, frameon=False)


def _categorical_order(values: list[str], preferred_order: list[str] | None = None) -> list[str]:
    """Return a stable categorical order with preferred items first."""
    preferred = [item for item in (preferred_order or []) if item in values]
    return preferred + [item for item in values if item not in preferred]


def _plot_metric_group_axis(
    ax: Any,
    metrics_df: pd.DataFrame,
    y_label: str,
    style: PlotStyleConfig,
    metric_view: str,
    group_order: list[str] | None,
    stats_by_metric: dict[str, dict[str, Any]] | None,
) -> None:
    """Plot one metric against ``Group`` using raw points, summary bars, or both."""
    metric_col = _require_column(metrics_df, y_label)
    frame = metrics_df[["Group", "Filename", metric_col]].dropna().copy()
    if frame.empty:
        return

    groups = _categorical_order(frame["Group"].astype(str).tolist(), preferred_order=group_order)
    unique_groups: list[str] = []
    for group in groups:
        if group not in unique_groups:
            unique_groups.append(group)
    x_positions = {group: index for index, group in enumerate(unique_groups)}

    if metric_view in {"summary", "both"}:
        summary = (
            frame.groupby("Group")[metric_col]
            .agg(["mean", "std"])
            .rename(columns={"mean": "Mean", "std": "SD"})
            .reindex(unique_groups)
            .fillna(0.0)
        )
        bars = ax.bar(
            list(range(len(unique_groups))),
            summary["Mean"].to_numpy(dtype=float),
            yerr=summary["SD"].to_numpy(dtype=float),
            color=[style.get_color(group) for group in unique_groups],
            alpha=0.45 if metric_view == "both" else 0.9,
            capsize=4,
        )
        metric_stats = (stats_by_metric or {}).get(metric_col)
        if metric_stats:
            summary_df = metric_stats.get("summary_df", pd.DataFrame()).copy()
            if not summary_df.empty and "Significance" in summary_df.columns:
                letter_map = {
                    str(row["Group"]): str(row["Significance"])
                    for _, row in summary_df.iterrows()
                    if "Group" in row.index
                }
                ymax = (
                    float(
                        np.nanmax(
                            summary["Mean"].to_numpy(dtype=float)
                            + summary["SD"].to_numpy(dtype=float)
                        )
                    )
                    if len(summary)
                    else 1.0
                )
                offset = max(ymax * 0.04, 0.02)
                for bar, group in zip(bars, unique_groups, strict=True):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + summary.loc[group, "SD"] + offset,
                        letter_map.get(group, ""),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )

    if metric_view in {"raw", "both"}:
        for group, group_frame in frame.groupby("Group", sort=False):
            positions = np.full(len(group_frame), x_positions[str(group)], dtype=float)
            jitter = (
                np.linspace(-0.12, 0.12, len(group_frame))
                if len(group_frame) > 1
                else np.array([0.0])
            )
            ax.scatter(
                positions + jitter,
                group_frame[metric_col].to_numpy(dtype=float),
                color=style.get_color(str(group)),
                alpha=0.9,
                s=26,
                label=str(group),
                edgecolors="white",
                linewidths=0.5,
            )

    ax.set_xticks(list(range(len(unique_groups))))
    ax.set_xticklabels(unique_groups, rotation=20, ha="right")
    ax.set_ylabel(axis_label(y_label))
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)


def _plot_metric_filename_axis(
    ax: Any,
    metrics_df: pd.DataFrame,
    y_label: str,
    style: PlotStyleConfig,
) -> None:
    """Plot one metric against ``Filename`` as per-file raw points."""
    metric_col = _require_column(metrics_df, y_label)
    frame = metrics_df[["Filename", "Group", metric_col]].dropna().copy().reset_index(drop=True)
    if frame.empty:
        return

    positions = np.arange(len(frame), dtype=float)
    colors = [style.get_color(str(group)) for group in frame["Group"].astype(str).tolist()]
    ax.scatter(positions, frame[metric_col].to_numpy(dtype=float), color=colors, s=28, alpha=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(frame["Filename"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylabel(axis_label(y_label))
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)


def _plot_metric_scatter_axis(
    ax: Any,
    metrics_df: pd.DataFrame,
    x_label: str,
    y_label: str,
    style: PlotStyleConfig,
) -> None:
    """Plot one numeric metric against another numeric metric."""
    x_col = _require_column(metrics_df, x_label)
    y_col = _require_column(metrics_df, y_label)
    frame = metrics_df[[x_col, y_col, "Group"]].dropna().copy()
    if frame.empty:
        return

    for group, group_frame in frame.groupby("Group", sort=False):
        ax.scatter(
            group_frame[x_col].to_numpy(dtype=float),
            group_frame[y_col].to_numpy(dtype=float),
            color=style.get_color(str(group)),
            label=str(group),
            s=28,
            alpha=0.9,
        )
    ax.set_xlabel(axis_label(x_label))
    ax.set_ylabel(axis_label(y_label))
    ax.grid(True, linestyle="--", alpha=0.25)


def _resolve_job_mode(mode: str, y_labels: list[str]) -> str:
    """Resolve ``auto`` or invalid modes into concrete panel or overlay behavior."""
    normalized = mode.lower().strip()
    if normalized == "overlay" and _units_compatible(y_labels):
        return "overlay"
    if normalized == "overlay":
        return "panel"
    if normalized == "panel":
        return "panel"
    return "overlay" if _units_compatible(y_labels) else "panel"


def _plot_trace_job(
    trace_df: pd.DataFrame,
    job: ResolvedGraphJob,
    style: PlotStyleConfig,
    output_dir: Path,
    figure_config: FigureConfig,
    allocated_stems: dict[str, int],
    group_order: list[str] | None = None,
) -> list[str]:
    """Render one trace-plot job and return saved file paths."""
    x_col = _require_column(trace_df, job.x_label)
    y_cols = [_require_column(trace_df, label) for label in job.y_labels]
    mode = _resolve_job_mode(job.mode, job.y_labels)
    figsize = figure_config.resolve_size()

    if mode == "overlay":
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for y_col, _y_label in zip(y_cols, job.y_labels, strict=True):
            _apply_curve_mode(
                ax,
                trace_df,
                x_col,
                y_col,
                style,
                curve_mode=job.curve_mode,
                band_mode=job.band_mode,
                group_order=group_order,
            )
            ax.set_ylabel(" / ".join(axis_label(label) for label in job.y_labels))
        ax.set_xlabel(axis_label(job.x_label))
        ax.set_title(f"{job.spec_title} [{job.x_label}]")
        ax.grid(True, linestyle="--", alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = _ordered_legend(handles, labels, group_order=group_order)
        if handles:
            ax.legend(handles, labels, frameon=False)
    else:
        fig, axes = plt.subplots(len(y_cols), 1, figsize=figsize, sharex=True)
        if len(y_cols) == 1:
            axes = [axes]
        for ax, y_col, y_label in zip(axes, y_cols, job.y_labels, strict=True):
            _apply_curve_mode(
                ax,
                trace_df,
                x_col,
                y_col,
                style,
                curve_mode=job.curve_mode,
                band_mode=job.band_mode,
                group_order=group_order,
            )
            ax.set_ylabel(axis_label(y_label))
            ax.grid(True, linestyle="--", alpha=0.25)
        axes[-1].set_xlabel(axis_label(job.x_label))
        axes[0].set_title(f"{job.spec_title} [{job.x_label}]")
        handles, labels = axes[0].get_legend_handles_labels()
        handles, labels = _ordered_legend(handles, labels, group_order=group_order)
        if handles:
            axes[0].legend(handles, labels, frameon=False)

    fig.tight_layout()
    path = _allocate_plot_path(output_dir, job.spec_title, job.x_label, allocated_stems)
    fig.savefig(path, dpi=figure_config.dpi, bbox_inches="tight")
    plt.close(fig)
    return [str(path)]


def _overlay_required_columns(overlay: CustomGraphOverlay) -> tuple[str, ...]:
    """Return the trace/QC columns required to render an overlay."""
    if overlay.key == "b1_start_to_peak1":
        return ("Bite1 Start Index", "Peak1 Index")
    if overlay.key == "peak1_to_b1_end":
        return ("Peak1 Index", "Bite1 End Index")
    if overlay.key == "b1_end_to_b2_start":
        return ("Bite1 End Index", "Bite2 Start Index")
    if overlay.key == "b2_start_to_peak2":
        return ("Bite2 Start Index", "Peak2 Index")
    if overlay.key == "peak2_to_b2_end":
        return ("Peak2 Index", "Bite2 End Index")
    if overlay.key == "hardness_peak1":
        return ("Peak1 Index",)
    if overlay.key == "adhesiveness":
        return ("Bite1 End Index", "Bite2 Start Index")
    if overlay.key == "modulus_window":
        return ("Modulus Strain Min (%)", "Modulus Strain Max (%)")
    return ()


def _render_segment_regression_overlay(
    ax: Any,
    trace_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    overlay: CustomGraphOverlay,
) -> str | None:
    """Render one fitted regression line per file over the already-sliced segment view."""
    segment_meta = SEMANTIC_SEGMENTS.get(overlay.key)
    overlay_label = (
        f"{segment_meta.label} regression"
        if segment_meta is not None
        else f"{overlay.key} regression"
    )
    drawn = False

    for file_key, raw_frame in trace_df.groupby("File", sort=False):
        frame = raw_frame.sort_values(x_col).reset_index(drop=True)
        if frame.empty:
            continue
        x_vals = frame[x_col].to_numpy(dtype=float)
        y_vals = frame[y_col].to_numpy(dtype=float)
        finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if int(finite_mask.sum()) < 2:
            continue
        x_fit = x_vals[finite_mask]
        y_fit = y_vals[finite_mask]
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        x_line = np.linspace(float(x_fit.min()), float(x_fit.max()), 48)
        file_name = _resolve_frame_file_name(file_key, frame)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label=f"{file_name} · {overlay_label}",
        )
        drawn = True

    if not drawn:
        return f"overlay '{overlay.key}' skipped: insufficient points for regression."
    return None


def _overlay_qc_file_key(row: pd.Series) -> str:
    """Return the normalized file key for one QC summary row."""
    for column in ("Filename", "File"):
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return ""


def _build_overlay_qc_lookup(qc_df: pd.DataFrame | None) -> dict[str, pd.Series]:
    """Index QC summary rows by filename for composed overlay rendering."""
    if qc_df is None or qc_df.empty:
        return {}
    lookup: dict[str, pd.Series] = {}
    for _, row in qc_df.iterrows():
        file_key = _overlay_qc_file_key(row)
        if file_key and file_key not in lookup:
            lookup[file_key] = row
    return lookup


def _overlay_value(row: pd.Series | None, column: str) -> Any | None:
    """Return one QC overlay value from a summary row."""
    if row is None or column not in row.index:
        return None
    value = row.get(column)
    return None if pd.isna(value) else value


def _overlay_index(frame: pd.DataFrame, row: pd.Series | None, column: str) -> int | None:
    """Return a clamped per-file overlay index."""
    raw_value = _overlay_value(row, column)
    if raw_value is None:
        return None
    try:
        index = int(float(raw_value))
    except (TypeError, ValueError):
        return None
    return int(np.clip(index, 0, max(len(frame) - 1, 0)))


def segment_index_columns(segment_key: str) -> tuple[str, str]:
    """Return the QC index columns for one semantic segment."""
    segment = SEMANTIC_SEGMENTS.get(segment_key)
    if segment is None:
        raise PlotSpecError(f"Unknown semantic segment: {segment_key}")
    if len(segment.qc_columns) != 2:
        raise PlotSpecError(f"Semantic segment '{segment_key}' does not define two QC columns.")
    if not _segment_is_index_based(segment_key):
        raise PlotSpecError(
            "Semantic segment "
            f"'{segment_key}' uses value-based QC markers "
            "and is not supported by the shared slice/rebase path."
        )
    start_column, end_column = segment.qc_columns
    return start_column, end_column


def _rebase_overlay_row(row: pd.Series, offset: int) -> pd.Series:
    """Shift QC index columns so they line up with a sliced segment frame."""
    if offset <= 0:
        return row.copy()

    rebased = row.copy()
    for column in rebased.index:
        if not str(column).endswith("Index"):
            continue
        value = rebased.get(column)
        if pd.isna(value):
            continue
        try:
            rebased[column] = int(float(value)) - offset
        except (TypeError, ValueError):
            continue
    return rebased


def _slice_trace_to_segment(
    frame: pd.DataFrame,
    qc_row: pd.Series,
    segment_key: str,
    x_label: str,
    *,
    rebase_x: bool,
) -> pd.DataFrame:
    start_column, end_column = segment_index_columns(segment_key)
    start_idx = _overlay_index(frame, qc_row, start_column)
    end_idx = _overlay_index(frame, qc_row, end_column)
    if start_idx is None:
        raise PlotSpecError(f"missing marker '{start_column}'")
    if end_idx is None:
        raise PlotSpecError(f"missing marker '{end_column}'")
    if end_idx < start_idx:
        raise PlotSpecError("end marker precedes start marker")

    sliced = frame.iloc[start_idx : end_idx + 1].copy()
    x_col = _require_column(sliced, x_label)
    if rebase_x and not sliced.empty:
        sliced[x_col] = (sliced[x_col].astype(float) - float(sliced[x_col].iloc[0])).round(10)
    return sliced


def _resolve_frame_file_name(file_key: Any, frame: pd.DataFrame) -> str:
    """Return the best available sample name for one trace frame."""
    file_name = str(file_key).strip()
    if not file_name and "Filename" in frame.columns:
        file_name = str(frame["Filename"].iloc[0]).strip()
    if not file_name and "File" in frame.columns:
        file_name = str(frame["File"].iloc[0]).strip()
    return file_name or "<unknown file>"


def _resolve_qc_row(
    file_key: Any, frame: pd.DataFrame, qc_lookup: dict[str, pd.Series]
) -> pd.Series | None:
    """Look up the QC summary row for one trace frame."""
    qc_row = qc_lookup.get(str(file_key).strip())
    if qc_row is None and "Filename" in frame.columns:
        qc_row = qc_lookup.get(str(frame["Filename"].iloc[0]).strip())
    if qc_row is None and "File" in frame.columns:
        qc_row = qc_lookup.get(str(frame["File"].iloc[0]).strip())
    return qc_row


def _prepare_segment_frame(
    frame: pd.DataFrame,
    file_key: Any,
    qc_lookup: dict[str, pd.Series],
    job: ResolvedComposedGraphJob,
) -> tuple[pd.DataFrame | None, pd.Series | None, str | None]:
    """Slice one frame to the selected segment and rebase QC markers when needed."""
    file_name = _resolve_frame_file_name(file_key, frame)
    qc_row = _resolve_qc_row(file_key, frame, qc_lookup)
    if qc_row is None:
        return (
            None,
            None,
            f"{job.spec_title}: segment '{job.segment_key}' skipped "
            f"for {file_name}: missing QC summary row.",
        )

    try:
        prepared_frame = _slice_trace_to_segment(
            frame, qc_row, job.segment_key or "", job.x_label, rebase_x=job.rebase_x
        )
    except PlotSpecError as exc:
        return (
            None,
            None,
            f"{job.spec_title}: segment '{job.segment_key}' skipped for {file_name}: {exc}.",
        )

    prepared_qc_row = qc_row
    if job.segment_key and job.rebase_x:
        start_column, _ = segment_index_columns(job.segment_key)
        start_idx = _overlay_index(frame, qc_row, start_column)
        prepared_qc_row = _rebase_overlay_row(qc_row, start_idx or 0)
    return prepared_frame, prepared_qc_row, None


def _prepare_grouped_trace_data(
    trace_df: pd.DataFrame,
    qc_lookup: dict[str, pd.Series],
    job: ResolvedComposedGraphJob,
    x_col: str,
) -> tuple[pd.DataFrame, dict[str, pd.Series], list[str]]:
    """Return grouped render data, slicing to one semantic segment when requested."""
    if not job.segment_key:
        return trace_df, qc_lookup, []

    warnings: list[str] = []
    segment_frames: list[pd.DataFrame] = []
    segment_qc_lookup: dict[str, pd.Series] = {}
    for file_key, raw_frame in trace_df.groupby("File", sort=False):
        frame = raw_frame.sort_values(x_col).reset_index(drop=True)
        if frame.empty:
            continue
        prepared_frame, prepared_qc_row, warning = _prepare_segment_frame(
            frame, file_key, qc_lookup, job
        )
        if warning is not None:
            warnings.append(warning)
            continue
        assert prepared_frame is not None and prepared_qc_row is not None
        prepared_frame = prepared_frame.copy()
        prepared_frame["File"] = frame["File"].iloc[0]
        if "Filename" in frame.columns:
            prepared_frame["Filename"] = frame["Filename"].iloc[0]
        if "Group" in frame.columns:
            prepared_frame["Group"] = frame["Group"].iloc[0]
        segment_frames.append(prepared_frame)
        segment_qc_lookup[_resolve_frame_file_name(file_key, frame)] = prepared_qc_row

    if not segment_frames:
        warnings.append(
            f"{job.spec_title}: segment '{job.segment_key}' is unavailable for all files."
        )
        raise PlotSpecError(" | ".join(warnings))
    return pd.concat(segment_frames, ignore_index=True), segment_qc_lookup, warnings


def _prepare_selected_sample_data(
    trace_df: pd.DataFrame,
    qc_lookup: dict[str, pd.Series],
    job: ResolvedComposedGraphJob,
    x_col: str,
) -> tuple[list[tuple[str, pd.DataFrame, pd.Series | None]], list[str]]:
    """Return one prepared frame per selected sample."""
    warnings: list[str] = []
    prepared_samples: list[tuple[str, pd.DataFrame, pd.Series | None]] = []
    filename_mask_source = (
        trace_df["Filename"].astype(str)
        if "Filename" in trace_df.columns
        else pd.Series("", index=trace_df.index, dtype=str)
    )

    for sample_name in job.selected_samples:
        sample_frame = trace_df[
            trace_df["File"].astype(str).eq(sample_name) | filename_mask_source.eq(sample_name)
        ].copy()
        if sample_frame.empty:
            warnings.append(
                f"{job.spec_title}: selected sample '{sample_name}' skipped: missing trace rows."
            )
            continue

        sample_frame = sample_frame.sort_values(x_col).reset_index(drop=True)
        sample_qc_row: pd.Series | None = _resolve_qc_row(sample_name, sample_frame, qc_lookup)
        if job.segment_key:
            prepared_frame, prepared_qc_row, warning = _prepare_segment_frame(
                sample_frame, sample_name, qc_lookup, job
            )
            if warning is not None:
                warnings.append(warning)
                continue
            assert prepared_frame is not None
            sample_frame = prepared_frame
            sample_qc_row = prepared_qc_row

        sample_frame = sample_frame.copy()
        sample_frame["File"] = sample_name
        if "Filename" in sample_frame.columns:
            sample_frame["Filename"] = sample_name
        prepared_samples.append((sample_name, sample_frame, sample_qc_row))

    return prepared_samples, warnings


def _render_composed_annotations(
    ax: Any,
    trace_df: pd.DataFrame,
    qc_lookup: dict[str, pd.Series],
    x_col: str,
    y_col: str,
    annotations: list[CustomGraphAnnotation],
) -> list[str]:
    """Render semantic annotations as markers and labels only."""
    warnings: list[str] = []
    for annotation in annotations:
        annotation_meta = ANNOTATION_COMPATIBILITY.get(annotation.key)
        if annotation_meta is None:
            warnings.append(f"annotation '{annotation.key}' skipped: unknown annotation.")
            continue

        drawn = False
        for file_key, raw_frame in trace_df.groupby("File", sort=False):
            frame = raw_frame.sort_values(x_col).reset_index(drop=True)
            if frame.empty:
                continue
            qc_row = _resolve_qc_row(file_key, frame, qc_lookup)
            if qc_row is None:
                continue
            x_vals = frame[x_col].to_numpy(dtype=float)
            y_vals = frame[y_col].to_numpy(dtype=float)

            columns = _overlay_required_columns(
                CustomGraphOverlay(kind="annotation", key=annotation.key)
            )
            if not columns:
                continue

            marker_points: list[tuple[float, float]] = []
            for column in columns:
                marker_idx = _overlay_index(frame, qc_row, column)
                if marker_idx is None:
                    marker_points = []
                    break
                marker_points.append((x_vals[marker_idx], y_vals[marker_idx]))
            if not marker_points:
                continue

            xs = [point[0] for point in marker_points]
            ys = [point[1] for point in marker_points]
            ax.scatter(
                xs,
                ys,
                color="#7C2D12",
                s=26,
                zorder=5,
                label=annotation_meta.label if not drawn else "",
            )
            anchor_idx = int(np.argmax(xs))
            ax.annotate(
                annotation_meta.label,
                (xs[anchor_idx], ys[anchor_idx]),
                textcoords="offset points",
                xytext=(5, 6),
                fontsize=8,
                color="#7C2D12",
            )
            drawn = True

        if not drawn:
            warnings.append(
                f"annotation '{annotation.key}' skipped: required QC values were unavailable."
            )
    return warnings


def _render_composed_overlay(
    ax: Any,
    trace_df: pd.DataFrame,
    qc_lookup: dict[str, pd.Series],
    x_col: str,
    y_col: str,
    overlay: CustomGraphOverlay,
) -> str | None:
    """Render a minimal overlay onto the composed figure, or return a warning."""
    if overlay.kind == "regression":
        return _render_segment_regression_overlay(ax, trace_df, x_col, y_col, overlay)

    qc_columns = {column for row in qc_lookup.values() for column in row.index}
    missing_columns = [
        column for column in _overlay_required_columns(overlay) if column not in qc_columns
    ]
    if missing_columns:
        return f"overlay '{overlay.key}' skipped: missing columns {', '.join(missing_columns)}"

    overlay_meta = OVERLAY_COMPATIBILITY.get(overlay.key)
    overlay_label = overlay_meta.label if overlay_meta is not None else overlay.key
    drawn = False

    for file_key, raw_frame in trace_df.groupby("File", sort=False):
        frame = raw_frame.sort_values(x_col).reset_index(drop=True)
        if frame.empty:
            continue
        qc_row = qc_lookup.get(str(file_key).strip())
        if qc_row is None and "Filename" in frame.columns:
            qc_row = qc_lookup.get(str(frame["Filename"].iloc[0]).strip())
        if qc_row is None:
            continue
        x_vals = frame[x_col].to_numpy(dtype=float)
        y_vals = frame[y_col].to_numpy(dtype=float)

        if overlay.key in {
            "b1_start_to_peak1",
            "peak1_to_b1_end",
            "b1_end_to_b2_start",
            "b2_start_to_peak2",
            "peak2_to_b2_end",
        }:
            start_column, end_column = _overlay_required_columns(overlay)
            start_idx = _overlay_index(frame, qc_row, start_column)
            end_idx = _overlay_index(frame, qc_row, end_column)
            if start_idx is None or end_idx is None:
                continue
            left, right = sorted((start_idx, end_idx))
            if right <= left:
                continue
            ax.plot(
                x_vals[left : right + 1],
                y_vals[left : right + 1],
                color="#111827",
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label=overlay_label if not drawn else "",
            )
            drawn = True
        elif overlay.key == "hardness_peak1":
            peak_idx = _overlay_index(frame, qc_row, "Peak1 Index")
            if peak_idx is None:
                continue
            ax.scatter(
                [x_vals[peak_idx]],
                [y_vals[peak_idx]],
                color="#7C2D12",
                s=22,
                zorder=5,
                label=overlay_label if not drawn else "",
            )
            ax.annotate(
                overlay_label,
                (x_vals[peak_idx], y_vals[peak_idx]),
                textcoords="offset points",
                xytext=(5, 6),
                fontsize=8,
                color="#7C2D12",
            )
            drawn = True
        elif overlay.key == "adhesiveness":
            start_idx = _overlay_index(frame, qc_row, "Bite1 End Index")
            end_idx = _overlay_index(frame, qc_row, "Bite2 Start Index")
            if start_idx is None or end_idx is None:
                continue
            left, right = sorted((start_idx, end_idx))
            if right <= left:
                continue
            x_seg = x_vals[left : right + 1]
            y_seg = y_vals[left : right + 1]
            where = y_seg < 0
            if int(where.sum()) < 2:
                continue
            ax.fill_between(
                x_seg,
                0.0,
                y_seg,
                where=where,
                interpolate=True,
                color="#FCA5A5",
                alpha=0.28,
                label=overlay_label if not drawn else "",
            )
            drawn = True
        elif overlay.key == "modulus_window":
            left_value = _overlay_value(qc_row, "Modulus Strain Min (%)")
            right_value = _overlay_value(qc_row, "Modulus Strain Max (%)")
            if left_value is None or right_value is None:
                continue
            try:
                left_float = float(left_value)
                right_float = float(right_value)
            except (TypeError, ValueError):
                continue
            if (
                not np.isfinite(left_float)
                or not np.isfinite(right_float)
                or right_float <= left_float
            ):
                continue
            ax.axvspan(
                left_float,
                right_float,
                color="#FDE68A",
                alpha=0.25,
                label=overlay_label if not drawn else "",
            )
            drawn = True

    if not drawn:
        return f"overlay '{overlay.key}' skipped: required QC values were unavailable."
    return None


def _capture_axis_artist_state(ax: Any) -> dict[str, list[Any]]:
    """Capture the current artist collections for rollback after overlay failures."""
    return {
        "artists": list(ax.artists),
        "collections": list(ax.collections),
        "lines": list(ax.lines),
        "patches": list(ax.patches),
        "texts": list(ax.texts),
    }


def _restore_axis_artist_state(ax: Any, before: dict[str, list[Any]]) -> None:
    """Remove artists added after ``before`` was captured."""
    for attribute, original_items in before.items():
        original_ids = {id(item) for item in original_items}
        for item in list(getattr(ax, attribute)):
            if id(item) in original_ids:
                continue
            item.remove()


def _plot_composed_trace_job(
    trace_df: pd.DataFrame,
    qc_lookup: dict[str, pd.Series],
    job: ResolvedComposedGraphJob,
    style: PlotStyleConfig,
    output_dir: Path,
    figure_config: FigureConfig,
    allocated_stems: dict[str, int],
    group_order: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Render one composed trace recipe and return saved paths plus warnings."""
    x_col = _require_column(trace_df, job.x_label)
    if job.data_scope == "selected_samples":
        return _plot_selected_sample_trace_job(
            trace_df=trace_df,
            qc_lookup=qc_lookup,
            job=job,
            style=style,
            output_dir=output_dir,
            figure_config=figure_config,
            allocated_stems=allocated_stems,
            group_order=group_order,
        )

    try:
        render_trace_df, render_qc_lookup, warnings = _prepare_grouped_trace_data(
            trace_df, qc_lookup, job, x_col
        )
    except PlotSpecError as exc:
        message = str(exc)
        if " | " in message:
            return [], [part for part in message.split(" | ") if part]
        raise
    return _render_composed_trace_figure(
        render_trace_df=render_trace_df,
        render_qc_lookup=render_qc_lookup,
        job=job,
        style=style,
        output_dir=output_dir,
        figure_config=figure_config,
        allocated_stems=allocated_stems,
        group_order=group_order,
        warnings=warnings,
    )


def _render_composed_trace_axis(
    ax_left: Any,
    render_trace_df: pd.DataFrame,
    render_qc_lookup: dict[str, pd.Series],
    job: ResolvedComposedGraphJob,
    style: PlotStyleConfig,
    group_order: list[str] | None,
) -> tuple[Any | None, list[str]]:
    """Render one composed trace payload on a provided axis."""
    warnings: list[str] = []
    x_col = _require_column(render_trace_df, job.x_label)

    for layer in job.left_layers:
        y_col = _require_column(render_trace_df, layer.variable)
        _apply_curve_mode(
            ax_left,
            render_trace_df,
            x_col,
            y_col,
            style,
            curve_mode=layer.curve_mode,
            band_mode=job.band_mode,
            group_order=group_order,
        )
    _style_trace_axis_side(ax_left, LEFT_AXIS_ACCENT, recolor_artists=False)

    ax_right = None
    if job.right_layer is not None:
        right_col = _require_column(render_trace_df, job.right_layer.variable)
        ax_right = ax_left.twinx()
        _apply_curve_mode(
            ax_right,
            render_trace_df,
            x_col,
            right_col,
            style,
            curve_mode=job.right_layer.curve_mode,
            band_mode=job.band_mode,
            group_order=group_order,
        )
        ax_right.set_ylabel(axis_label(job.right_layer.variable))
        ax_right.grid(False)
        _style_trace_axis_side(ax_right, RIGHT_AXIS_ACCENT, recolor_artists=True)

    if job.overlays and job.left_layers:
        overlay_ref_col = _require_column(render_trace_df, job.left_layers[0].variable)
        for overlay in job.overlays:
            overlay_artist_state = _capture_axis_artist_state(ax_left)
            try:
                overlay_warning = _render_composed_overlay(
                    ax_left,
                    render_trace_df,
                    render_qc_lookup,
                    x_col,
                    overlay_ref_col,
                    overlay,
                )
            except Exception as exc:
                _restore_axis_artist_state(ax_left, overlay_artist_state)
                overlay_warning = str(exc) or f"overlay '{overlay.key}' skipped during rendering."
            if overlay_warning is not None:
                warnings.append(f"{job.spec_title}: {overlay_warning}")

    if job.annotations and job.left_layers:
        annotation_ref_col = _require_column(render_trace_df, job.left_layers[0].variable)
        for annotation_warning in _render_composed_annotations(
            ax_left,
            render_trace_df,
            render_qc_lookup,
            x_col,
            annotation_ref_col,
            job.annotations,
        ):
            warnings.append(f"{job.spec_title}: {annotation_warning}")

    return ax_right, warnings


def _render_composed_trace_figure(
    render_trace_df: pd.DataFrame,
    render_qc_lookup: dict[str, pd.Series],
    job: ResolvedComposedGraphJob,
    style: PlotStyleConfig,
    output_dir: Path,
    figure_config: FigureConfig,
    allocated_stems: dict[str, int],
    group_order: list[str] | None,
    warnings: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Render one grouped composed figure and save it."""
    fig, ax_left = plt.subplots(1, 1, figsize=figure_config.resolve_size())
    warnings = list(warnings or [])
    ax_right, render_warnings = _render_composed_trace_axis(
        ax_left,
        render_trace_df,
        render_qc_lookup,
        job,
        style,
        group_order,
    )
    warnings.extend(render_warnings)

    ax_left.set_xlabel(axis_label(job.x_label))
    ax_left.set_ylabel(" / ".join(axis_label(layer.variable) for layer in job.left_layers))
    ax_left.set_title(f"{job.spec_title} [{job.x_label}]")
    ax_left.grid(True, linestyle="--", alpha=0.25)
    _apply_axis_legend(
        ax_left, group_order=group_order, extra_axes=[ax_right] if ax_right is not None else None
    )

    fig.tight_layout()
    path = _allocate_plot_path(
        output_dir,
        job.spec_title,
        job.x_label,
        allocated_stems,
        export_stem_suffix=job.export_stem_suffix,
    )
    fig.savefig(path, dpi=figure_config.dpi, bbox_inches="tight")
    plt.close(fig)
    return [str(path)], warnings


def _render_selected_sample_overlay_axis(
    ax_left: Any,
    prepared_samples: list[tuple[str, pd.DataFrame, pd.Series | None]],
    job: ResolvedComposedGraphJob,
    style: PlotStyleConfig,
) -> tuple[Any | None, list[str]]:
    """Render explicitly selected samples together on one shared axis."""
    warnings: list[str] = []
    sample_names = [sample_name for sample_name, _, _ in prepared_samples]
    style.ensure_group_colors(sample_names)

    left_styles = ["-", "--"]
    for layer_index, layer in enumerate(job.left_layers):
        linestyle = left_styles[layer_index % len(left_styles)]
        for sample_name, sample_frame, _ in prepared_samples:
            x_col = _require_column(sample_frame, job.x_label)
            y_col = _require_column(sample_frame, layer.variable)
            ordered = sample_frame.sort_values(x_col)
            label = (
                sample_name
                if len(job.left_layers) == 1 and job.right_layer is None
                else f"{sample_name} · {layer.variable}"
            )
            ax_left.plot(
                ordered[x_col],
                ordered[y_col],
                color=style.get_color(sample_name),
                linestyle=linestyle,
                linewidth=style.mean_linewidth,
                alpha=0.95,
                label=label,
            )
    _style_trace_axis_side(ax_left, LEFT_AXIS_ACCENT, recolor_artists=False)

    ax_right = None
    if job.right_layer is not None:
        ax_right = ax_left.twinx()
        for sample_name, sample_frame, _ in prepared_samples:
            x_col = _require_column(sample_frame, job.x_label)
            y_col = _require_column(sample_frame, job.right_layer.variable)
            ordered = sample_frame.sort_values(x_col)
            ax_right.plot(
                ordered[x_col],
                ordered[y_col],
                color=style.get_color(sample_name),
                linestyle=":",
                linewidth=style.mean_linewidth,
                alpha=0.95,
                label=f"{sample_name} · {job.right_layer.variable}",
            )
        ax_right.set_ylabel(axis_label(job.right_layer.variable))
        ax_right.grid(False)
        _style_trace_axis_side(ax_right, RIGHT_AXIS_ACCENT, recolor_artists=True)

    combined_trace = pd.concat([frame for _, frame, _ in prepared_samples], ignore_index=True)
    combined_lookup = {
        sample_name: qc_row for sample_name, _, qc_row in prepared_samples if qc_row is not None
    }
    x_col = _require_column(combined_trace, job.x_label)

    if job.overlays and job.left_layers:
        overlay_ref_col = _require_column(combined_trace, job.left_layers[0].variable)
        for overlay in job.overlays:
            overlay_artist_state = _capture_axis_artist_state(ax_left)
            try:
                overlay_warning = _render_composed_overlay(
                    ax_left,
                    combined_trace,
                    combined_lookup,
                    x_col,
                    overlay_ref_col,
                    overlay,
                )
            except Exception as exc:
                _restore_axis_artist_state(ax_left, overlay_artist_state)
                overlay_warning = str(exc) or f"overlay '{overlay.key}' skipped during rendering."
            if overlay_warning is not None:
                warnings.append(f"{job.spec_title}: {overlay_warning}")

    if job.annotations and job.left_layers:
        annotation_ref_col = _require_column(combined_trace, job.left_layers[0].variable)
        for annotation_warning in _render_composed_annotations(
            ax_left,
            combined_trace,
            combined_lookup,
            x_col,
            annotation_ref_col,
            job.annotations,
        ):
            warnings.append(f"{job.spec_title}: {annotation_warning}")

    return ax_right, warnings


def _plot_selected_sample_trace_job(
    trace_df: pd.DataFrame,
    qc_lookup: dict[str, pd.Series],
    job: ResolvedComposedGraphJob,
    style: PlotStyleConfig,
    output_dir: Path,
    figure_config: FigureConfig,
    allocated_stems: dict[str, int],
    group_order: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Render one composed trace job for explicitly selected samples."""
    x_col = _require_column(trace_df, job.x_label)
    prepared_samples, warnings = _prepare_selected_sample_data(trace_df, qc_lookup, job, x_col)
    if not prepared_samples:
        return [], warnings

    if job.display_mode == "overlay":
        fig, ax_left = plt.subplots(1, 1, figsize=figure_config.resolve_size())
        ax_right, render_warnings = _render_selected_sample_overlay_axis(
            ax_left,
            prepared_samples,
            job,
            style,
        )
        warnings.extend(render_warnings)
        ax_left.set_xlabel(axis_label(job.x_label))
        ax_left.set_ylabel(" / ".join(axis_label(layer.variable) for layer in job.left_layers))
        ax_left.set_title(f"{job.spec_title} [{job.x_label}]")
        ax_left.grid(True, linestyle="--", alpha=0.25)
        _apply_axis_legend(
            ax_left, group_order=None, extra_axes=[ax_right] if ax_right is not None else None
        )
        fig.tight_layout()
        path = _allocate_plot_path(output_dir, job.spec_title, job.x_label, allocated_stems)
        fig.savefig(path, dpi=figure_config.dpi, bbox_inches="tight")
        plt.close(fig)
        return [str(path)], warnings

    if job.display_mode == "individual":
        saved_paths: list[str] = []
        for sample_name, sample_frame, sample_qc_row in prepared_samples:
            sample_lookup = {sample_name: sample_qc_row} if sample_qc_row is not None else {}
            sample_job = ResolvedComposedGraphJob(
                spec_title=job.spec_title,
                x_label=job.x_label,
                left_layers=job.left_layers,
                right_layer=job.right_layer,
                overlays=job.overlays,
                band_mode=job.band_mode,
                segment_key=job.segment_key,
                annotations=job.annotations,
                data_scope=job.data_scope,
                selected_samples=job.selected_samples,
                display_mode=job.display_mode,
                rebase_x=job.rebase_x,
                export_stem_suffix=sample_name,
            )
            paths, render_warnings = _render_composed_trace_figure(
                render_trace_df=sample_frame,
                render_qc_lookup=sample_lookup,
                job=sample_job,
                style=style,
                output_dir=output_dir,
                figure_config=figure_config,
                allocated_stems=allocated_stems,
                group_order=group_order,
            )
            saved_paths.extend(paths)
            warnings.extend(render_warnings)
        return saved_paths, warnings

    fig, axes = plt.subplots(
        len(prepared_samples),
        1,
        figsize=figure_config.resolve_size(default=(10.0, max(4.0, 3.0 * len(prepared_samples)))),
        sharex=True,
    )
    if len(prepared_samples) == 1:
        axes = [axes]

    legend_axes: list[Any] = []
    for ax, (sample_name, sample_frame, sample_qc_row) in zip(
        axes, prepared_samples, strict=True
    ):
        sample_lookup = {sample_name: sample_qc_row} if sample_qc_row is not None else {}
        ax_right, render_warnings = _render_composed_trace_axis(
            ax,
            sample_frame,
            sample_lookup,
            job,
            style,
            group_order=None,
        )
        warnings.extend(render_warnings)
        legend_axes.append(ax)
        if ax_right is not None:
            legend_axes.append(ax_right)
        ax.set_ylabel(" / ".join(axis_label(layer.variable) for layer in job.left_layers))
        ax.set_title(sample_name)
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[-1].set_xlabel(axis_label(job.x_label))
    axes[0].figure.suptitle(f"{job.spec_title} [{job.x_label}]")
    if legend_axes:
        _apply_axis_legend(axes[0], group_order=group_order, extra_axes=legend_axes[1:])
    fig.tight_layout()
    path = _allocate_plot_path(output_dir, job.spec_title, job.x_label, allocated_stems)
    fig.savefig(path, dpi=figure_config.dpi, bbox_inches="tight")
    plt.close(fig)
    return [str(path)], warnings


def _plot_metric_job(
    metrics_df: pd.DataFrame,
    job: ResolvedGraphJob,
    style: PlotStyleConfig,
    output_dir: Path,
    figure_config: FigureConfig,
    allocated_stems: dict[str, int],
    group_order: list[str] | None = None,
    stats_by_metric: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Render one metric-plot job and return saved file paths."""
    mode = _resolve_job_mode(job.mode, job.y_labels)
    figsize = figure_config.resolve_size()
    x_meta = registry_entry(job.x_label)

    if len(job.y_labels) > 1:
        mode = "panel"

    if x_meta.column == "Filename":
        mode = "panel"
    elif x_meta.column != "Group" and x_meta.scale == "numeric":
        mode = _resolve_job_mode(job.mode, job.y_labels)

    if mode == "overlay" and x_meta.column == "Group":
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for y_label in job.y_labels:
            _plot_metric_group_axis(
                ax, metrics_df, y_label, style, job.metric_view, group_order, stats_by_metric
            )
        ax.set_xlabel("Group")
        ax.set_title(f"{job.spec_title} [{job.x_label}]")
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = _ordered_legend(handles, labels, group_order=group_order)
        if handles:
            ax.legend(handles, labels, frameon=False)
    else:
        fig, axes = plt.subplots(len(job.y_labels), 1, figsize=figsize, sharex=False)
        if len(job.y_labels) == 1:
            axes = [axes]
        for ax, y_label in zip(axes, job.y_labels, strict=True):
            if x_meta.column == "Group":
                _plot_metric_group_axis(
                    ax, metrics_df, y_label, style, job.metric_view, group_order, stats_by_metric
                )
                ax.set_xlabel("Group")
            elif x_meta.column == "Filename":
                _plot_metric_filename_axis(ax, metrics_df, y_label, style)
                ax.set_xlabel("Filename")
            else:
                _plot_metric_scatter_axis(ax, metrics_df, job.x_label, y_label, style)
        axes[0].set_title(f"{job.spec_title} [{job.x_label}]")
        handles, labels = axes[0].get_legend_handles_labels()
        handles, labels = _ordered_legend(handles, labels, group_order=group_order)
        if handles:
            axes[0].legend(handles, labels, frameon=False)

    fig.tight_layout()
    path = _allocate_plot_path(output_dir, job.spec_title, job.x_label, allocated_stems)
    fig.savefig(path, dpi=figure_config.dpi, bbox_inches="tight")
    plt.close(fig)
    return [str(path)]


def plot_trace_stack(
    trace_df: pd.DataFrame,
    spec: dict[str, Any] | None,
    style: PlotStyleConfig,
    output_path: str | Path,
    figure_config: FigureConfig | None = None,
) -> str:
    """Generate the default two-panel vertical stack plot."""
    figure_config = figure_config or FigureConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_label = "Time (s)"
    y_labels = ["Force (N)", "Deformation (mm)"]
    curve_mode = str((spec or {}).get("curve_mode", "individual"))
    band_mode = str((spec or {}).get("band_mode", "sd"))
    group_order = [
        str(group).strip() for group in (spec or {}).get("group_order", []) if str(group).strip()
    ]
    x_col = _require_column(trace_df, x_label)

    fig, axes = plt.subplots(
        2, 1, figsize=figure_config.resolve_size(default=(10.0, 8.0)), sharex=True
    )
    for idx, y_label in enumerate(y_labels):
        y_col = _require_column(trace_df, y_label)
        ax = axes[idx]
        _apply_curve_mode(
            ax, trace_df, x_col, y_col, style, curve_mode, band_mode, group_order=group_order
        )
        ax.set_ylabel(axis_label(y_label))
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[-1].set_xlabel(axis_label(x_label))
    axes[0].set_title("TPA Default Trace Stack")
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = _ordered_legend(handles, labels, group_order=group_order)
    if handles:
        axes[0].legend(handles, labels, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=figure_config.dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_custom_graphs(
    trace_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    graph_specs: list[GraphSpec | CustomGraphSpec | dict[str, Any]],
    style: PlotStyleConfig,
    output_dir: str | Path,
    figure_config: FigureConfig | None = None,
    group_order: list[str] | None = None,
    stats_by_metric: dict[str, dict[str, Any]] | None = None,
    qc_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Render all custom graph specifications and return saved paths and warnings."""
    figure_config = figure_config or FigureConfig()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    warnings: list[str] = []
    allocated_stems: dict[str, int] = {}
    qc_lookup = _build_overlay_qc_lookup(qc_df)

    for raw_spec in graph_specs:
        try:
            spec = (
                normalize_composed_graph_spec(raw_spec)
                if _is_composed_graph_payload(raw_spec)
                else normalize_graph_spec(raw_spec)
            )
        except PlotSpecError as exc:
            warnings.append(f"{_graph_spec_title(raw_spec)}: {exc}")
            continue
        if not spec.enabled:
            continue
        if isinstance(spec, CustomGraphSpec):
            try:
                jobs = expand_composed_graph_spec(spec)
            except PlotSpecError as exc:
                warnings.append(f"{spec.title}: {exc}")
                continue

            for job in jobs:
                try:
                    paths, job_warnings = _plot_composed_trace_job(
                        trace_df=trace_df,
                        qc_lookup=qc_lookup,
                        job=job,
                        style=style,
                        output_dir=output_root,
                        figure_config=figure_config,
                        allocated_stems=allocated_stems,
                        group_order=group_order,
                    )
                    saved_paths.extend(paths)
                    warnings.extend(job_warnings)
                except Exception as exc:
                    warnings.append(f"{job.spec_title} [{job.x_label}]: {exc}")
            continue

        try:
            jobs = expand_graph_spec_jobs(spec)
        except PlotSpecError as exc:
            warnings.append(f"{spec.title}: {exc}")
            continue

        for job in jobs:
            try:
                if job.plot_type == "trace":
                    saved_paths.extend(
                        _plot_trace_job(
                            trace_df=trace_df,
                            job=job,
                            style=style,
                            output_dir=output_root,
                            figure_config=figure_config,
                            allocated_stems=allocated_stems,
                            group_order=group_order,
                        )
                    )
                else:
                    saved_paths.extend(
                        _plot_metric_job(
                            metrics_df=metrics_df,
                            job=job,
                            style=style,
                            output_dir=output_root,
                            figure_config=figure_config,
                            allocated_stems=allocated_stems,
                            group_order=group_order,
                            stats_by_metric=stats_by_metric,
                        )
                    )
            except Exception as exc:
                warnings.append(f"{job.spec_title} [{job.x_label}]: {exc}")
    return {"paths": saved_paths, "warnings": warnings}


def export_qc_report(
    trace_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    output_dir: str | Path,
    figure_config: FigureConfig | None = None,
) -> dict[str, Any]:
    """Export QC tables and per-file annotated QC figures."""
    figure_config = figure_config or FigureConfig()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    saved_paths: list[str] = []

    guide_source = Path(__file__).resolve().parents[3] / "QC_REPORT_INTERPRETATION.md"
    guide_target = output_root / "QC_REPORT_INTERPRETATION.md"
    try:
        if guide_source.exists():
            guide_target.write_text(guide_source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            guide_target.write_text(
                "QC guide missing in project root. "
                "See repository file QC_REPORT_INTERPRETATION.md.",
                encoding="utf-8",
            )
            warnings.append("QC interpretation guide source file was not found in project root.")
    except Exception as exc:
        warnings.append(f"Could not write QC interpretation guide: {exc}")

    if trace_df.empty:
        return {"paths": saved_paths, "warnings": ["QC report skipped: trace dataframe is empty."]}
    if qc_df.empty:
        return {
            "paths": saved_paths,
            "warnings": ["QC report skipped: QC summary dataframe is empty."],
        }

    qc_sorted = qc_df.copy()
    for key in ["Group", "Filename"]:
        if key not in qc_sorted.columns:
            qc_sorted[key] = ""
    qc_sorted = qc_sorted.sort_values(["Group", "Filename"], kind="stable").reset_index(drop=True)
    qc_sorted.to_csv(output_root / "qc_summary.csv", index=False)

    control_cols = [
        "Filename",
        "Group",
        "Baseline Offset (N)",
        "Trigger Force (N)",
        "Peak Prominence (N)",
        "Peak Distance (pts)",
        "Modulus Strain Min (%)",
        "Modulus Strain Max (%)",
    ]
    marker_cols = [
        "Filename",
        "Group",
        "Peak1 Index",
        "Peak2 Index",
        "Bite1 Start Index",
        "Bite1 End Index",
        "Bite2 Start Index",
        "Bite2 End Index",
        "Peak1 Time (s)",
        "Peak2 Time (s)",
        "A1 Area (N*s)",
        "A2 Area (N*s)",
        "A1 Up Area (N*s)",
        "A1 Down Area (N*s)",
        "Adhesiveness Area (N*s)",
    ]
    qc_sorted[[column for column in control_cols if column in qc_sorted.columns]].to_csv(
        output_root / "qc_control_parameters.csv",
        index=False,
    )
    qc_sorted[[column for column in marker_cols if column in qc_sorted.columns]].to_csv(
        output_root / "qc_markers_and_areas.csv",
        index=False,
    )

    files_dir = output_root / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    def _safe_int(row: pd.Series, key: str, limit: int) -> int:
        """Clamp QC indices into the valid plotting range."""
        raw = row.get(key, 0)
        try:
            index = int(float(raw))
        except (TypeError, ValueError):
            index = 0
        return int(np.clip(index, 0, max(limit - 1, 0)))

    def _safe_float(row: pd.Series, key: str, default: float = float("nan")) -> float:
        """Safely convert a QC value to float."""
        raw = row.get(key, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    for _, row in qc_sorted.iterrows():
        filename = str(row.get("Filename", "")).strip()
        if not filename:
            warnings.append("Skipped one QC row with missing filename.")
            continue

        frame = trace_df[trace_df["File"].astype(str) == filename].copy()
        if frame.empty:
            warnings.append(f"{filename}: missing trace rows; QC figure skipped.")
            continue

        frame = frame.sort_values("Time (s)").reset_index(drop=True)
        n_points = len(frame)
        if n_points < 3:
            warnings.append(f"{filename}: too few points ({n_points}) for QC figure.")
            continue

        time_vals = frame["Time (s)"].to_numpy(dtype=float)
        force_vals = frame["Force Corrected (N)"].to_numpy(dtype=float)
        strain_vals = frame["True Strain (%)"].to_numpy(dtype=float)
        stress_vals = frame["True Stress (kPa)"].to_numpy(dtype=float)

        p1 = _safe_int(row, "Peak1 Index", n_points)
        p2 = _safe_int(row, "Peak2 Index", n_points)
        b1s = _safe_int(row, "Bite1 Start Index", n_points)
        b1e = _safe_int(row, "Bite1 End Index", n_points)
        b2s = _safe_int(row, "Bite2 Start Index", n_points)
        b2e = _safe_int(row, "Bite2 End Index", n_points)

        trigger = _safe_float(row, "Trigger Force (N)", 0.0)
        mmin = _safe_float(row, "Modulus Strain Min (%)")
        mmax = _safe_float(row, "Modulus Strain Max (%)")

        fig, (ax_force, ax_stress) = plt.subplots(
            2,
            1,
            figsize=figure_config.resolve_size(default=(11.0, 8.5)),
            sharex=False,
        )
        ax_force.plot(
            time_vals, force_vals, color="#1F2937", linewidth=1.6, label="Force corrected"
        )
        ax_force.axhline(0.0, color="#64748B", linewidth=0.9, linestyle="--", alpha=0.7)
        ax_force.axhline(
            trigger, color="#0EA5E9", linewidth=0.9, linestyle=":", alpha=0.9, label="Trigger"
        )

        def _fill_segment(
            start_idx: int,
            end_idx: int,
            color: str,
            label: str,
            positive_only: bool | None,
            *,
            time_points: np.ndarray,
            force_points: np.ndarray,
            force_axis: Any,
        ) -> None:
            """Fill an area between markers on the force trace."""
            left, right = sorted((start_idx, end_idx))
            if right - left < 1:
                return
            x_seg = time_points[left : right + 1]
            y_seg = force_points[left : right + 1]
            if len(x_seg) < 2:
                return
            if positive_only is True:
                where = y_seg > 0
            elif positive_only is False:
                where = y_seg < 0
            else:
                where = np.ones_like(y_seg, dtype=bool)
            if int(where.sum()) < 2:
                return
            force_axis.fill_between(
                x_seg,
                0.0,
                y_seg,
                where=where,
                interpolate=True,
                color=color,
                alpha=0.25,
                label=label,
            )

        _fill_segment(
            b1s,
            b1e,
            "#93C5FD",
            "A1",
            True,
            time_points=time_vals,
            force_points=force_vals,
            force_axis=ax_force,
        )
        _fill_segment(
            b2s,
            b2e,
            "#86EFAC",
            "A2",
            True,
            time_points=time_vals,
            force_points=force_vals,
            force_axis=ax_force,
        )
        _fill_segment(
            b1e,
            b2s,
            "#FCA5A5",
            "Adhesiveness",
            False,
            time_points=time_vals,
            force_points=force_vals,
            force_axis=ax_force,
        )

        marker_specs = [
            (b1s, "B1 start", "#0EA5E9"),
            (p1, "Peak1", "#1D4ED8"),
            (b1e, "B1 end", "#0EA5E9"),
            (b2s, "B2 start", "#16A34A"),
            (p2, "Peak2", "#15803D"),
            (b2e, "B2 end", "#16A34A"),
        ]
        for index, label, color in marker_specs:
            x_val = time_vals[index]
            y_val = force_vals[index]
            ax_force.scatter([x_val], [y_val], s=20, color=color, zorder=4)
            ax_force.annotate(
                label,
                (x_val, y_val),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7.5,
                color=color,
            )

        text_lines = [
            f"A1={_safe_float(row, 'A1 Area (N*s)', 0.0):.3f} N*s",
            f"A2={_safe_float(row, 'A2 Area (N*s)', 0.0):.3f} N*s",
            f"Adhesiveness={_safe_float(row, 'Adhesiveness Area (N*s)', 0.0):.3f} N*s",
            f"Hardness={_safe_float(row, 'Hardness (N)', 0.0):.3f} N",
            f"Cohesiveness={_safe_float(row, 'Cohesiveness', 0.0):.3f}",
        ]
        ax_force.text(
            0.995,
            0.97,
            "\n".join(text_lines),
            transform=ax_force.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "#FFFFFF", "edgecolor": "#CBD5E1", "boxstyle": "round,pad=0.3"},
        )
        ax_force.set_title(f"QC Force Map: {filename}")
        ax_force.set_xlabel("Time (s)")
        ax_force.set_ylabel("Force Corrected (N)")
        ax_force.grid(True, linestyle="--", alpha=0.25)
        h_force, l_force = ax_force.get_legend_handles_labels()
        if h_force:
            ax_force.legend(h_force, l_force, frameon=False, fontsize=8)

        finite_mask = np.isfinite(strain_vals) & np.isfinite(stress_vals)
        if int(finite_mask.sum()) >= 3:
            strain_plot = strain_vals[finite_mask]
            stress_plot = stress_vals[finite_mask]
            ax_stress.plot(
                strain_plot, stress_plot, color="#B45309", linewidth=1.6, label="Stress-strain"
            )
            if np.isfinite(mmin) and np.isfinite(mmax) and mmax > mmin:
                ax_stress.axvspan(mmin, mmax, color="#FDE68A", alpha=0.35, label="Modulus window")

            comp_left, comp_right = sorted((b1s, p1))
            comp_mask = np.zeros(n_points, dtype=bool)
            comp_mask[comp_left : comp_right + 1] = True
            fit_mask = (
                comp_mask
                & np.isfinite(strain_vals)
                & np.isfinite(stress_vals)
                & (strain_vals >= mmin)
                & (strain_vals <= mmax)
            )
            if int(fit_mask.sum()) >= 2:
                x_fit = strain_vals[fit_mask] / 100.0
                y_fit = stress_vals[fit_mask]
                slope, intercept = np.polyfit(x_fit, y_fit, 1)
                x_line = np.linspace(float(x_fit.min()), float(x_fit.max()), 48)
                ax_stress.plot(
                    x_line * 100.0,
                    slope * x_line + intercept,
                    color="#7C2D12",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Fit slope={slope:.1f} kPa",
                )
        else:
            ax_stress.text(
                0.5,
                0.5,
                "Stress/strain unavailable",
                transform=ax_stress.transAxes,
                ha="center",
                va="center",
            )

        ax_stress.set_title("Modulus Context (True Stress vs True Strain)")
        ax_stress.set_xlabel("True Strain (%)")
        ax_stress.set_ylabel("True Stress (kPa)")
        ax_stress.grid(True, linestyle="--", alpha=0.25)
        h_stress, l_stress = ax_stress.get_legend_handles_labels()
        if h_stress:
            ax_stress.legend(h_stress, l_stress, frameon=False, fontsize=8)

        fig.tight_layout()
        save_path = files_dir / f"{_slugify(filename)}_qc.png"
        fig.savefig(save_path, dpi=figure_config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(save_path))

    return {"paths": saved_paths, "warnings": warnings}


def plot_grouped_metrics(
    stats_by_metric: dict[str, dict[str, Any]],
    style: PlotStyleConfig,
    output_path: str | Path,
    figure_config: FigureConfig | None = None,
) -> str:
    """Plot grouped summary charts for all available computed metrics."""
    figure_config = figure_config or FigureConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    present_metrics = [metric for metric in COMPUTED_METRICS if metric in stats_by_metric]
    if not present_metrics:
        raise PlotSpecError("No supported metrics found for grouped bar plot.")

    ordered_groups_union: list[str] = []
    for metric in present_metrics:
        summary = stats_by_metric[metric]["summary_df"]
        group_col = stats_by_metric[metric]["test_info"]["group_col"]
        for group in summary[group_col].astype(str).tolist():
            if group not in ordered_groups_union:
                ordered_groups_union.append(group)
    style.ensure_group_colors(ordered_groups_union)

    cols = 2
    rows = ceil(len(present_metrics) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=figure_config.resolve_size(default=(12.0, max(8.0, rows * 3.5)))
    )
    flat_axes = np.atleast_1d(axes).flatten()

    for ax in flat_axes[len(present_metrics) :]:
        ax.axis("off")

    for index, metric in enumerate(present_metrics):
        ax = flat_axes[index]
        summary = stats_by_metric[metric]["summary_df"].copy()
        group_col = stats_by_metric[metric]["test_info"]["group_col"]
        groups = summary[group_col].astype(str).tolist()
        means = summary["Mean"].to_numpy(dtype=float)
        sds = summary["SD"].to_numpy(dtype=float)
        letters = summary["Significance"].astype(str).tolist()
        bars = ax.bar(
            groups,
            means,
            yerr=sds,
            capsize=4,
            color=[style.get_color(group) for group in groups],
            alpha=0.9,
        )

        ymax = float(np.nanmax(means + sds)) if len(means) else 1.0
        offset = max(ymax * 0.04, 0.02)
        for bar, letter, mean, sd in zip(bars, letters, means, sds, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + sd + offset,
                letter,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
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
    """Plot per-group overlay traces as individual, mean-band, or both."""
    figure_config = figure_config or FigureConfig()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    mode = str(overlay_spec.get("mode", "mean_band"))
    band_mode = str(overlay_spec.get("band_mode", "sd"))
    x_label = str(overlay_spec.get("x_col", "Aligned Time (s)"))
    y_labels = list(overlay_spec.get("y_cols", ["Force (N)", "Deformation (mm)"]))
    group_order = [
        str(group).strip() for group in overlay_spec.get("group_order", []) if str(group).strip()
    ]

    x_col = _require_column(trace_df, x_label)
    y_cols = [_require_column(trace_df, label) for label in y_labels]
    ordered_groups = _categorical_order(
        trace_df["Group"].dropna().astype(str).tolist(), preferred_order=group_order
    )
    unique_groups: list[str] = []
    for group in ordered_groups:
        if group not in unique_groups:
            unique_groups.append(group)

    saved_paths: list[str] = []
    for group_name in unique_groups:
        group_frame = trace_df[trace_df["Group"].astype(str) == group_name]
        if group_frame.empty:
            continue
        fig, axes = plt.subplots(
            len(y_cols), 1, figsize=figure_config.resolve_size(default=(10.0, 7.5)), sharex=True
        )
        if len(y_cols) == 1:
            axes = [axes]
        for ax, y_col, y_label in zip(axes, y_cols, y_labels, strict=True):
            _apply_curve_mode(
                ax,
                group_frame,
                x_col,
                y_col,
                style,
                curve_mode=mode,
                band_mode=band_mode,
                group_order=[group_name],
            )
            ax.set_ylabel(axis_label(y_label))
            ax.grid(True, linestyle="--", alpha=0.25)
        axes[0].set_title(f"Overlay: {group_name}")
        axes[-1].set_xlabel(axis_label(x_label))
        save_path = output_root / f"overlay_{_slugify(group_name)}.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=figure_config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(save_path))
    return saved_paths


def _serialize_graph_spec(spec: Any) -> dict[str, Any]:
    """Convert a graph-spec payload into plain JSON-compatible data."""
    if is_dataclass(spec):
        return asdict(spec)
    if isinstance(spec, dict):
        return {
            key: _serialize_graph_spec(value)
            if is_dataclass(value) or isinstance(value, dict)
            else value
            for key, value in spec.items()
        }
    raise TypeError(f"Unsupported graph spec payload: {type(spec)!r}")


def serialize_graph_specs(
    graph_specs: list[GraphSpec | CustomGraphSpec | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize graph specs for session persistence."""
    return [_serialize_graph_spec(spec) for spec in graph_specs]
