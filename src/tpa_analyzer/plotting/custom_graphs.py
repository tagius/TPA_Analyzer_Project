"""Compatibility metadata for custom graph choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


PlotItemType = Literal["trace", "segment", "annotation", "window", "inset_bar"]


@dataclass(frozen=True)
class CompatiblePlotItem:
    """Declarative compatibility metadata for a selectable plot item."""

    key: str
    label: str
    item_type: PlotItemType
    allowed_x_domains: tuple[str, ...]
    requires_analysis: bool = False
    requires_left_variables: tuple[str, ...] = ()
    blocks_with: tuple[str, ...] = ()


TRACE_COMPATIBILITY: Final[dict[str, CompatiblePlotItem]] = {
    "Force (N)": CompatiblePlotItem(
        key="Force (N)",
        label="Force (N)",
        item_type="trace",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
    ),
    "Force Corrected (N)": CompatiblePlotItem(
        key="Force Corrected (N)",
        label="Force Corrected (N)",
        item_type="trace",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        requires_analysis=True,
    ),
    "Deformation (mm)": CompatiblePlotItem(
        key="Deformation (mm)",
        label="Deformation (mm)",
        item_type="trace",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        requires_analysis=True,
    ),
    "True Stress (kPa)": CompatiblePlotItem(
        key="True Stress (kPa)",
        label="True Stress (kPa)",
        item_type="trace",
        allowed_x_domains=("True Strain (%)",),
        requires_analysis=True,
    ),
}


OVERLAY_COMPATIBILITY: Final[dict[str, CompatiblePlotItem]] = {
    "b1_start_to_peak1": CompatiblePlotItem(
        key="b1_start_to_peak1",
        label="B1 start -> Peak1",
        item_type="segment",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        requires_analysis=True,
        requires_left_variables=("Force Corrected (N)",),
    ),
    "peak1_to_b1_end": CompatiblePlotItem(
        key="peak1_to_b1_end",
        label="Peak1 -> B1 end",
        item_type="segment",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        requires_analysis=True,
        requires_left_variables=("Force Corrected (N)",),
    ),
    "b1_end_to_b2_start": CompatiblePlotItem(
        key="b1_end_to_b2_start",
        label="B1 end -> B2 start",
        item_type="segment",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        requires_analysis=True,
        requires_left_variables=("Force Corrected (N)",),
    ),
    "b2_start_to_peak2": CompatiblePlotItem(
        key="b2_start_to_peak2",
        label="B2 start -> Peak2",
        item_type="segment",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        requires_analysis=True,
        requires_left_variables=("Force Corrected (N)",),
    ),
    "hardness_peak1": CompatiblePlotItem(
        key="hardness_peak1",
        label="Hardness at Peak1",
        item_type="annotation",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        requires_analysis=True,
        requires_left_variables=("Force Corrected (N)",),
    ),
    "adhesiveness": CompatiblePlotItem(
        key="adhesiveness",
        label="Adhesiveness",
        item_type="annotation",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        requires_analysis=True,
        requires_left_variables=("Force Corrected (N)",),
    ),
    "modulus_window": CompatiblePlotItem(
        key="modulus_window",
        label="Modulus window",
        item_type="window",
        allowed_x_domains=("True Strain (%)",),
        requires_analysis=True,
        requires_left_variables=("True Stress (kPa)",),
    ),
}


def _eligible_items(
    items: dict[str, CompatiblePlotItem],
    x_domain: str,
    left_variables: list[str],
    analysis_ready: bool,
) -> list[str]:
    """Return keys whose compatibility metadata matches the current graph state."""
    selected_left_variables = {str(variable).strip() for variable in left_variables if str(variable).strip()}
    eligible: list[str] = []
    for key, item in items.items():
        if x_domain not in item.allowed_x_domains:
            continue
        if item.requires_analysis and not analysis_ready:
            continue
        if item.requires_left_variables and not set(item.requires_left_variables).issubset(selected_left_variables):
            continue
        eligible.append(key)
    return eligible


def eligible_left_axis_variables(x_domain: str, analysis_ready: bool) -> list[str]:
    """Return left-axis trace variables eligible for the selected X domain."""
    eligible: list[str] = []
    for key, item in TRACE_COMPATIBILITY.items():
        if x_domain not in item.allowed_x_domains:
            continue
        if item.requires_analysis and not analysis_ready:
            continue
        eligible.append(key)
    return eligible


def eligible_right_axis_variables(x_domain: str, left_variables: list[str], analysis_ready: bool) -> list[str]:
    """Return right-axis trace variables compatible with the current left-axis selection."""
    left_axis_candidates = eligible_left_axis_variables(x_domain=x_domain, analysis_ready=analysis_ready)
    selected_left_variables = {str(variable).strip() for variable in left_variables if str(variable).strip()}
    return [variable for variable in left_axis_candidates if variable not in selected_left_variables]


def eligible_overlay_keys(x_domain: str, left_variables: list[str], analysis_ready: bool) -> list[str]:
    """Return overlay keys that match the current graph composition."""
    return _eligible_items(
        OVERLAY_COMPATIBILITY,
        x_domain=x_domain,
        left_variables=left_variables,
        analysis_ready=analysis_ready,
    )
