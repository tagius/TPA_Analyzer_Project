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


@dataclass(frozen=True)
class SemanticSegment:
    """Compatibility metadata for a semantic segment choice."""

    key: str
    label: str
    allowed_x_domains: tuple[str, ...]
    qc_columns: tuple[str, ...]


@dataclass(frozen=True)
class SegmentAnnotation:
    """Compatibility metadata for an annotation tied to one or more segments."""

    key: str
    label: str
    allowed_segments: tuple[str, ...]
    required_left_variables: tuple[str, ...]


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


SEMANTIC_SEGMENTS: Final[dict[str, SemanticSegment]] = {
    "b1_start_to_peak1": SemanticSegment(
        key="b1_start_to_peak1",
        label="B1 start -> Peak1",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        qc_columns=("Bite1 Start Index", "Peak1 Index"),
    ),
    "peak1_to_b1_end": SemanticSegment(
        key="peak1_to_b1_end",
        label="Peak1 -> B1 end",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        qc_columns=("Peak1 Index", "Bite1 End Index"),
    ),
    "b1_end_to_b2_start": SemanticSegment(
        key="b1_end_to_b2_start",
        label="B1 end -> B2 start",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        qc_columns=("Bite1 End Index", "Bite2 Start Index"),
    ),
    "b2_start_to_peak2": SemanticSegment(
        key="b2_start_to_peak2",
        label="B2 start -> Peak2",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        qc_columns=("Bite2 Start Index", "Peak2 Index"),
    ),
    "peak2_to_b2_end": SemanticSegment(
        key="peak2_to_b2_end",
        label="Peak2 -> B2 end",
        allowed_x_domains=("Time (s)", "Aligned Time (s)"),
        qc_columns=("Peak2 Index", "Bite2 End Index"),
    ),
    "modulus_window": SemanticSegment(
        key="modulus_window",
        label="Modulus window",
        allowed_x_domains=("True Strain (%)",),
        qc_columns=("Modulus Strain Min (%)", "Modulus Strain Max (%)"),
    ),
}


ANNOTATION_COMPATIBILITY: Final[dict[str, SegmentAnnotation]] = {
    "hardness_peak1": SegmentAnnotation(
        key="hardness_peak1",
        label="Hardness at Peak1",
        allowed_segments=("b1_start_to_peak1",),
        required_left_variables=("Force Corrected (N)",),
    ),
    "adhesiveness": SegmentAnnotation(
        key="adhesiveness",
        label="Adhesiveness",
        allowed_segments=("b1_end_to_b2_start",),
        required_left_variables=("Force Corrected (N)",),
    ),
    "modulus_window": SegmentAnnotation(
        key="modulus_window",
        label="Modulus window",
        allowed_segments=("modulus_window",),
        required_left_variables=("True Stress (kPa)",),
    ),
}


def _legacy_overlay_item(segment: SemanticSegment) -> CompatiblePlotItem:
    """Translate segment metadata to the legacy composed-graph compatibility shape."""
    item_type = "window" if segment.key == "modulus_window" else "segment"
    return CompatiblePlotItem(
        key=segment.key,
        label=segment.label,
        item_type=item_type,
        allowed_x_domains=segment.allowed_x_domains,
        requires_analysis=True,
        requires_left_variables=("True Stress (kPa)",) if segment.key == "modulus_window" else ("Force Corrected (N)",),
    )


def _legacy_annotation_item(annotation: SegmentAnnotation) -> CompatiblePlotItem:
    """Translate annotation metadata to the legacy composed-graph compatibility shape."""
    item_type = "window" if annotation.key == "modulus_window" else "annotation"
    return CompatiblePlotItem(
        key=annotation.key,
        label=annotation.label,
        item_type=item_type,
        allowed_x_domains=_legacy_annotation_x_domains(annotation),
        requires_analysis=True,
        requires_left_variables=annotation.required_left_variables,
    )


def _legacy_annotation_x_domains(annotation: SegmentAnnotation) -> tuple[str, ...]:
    """Return the legacy x-domain compatibility for one annotation."""
    if annotation.key == "modulus_window":
        return ("True Strain (%)",)
    return ("Time (s)", "Aligned Time (s)")


OVERLAY_COMPATIBILITY: Final[dict[str, CompatiblePlotItem]] = {
    "b1_start_to_peak1": _legacy_overlay_item(SEMANTIC_SEGMENTS["b1_start_to_peak1"]),
    "peak1_to_b1_end": _legacy_overlay_item(SEMANTIC_SEGMENTS["peak1_to_b1_end"]),
    "b1_end_to_b2_start": _legacy_overlay_item(SEMANTIC_SEGMENTS["b1_end_to_b2_start"]),
    "b2_start_to_peak2": _legacy_overlay_item(SEMANTIC_SEGMENTS["b2_start_to_peak2"]),
    "hardness_peak1": _legacy_annotation_item(ANNOTATION_COMPATIBILITY["hardness_peak1"]),
    "adhesiveness": _legacy_annotation_item(ANNOTATION_COMPATIBILITY["adhesiveness"]),
    "modulus_window": _legacy_annotation_item(ANNOTATION_COMPATIBILITY["modulus_window"]),
}


def semantic_segment_keys() -> list[str]:
    """Return semantic segment keys in registry order."""
    return list(SEMANTIC_SEGMENTS)


def eligible_segment_keys(x_domain: str, analysis_ready: bool) -> list[str]:
    """Return semantic segments that can be selected for the current graph state."""
    if not analysis_ready:
        return []
    return [key for key, segment in SEMANTIC_SEGMENTS.items() if x_domain in segment.allowed_x_domains]


def eligible_annotation_keys(segment_key: str, left_variables: list[str] | None = None) -> list[str]:
    """Return annotations compatible with one semantic segment."""
    if segment_key not in SEMANTIC_SEGMENTS:
        return []

    selected_left_variables = (
        {str(variable).strip() for variable in left_variables if str(variable).strip()}
        if left_variables is not None
        else None
    )

    eligible: list[str] = []
    for key, annotation in ANNOTATION_COMPATIBILITY.items():
        if segment_key not in annotation.allowed_segments:
            continue
        if selected_left_variables is not None and annotation.required_left_variables:
            if not set(annotation.required_left_variables).issubset(selected_left_variables):
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
