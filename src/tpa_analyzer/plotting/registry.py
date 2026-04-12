"""Variable registry helpers for custom plotting."""

from __future__ import annotations

from typing import Final

from tpa_analyzer.core.constants import COMPUTED_METRICS
from tpa_analyzer.core.models import PlotVariable


VARIABLE_REGISTRY: Final[dict[str, PlotVariable]] = {
    "Time (s)": PlotVariable(
        label="Time (s)",
        column="Time (s)",
        axis="Time (s)",
        unit="s",
        kind="x",
        source="trace",
    ),
    "Aligned Time (s)": PlotVariable(
        label="Aligned Time (s)",
        column="Aligned Time (s)",
        axis="Aligned Time (s)",
        unit="s",
        kind="x",
        source="trace",
    ),
    "True Strain (%)": PlotVariable(
        label="True Strain (%)",
        column="True Strain (%)",
        axis="True Strain (%)",
        unit="%",
        kind="x",
        source="trace",
    ),
    "Force (N)": PlotVariable(
        label="Force (N)",
        column="Force (N)",
        axis="Force (N)",
        unit="N",
        kind="y",
        source="trace",
    ),
    "Force Corrected (N)": PlotVariable(
        label="Force Corrected (N)",
        column="Force Corrected (N)",
        axis="Force Corrected (N)",
        unit="N",
        kind="y",
        source="trace",
    ),
    "Deformation (mm)": PlotVariable(
        label="Deformation (mm)",
        column="Deformation (mm)",
        axis="Deformation (mm)",
        unit="mm",
        kind="y",
        source="trace",
    ),
    "True Stress (kPa)": PlotVariable(
        label="True Stress (kPa)",
        column="True Stress (kPa)",
        axis="True Stress (kPa)",
        unit="kPa",
        kind="y",
        source="trace",
    ),
    "Group": PlotVariable(
        label="Group",
        column="Group",
        axis="Group",
        unit="",
        kind="x",
        source="metric",
        scale="categorical",
    ),
    "Filename": PlotVariable(
        label="Filename",
        column="Filename",
        axis="Filename",
        unit="",
        kind="x",
        source="metric",
        scale="categorical",
    ),
}

for metric_label in COMPUTED_METRICS:
    unit = ""
    if "(" in metric_label and ")" in metric_label:
        unit = metric_label.split("(")[-1].rstrip(")")
    VARIABLE_REGISTRY[metric_label] = PlotVariable(
        label=metric_label,
        column=metric_label,
        axis=metric_label,
        unit=unit,
        kind="y",
        source="metric",
    )


def axis_label(label: str) -> str:
    """Return the display label used for an axis."""
    return VARIABLE_REGISTRY.get(label, PlotVariable(label, label, label, "", "x", "trace")).axis


def registry_entry(label: str) -> PlotVariable:
    """Return the registered variable metadata for ``label``."""
    return VARIABLE_REGISTRY[label]


def options_for(kind: str, source: str | None = None) -> list[str]:
    """Return sorted variable labels for a given axis kind and optional source."""
    values = [
        label
        for label, meta in VARIABLE_REGISTRY.items()
        if (meta.kind == kind or (kind == "x" and meta.source == "metric" and meta.scale == "numeric"))
        and (source is None or meta.source == source)
    ]
    return sorted(values, key=str.lower)
