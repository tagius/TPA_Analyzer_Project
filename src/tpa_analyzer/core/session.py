"""Session persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tpa_analyzer.core.constants import SESSION_FILE_NAME, SESSION_SCHEMA_VERSION
from tpa_analyzer.core.errors import PlotSpecError, SessionError
from tpa_analyzer.core.models import (
    CustomGraphAnnotation,
    CustomGraphOverlay,
    CustomGraphSpec,
    GraphSpec,
)
from tpa_analyzer.plotting.engine import normalize_composed_graph_spec, normalize_graph_spec, serialize_graph_specs


def session_path(directory: Path) -> Path:
    """Return the session file path for a data directory."""
    return directory / SESSION_FILE_NAME


def _is_composed_graph_payload(spec: Any) -> bool:
    """Return ``True`` for payloads shaped like ``CustomGraphSpec`` data."""
    return isinstance(spec, dict) and any(key in spec for key in ("x_domain", "left_axis", "right_axis", "overlay"))


def _normalize_overlay_payload(raw_overlay: Any) -> CustomGraphOverlay:
    """Normalize one overlay payload into ``CustomGraphOverlay``."""
    if isinstance(raw_overlay, CustomGraphOverlay):
        return raw_overlay
    if not isinstance(raw_overlay, dict):
        raise PlotSpecError("Invalid overlay payload.")
    return CustomGraphOverlay(
        kind=str(raw_overlay.get("kind", "")).strip(),
        key=str(raw_overlay.get("key", "")).strip(),
    )


def _normalize_annotation_payload(raw_annotation: Any) -> CustomGraphAnnotation:
    """Normalize one annotation payload into ``CustomGraphAnnotation``."""
    if isinstance(raw_annotation, CustomGraphAnnotation):
        return raw_annotation
    if not isinstance(raw_annotation, dict):
        raise PlotSpecError("Invalid annotation payload.")
    return CustomGraphAnnotation(
        kind=str(raw_annotation.get("kind", "annotation")).strip() or "annotation",
        key=str(raw_annotation.get("key", "")).strip(),
    )


def _dedupe_annotations(raw_annotations: list[Any]) -> list[CustomGraphAnnotation]:
    """Normalize annotations and keep the first occurrence of each key."""
    deduped: list[CustomGraphAnnotation] = []
    seen_keys: set[str] = set()
    for item in raw_annotations:
        annotation = _normalize_annotation_payload(item)
        if annotation.key in seen_keys:
            continue
        seen_keys.add(annotation.key)
        deduped.append(annotation)
    return deduped


def _migrate_legacy_composed_graph_payload(spec: dict[str, Any]) -> dict[str, Any]:
    """Promote legacy composed-graph overlay recipes into the new recipe fields."""
    migrated = dict(spec)
    overlay_value = migrated.get("overlay")
    if not overlay_value:
        return migrated

    overlay = _normalize_overlay_payload(overlay_value)

    if overlay.kind == "segment":
        migrated["view_domain"] = "semantic_segment"
        migrated["segment_key"] = overlay.key
        migrated["rebase_x"] = True
        migrated["overlay"] = None
        return migrated

    if overlay.kind == "annotation":
        annotations = _dedupe_annotations(list(migrated.get("annotations", [])))
        if overlay.key not in {annotation.key for annotation in annotations}:
            annotations.append(CustomGraphAnnotation(kind="annotation", key=overlay.key))
        migrated["annotations"] = annotations
        if migrated.get("segment_key"):
            migrated["view_domain"] = "semantic_segment"
            migrated["rebase_x"] = True
        migrated["overlay"] = None
    return migrated


def _is_trace_graph_payload(spec: Any) -> bool:
    """Return ``True`` for payloads shaped like legacy or current trace graph data."""
    if not isinstance(spec, dict):
        return False
    return any(key in spec for key in ("plot_type", "x_cols", "x_col", "y_cols", "y_vars"))


def _normalize_session_graph_spec(spec: Any) -> GraphSpec | CustomGraphSpec:
    """Normalize one saved graph spec into a typed model."""
    if isinstance(spec, (GraphSpec, CustomGraphSpec)):
        return spec
    if not isinstance(spec, dict):
        raise TypeError("Graph spec payload must be a mapping or typed spec.")
    if _is_composed_graph_payload(spec):
        return normalize_composed_graph_spec(_migrate_legacy_composed_graph_payload(spec))
    if _is_trace_graph_payload(spec):
        return normalize_graph_spec(spec)
    raise PlotSpecError("Unrecognized graph spec payload.")


def _parse_schema_version(value: Any) -> int:
    """Parse a persisted schema version."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SessionError(f"Session load failed: invalid schema_version {value!r}") from exc


def migrate_graph_specs(raw_specs: list[Any]) -> list[GraphSpec | CustomGraphSpec]:
    """Migrate legacy graph-spec payloads into the current typed model."""
    migrated: list[GraphSpec | CustomGraphSpec] = []
    for index, item in enumerate(raw_specs):
        try:
            migrated.append(_normalize_session_graph_spec(item))
        except (PlotSpecError, TypeError, ValueError) as exc:
            raise SessionError(f"Session restore failed: invalid graph spec payload at index {index}.") from exc
    return migrated


def load_session_data(path: Path) -> dict[str, Any]:
    """Load and minimally normalize a saved session payload."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SessionError(f"Session load failed: {exc}") from exc
    if not isinstance(data, dict):
        raise SessionError("Session file does not contain a valid object payload.")
    data["schema_version"] = _parse_schema_version(data.get("schema_version", 1))
    return data


def save_session_data(path: Path, payload: dict[str, Any]) -> None:
    """Persist the current UI session to disk."""
    output = dict(payload)
    output["schema_version"] = SESSION_SCHEMA_VERSION
    if "graph_specs" in output and isinstance(output["graph_specs"], list):
        specs: list[GraphSpec | CustomGraphSpec | dict[str, Any]] = []
        for item in output["graph_specs"]:
            if isinstance(item, dict):
                try:
                    specs.append(_normalize_session_graph_spec(item))
                except (PlotSpecError, TypeError, ValueError):
                    specs.append(dict(item))
            elif isinstance(item, (GraphSpec, CustomGraphSpec)):
                specs.append(item)
        output["graph_specs"] = serialize_graph_specs(specs)
    try:
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    except Exception as exc:
        raise SessionError(f"Session save failed: {exc}") from exc
