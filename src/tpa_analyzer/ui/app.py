"""Textual application for the TPA Analyzer."""

from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from textual import events, on, work
from textual.app import App, ComposeResult, ScreenStackError
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from tpa_analyzer.analysis.tpa import TPAConfig, build_metrics_row, calculate_tpa, parse_zwick_data
from tpa_analyzer.config.logging import configure_logging, get_logger
from tpa_analyzer.config.settings import AppSettings
from tpa_analyzer.core.constants import (
    COMPUTED_METRICS,
    DEFAULT_TRACE_Y_VARIABLES,
    SESSION_SCHEMA_VERSION,
    SUPPORTED_DATA_EXTENSIONS,
)
from tpa_analyzer.core.errors import AnalysisError, DataParseError, PlotSpecError, SessionError
from tpa_analyzer.core.exporting import (
    current_export_root,
    export_plot_bundle,
    export_tables_bundle,
)
from tpa_analyzer.core.models import (
    CustomGraphAnnotation,
    CustomGraphAxisLayer,
    CustomGraphSpec,
    FigureConfig,
    GraphSpec,
    PlotStyleConfig,
    is_hex_color,
)
from tpa_analyzer.core.session import (
    load_session_data,
    migrate_graph_specs,
    save_session_data,
    session_path,
)
from tpa_analyzer.plotting.custom_graphs import (
    ANNOTATION_COMPATIBILITY,
    OVERLAY_COMPATIBILITY,
    TRACE_COMPATIBILITY,
    eligible_annotation_keys,
    eligible_left_axis_variables,
    eligible_right_axis_variables,
    eligible_segment_keys,
)
from tpa_analyzer.plotting.engine import expand_composed_graph_spec
from tpa_analyzer.plotting.registry import registry_entry
from tpa_analyzer.stats.engine import run_statistics
from tpa_analyzer.ui.layout import resolve_layout_mode

PARAM_INFO: dict[str, dict[str, str]] = {
    "sample_height": {
        "label": "Sample Height (mm)",
        "help": "Initial sample thickness before compression. Used to compute true strain and modulus.",
    },
    "contact_area": {
        "label": "Contact Area (mm2)",
        "help": "Contact surface area between probe and sample. Used to convert force into true stress.",
    },
    "baseline_points": {
        "label": "Baseline Points",
        "help": "Number of first points used to estimate force baseline offset before peak detection.",
    },
    "trigger_force": {
        "label": "Trigger Force (N)",
        "help": "Force threshold used to detect start and end of compression cycles.",
    },
    "peak_prominence": {
        "label": "Peak Prominence (N)",
        "help": "Minimum prominence required for peaks. Increase it to ignore noise.",
    },
    "peak_distance": {
        "label": "Peak Distance (pts)",
        "help": "Minimum spacing between detected peaks in data points.",
    },
    "modulus_min": {
        "label": "Modulus Strain Min (%)",
        "help": "Lower strain bound for modulus fitting during first compression.",
    },
    "modulus_max": {
        "label": "Modulus Strain Max (%)",
        "help": "Upper strain bound for modulus fitting during first compression.",
    },
    "stats_mode": {
        "label": "Stats Mode",
        "help": "Select statistical test family. Auto chooses based on assumptions.",
    },
}


CUSTOM_GRAPH_NONE = "__none__"


def custom_graph_overlay_qc_columns(overlay_key: str) -> tuple[str, ...]:
    """Return the QC summary columns required for one overlay option."""
    if overlay_key == "b1_start_to_peak1":
        return ("Bite1 Start Index", "Peak1 Index")
    if overlay_key == "peak1_to_b1_end":
        return ("Peak1 Index", "Bite1 End Index")
    if overlay_key == "b1_end_to_b2_start":
        return ("Bite1 End Index", "Bite2 Start Index")
    if overlay_key == "b2_start_to_peak2":
        return ("Bite2 Start Index", "Peak2 Index")
    if overlay_key == "hardness_peak1":
        return ("Peak1 Index",)
    if overlay_key == "adhesiveness":
        return ("Bite1 End Index", "Bite2 Start Index")
    if overlay_key == "modulus_window":
        return ("Modulus Strain Min (%)", "Modulus Strain Max (%)")
    return ()


def custom_graph_x_domains() -> list[str]:
    """Return X-domain options supported by the composed graph builder."""
    ordered: list[str] = []
    for item in [*TRACE_COMPATIBILITY.values(), *OVERLAY_COMPATIBILITY.values()]:
        for x_domain in item.allowed_x_domains:
            if x_domain not in ordered:
                ordered.append(x_domain)
    return ordered


def _left_axis_variables_compatible(variables: list[str]) -> bool:
    """Return whether selected left-axis variables share one unit."""
    units = {
        registry_entry(str(variable).strip()).unit
        for variable in variables
        if str(variable).strip() and registry_entry(str(variable).strip()).unit
    }
    return len(units) <= 1


def _eligible_left_axis_variables_for_selection(
    x_domain: str,
    selected_left_variables: list[str],
    analysis_ready: bool,
) -> list[str]:
    """Return left-axis variables compatible with the current selection."""
    eligible = eligible_left_axis_variables(x_domain=x_domain, analysis_ready=analysis_ready)
    selected = [str(variable).strip() for variable in selected_left_variables if str(variable).strip()]
    if not selected:
        return eligible

    selected_units = {registry_entry(variable).unit for variable in selected if registry_entry(variable).unit}
    if len(selected_units) != 1:
        return eligible

    selected_unit = next(iter(selected_units))
    return [
        variable
        for variable in eligible
        if variable in selected or registry_entry(variable).unit == selected_unit
    ]


def _custom_graph_overlay_available(qc_df: pd.DataFrame, overlay_key: str) -> bool:
    """Return whether the current QC summary contains prerequisites for one overlay."""
    required_columns = custom_graph_overlay_qc_columns(overlay_key)
    if not required_columns:
        return False
    qc_df = _filter_assigned_group_rows(qc_df)
    if qc_df.empty:
        return False

    row_mask = pd.Series(True, index=qc_df.index, dtype=bool)
    for column in required_columns:
        if column not in qc_df.columns:
            return False
        values = qc_df[column]
        value_mask = values.notna()
        if values.dtype == object:
            value_mask &= values.astype(str).str.strip().ne("")
        row_mask &= value_mask
    return bool(row_mask.any())


def derive_group_order_from_file_records(file_records: Any) -> list[str]:
    """Derive stable first-seen group order from serialized file records."""
    if not isinstance(file_records, list):
        return []
    derived_order: list[str] = []
    seen: set[str] = set()
    for item in file_records:
        if not isinstance(item, dict):
            continue
        group = str(item.get("group", "")).strip()
        if group and group not in seen:
            seen.add(group)
            derived_order.append(group)
    return derived_order


def _filter_assigned_group_rows(frame: pd.DataFrame, group_column: str = "Group") -> pd.DataFrame:
    """Return only rows with a non-empty assigned group."""
    if frame.empty or group_column not in frame.columns:
        return frame.copy()
    groups = frame[group_column].fillna("").astype(str).str.strip()
    return frame.loc[groups.ne("")].copy()


def filter_assigned_plot_export_payload(
    trace_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    stats_results: dict[str, dict[str, Any]],
    group_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], list[str]]:
    """Remove unassigned-group rows from plot and QC export payloads."""
    filtered_trace = _filter_assigned_group_rows(trace_df)
    filtered_metrics = _filter_assigned_group_rows(metrics_df)
    filtered_qc = _filter_assigned_group_rows(qc_df)
    filtered_group_order: list[str] = []
    seen_groups: set[str] = set()
    for group in group_order:
        normalized = str(group).strip()
        if normalized and normalized not in seen_groups:
            seen_groups.add(normalized)
            filtered_group_order.append(normalized)

    filtered_stats: dict[str, dict[str, Any]] = {}
    for metric, result in stats_results.items():
        summary_df = _filter_assigned_group_rows(result.get("summary_df", pd.DataFrame()), "Group")
        pairwise_df = result.get("pairwise_df", pd.DataFrame())
        if isinstance(pairwise_df, pd.DataFrame):
            pairwise_df = pairwise_df.copy()
            for column in ("Group1", "Group2"):
                if column in pairwise_df.columns:
                    pairwise_df = pairwise_df.loc[pairwise_df[column].fillna("").astype(str).str.strip().ne("")]
        else:
            pairwise_df = pd.DataFrame()
        if summary_df.empty:
            continue
        test_info = dict(result.get("test_info", {}))
        test_info["group_order"] = filtered_group_order.copy()
        filtered_stats[metric] = {
            **result,
            "summary_df": summary_df,
            "pairwise_df": pairwise_df.reset_index(drop=True),
            "test_info": test_info,
        }

    return filtered_trace, filtered_metrics, filtered_qc, filtered_stats, filtered_group_order


class ParameterInfoModal(ModalScreen[None]):
    """Simple accessibility modal for parameter help text."""

    BINDINGS = [
        ("escape", "dismiss_modal", "Close"),
        ("enter", "dismiss_modal", "Close"),
        ("q", "dismiss_modal", "Close"),
    ]

    CSS = """
    #param-help-root {
        align: center middle;
        width: 72;
        max-width: 96%;
        height: auto;
        border: round $border;
        background: $panel;
        padding: 1 2;
    }
    #param-help-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    #param-help-body {
        margin-bottom: 1;
    }
    """

    def __init__(self, title: str, body: str) -> None:
        """Initialize the modal with a title and body."""
        super().__init__()
        self.title = title
        self.body = body

    def compose(self) -> ComposeResult:
        """Compose the modal UI."""
        with Vertical(id="param-help-root"):
            yield Static(self.title, id="param-help-title")
            yield Static(self.body, id="param-help-body")
            yield Static("Press Esc, Enter, or q to close.", classes="small-label")

    def action_dismiss_modal(self) -> None:
        """Dismiss the modal."""
        self.dismiss(None)


class TPAAnalyzerApp(App):
    """Terminal TPA analyzer with responsive layout and typed custom plots."""

    CSS = """
    Screen {
        background: $surface;
        color: $foreground;
    }

    #studio {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 1fr 1.25fr 1fr;
        height: 1fr;
    }

    #studio.medium {
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
    }

    #studio.medium #center-pane {
        column-span: 2;
    }

    #studio.narrow {
        grid-size: 1 3;
        grid-columns: 1fr;
    }

    .pane {
        border: round $border;
        background: $panel;
        padding: 1;
        margin: 1;
        height: 1fr;
    }

    .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    .small-label {
        color: $foreground 60%;
    }

    .option-box {
        border: round $border;
        height: 12;
        margin-bottom: 1;
        padding: 0 1;
    }

    #file_list {
        height: 1fr;
        min-height: 8;
    }

    #group_order_list {
        height: 6;
        margin-bottom: 1;
    }

    DataTable {
        height: 1fr;
        margin-bottom: 1;
        border: round $border;
    }

    RichLog {
        height: 10;
        border: round $border;
    }

    .action-row {
        height: auto;
        margin-top: 1;
    }

    Input, Select {
        margin-bottom: 1;
    }

    #graph-spec-list {
        border: round $border;
        padding: 1;
        height: 8;
    }

    #custom_graph_summary {
        border: round $border;
        padding: 1;
        height: 7;
        margin-bottom: 1;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("space", "toggle_highlighted_file_selection", "Toggle File"),
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    def __init__(self, settings: AppSettings | None = None) -> None:
        """Initialize the application state."""
        super().__init__()
        self.settings = settings or AppSettings.from_env()
        configure_logging(self.settings)
        self.logger = get_logger(__name__)
        self.base_dir = Path.cwd()
        self.active_directory: Path | None = None
        self._loading_session = False
        self._pending_color_group: str | None = None
        self.file_records: list[dict[str, str]] = []
        self.selected_file_index: int | None = None
        self.selected_file_indices: set[int] = set()
        self.group_order: list[str] = []
        self.selected_group_order_index: int | None = None
        self.metrics_df = pd.DataFrame()
        self.trace_df = pd.DataFrame()
        self.qc_df = pd.DataFrame()
        self.stats_results: dict[str, dict[str, Any]] = {}
        self.plot_style = PlotStyleConfig()
        self.graph_specs: list[GraphSpec | CustomGraphSpec] = []
        self._custom_graph_selected_samples: list[str] = []
        self._syncing_custom_graph_builder = False
        self._custom_graph_builder_internal_update_counts: dict[str, int] = {}

    def compose(self) -> ComposeResult:
        """Compose the full Textual UI."""
        yield Header(show_clock=True)
        with Container(id="studio"):
            with VerticalScroll(id="left-pane", classes="pane"):
                yield Static("Data & Grouping", classes="section-title")
                yield Label("Directory", classes="small-label")
                yield Input(value=str(self.settings.default_data_dir), id="input_dir", placeholder="Relative or absolute path")
                yield Button("Refresh Directory", id="btn_refresh", variant="primary")
                yield Label("Detected Files", classes="small-label")
                yield DataTable(id="file_list", cursor_type="row", show_row_labels=False, zebra_stripes=False)
                yield Static("Selected: none", id="selected_file_info", classes="small-label")
                yield Static("Selected files: 0", id="selected_file_count", classes="small-label")
                yield Static("Groups: none", id="group_summary", classes="small-label")
                yield Label("Group Display Order", classes="small-label")
                yield OptionList(id="group_order_list", markup=False)
                with Horizontal(classes="action-row"):
                    yield Button("Group Up", id="btn_group_up")
                    yield Button("Group Down", id="btn_group_down")
                yield Label("Group Name", classes="small-label")
                yield Input(id="input_group_name", placeholder="Add or rename a group")
                with Horizontal(classes="action-row"):
                    yield Button("Add Group", id="btn_group_add", variant="primary")
                    yield Button("Rename Group", id="btn_group_rename")
                    yield Button("Delete Group", id="btn_group_delete", variant="warning")
                with Horizontal(classes="action-row"):
                    yield Button("Assign to Active Group", id="btn_assign_active", variant="success")
                    yield Button("Clear Assignment", id="btn_clear_assignment", variant="warning")
                yield Static("Color Style", classes="section-title")
                yield Label("Target Group", classes="small-label")
                yield Select([("No groups", "__none__")], id="select_color_group", allow_blank=False)
                yield Label("Group Hex", classes="small-label")
                yield Input(value="#2563EB", id="input_group_hex", placeholder="#RRGGBB")
                with Horizontal(classes="action-row"):
                    yield Button("Apply Color", id="btn_apply_colors", variant="success")
                    yield Button("Reset Palette", id="btn_reset_palette", variant="warning")

            with Vertical(id="center-pane", classes="pane"):
                yield Static("Analysis Results", classes="section-title")
                yield DataTable(id="results_table")
                yield Static("Status", classes="section-title")
                yield RichLog(id="log_stream", markup=True)
                yield Static("Ready", id="status-msg")

            with VerticalScroll(id="right-pane", classes="pane"):
                yield Static("Controls", classes="section-title")
                with TabbedContent(id="right-tabs"):
                    with TabPane("Analysis Params"):
                        yield Label(PARAM_INFO["sample_height"]["label"], classes="small-label")
                        yield Input(value="10.0", id="input_height")
                        yield Label(PARAM_INFO["contact_area"]["label"], classes="small-label")
                        yield Input(value="100.0", id="input_area")
                        yield Label(PARAM_INFO["baseline_points"]["label"], classes="small-label")
                        yield Input(value="10", id="input_baseline_points")
                        yield Label(PARAM_INFO["trigger_force"]["label"], classes="small-label")
                        yield Input(value="0.05", id="input_trigger")
                        yield Label(PARAM_INFO["peak_prominence"]["label"], classes="small-label")
                        yield Input(value="0.5", id="input_prominence")
                        yield Label(PARAM_INFO["peak_distance"]["label"], classes="small-label")
                        yield Input(value="200", id="input_peak_distance")
                        yield Label(PARAM_INFO["modulus_min"]["label"], classes="small-label")
                        yield Input(value="10", id="input_mod_min")
                        yield Label(PARAM_INFO["modulus_max"]["label"], classes="small-label")
                        yield Input(value="30", id="input_mod_max")
                        yield Label(PARAM_INFO["stats_mode"]["label"], classes="small-label")
                        yield Select(
                            [("Auto", "auto"), ("Parametric", "parametric"), ("Nonparametric", "nonparametric")],
                            value="auto",
                            id="select_stats_mode",
                            allow_blank=False,
                        )
                        yield Button("Run Analysis", id="btn_analyze", variant="primary")

                    with TabPane("Plot Builder"):
                        yield Label("View Domain", classes="small-label")
                        yield Select(
                            [("Full Curve", "full_curve"), ("Semantic Segment", "semantic_segment")],
                            value="full_curve",
                            id="select_custom_view_domain",
                            allow_blank=False,
                        )
                        yield Label("Segment", classes="small-label", id="label_custom_segment")
                        yield Select(
                            [("None", CUSTOM_GRAPH_NONE)],
                            value=CUSTOM_GRAPH_NONE,
                            id="select_custom_segment",
                            allow_blank=False,
                        )
                        yield Label("Curves", classes="small-label")
                        yield Label("X Domain", classes="small-label")
                        yield Select(
                            [(label, label) for label in custom_graph_x_domains()],
                            value="Time (s)",
                            id="select_custom_x_domain",
                            allow_blank=False,
                        )
                        yield Label("Left Axis (choose up to two)", classes="small-label")
                        with VerticalScroll(id="custom_left_axis_choices", classes="option-box"):
                            for label in TRACE_COMPATIBILITY:
                                yield Checkbox(label, value=(label == "Force (N)"), classes="custom-left-axis-choice")
                        yield Label("Right Axis", classes="small-label")
                        yield Select([("None", CUSTOM_GRAPH_NONE)], value=CUSTOM_GRAPH_NONE, id="select_custom_right_axis", allow_blank=False)
                        yield Label("Annotations", classes="small-label", id="label_custom_annotation")
                        yield Select(
                            [("None", CUSTOM_GRAPH_NONE)],
                            value=CUSTOM_GRAPH_NONE,
                            id="select_custom_annotation",
                            allow_blank=False,
                        )
                        yield Label("Data Scope", classes="small-label")
                        yield Select(
                            [("Grouped", "grouped"), ("Selected Samples", "selected_samples")],
                            value="grouped",
                            id="select_custom_data_scope",
                            allow_blank=False,
                        )
                        yield Label("Sample Selection", classes="small-label", id="label_custom_sample_list")
                        yield OptionList(id="custom_graph_sample_list", markup=False)
                        yield Label("Display Mode", classes="small-label")
                        yield Select(
                            [("Stacked", "stacked"), ("Individual", "individual")],
                            value="stacked",
                            id="select_custom_display_mode",
                            allow_blank=False,
                        )
                        yield Label("Band Type", classes="small-label")
                        yield Select([("SD", "sd"), ("95% CI", "ci95")], value="sd", id="select_band_mode", allow_blank=False)
                        yield Label("Live Summary", classes="small-label")
                        yield Static("", id="custom_graph_summary")
                        yield Label("Graph Title", classes="small-label")
                        yield Input(value="Custom Graph", id="input_graph_title")
                        with Horizontal(classes="action-row"):
                            yield Button("Add Graph", id="btn_add_graph")
                            yield Button("Clear Graphs", id="btn_clear_graphs", variant="warning")
                        yield Static("(No custom graphs yet)", id="graph-spec-list")
                        yield Label("Group Overlay Mode", classes="small-label")
                        yield Select(
                            [("Mean + Band", "mean_band"), ("Individual", "individual"), ("Both", "both")],
                            value="mean_band",
                            id="select_overlay_mode",
                            allow_blank=False,
                        )

                    with TabPane("Style / Theme"):
                        yield Label("Ratio Preset", classes="small-label")
                        yield Select(
                            [("1:1", "1:1"), ("4:3", "4:3"), ("16:9", "16:9"), ("A4 portrait", "A4 portrait"), ("A4 landscape", "A4 landscape")],
                            value="4:3",
                            id="select_ratio",
                            allow_blank=False,
                        )
                        yield Label("Width (in, optional)", classes="small-label")
                        yield Input(value="", id="input_width")
                        yield Label("Height (in, optional)", classes="small-label")
                        yield Input(value="", id="input_height_fig")
                        yield Label("DPI", classes="small-label")
                        yield Input(value="300", id="input_dpi")
                        yield Static("Effective Size: 10.00 x 7.50 in", id="fig-preview")

                    with TabPane("Export"):
                        yield Button("Export Tables", id="btn_export_tables", variant="success")
                        yield Button("Export Plots", id="btn_export_plots", variant="primary")
                        yield Button("Export All", id="btn_export_all", variant="warning")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize widgets and load the current directory."""
        file_table = self.query_one("#file_list", DataTable)
        file_table.add_columns("#", "Group", "Filename")

        table = self.query_one("#results_table", DataTable)
        table.add_columns("Filename", "Group", *COMPUTED_METRICS)

        self._apply_layout_mode()
        self._sync_custom_graph_builder_state()
        self._refresh_directory()
        self._update_figure_preview()
        self._log("App started.")

    def on_resize(self, event: events.Resize) -> None:
        """Update layout classes whenever the terminal is resized."""
        _ = event
        self._apply_layout_mode()

    def _apply_layout_mode(self) -> None:
        """Apply the current responsive layout mode to the studio container."""
        mode = resolve_layout_mode(self.size.width)
        studio = self.query_one("#studio", Container)
        studio.remove_class("wide", "medium", "narrow")
        studio.add_class(mode)

    def _log(self, message: str) -> None:
        """Write a timestamped message to the in-app log."""
        self.logger.info(message)
        if not self._widgets_ready():
            return
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        self.query_one("#log_stream", RichLog).write(f"[{timestamp}] {message}")

    def _widgets_ready(self) -> bool:
        """Return ``True`` when widget queries can safely access the live screen."""
        try:
            _ = self.screen
        except ScreenStackError:
            return False
        return True

    def _set_status(self, text: str) -> None:
        """Update the footer status line."""
        if not self._widgets_ready():
            return
        self.query_one("#status-msg", Static).update(text)

    def _set_buttons_disabled(self, disabled: bool) -> None:
        """Enable or disable the long-running action buttons."""
        if not self._widgets_ready():
            return
        for button_id in ["#btn_analyze", "#btn_export_tables", "#btn_export_plots", "#btn_export_all", "#btn_refresh"]:
            self.query_one(button_id, Button).disabled = disabled
        if disabled:
            self.query_one("#btn_group_up", Button).disabled = True
            self.query_one("#btn_group_down", Button).disabled = True
            self.query_one("#btn_assign_active", Button).disabled = True
            self.query_one("#btn_clear_assignment", Button).disabled = True
        else:
            self._update_group_order_buttons()
            self._update_assignment_buttons()

    def _resolve_directory(self) -> Path:
        """Resolve the current directory input into an absolute path."""
        raw = str(self.settings.default_data_dir).strip() or "."
        if self._widgets_ready():
            raw = self.query_one("#input_dir", Input).value.strip() or raw
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (self.base_dir / path).resolve()
        return path

    def _set_input_if_present(self, widget_id: str, value: str) -> None:
        """Set an input value when the widget exists."""
        if not self._widgets_ready():
            return
        try:
            self.query_one(widget_id, Input).value = value
        except Exception:
            return

    def _set_select_if_present(self, widget_id: str, value: str) -> None:
        """Set a select value when the widget exists."""
        if not self._widgets_ready():
            return
        try:
            self.query_one(widget_id, Select).value = value
        except Exception:
            return

    def _set_checkbox_values(self, container_id: str, values: list[str]) -> None:
        """Apply selected values to all checkboxes inside a container."""
        if not self._widgets_ready():
            return
        container = self.query_one(container_id)
        selected = set(values)
        for checkbox in container.query(Checkbox):
            checkbox.value = str(checkbox.label) in selected

    def _selected_checkbox_values(self, container_id: str) -> list[str]:
        """Return selected checkbox labels in DOM order for a container."""
        if not self._widgets_ready():
            return []
        container = self.query_one(container_id)
        selected: list[str] = []
        for checkbox in container.query(Checkbox):
            if checkbox.value:
                selected.append(str(checkbox.label))
        return selected

    def watch_theme(self, theme_name: str) -> None:
        """Autosave the session when the theme changes."""
        _ = theme_name
        if not self._widgets_ready() or self._loading_session:
            return
        self._autosave_session()

    def _collect_session_payload(self) -> dict[str, Any]:
        """Collect the serializable session payload from current UI state."""
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "saved_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "directory": str(self.active_directory) if self.active_directory else "",
            "file_records": [{"filename": record.get("filename", ""), "group": record.get("group", "")} for record in self.file_records],
            "group_order": self.group_order.copy(),
            "active_group": self._active_group_name(),
            "selected_file_index": self.selected_file_index,
            "ui": {"theme": str(self.theme)},
            "analysis_params": {
                "sample_height": self.query_one("#input_height", Input).value,
                "contact_area": self.query_one("#input_area", Input).value,
                "baseline_points": self.query_one("#input_baseline_points", Input).value,
                "trigger_force": self.query_one("#input_trigger", Input).value,
                "peak_prominence": self.query_one("#input_prominence", Input).value,
                "peak_distance": self.query_one("#input_peak_distance", Input).value,
                "modulus_min": self.query_one("#input_mod_min", Input).value,
                "modulus_max": self.query_one("#input_mod_max", Input).value,
                "stats_mode": str(self.query_one("#select_stats_mode", Select).value),
            },
            "plot_builder": {
                "view_domain": str(self.query_one("#select_custom_view_domain", Select).value),
                "segment_key": str(self.query_one("#select_custom_segment", Select).value),
                "x_domain": str(self.query_one("#select_custom_x_domain", Select).value),
                "left_axis": self._selected_custom_left_axis_variables(),
                "right_axis": str(self.query_one("#select_custom_right_axis", Select).value),
                "annotation": str(self.query_one("#select_custom_annotation", Select).value),
                "data_scope": str(self.query_one("#select_custom_data_scope", Select).value),
                "selected_samples": self._selected_custom_graph_samples(),
                "display_mode": str(self.query_one("#select_custom_display_mode", Select).value),
                "band_mode": str(self.query_one("#select_band_mode", Select).value),
                "graph_title": self.query_one("#input_graph_title", Input).value,
                "overlay_mode": str(self.query_one("#select_overlay_mode", Select).value),
            },
            "figure_style": {
                "ratio": str(self.query_one("#select_ratio", Select).value),
                "width_in": self.query_one("#input_width", Input).value,
                "height_in": self.query_one("#input_height_fig", Input).value,
                "dpi": self.query_one("#input_dpi", Input).value,
            },
            "colors": {
                "group_colors": self.plot_style.group_colors.copy(),
                "selected_color_group": str(self.query_one("#select_color_group", Select).value),
                "group_hex_input": self.query_one("#input_group_hex", Input).value,
            },
            "graph_specs": [asdict(spec) for spec in self.graph_specs],
        }

    def _autosave_session(self) -> None:
        """Persist the current session when autosave is enabled."""
        if not self._widgets_ready() or self._loading_session or self.active_directory is None or not self.settings.session_autosave_enabled:
            return
        try:
            save_session_data(session_path(self.active_directory), self._collect_session_payload())
        except SessionError as exc:
            self._log(str(exc))

    def _has_cached_analysis_results(self) -> bool:
        """Return ``True`` when analysis-derived artifacts are currently cached."""
        return not self.metrics_df.empty or not self.trace_df.empty or not self.qc_df.empty or bool(self.stats_results)

    def _invalidate_analysis_results_for_grouping_change(self) -> bool:
        """Clear cached analysis results after a grouping mutation."""
        if not self._has_cached_analysis_results():
            return False
        self.metrics_df = pd.DataFrame()
        self.trace_df = pd.DataFrame()
        self.qc_df = pd.DataFrame()
        self.stats_results = {}
        self._rebuild_results_table()
        self._sync_custom_graph_builder_state()
        self._log("Grouping changed. Run analysis again before exporting.")
        return True

    def _load_session_for_directory(self, directory: Path) -> tuple[bool, int | None]:
        """Load session state for a directory and apply it to the UI."""
        path = session_path(directory)
        if not path.exists() or not path.is_file():
            return False, None

        selected_idx: int | None = None
        self._loading_session = True
        try:
            data = load_session_data(path)
            ui = data.get("ui", {})
            if isinstance(ui, dict):
                saved_theme = str(ui.get("theme", "")).strip()
                if saved_theme and self.get_theme(saved_theme) is not None:
                    self.theme = saved_theme

            files_data = data.get("file_records", [])
            group_map: dict[str, str] = {}
            if isinstance(files_data, list):
                for item in files_data:
                    if isinstance(item, dict):
                        filename = str(item.get("filename", "")).strip()
                        group = str(item.get("group", "")).strip()
                        if filename:
                            group_map[filename] = group
            for record in self.file_records:
                filename = record.get("filename", "")
                if filename in group_map:
                    record["group"] = group_map[filename]

            saved_order = data.get("group_order", [])
            restored_order = [str(group).strip() for group in saved_order if str(group).strip()] if isinstance(saved_order, list) else []
            self.group_order = restored_order or derive_group_order_from_file_records(files_data)
            valid_groups = set(self.group_order)
            for record in self.file_records:
                group = record.get("group", "").strip()
                if group and group not in valid_groups:
                    record["group"] = ""
            self.selected_group_order_index = None
            saved_active_group = str(data.get("active_group", "")).strip()
            if saved_active_group and saved_active_group in self.group_order:
                self.selected_group_order_index = self.group_order.index(saved_active_group)

            analysis = data.get("analysis_params", {})
            if isinstance(analysis, dict):
                self._set_input_if_present("#input_height", str(analysis.get("sample_height", "10.0")))
                self._set_input_if_present("#input_area", str(analysis.get("contact_area", "100.0")))
                self._set_input_if_present("#input_baseline_points", str(analysis.get("baseline_points", "10")))
                self._set_input_if_present("#input_trigger", str(analysis.get("trigger_force", "0.05")))
                self._set_input_if_present("#input_prominence", str(analysis.get("peak_prominence", "0.5")))
                self._set_input_if_present("#input_peak_distance", str(analysis.get("peak_distance", "200")))
                self._set_input_if_present("#input_mod_min", str(analysis.get("modulus_min", "10")))
                self._set_input_if_present("#input_mod_max", str(analysis.get("modulus_max", "30")))
                self._set_select_if_present("#select_stats_mode", str(analysis.get("stats_mode", "auto")))

            builder = data.get("plot_builder", {})
            if isinstance(builder, dict):
                overlay_value = str(builder.get("overlay", CUSTOM_GRAPH_NONE))
                view_domain = str(builder.get("view_domain", "full_curve"))
                segment_key = str(builder.get("segment_key", CUSTOM_GRAPH_NONE))
                annotation_value = str(builder.get("annotation", CUSTOM_GRAPH_NONE))
                if overlay_value != CUSTOM_GRAPH_NONE:
                    overlay_meta = OVERLAY_COMPATIBILITY.get(overlay_value)
                    if overlay_meta is not None:
                        if overlay_meta.item_type == "segment":
                            view_domain = "semantic_segment"
                            segment_key = overlay_value
                        elif overlay_meta.item_type == "annotation":
                            annotation_value = overlay_value

                self._set_select_if_present("#select_custom_view_domain", view_domain)
                self._set_select_if_present(
                    "#select_custom_x_domain",
                    str(builder.get("x_domain", builder.get("trace_x", ["Time (s)"])[0])),
                )
                legacy_left_axis = list(builder.get("trace_y", list(DEFAULT_TRACE_Y_VARIABLES)))
                self._set_custom_left_axis_selections(list(builder.get("left_axis", legacy_left_axis)))
                self._sync_custom_graph_builder_state()
                self._set_select_if_present("#select_custom_right_axis", str(builder.get("right_axis", CUSTOM_GRAPH_NONE)))
                self._set_select_if_present("#select_custom_segment", segment_key)
                self._set_select_if_present("#select_custom_annotation", annotation_value)
                self._set_select_if_present("#select_custom_data_scope", str(builder.get("data_scope", "grouped")))
                self._custom_graph_selected_samples = [
                    str(item).strip() for item in builder.get("selected_samples", []) if str(item).strip()
                ]
                self._set_select_if_present("#select_custom_display_mode", str(builder.get("display_mode", "stacked")))
                self._set_select_if_present("#select_band_mode", str(builder.get("band_mode", "sd")))
                self._set_input_if_present("#input_graph_title", str(builder.get("graph_title", "Custom Graph")))
                self._set_select_if_present("#select_overlay_mode", str(builder.get("overlay_mode", "mean_band")))
                self._sync_custom_graph_builder_state()

            figure = data.get("figure_style", {})
            if isinstance(figure, dict):
                self._set_select_if_present("#select_ratio", str(figure.get("ratio", "4:3")))
                self._set_input_if_present("#input_width", str(figure.get("width_in", "")))
                self._set_input_if_present("#input_height_fig", str(figure.get("height_in", "")))
                self._set_input_if_present("#input_dpi", str(figure.get("dpi", "300")))

            colors = data.get("colors", {})
            if isinstance(colors, dict):
                group_colors = colors.get("group_colors", {})
                if isinstance(group_colors, dict):
                    self.plot_style.group_colors = {str(key): str(value).upper() for key, value in group_colors.items()}
                self._pending_color_group = str(colors.get("selected_color_group", "")).strip() or None
                self._set_input_if_present("#input_group_hex", str(colors.get("group_hex_input", "#2563EB")))

            self.graph_specs = migrate_graph_specs(data.get("graph_specs", []))
            self._render_graph_specs()
            raw_selected = data.get("selected_file_index")
            if isinstance(raw_selected, int):
                selected_idx = raw_selected
        except SessionError as exc:
            self._log(str(exc))
            return False, None
        finally:
            self._loading_session = False

        self._sync_custom_graph_builder_state()
        return True, selected_idx

    def _refresh_directory(self) -> None:
        """Refresh the active directory and rebuild the file list."""
        try:
            directory = self._resolve_directory()
            if not directory.exists() or not directory.is_dir():
                self.active_directory = None
                self.file_records = []
                self.group_order = []
                self.selected_file_index = None
                self.selected_file_indices = set()
                self._render_group_order_list()
                self._render_file_list()
                self._update_color_group_select()
                self._set_status(f"Invalid directory: {directory}")
                self._log(f"Invalid directory: {directory}")
                return

            files = sorted(
                [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS],
                key=lambda item: item.name.lower(),
            )
            self.active_directory = directory
            self.file_records = [{"filename": path.name, "path": str(path), "group": ""} for path in files]
            self.group_order = []
            self.selected_file_index = None
            self.selected_file_indices = set()
            loaded_session, selected_idx = self._load_session_for_directory(directory)
            self._sync_group_order_from_records()
            self._render_file_list(selected_idx=selected_idx)
            self._update_color_group_select()
            if self._widgets_ready() and self._pending_color_group and self._pending_color_group in self.group_order:
                self.query_one("#select_color_group", Select).value = self._pending_color_group
                self._sync_color_inputs_for_group(self._pending_color_group)
            self._pending_color_group = None
            self._update_figure_preview()
            self._autosave_session()
            if self.file_records:
                status = f"Loaded {len(self.file_records)} files from {directory}"
                if loaded_session:
                    status += " (session restored)"
                self._set_status(status)
                self._log(status)
            else:
                self._set_status(f"No .csv/.tra files found in {directory}")
                self._log(f"Directory refreshed: {directory} (no compatible files)")
        except Exception as exc:
            self.logger.exception("Refresh failed", extra={"directory": str(self.active_directory or self.settings.default_data_dir)})
            self.active_directory = None
            self.file_records = []
            self.group_order = []
            self.selected_file_index = None
            self.selected_file_indices = set()
            self._sync_group_order_from_records()
            self._render_file_list()
            self._update_color_group_select()
            self._set_status(f"Refresh failed: {exc}")
            self._log(f"Refresh failed: {exc}")

    def _render_file_list(self, selected_idx: int | None = None) -> None:
        """Render the detected file table and selected-row state."""
        self.selected_file_indices = {idx for idx in self.selected_file_indices if 0 <= idx < len(self.file_records)}
        if not self.file_records:
            self.selected_file_index = None
            if not self._widgets_ready():
                return
            table = self.query_one("#file_list", DataTable)
            table.clear(columns=False)
            self.query_one("#selected_file_info", Static).update("Selected: none")
            self.query_one("#selected_file_count", Static).update("Selected files: 0")
            self._update_group_summary()
            self._update_assignment_buttons()
            return

        target_idx = max(0, min(selected_idx if selected_idx is not None else 0, len(self.file_records) - 1))
        if not self._widgets_ready():
            self.selected_file_index = target_idx
            return

        table = self.query_one("#file_list", DataTable)
        table.clear(columns=False)
        for idx, record in enumerate(self.file_records):
            group = record["group"].strip() or "UNASSIGNED"
            marker = "*" if idx in self.selected_file_indices else " "
            table.add_row(f"{idx + 1:02d}{marker}", group, record["filename"], key=str(idx))

        table.move_cursor(row=target_idx, column=0, animate=False, scroll=True)
        self._set_selected_file(target_idx, update_status=False)
        self._update_selected_file_count()
        self._update_group_summary()
        self._update_assignment_buttons()

    def _set_selected_file(self, idx: int, update_status: bool = True) -> None:
        """Update the current file selection."""
        if idx < 0 or idx >= len(self.file_records):
            return
        self.selected_file_index = idx
        if not self._widgets_ready():
            return
        group = self.file_records[idx]["group"].strip() or "UNASSIGNED"
        filename = self.file_records[idx]["filename"]
        self.query_one("#selected_file_info", Static).update(f"Selected: {filename} -> {group}")
        if update_status:
            self._set_status(f"Selected {filename}")

    def _update_selected_file_count(self) -> None:
        """Refresh the selected-file count label."""
        if not self._widgets_ready():
            return
        count = len(self.selected_file_indices)
        self.query_one("#selected_file_count", Static).update(f"Selected files: {count}")

    def _update_group_summary(self) -> None:
        """Refresh the compact group summary label."""
        if not self._widgets_ready():
            return
        summary_widget = self.query_one("#group_summary", Static)
        if not self.file_records:
            summary_widget.update("Groups: none")
            return
        counts: dict[str, int] = {}
        for record in self.file_records:
            group = record["group"].strip() or "UNASSIGNED"
            counts[group] = counts.get(group, 0) + 1
        ordered_group_names = [group for group in self.group_order if group in counts]
        ordered_group_names.extend([group for group in counts if group not in ordered_group_names])
        pieces = [f"{name}({counts[name]})" for name in ordered_group_names]
        preview = ", ".join(pieces[:6])
        if len(pieces) > 6:
            preview += f", +{len(pieces) - 6} more"
        summary_widget.update(f"Groups: {preview}")

    def _sync_group_order_from_records(self) -> None:
        """Synchronize stored group ordering with the current file records."""
        normalized_order: list[str] = []
        seen: set[str] = set()
        for group in self.group_order:
            normalized = group.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_order.append(normalized)
        self.group_order = normalized_order
        self.plot_style.ensure_group_colors(self.group_order)
        self._reorder_existing_stats_results()
        self._render_group_order_list()

    def _render_group_order_list(self) -> None:
        """Render the group-order option list."""
        if not self.group_order:
            self.selected_group_order_index = None
            self._set_input_if_present("#input_group_name", "")
            if not self._widgets_ready():
                return
            list_widget = self.query_one("#group_order_list", OptionList)
            list_widget.clear_options()
            list_widget.add_option("No groups")
            self._update_group_order_buttons()
            self._update_assignment_buttons()
            return
        if self.selected_group_order_index is None:
            self.selected_group_order_index = 0
        self.selected_group_order_index = max(0, min(self.selected_group_order_index, len(self.group_order) - 1))
        self._set_input_if_present("#input_group_name", self.group_order[self.selected_group_order_index])
        if not self._widgets_ready():
            return
        list_widget = self.query_one("#group_order_list", OptionList)
        list_widget.clear_options()
        list_widget.add_options([f"{idx + 1:02d}. {group}" for idx, group in enumerate(self.group_order)])
        list_widget.highlighted = self.selected_group_order_index
        self._update_group_order_buttons()
        self._update_assignment_buttons()

    def _update_group_order_buttons(self) -> None:
        """Enable or disable group-order buttons based on the current selection."""
        if not self._widgets_ready():
            return
        up = self.query_one("#btn_group_up", Button)
        down = self.query_one("#btn_group_down", Button)
        idx = self.selected_group_order_index
        if idx is None or not self.group_order:
            up.disabled = True
            down.disabled = True
            return
        up.disabled = idx <= 0
        down.disabled = idx >= len(self.group_order) - 1

    def _update_assignment_buttons(self) -> None:
        """Enable or disable assignment controls from current selection state."""
        if not self._widgets_ready():
            return
        selected_count = len(self.selected_file_indices)
        assign = self.query_one("#btn_assign_active", Button)
        clear = self.query_one("#btn_clear_assignment", Button)
        assign.disabled = selected_count == 0 or self._active_group_name() is None
        clear.disabled = selected_count == 0

    def _reorder_existing_stats_results(self) -> None:
        """Reorder existing stats summaries to match current group ordering."""
        if not self.stats_results:
            return
        for _, result in self.stats_results.items():
            summary = result.get("summary_df")
            info = result.get("test_info", {})
            if summary is None or summary.empty:
                continue
            group_col = info.get("group_col", "Group")
            if group_col not in summary.columns:
                continue
            order_map = {group: idx for idx, group in enumerate(self.group_order)}
            reordered = summary.copy()
            reordered["_order"] = reordered[group_col].astype(str).map(order_map).fillna(10_000)
            result["summary_df"] = reordered.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
            result.setdefault("test_info", {})["group_order"] = self.group_order.copy()

    def _selected_group_name(self) -> str | None:
        """Return the currently highlighted group, if any."""
        return self._active_group_name()

    def _active_group_name(self) -> str | None:
        """Return the active group from the explicit group order list."""
        idx = self.selected_group_order_index
        if idx is None or idx < 0 or idx >= len(self.group_order):
            return None
        return self.group_order[idx]

    def _assign_selected_files_to_active_group(self) -> int:
        """Assign the selected files to the active group."""
        group = self._active_group_name()
        if group is None:
            return 0
        updated = 0
        for idx in sorted(self.selected_file_indices):
            if idx < 0 or idx >= len(self.file_records):
                continue
            if self.file_records[idx].get("group", "").strip() == group:
                continue
            self.file_records[idx]["group"] = group
            updated += 1
        if updated:
            self._invalidate_analysis_results_for_grouping_change()
            self._render_file_list(selected_idx=self.selected_file_index)
        return updated

    def _clear_selected_file_assignments(self) -> int:
        """Clear group assignments for the selected files."""
        updated = 0
        for idx in sorted(self.selected_file_indices):
            if idx < 0 or idx >= len(self.file_records):
                continue
            if not self.file_records[idx].get("group", "").strip():
                continue
            self.file_records[idx]["group"] = ""
            updated += 1
        if updated:
            self._invalidate_analysis_results_for_grouping_change()
            self._render_file_list(selected_idx=self.selected_file_index)
        return updated

    def _toggle_selected_file_index(self, idx: int) -> bool:
        """Toggle a file row in the current multi-selection."""
        if idx < 0 or idx >= len(self.file_records):
            return False
        self.selected_file_index = idx
        if idx in self.selected_file_indices:
            self.selected_file_indices.remove(idx)
            is_selected = False
        else:
            self.selected_file_indices.add(idx)
            is_selected = True
        self._render_file_list(selected_idx=idx)
        return is_selected

    def _toggle_file_selection_with_status(self, idx: int) -> None:
        """Toggle a file row and report the updated selection count."""
        if idx < 0 or idx >= len(self.file_records):
            return
        is_selected = self._toggle_selected_file_index(idx)
        filename = self.file_records[idx]["filename"]
        verb = "Selected" if is_selected else "Cleared"
        self._set_status(f"{verb} {filename} ({len(self.selected_file_indices)} selected).")

    def action_toggle_highlighted_file_selection(self) -> None:
        """Keyboard fallback for file-row and plot-builder sample toggles."""
        if not self._widgets_ready():
            return
        sample_list = self.query_one("#custom_graph_sample_list", OptionList)
        if self.focused is sample_list and sample_list.display and not sample_list.disabled:
            highlighted = sample_list.highlighted
            if highlighted is None:
                return
            if self._toggle_custom_graph_sample_by_index(int(highlighted)):
                self._sync_custom_graph_builder_state()
                self._autosave_session()
            return
        if not self.file_records:
            return
        file_table = self.query_one("#file_list", DataTable)
        if self.focused is not file_table:
            return
        self._toggle_file_selection_with_status(int(file_table.cursor_row))

    def _add_group(self, name: str) -> bool:
        """Append a new group to the explicit display order."""
        normalized = name.strip()
        if not normalized or normalized in self.group_order:
            return False
        self.group_order.append(normalized)
        self.selected_group_order_index = len(self.group_order) - 1
        self.plot_style.ensure_group_colors(self.group_order)
        self._reorder_existing_stats_results()
        self._invalidate_analysis_results_for_grouping_change()
        self._render_group_order_list()
        self._update_color_group_select()
        return True

    def _rename_group(self, old: str, new: str) -> bool:
        """Rename an explicit group and update any assigned files."""
        current = old.strip()
        replacement = new.strip()
        if not current or not replacement or current not in self.group_order or current == replacement or replacement in self.group_order:
            return False
        idx = self.group_order.index(current)
        preserved_color = self.plot_style.group_colors.pop(current, None)
        self.group_order[idx] = replacement
        for record in self.file_records:
            if record.get("group", "").strip() == current:
                record["group"] = replacement
        self.plot_style.ensure_group_colors(self.group_order)
        if preserved_color:
            self.plot_style.group_colors[replacement] = preserved_color
        self.selected_group_order_index = idx
        self._reorder_existing_stats_results()
        self._invalidate_analysis_results_for_grouping_change()
        self._render_group_order_list()
        self._render_file_list(selected_idx=self.selected_file_index)
        self._update_group_summary()
        self._update_color_group_select()
        return True

    def _delete_group(self, name: str) -> bool:
        """Delete an explicit group and clear assignments that used it."""
        normalized = name.strip()
        if not normalized or normalized not in self.group_order:
            return False
        idx = self.group_order.index(normalized)
        self.group_order.pop(idx)
        for record in self.file_records:
            if record.get("group", "").strip() == normalized:
                record["group"] = ""
        self.plot_style.group_colors.pop(normalized, None)
        self.plot_style.ensure_group_colors(self.group_order)
        if not self.group_order:
            self.selected_group_order_index = None
        else:
            self.selected_group_order_index = min(idx, len(self.group_order) - 1)
        self._reorder_existing_stats_results()
        self._invalidate_analysis_results_for_grouping_change()
        self._render_group_order_list()
        self._render_file_list(selected_idx=self.selected_file_index)
        self._update_group_summary()
        self._update_color_group_select()
        return True

    def _update_color_group_select(self) -> None:
        """Refresh the group-color target selector."""
        groups = [group for group in self.group_order if group.strip()]
        self.plot_style.ensure_group_colors(groups)
        if not self._widgets_ready():
            return
        select = self.query_one("#select_color_group", Select)
        if not groups:
            select.set_options([("No groups", "__none__")])
            select.value = "__none__"
            return
        select.set_options([(group, group) for group in groups])
        if select.value not in groups:
            select.value = groups[0]
            self._sync_color_inputs_for_group(groups[0])

    def _sync_color_inputs_for_group(self, group: str) -> None:
        """Copy the selected group's color into the edit input."""
        if not self._widgets_ready():
            return
        self.query_one("#input_group_hex", Input).value = self.plot_style.get_color(group)

    def _collect_tpa_config(self) -> TPAConfig:
        """Collect the TPA configuration from UI fields."""
        def as_float(widget_id: str, default: float) -> float:
            """Parse a float input or return the provided default."""
            try:
                return float(self.query_one(widget_id, Input).value.strip())
            except ValueError:
                return default

        def as_int(widget_id: str, default: int) -> int:
            """Parse an integer input or return the provided default."""
            try:
                return int(float(self.query_one(widget_id, Input).value.strip()))
            except ValueError:
                return default

        return TPAConfig(
            sample_height_mm=as_float("#input_height", 10.0),
            contact_area_mm2=as_float("#input_area", 100.0),
            baseline_points=max(as_int("#input_baseline_points", 10), 1),
            trigger_force_n=max(as_float("#input_trigger", 0.05), 0.0),
            peak_prominence_n=max(as_float("#input_prominence", 0.5), 0.0),
            peak_distance_pts=max(as_int("#input_peak_distance", 200), 1),
            modulus_strain_min_pct=as_float("#input_mod_min", 10.0),
            modulus_strain_max_pct=as_float("#input_mod_max", 30.0),
        )

    def _collect_figure_config(self) -> FigureConfig:
        """Collect the figure export configuration from UI fields."""
        ratio = str(self.query_one("#select_ratio", Select).value)

        def parse_float_or_none(widget_id: str) -> float | None:
            """Parse an optional float input, returning ``None`` when blank or invalid."""
            value = self.query_one(widget_id, Input).value.strip()
            if not value:
                return None
            try:
                return float(value)
            except ValueError:
                return None

        width = parse_float_or_none("#input_width")
        height = parse_float_or_none("#input_height_fig")
        try:
            dpi = int(float(self.query_one("#input_dpi", Input).value.strip()))
        except ValueError:
            dpi = 300
        return FigureConfig(ratio_preset=ratio, width_in=width, height_in=height, dpi=int(max(72, min(dpi, 1200))))

    def _update_figure_preview(self) -> None:
        """Refresh the figure preview summary."""
        if not self._widgets_ready():
            return
        fig_cfg = self._collect_figure_config()
        width, height = fig_cfg.resolve_size(default=(10.0, 7.5))
        self.query_one("#fig-preview", Static).update(f"Effective Size: {width:.2f} x {height:.2f} in @ {fig_cfg.dpi} DPI")

    def _custom_graph_analysis_ready(self) -> bool:
        """Return whether analysis-derived custom graph controls may be enabled."""
        return not self.trace_df.empty

    def _custom_left_axis_checkboxes(self) -> list[Checkbox]:
        """Return all left-axis checkboxes in builder order."""
        if not self._widgets_ready():
            return []
        return list(self.query_one("#custom_left_axis_choices").query(Checkbox))

    def _selected_custom_left_axis_variables(self) -> list[str]:
        """Return checked left-axis variables in DOM order."""
        return [str(checkbox.label) for checkbox in self._custom_left_axis_checkboxes() if checkbox.value]

    def _set_custom_left_axis_selections(self, values: list[str]) -> None:
        """Set left-axis selections while preserving checkbox order."""
        selected = {value for value in values[:2] if value}
        for checkbox in self._custom_left_axis_checkboxes():
            checkbox.value = str(checkbox.label) in selected

    def _set_custom_select_options(self, widget_id: str, options: list[tuple[str, str]], value: str, disabled: bool) -> None:
        """Replace select options while keeping a valid current value."""
        select = self.query_one(widget_id, Select)
        current_options = list(getattr(select, "_options", []))
        current_value = str(select.value)
        if current_options == options and current_value == value and select.disabled == disabled:
            return
        available_values = {option_value for _, option_value in options}
        next_value = value if value in available_values else options[0][1]
        if current_value != next_value:
            self._mark_custom_graph_internal_update(widget_id)
        select.set_options(options)
        select.value = next_value
        select.disabled = disabled

    def _set_custom_sample_options(self, samples: list[str], selected: list[str]) -> None:
        """Replace sample-list options without churn when nothing changed."""
        sample_list = self.query_one("#custom_graph_sample_list", OptionList)
        current_samples = [str(option.prompt) for option in sample_list._options]
        current_selected = self._selected_custom_graph_samples()
        normalized_selected = [sample for sample in selected if sample in samples]
        if current_samples == samples and current_selected == normalized_selected:
            return
        sample_list.clear_options()
        if samples:
            sample_list.add_options(samples)
            if normalized_selected:
                first_selected = normalized_selected[0]
                if first_selected in samples:
                    sample_list.highlighted = samples.index(first_selected)
            elif sample_list.highlighted is None:
                sample_list.highlighted = 0
        self._custom_graph_selected_samples = normalized_selected

    def _available_custom_graph_samples(self) -> list[str]:
        """Return sample names available to the plot builder in stable order."""
        available: list[str] = []
        seen: set[str] = set()
        if not self.trace_df.empty and "Filename" in self.trace_df.columns:
            for filename in self.trace_df["Filename"].fillna("").astype(str):
                normalized = filename.strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    available.append(normalized)
            return available

        for record in self.file_records:
            normalized = str(record.get("filename", "")).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                available.append(normalized)
        return available

    def _selected_custom_graph_samples(self) -> list[str]:
        """Return the current plot-builder sample selection."""
        available = set(self._available_custom_graph_samples())
        return [sample for sample in self._custom_graph_selected_samples if sample in available]

    def _mark_custom_graph_internal_update(self, widget_id: str | None) -> None:
        """Suppress the next builder event emitted by an internal widget refresh."""
        if not widget_id:
            return
        widget_id = widget_id.removeprefix("#")
        self._custom_graph_builder_internal_update_counts[widget_id] = (
            self._custom_graph_builder_internal_update_counts.get(widget_id, 0) + 1
        )

    def _consume_custom_graph_internal_update(self, widget_id: str | None) -> bool:
        """Return ``True`` when a builder event came from an internal refresh."""
        if not widget_id:
            return False
        widget_id = widget_id.removeprefix("#")
        pending = self._custom_graph_builder_internal_update_counts.get(widget_id, 0)
        if pending <= 0:
            return False
        if pending == 1:
            self._custom_graph_builder_internal_update_counts.pop(widget_id, None)
        else:
            self._custom_graph_builder_internal_update_counts[widget_id] = pending - 1
        return True

    def _toggle_custom_graph_sample_by_index(self, option_index: int) -> bool:
        """Toggle one builder sample selection by list index."""
        samples = self._available_custom_graph_samples()
        if option_index < 0 or option_index >= len(samples):
            return False
        sample = samples[option_index]
        selected = set(self._selected_custom_graph_samples())
        if sample in selected:
            selected.remove(sample)
        else:
            selected.add(sample)
        self._custom_graph_selected_samples = [item for item in samples if item in selected]
        return True

    def _refresh_custom_graph_sample_list(self) -> None:
        """Refresh the dedicated Plot Builder sample list."""
        if not self._widgets_ready():
            return
        samples = self._available_custom_graph_samples()
        self._set_custom_sample_options(samples, self._custom_graph_selected_samples)

    def _overlay_label(self, overlay_key: str | None) -> str:
        """Return the user-facing label for an overlay selection."""
        if not overlay_key or overlay_key == CUSTOM_GRAPH_NONE:
            return "None"
        overlay_meta = OVERLAY_COMPATIBILITY.get(overlay_key)
        return overlay_meta.label if overlay_meta is not None else overlay_key

    def _annotation_label(self, annotation_key: str | None) -> str:
        """Return the user-facing label for an annotation selection."""
        if not annotation_key or annotation_key == CUSTOM_GRAPH_NONE:
            return "None"
        annotation_meta = ANNOTATION_COMPATIBILITY.get(annotation_key)
        return annotation_meta.label if annotation_meta is not None else annotation_key

    def _sync_custom_graph_builder_state(self) -> None:
        """Synchronize composed graph builder controls from the current builder state."""
        if not self._widgets_ready():
            return

        self._syncing_custom_graph_builder = True
        try:
            view_domain = str(self.query_one("#select_custom_view_domain", Select).value)
            x_domain = str(self.query_one("#select_custom_x_domain", Select).value)
            analysis_ready = self._custom_graph_analysis_ready()
            current_selected_left = self._selected_custom_left_axis_variables()
            eligible_left = set(
                _eligible_left_axis_variables_for_selection(
                    x_domain=x_domain,
                    selected_left_variables=current_selected_left,
                    analysis_ready=analysis_ready,
                )
            )

            selected_left: list[str] = []
            for checkbox in self._custom_left_axis_checkboxes():
                label = str(checkbox.label)
                allowed = label in eligible_left
                checkbox.display = allowed
                if not allowed and checkbox.value:
                    checkbox.value = False
                if checkbox.value and allowed:
                    selected_left.append(label)

            if len(selected_left) > 2:
                selected_left = selected_left[:2]
                self._set_custom_left_axis_selections(selected_left)

            limit_reached = len(selected_left) >= 2
            for checkbox in self._custom_left_axis_checkboxes():
                label = str(checkbox.label)
                allowed = label in eligible_left
                checkbox.disabled = (not allowed) or (limit_reached and not checkbox.value)

            right_axis = self.query_one("#select_custom_right_axis", Select)
            current_right_axis = str(right_axis.value)
            right_axis_options = [( "None", CUSTOM_GRAPH_NONE )]
            right_axis_options.extend(
                (variable, variable)
                for variable in eligible_right_axis_variables(
                    x_domain=x_domain,
                    left_variables=selected_left,
                    analysis_ready=analysis_ready,
                )
            )
            self._set_custom_select_options(
                "#select_custom_right_axis",
                right_axis_options,
                current_right_axis,
                disabled=len(right_axis_options) == 1,
            )

            segment_select = self.query_one("#select_custom_segment", Select)
            current_segment = str(segment_select.value)
            segment_keys = [
                key
                for key in eligible_segment_keys(x_domain=x_domain, analysis_ready=analysis_ready)
                if _custom_graph_overlay_available(self.qc_df, key)
            ]
            if current_segment not in segment_keys:
                current_segment = CUSTOM_GRAPH_NONE
            segment_options = [("None", CUSTOM_GRAPH_NONE)]
            segment_options.extend((OVERLAY_COMPATIBILITY[key].label, key) for key in segment_keys)
            self._set_custom_select_options(
                "#select_custom_segment",
                segment_options,
                current_segment,
                disabled=(view_domain != "semantic_segment") or (not analysis_ready) or len(segment_options) == 1,
            )

            segment_label = self.query_one("#label_custom_segment", Label)
            active_segment = str(self.query_one("#select_custom_segment", Select).value)
            show_segment = view_domain == "semantic_segment"
            segment_label.display = show_segment
            segment_select.display = show_segment

            annotation_select = self.query_one("#select_custom_annotation", Select)
            current_annotation = str(annotation_select.value)
            annotation_keys = (
                [
                    key
                    for key in eligible_annotation_keys(active_segment, selected_left)
                    if _custom_graph_overlay_available(self.qc_df, key)
                ]
                if active_segment != CUSTOM_GRAPH_NONE
                else []
            )
            annotation_options = [("None", CUSTOM_GRAPH_NONE)]
            annotation_options.extend((self._annotation_label(key), key) for key in annotation_keys)
            self._set_custom_select_options(
                "#select_custom_annotation",
                annotation_options,
                current_annotation,
                disabled=(not show_segment) or active_segment == CUSTOM_GRAPH_NONE or len(annotation_options) == 1,
            )

            data_scope = str(self.query_one("#select_custom_data_scope", Select).value)
            sample_label = self.query_one("#label_custom_sample_list", Label)
            sample_list = self.query_one("#custom_graph_sample_list", OptionList)
            selected_samples_mode = data_scope == "selected_samples"
            self._refresh_custom_graph_sample_list()
            available_samples = self._available_custom_graph_samples()
            sample_label.display = selected_samples_mode
            sample_list.display = selected_samples_mode
            sample_list.disabled = (not selected_samples_mode) or not available_samples

            display_mode = self.query_one("#select_custom_display_mode", Select)
            display_mode.disabled = not selected_samples_mode

            right_axis_value = str(self.query_one("#select_custom_right_axis", Select).value)
            annotation_value = str(self.query_one("#select_custom_annotation", Select).value)
            selected_samples = self._selected_custom_graph_samples()
            summary = (
                f"View domain: {'Semantic segment' if view_domain == 'semantic_segment' else 'Full curve'}\n"
                f"Segment: {OVERLAY_COMPATIBILITY.get(active_segment).label if active_segment in OVERLAY_COMPATIBILITY else 'None'}\n"
                f"Curves: {', '.join(selected_left) if selected_left else 'None'}"
                f"{'' if right_axis_value == CUSTOM_GRAPH_NONE else f' | Right axis: {right_axis_value}'}\n"
                f"Annotations: {self._annotation_label(annotation_value)}\n"
                f"Data scope: {'Selected samples' if selected_samples_mode else 'Grouped'}"
                f"{'' if not selected_samples_mode else f' ({len(selected_samples)} selected)'}\n"
                f"Display mode: {str(display_mode.value)}"
            )
            if view_domain == "semantic_segment" and active_segment == CUSTOM_GRAPH_NONE:
                summary += "\nWarning: choose a segment before adding the graph."
            if selected_samples_mode and not selected_samples:
                summary += "\nWarning: no samples selected."
            summary += (
                f"\nX domain: {x_domain}\n"
                f"Right axis: {right_axis_value if right_axis_value != CUSTOM_GRAPH_NONE else 'None'}\n"
                f"Band type: {str(self.query_one('#select_band_mode', Select).value).upper()}"
            )
            self.query_one("#custom_graph_summary", Static).update(summary)
        finally:
            self._syncing_custom_graph_builder = False

    def _collect_graph_spec_from_ui(self) -> CustomGraphSpec:
        """Collect and validate a composed custom graph specification from the UI."""
        left_axis = [
            CustomGraphAxisLayer(variable=variable, role="left")
            for variable in self._selected_custom_left_axis_variables()
        ]
        view_domain = str(self.query_one("#select_custom_view_domain", Select).value)
        right_axis_value = str(self.query_one("#select_custom_right_axis", Select).value)
        segment_value = str(self.query_one("#select_custom_segment", Select).value)
        annotation_value = str(self.query_one("#select_custom_annotation", Select).value)
        data_scope = str(self.query_one("#select_custom_data_scope", Select).value)

        spec = CustomGraphSpec(
            title=self.query_one("#input_graph_title", Input).value.strip() or "Custom Graph",
            x_domain=str(self.query_one("#select_custom_x_domain", Select).value),
            left_axis=left_axis,
            right_axis=(
                None
                if right_axis_value == CUSTOM_GRAPH_NONE
                else CustomGraphAxisLayer(variable=right_axis_value, role="right")
            ),
            view_domain=view_domain,
            segment_key=None if view_domain == "full_curve" or segment_value == CUSTOM_GRAPH_NONE else segment_value,
            rebase_x=view_domain == "semantic_segment",
            annotations=[]
            if annotation_value == CUSTOM_GRAPH_NONE
            else [CustomGraphAnnotation(kind="annotation", key=annotation_value)],
            data_scope=data_scope,
            selected_samples=[] if data_scope == "grouped" else self._selected_custom_graph_samples(),
            display_mode=str(self.query_one("#select_custom_display_mode", Select).value),
            enabled=True,
            band_mode=str(self.query_one("#select_band_mode", Select).value),
        )
        expand_composed_graph_spec(spec)
        return spec

    def _render_graph_specs(self) -> None:
        """Render the current custom graph specification summary."""
        if not self._widgets_ready():
            return
        if not self.graph_specs:
            self.query_one("#graph-spec-list", Static).update("(No custom graphs yet)")
            return
        lines = []
        for index, spec in enumerate(self.graph_specs, start=1):
            if isinstance(spec, CustomGraphSpec):
                left_axis = ", ".join(layer.variable for layer in spec.left_axis) or "None"
                right_axis = spec.right_axis.variable if spec.right_axis is not None else "None"
                segment = OVERLAY_COMPATIBILITY.get(spec.segment_key or "")
                annotation = ", ".join(self._annotation_label(item.key) for item in spec.annotations) or "None"
                scope = "Selected samples" if spec.data_scope == "selected_samples" else "Grouped"
                if spec.data_scope == "selected_samples" and spec.selected_samples:
                    scope = f"{scope} ({', '.join(spec.selected_samples)})"
                lines.append(
                    f"{index}. {spec.title}\n"
                    f"   View domain: {'Semantic segment' if spec.view_domain == 'semantic_segment' else 'Full curve'}\n"
                    f"   Segment: {segment.label if segment is not None else 'None'}\n"
                    f"   Curves: {left_axis}\n"
                    f"   Right axis: {right_axis}\n"
                    f"   Annotations: {annotation}\n"
                    f"   Data scope: {scope}\n"
                    f"   Display mode: {spec.display_mode}"
                )
                continue

            lines.append(
                f"{index}. {spec.title}\n"
                f"   Recipe: {spec.plot_type} graph\n"
                f"   Anchor domain: {', '.join(spec.x_cols) or 'None'}\n"
                f"   Primary axis: {', '.join(spec.y_cols) or 'None'}\n"
                f"   Secondary axis: None\n"
                f"   Overlay: None"
            )
        self.query_one("#graph-spec-list", Static).update("\n".join(lines))

    def _rebuild_results_table(self) -> None:
        """Rebuild the metrics results table from the latest analysis."""
        if not self._widgets_ready():
            return
        table = self.query_one("#results_table", DataTable)
        table.clear(columns=False)
        if self.metrics_df.empty:
            return
        for _, row in self.metrics_df.iterrows():
            values = [str(row.get("Filename", "")), str(row.get("Group", ""))]
            values.extend(str(row.get(metric, "")) for metric in COMPUTED_METRICS)
            table.add_row(*values)

    @on(Button.Pressed, "#btn_refresh")
    def handle_refresh(self) -> None:
        """Refresh the selected input directory."""
        self._refresh_directory()

    @on(DataTable.RowSelected, "#file_list")
    def handle_file_selected(self, event: DataTable.RowSelected) -> None:
        """Toggle the highlighted file row in the current selection."""
        if not self.file_records:
            return
        self._toggle_file_selection_with_status(int(event.cursor_row))

    @on(DataTable.RowHighlighted, "#file_list")
    def handle_file_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Update the highlighted file preview."""
        if self.file_records:
            self._set_selected_file(int(event.cursor_row), update_status=False)

    @on(OptionList.OptionHighlighted, "#group_order_list")
    def handle_group_order_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Track the currently highlighted group-order row."""
        if not self.group_order:
            self.selected_group_order_index = None
        elif 0 <= int(event.option_index) < len(self.group_order):
            self.selected_group_order_index = int(event.option_index)
            self._set_input_if_present("#input_group_name", self.group_order[self.selected_group_order_index])
        self._update_group_order_buttons()
        self._update_assignment_buttons()

    @on(Button.Pressed, "#btn_group_up")
    def handle_group_up(self) -> None:
        """Move the selected group upward in display order."""
        idx = self.selected_group_order_index
        if idx is None or idx <= 0 or idx >= len(self.group_order):
            return
        self.group_order[idx - 1], self.group_order[idx] = self.group_order[idx], self.group_order[idx - 1]
        self.selected_group_order_index = idx - 1
        self._render_group_order_list()
        self._update_color_group_select()
        self._update_group_summary()
        self._reorder_existing_stats_results()
        invalidated = self._invalidate_analysis_results_for_grouping_change()
        status = "Moved selected group up."
        if invalidated:
            status += " Grouping changed. Run analysis again before exporting."
        self._set_status(status)
        self._autosave_session()

    @on(Button.Pressed, "#btn_group_down")
    def handle_group_down(self) -> None:
        """Move the selected group downward in display order."""
        idx = self.selected_group_order_index
        if idx is None or idx < 0 or idx >= len(self.group_order) - 1:
            return
        self.group_order[idx + 1], self.group_order[idx] = self.group_order[idx], self.group_order[idx + 1]
        self.selected_group_order_index = idx + 1
        self._render_group_order_list()
        self._update_color_group_select()
        self._update_group_summary()
        self._reorder_existing_stats_results()
        invalidated = self._invalidate_analysis_results_for_grouping_change()
        status = "Moved selected group down."
        if invalidated:
            status += " Grouping changed. Run analysis again before exporting."
        self._set_status(status)
        self._autosave_session()

    @on(Button.Pressed, "#btn_group_add")
    def handle_group_add(self) -> None:
        """Add a new explicit group."""
        group_name = self.query_one("#input_group_name", Input).value.strip()
        if not group_name:
            self._set_status("Group name cannot be empty.")
            return
        had_results = self._has_cached_analysis_results()
        if not self._add_group(group_name):
            self._set_status(f"Group '{group_name}' already exists.")
            return
        status = f"Added group '{group_name}'."
        if had_results:
            status += " Grouping changed. Run analysis again before exporting."
        self._set_status(status)
        self._autosave_session()

    @on(Button.Pressed, "#btn_group_rename")
    def handle_group_rename(self) -> None:
        """Rename the selected explicit group."""
        current_group = self._selected_group_name()
        if current_group is None:
            self._set_status("Select a group first.")
            return
        new_name = self.query_one("#input_group_name", Input).value.strip()
        if not new_name:
            self._set_status("Group name cannot be empty.")
            return
        had_results = self._has_cached_analysis_results()
        if not self._rename_group(current_group, new_name):
            if current_group == new_name:
                self._set_status("Enter a different group name.")
            else:
                self._set_status(f"Cannot rename '{current_group}' to '{new_name}'.")
            return
        status = f"Renamed group '{current_group}' to '{new_name}'."
        if had_results:
            status += " Grouping changed. Run analysis again before exporting."
        self._set_status(status)
        self._autosave_session()

    @on(Button.Pressed, "#btn_group_delete")
    def handle_group_delete(self) -> None:
        """Delete the selected explicit group."""
        current_group = self._selected_group_name()
        if current_group is None:
            self._set_status("Select a group first.")
            return
        had_results = self._has_cached_analysis_results()
        if not self._delete_group(current_group):
            self._set_status(f"Could not delete group '{current_group}'.")
            return
        status = f"Deleted group '{current_group}'."
        if had_results:
            status += " Grouping changed. Run analysis again before exporting."
        self._set_status(status)
        self._autosave_session()

    @on(Button.Pressed, "#btn_assign_active")
    def handle_assign_active(self) -> None:
        """Assign selected files to the active group."""
        current_group = self._active_group_name()
        if current_group is None:
            self._set_status("Select a group first.")
            return
        if not self.selected_file_indices:
            self._set_status("Select one or more files first.")
            return
        had_results = self._has_cached_analysis_results()
        updated = self._assign_selected_files_to_active_group()
        if updated == 0:
            self._set_status(f"Selected files already belong to '{current_group}'.")
            return
        status = f"Assigned {updated} file(s) to '{current_group}'."
        if had_results:
            status += " Grouping changed. Run analysis again before exporting."
        self._set_status(status)
        self._autosave_session()

    @on(Button.Pressed, "#btn_clear_assignment")
    def handle_clear_assignment(self) -> None:
        """Clear selected file assignments."""
        if not self.selected_file_indices:
            self._set_status("Select one or more files first.")
            return
        had_results = self._has_cached_analysis_results()
        updated = self._clear_selected_file_assignments()
        if updated == 0:
            self._set_status("Selected files are already unassigned.")
            return
        status = f"Cleared assignments for {updated} file(s)."
        if had_results:
            status += " Grouping changed. Run analysis again before exporting."
        self._set_status(status)
        self._autosave_session()

    @on(Select.Changed, "#select_color_group")
    def handle_color_group_changed(self, event: Select.Changed) -> None:
        """Update the color input when the selected color group changes."""
        group = str(event.value)
        if group != "__none__":
            self._sync_color_inputs_for_group(group)
            self._autosave_session()

    @on(Select.Changed, "#select_custom_view_domain")
    @on(Select.Changed, "#select_custom_segment")
    @on(Select.Changed, "#select_custom_annotation")
    @on(Select.Changed, "#select_custom_x_domain")
    @on(Select.Changed, "#select_custom_right_axis")
    @on(Select.Changed, "#select_custom_data_scope")
    @on(Select.Changed, "#select_custom_display_mode")
    def handle_custom_graph_select_changed(self, event: Select.Changed) -> None:
        """Refresh custom graph builder state after a builder select changes."""
        if self._syncing_custom_graph_builder:
            return
        if self._consume_custom_graph_internal_update(event.select.id):
            return
        self._sync_custom_graph_builder_state()
        self._autosave_session()

    @on(OptionList.OptionSelected, "#custom_graph_sample_list")
    def handle_custom_graph_sample_selected(self, event: OptionList.OptionSelected) -> None:
        """Toggle the currently chosen plot-builder sample."""
        if self._syncing_custom_graph_builder:
            return
        if not self._toggle_custom_graph_sample_by_index(int(event.option_index)):
            return
        self._sync_custom_graph_builder_state()
        self._autosave_session()

    @on(Checkbox.Changed, "#custom_left_axis_choices Checkbox")
    def handle_custom_left_axis_changed(self, event: Checkbox.Changed) -> None:
        """Keep the custom left-axis list constrained to two active selections."""
        if self._syncing_custom_graph_builder:
            return
        if event.checkbox.value and len(self._selected_custom_left_axis_variables()) > 2:
            event.checkbox.value = False
            self._set_status("Select at most two left-axis variables.")
            return
        if event.checkbox.value:
            selected_left = self._selected_custom_left_axis_variables()
            if not _left_axis_variables_compatible(selected_left):
                event.checkbox.value = False
                self._set_status("Left-axis variables must share the same unit.")
                return
        self._sync_custom_graph_builder_state()
        self._autosave_session()

    @on(Button.Pressed)
    def handle_parameter_info_click(self, event: Button.Pressed) -> None:
        """Show contextual help for parameter info buttons."""
        button_id = event.button.id or ""
        if not button_id.startswith("info_"):
            return
        info_key = button_id.removeprefix("info_")
        info = PARAM_INFO.get(info_key)
        if info is None:
            self._set_status("No help text available for this parameter.")
            return
        self.push_screen(ParameterInfoModal(title=info["label"], body=info["help"]))
        event.stop()

    @on(Button.Pressed, "#btn_apply_colors")
    def apply_colors(self) -> None:
        """Apply the custom color override for the selected group."""
        group = str(self.query_one("#select_color_group", Select).value)
        if group == "__none__":
            self._set_status("No group selected for color update.")
            return
        group_hex = self.query_one("#input_group_hex", Input).value.strip().upper()
        if not is_hex_color(group_hex):
            self._set_status("Invalid hex color. Use #RRGGBB format.")
            return
        self.plot_style.group_colors[group] = group_hex
        self._set_status(f"Applied custom color to group '{group}'.")
        self._autosave_session()

    @on(Button.Pressed, "#btn_reset_palette")
    def reset_palette(self) -> None:
        """Reset group color overrides to the default palette."""
        self.plot_style = PlotStyleConfig()
        self.plot_style.ensure_group_colors(self.group_order)
        group = str(self.query_one("#select_color_group", Select).value)
        if group != "__none__":
            self._sync_color_inputs_for_group(group)
        self._set_status("Reset color overrides to auto palette.")
        self._autosave_session()

    @on(Button.Pressed, "#btn_add_graph")
    def add_graph_spec(self) -> None:
        """Add the current graph builder selection to saved specs."""
        try:
            spec = self._collect_graph_spec_from_ui()
        except PlotSpecError as exc:
            self._set_status(str(exc))
            return
        self.graph_specs.append(spec)
        self._render_graph_specs()
        self._set_status(f"Added graph spec: {spec.title}")
        self._autosave_session()

    @on(Button.Pressed, "#btn_clear_graphs")
    def clear_graph_specs(self) -> None:
        """Clear all saved custom graph specifications."""
        self.graph_specs.clear()
        self._render_graph_specs()
        self._set_status("Cleared custom graph specs.")
        self._autosave_session()

    @on(Input.Changed, "#input_width")
    @on(Input.Changed, "#input_height_fig")
    @on(Input.Changed, "#input_dpi")
    @on(Select.Changed, "#select_ratio")
    def handle_figure_inputs_changed(self) -> None:
        """Refresh figure preview when size inputs change."""
        self._update_figure_preview()
        self._autosave_session()

    @on(Input.Changed)
    def handle_persistent_input_changed(self, event: Input.Changed) -> None:
        """Autosave any input change."""
        _ = event
        self._autosave_session()

    @on(Checkbox.Changed)
    def handle_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Autosave any checkbox change."""
        if event.checkbox.id is None and event.checkbox.has_class("custom-left-axis-choice"):
            return
        self._autosave_session()

    @on(Select.Changed)
    def handle_persistent_select_changed(self, event: Select.Changed) -> None:
        """Autosave select changes not handled elsewhere."""
        if event.select.id in {
            "select_custom_view_domain",
            "select_custom_segment",
            "select_custom_annotation",
            "select_custom_x_domain",
            "select_custom_right_axis",
            "select_custom_data_scope",
            "select_custom_display_mode",
        }:
            return
        self._autosave_session()

    @on(Button.Pressed, "#btn_analyze")
    def trigger_analysis(self) -> None:
        """Start the analysis worker."""
        if not self.file_records:
            self._set_status("No files loaded.")
            return
        self._sync_group_order_from_records()
        self.run_analysis_worker(
            config=self._collect_tpa_config(),
            stats_mode=str(self.query_one("#select_stats_mode", Select).value),
            records=[record.copy() for record in self.file_records],
            group_order=self.group_order.copy(),
        )

    @work(thread=True, exclusive=True)
    def run_analysis_worker(
        self,
        config: TPAConfig,
        stats_mode: str,
        records: list[dict[str, str]],
        group_order: list[str],
    ) -> None:
        """Analyze all files in the current directory."""
        self.call_from_thread(self._set_buttons_disabled, True)
        self.call_from_thread(self._set_status, "Running analysis...")
        self.call_from_thread(self._log, f"Analysis started with config: {asdict(config)}")

        metric_rows: list[dict[str, Any]] = []
        traces: list[pd.DataFrame] = []
        qc_rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        failures: list[str] = []

        for idx, record in enumerate(records, start=1):
            filename = record["filename"]
            group = record["group"]
            path = record["path"]
            self.call_from_thread(self._set_status, f"Processing {idx}/{len(records)}: {filename}")
            try:
                parsed = parse_zwick_data(path)
                result = calculate_tpa(parsed, config=config, file_id=filename, group=group)
                metric_rows.append(build_metrics_row(result, filename, group))
                trace = result.get("Trace Data")
                if isinstance(trace, pd.DataFrame) and not trace.empty:
                    traces.append(trace)
                qc_summary = result.get("QC Summary")
                if isinstance(qc_summary, dict):
                    qc_rows.append(qc_summary)
                warnings.extend([f"{filename}: {warning}" for warning in result.get("Warnings", [])])
            except (DataParseError, AnalysisError) as exc:
                failures.append(f"{filename}: {exc}")
            except Exception as exc:
                self.logger.exception("Unexpected analysis failure", extra={"filename": filename, "group": group})
                failures.append(f"{filename}: {exc}")

        metrics_df = pd.DataFrame(metric_rows)
        trace_df = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
        qc_df = pd.DataFrame(qc_rows)
        stats_results: dict[str, dict[str, Any]] = {}
        if not metrics_df.empty:
            for metric in COMPUTED_METRICS:
                if metric not in metrics_df.columns:
                    continue
                metric_frame = metrics_df[["Group", metric]].dropna()
                if metric_frame.empty:
                    continue
                try:
                    stats_results[metric] = run_statistics(metric_frame, group_col="Group", metric_col=metric, alpha=0.05, mode=stats_mode, group_order=group_order)
                except Exception as exc:
                    warnings.append(f"Stats failed for {metric}: {exc}")

        self.call_from_thread(self._apply_analysis_results, metrics_df, trace_df, qc_df, stats_results, warnings, failures)
        self.call_from_thread(self._set_buttons_disabled, False)

    def _apply_analysis_results(
        self,
        metrics_df: pd.DataFrame,
        trace_df: pd.DataFrame,
        qc_df: pd.DataFrame,
        stats_results: dict[str, dict[str, Any]],
        warnings: list[str],
        failures: list[str],
    ) -> None:
        """Apply completed analysis results to UI state."""
        self.metrics_df = metrics_df
        self.trace_df = trace_df
        self.qc_df = qc_df
        self.stats_results = stats_results
        self._sync_group_order_from_records()
        self._rebuild_results_table()
        self._update_color_group_select()
        if failures:
            self._log(f"Failures ({len(failures)}):")
            for failure in failures[:12]:
                self._log(f"- {failure}")
        if warnings:
            self._log(f"Warnings ({len(warnings)}):")
            for warning in warnings[:12]:
                self._log(f"- {warning}")
        self._sync_custom_graph_builder_state()
        self._set_status(f"Analysis done. Valid files: {len(metrics_df)} | Stats metrics: {len(stats_results)} | Failures: {len(failures)}")
        self._log("Analysis completed.")
        self._autosave_session()

    @on(Button.Pressed, "#btn_export_tables")
    def trigger_export_tables(self) -> None:
        """Export result tables only."""
        if self.metrics_df.empty:
            self._set_status("No analysis results to export.")
            return
        self.export_tables_worker(self.metrics_df.copy(), self.qc_df.copy(), self.stats_results.copy())

    @work(thread=True, exclusive=True)
    def export_tables_worker(
        self,
        metrics_df: pd.DataFrame,
        qc_df: pd.DataFrame,
        stats_results: dict[str, dict[str, Any]],
    ) -> None:
        """Write table exports in a worker thread."""
        self.call_from_thread(self._set_buttons_disabled, True)
        try:
            export_root = current_export_root(self.settings.export_root_name)
            export_tables_bundle(export_root, metrics_df, qc_df, stats_results)
            self.call_from_thread(self._set_status, f"Tables exported to {export_root}")
            self.call_from_thread(self._log, f"Tables exported: {export_root}")
        except Exception as exc:
            self.logger.exception("Table export failed")
            self.call_from_thread(self._set_status, f"Table export failed: {exc}")
            self.call_from_thread(self._log, f"Table export failed: {exc}")
        finally:
            self.call_from_thread(self._set_buttons_disabled, False)

    def _current_plot_style(self) -> PlotStyleConfig:
        """Clone the current plot style state for worker use."""
        style = PlotStyleConfig(
            group_colors=self.plot_style.group_colors.copy(),
            palette_name=self.plot_style.palette_name,
            replicate_alpha=self.plot_style.replicate_alpha,
            replicate_linewidth=self.plot_style.replicate_linewidth,
            mean_linewidth=self.plot_style.mean_linewidth,
        )
        style.ensure_group_colors(self.group_order.copy())
        return style

    @on(Button.Pressed, "#btn_export_plots")
    def trigger_export_plots(self) -> None:
        """Export plot assets only."""
        if self.trace_df.empty:
            self._set_status("No trace data to plot. Run analysis first.")
            return
        trace_df, metrics_df, qc_df, stats_results, group_order = filter_assigned_plot_export_payload(
            trace_df=self.trace_df.copy(),
            metrics_df=self.metrics_df.copy(),
            qc_df=self.qc_df.copy(),
            stats_results=self.stats_results.copy(),
            group_order=self.group_order.copy(),
        )
        self.export_plots_worker(
            trace_df=trace_df,
            metrics_df=metrics_df,
            qc_df=qc_df,
            stats_results=stats_results,
            graph_specs=copy.deepcopy(self.graph_specs),
            style=self._current_plot_style(),
            fig_cfg=self._collect_figure_config(),
            overlay_mode=str(self.query_one("#select_overlay_mode", Select).value),
            band_mode=str(self.query_one("#select_band_mode", Select).value),
            group_order=group_order,
        )

    @work(thread=True, exclusive=True)
    def export_plots_worker(
        self,
        trace_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        qc_df: pd.DataFrame,
        stats_results: dict[str, dict[str, Any]],
        graph_specs: list[GraphSpec | CustomGraphSpec],
        style: PlotStyleConfig,
        fig_cfg: FigureConfig,
        overlay_mode: str,
        band_mode: str,
        group_order: list[str],
    ) -> None:
        """Write plot exports in a worker thread."""
        self.call_from_thread(self._set_buttons_disabled, True)
        try:
            output_root = current_export_root(self.settings.plots_root_name)
            warnings = export_plot_bundle(
                root=output_root,
                trace_df=trace_df,
                metrics_df=metrics_df,
                qc_df=qc_df,
                stats_results=stats_results,
                graph_specs=graph_specs,
                style=style,
                fig_cfg=fig_cfg,
                overlay_mode=overlay_mode,
                band_mode=band_mode,
                group_order=group_order,
                include_plots_dir=False,
            )
            for warning in warnings:
                self.call_from_thread(self._log, warning)
            self.call_from_thread(self._set_status, f"Plots exported to {output_root}")
            self.call_from_thread(self._log, f"Plots exported: {output_root}")
        except Exception as exc:
            self.logger.exception("Plot export failed")
            self.call_from_thread(self._set_status, f"Plot export failed: {exc}")
            self.call_from_thread(self._log, f"Plot export failed: {exc}")
        finally:
            self.call_from_thread(self._set_buttons_disabled, False)

    @on(Button.Pressed, "#btn_export_all")
    def trigger_export_all(self) -> None:
        """Export tables and plots together."""
        if self.metrics_df.empty:
            self._set_status("No analysis results. Run analysis first.")
            return
        trace_df, metrics_df, qc_df, stats_results, group_order = filter_assigned_plot_export_payload(
            trace_df=self.trace_df.copy(),
            metrics_df=self.metrics_df.copy(),
            qc_df=self.qc_df.copy(),
            stats_results=self.stats_results.copy(),
            group_order=self.group_order.copy(),
        )
        self.export_all_worker(
            metrics_df=metrics_df,
            trace_df=trace_df,
            qc_df=qc_df,
            stats_results=stats_results,
            graph_specs=copy.deepcopy(self.graph_specs),
            style=self._current_plot_style(),
            fig_cfg=self._collect_figure_config(),
            overlay_mode=str(self.query_one("#select_overlay_mode", Select).value),
            band_mode=str(self.query_one("#select_band_mode", Select).value),
            group_order=group_order,
        )

    @work(thread=True, exclusive=True)
    def export_all_worker(
        self,
        metrics_df: pd.DataFrame,
        trace_df: pd.DataFrame,
        qc_df: pd.DataFrame,
        stats_results: dict[str, dict[str, Any]],
        graph_specs: list[GraphSpec | CustomGraphSpec],
        style: PlotStyleConfig,
        fig_cfg: FigureConfig,
        overlay_mode: str,
        band_mode: str,
        group_order: list[str],
    ) -> None:
        """Write tables and plots into a combined export bundle."""
        self.call_from_thread(self._set_buttons_disabled, True)
        try:
            root = current_export_root(self.settings.export_root_name)
            export_tables_bundle(root, metrics_df, qc_df, stats_results)
            warnings = export_plot_bundle(
                root=root,
                trace_df=trace_df,
                metrics_df=metrics_df,
                qc_df=qc_df,
                stats_results=stats_results,
                graph_specs=graph_specs,
                style=style,
                fig_cfg=fig_cfg,
                overlay_mode=overlay_mode,
                band_mode=band_mode,
                group_order=group_order,
                include_plots_dir=True,
            )
            for warning in warnings:
                self.call_from_thread(self._log, warning)
            self.call_from_thread(self._set_status, f"Export all completed: {root}")
            self.call_from_thread(self._log, f"Export all completed: {root}")
        except Exception as exc:
            self.logger.exception("Export all failed")
            self.call_from_thread(self._set_status, f"Export all failed: {exc}")
            self.call_from_thread(self._log, f"Export all failed: {exc}")
        finally:
            self.call_from_thread(self._set_buttons_disabled, False)
