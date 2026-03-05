from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import pandas as pd
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
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

from stats_engine import run_statistics
from tpa_engine import TPAConfig, calculate_tpa, infer_group_from_filename, parse_zwick_data
from viz_engine import FigureConfig, GraphSpec, PlotStyleConfig, VARIABLE_REGISTRY, plot_custom_graphs, plot_grouped_metrics, plot_overlay_traces, plot_trace_stack


def _is_hex_color(value: str) -> bool:
    return bool(re.fullmatch(r"#[0-9a-fA-F]{6}", (value or "").strip()))


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
        "help": "Force threshold used to detect start/end of compression cycles. Higher values can trim low-force regions.",
    },
    "peak_prominence": {
        "label": "Peak Prominence (N)",
        "help": "Minimum prominence required for peaks. Increase it to ignore noise and minor local peaks.",
    },
    "peak_distance": {
        "label": "Peak Distance (pts)",
        "help": "Minimum spacing between detected peaks in data points. Helps separate first and second compression peaks.",
    },
    "modulus_min": {
        "label": "Modulus Strain Min (%)",
        "help": "Lower strain bound for modulus fitting window during first compression.",
    },
    "modulus_max": {
        "label": "Modulus Strain Max (%)",
        "help": "Upper strain bound for modulus fitting window during first compression.",
    },
    "stats_mode": {
        "label": "Stats Mode",
        "help": "Select statistical test family. Auto chooses parametric or nonparametric based on assumptions.",
    },
}


class ParameterInfoModal(ModalScreen[None]):
    """Simple click-to-read accessibility help modal for analysis parameters."""

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
        border: round #64748B;
        background: #FFFFFF;
        padding: 1 2;
    }
    #param-help-title {
        text-style: bold;
        color: #2563EB;
        margin-bottom: 1;
    }
    #param-help-body {
        margin-bottom: 1;
    }
    """

    def __init__(self, title: str, body: str):
        super().__init__()
        self.title = title
        self.body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="param-help-root"):
            yield Static(self.title, id="param-help-title")
            yield Static(self.body, id="param-help-body")
            yield Static("Press Esc, Enter, or q to close.", classes="small-label")

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)


class TPAAnalyzerApp(App):
    """Terminal TPA analyzer with grouped statistics and advanced plot studio controls."""

    CSS = """
    Screen {
        background: #F4F6F8;
        color: #0F172A;
    }

    #studio {
        height: 1fr;
    }

    .pane {
        border: round #CBD5E1;
        background: #FFFFFF;
        padding: 1;
        margin: 1;
    }

    #left-pane {
        width: 27%;
        min-width: 34;
    }

    #center-pane {
        width: 44%;
        min-width: 48;
    }

    #right-pane {
        width: 29%;
        min-width: 42;
    }

    .section-title {
        text-style: bold;
        color: #2563EB;
        margin-bottom: 1;
    }

    .small-label {
        color: #64748B;
    }

    .param-row {
        height: auto;
        align: left middle;
        margin-bottom: 0;
    }

    .info-btn {
        width: 3;
        min-width: 3;
        height: 1;
        margin-left: 1;
        padding: 0;
        background: #EEF2FF;
        color: #1D4ED8;
        border: round #BFDBFE;
        text-style: bold;
    }

    #file_list {
        height: 1fr;
        border: round #CBD5E1;
        margin-bottom: 1;
        min-height: 8;
    }

    #group_order_list {
        height: 6;
        border: round #CBD5E1;
        margin-bottom: 1;
    }

    DataTable {
        height: 1fr;
        margin-bottom: 1;
    }

    RichLog {
        height: 10;
        border: round #CBD5E1;
    }

    .action-row {
        height: auto;
        margin-top: 1;
    }

    Input, Select {
        margin-bottom: 1;
    }

    #status-msg {
        color: #64748B;
        margin-top: 1;
    }

    #graph-spec-list {
        border: round #CBD5E1;
        padding: 1;
        height: 8;
        overflow-y: auto;
    }

    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.base_dir = Path.cwd()
        self.file_records: list[dict[str, str]] = []
        self.selected_file_index: int | None = None
        self.group_order: list[str] = []
        self.selected_group_order_index: int | None = None

        self.metrics_df = pd.DataFrame()
        self.trace_df = pd.DataFrame()
        self.stats_results: dict[str, dict[str, Any]] = {}

        self.plot_style = PlotStyleConfig()
        self.graph_specs: list[GraphSpec] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="studio"):
            with Vertical(id="left-pane", classes="pane"):
                yield Static("Data & Grouping", classes="section-title")

                yield Label("Directory", classes="small-label")
                yield Input(value=".", id="input_dir", placeholder="Relative or absolute path")
                yield Button("Refresh Directory", id="btn_refresh", variant="primary")

                yield Label("Detected Files", classes="small-label")
                yield OptionList(id="file_list", markup=False)
                yield Static("Selected: none", id="selected_file_info", classes="small-label")
                yield Static("Groups: none", id="group_summary", classes="small-label")
                yield Label("Group Display Order", classes="small-label")
                yield OptionList(id="group_order_list", markup=False)
                with Horizontal(classes="action-row"):
                    yield Button("Group Up", id="btn_group_up", variant="default")
                    yield Button("Group Down", id="btn_group_down", variant="default")

                yield Label("Group Name", classes="small-label")
                yield Input(id="input_group", placeholder="Assign selected file to group")
                yield Label("Batch Match Terms (comma-separated)", classes="small-label")
                yield Input(id="input_group_filter", placeholder="e.g., n1, n2, n3 (blank = all files)")
                with Horizontal(classes="action-row"):
                    yield Button("Assign + Next", id="btn_assign_group", variant="default")
                    yield Button("Assign Terms", id="btn_assign_matching", variant="primary")

                yield Static("Color Style", classes="section-title")
                yield Label("Target Group", classes="small-label")
                yield Select([("No groups", "__none__")], id="select_color_group", allow_blank=False)

                yield Label("Force Hex", classes="small-label")
                yield Input(value="#2563EB", id="input_force_hex", placeholder="#RRGGBB")

                yield Label("Deformation Hex", classes="small-label")
                yield Input(value="#059669", id="input_defo_hex", placeholder="#RRGGBB")

                yield Label("Stress Hex", classes="small-label")
                yield Input(value="#D97706", id="input_stress_hex", placeholder="#RRGGBB")

                with Horizontal(classes="action-row"):
                    yield Button("Apply Colors", id="btn_apply_colors", variant="success")
                    yield Button("Reset Palette", id="btn_reset_palette", variant="warning")

            with Vertical(id="center-pane", classes="pane"):
                yield Static("Analysis Results", classes="section-title")
                yield DataTable(id="results_table")
                yield Static("Status", classes="section-title")
                yield RichLog(id="log_stream", markup=True)
                yield Static("Ready", id="status-msg")

            with Vertical(id="right-pane", classes="pane"):
                yield Static("Controls", classes="section-title")
                with TabbedContent(id="right-tabs"):
                    with TabPane("Analysis Params"):
                        with Horizontal(classes="param-row"):
                            yield Label(PARAM_INFO["sample_height"]["label"], classes="small-label")
                            yield Button("i", id="info_sample_height", classes="info-btn", tooltip=PARAM_INFO["sample_height"]["help"])
                        yield Input(value="10.0", id="input_height")
                        with Horizontal(classes="param-row"):
                            yield Label(PARAM_INFO["contact_area"]["label"], classes="small-label")
                            yield Button("i", id="info_contact_area", classes="info-btn", tooltip=PARAM_INFO["contact_area"]["help"])
                        yield Input(value="100.0", id="input_area")
                        with Horizontal(classes="param-row"):
                            yield Label(PARAM_INFO["baseline_points"]["label"], classes="small-label")
                            yield Button("i", id="info_baseline_points", classes="info-btn", tooltip=PARAM_INFO["baseline_points"]["help"])
                        yield Input(value="10", id="input_baseline_points")
                        with Horizontal(classes="param-row"):
                            yield Label(PARAM_INFO["trigger_force"]["label"], classes="small-label")
                            yield Button("i", id="info_trigger_force", classes="info-btn", tooltip=PARAM_INFO["trigger_force"]["help"])
                        yield Input(value="0.05", id="input_trigger")
                        with Horizontal(classes="param-row"):
                            yield Label(PARAM_INFO["peak_prominence"]["label"], classes="small-label")
                            yield Button("i", id="info_peak_prominence", classes="info-btn", tooltip=PARAM_INFO["peak_prominence"]["help"])
                        yield Input(value="0.5", id="input_prominence")
                        with Horizontal(classes="param-row"):
                            yield Label(PARAM_INFO["peak_distance"]["label"], classes="small-label")
                            yield Button("i", id="info_peak_distance", classes="info-btn", tooltip=PARAM_INFO["peak_distance"]["help"])
                        yield Input(value="200", id="input_peak_distance")
                        with Horizontal(classes="param-row"):
                            yield Label(PARAM_INFO["modulus_min"]["label"], classes="small-label")
                            yield Button("i", id="info_modulus_min", classes="info-btn", tooltip=PARAM_INFO["modulus_min"]["help"])
                        yield Input(value="10", id="input_mod_min")
                        with Horizontal(classes="param-row"):
                            yield Label(PARAM_INFO["modulus_max"]["label"], classes="small-label")
                            yield Button("i", id="info_modulus_max", classes="info-btn", tooltip=PARAM_INFO["modulus_max"]["help"])
                        yield Input(value="30", id="input_mod_max")
                        with Horizontal(classes="param-row"):
                            yield Label(PARAM_INFO["stats_mode"]["label"], classes="small-label")
                            yield Button("i", id="info_stats_mode", classes="info-btn", tooltip=PARAM_INFO["stats_mode"]["help"])
                        yield Select(
                            [
                                ("Auto", "auto"),
                                ("Parametric", "parametric"),
                                ("Nonparametric", "nonparametric"),
                            ],
                            value="auto",
                            id="select_stats_mode",
                            allow_blank=False,
                        )
                        yield Button("Run Analysis", id="btn_analyze", variant="primary")

                    with TabPane("Plot Builder"):
                        x_options = [(key, key) for key, meta in VARIABLE_REGISTRY.items() if meta["kind"] == "x"]
                        yield Label("X Variable", classes="small-label")
                        yield Select(x_options, value="Time (s)", id="select_x_var", allow_blank=False)
                        yield Label("Y Variables (comma-separated)", classes="small-label")
                        yield Input(
                            value="Force (N), Deformation (mm)",
                            id="input_y_vars",
                            placeholder="Force (N), Deformation (mm)",
                        )
                        yield Label("Graph Mode", classes="small-label")
                        yield Select(
                            [("Panel", "panel"), ("Overlay", "overlay")],
                            value="panel",
                            id="select_graph_mode",
                            allow_blank=False,
                        )
                        yield Label("Curve Mode", classes="small-label")
                        yield Select(
                            [
                                ("Individual Replicates", "individual"),
                                ("Mean + Band", "mean_band"),
                                ("Both", "both"),
                            ],
                            value="mean_band",
                            id="select_curve_mode",
                            allow_blank=False,
                        )
                        yield Label("Band Type", classes="small-label")
                        yield Select(
                            [("SD", "sd"), ("95% CI", "ci95")],
                            value="sd",
                            id="select_band_mode",
                            allow_blank=False,
                        )
                        yield Label("Graph Title", classes="small-label")
                        yield Input(value="Custom Graph", id="input_graph_title")
                        with Horizontal(classes="action-row"):
                            yield Button("Add Graph", id="btn_add_graph", variant="default")
                            yield Button("Clear Graphs", id="btn_clear_graphs", variant="warning")
                        yield Static("(No custom graphs yet)", id="graph-spec-list")

                        yield Label("Group Overlay Mode", classes="small-label")
                        yield Select(
                            [
                                ("Mean + Band", "mean_band"),
                                ("Individual", "individual"),
                                ("Both", "both"),
                            ],
                            value="mean_band",
                            id="select_overlay_mode",
                            allow_blank=False,
                        )

                    with TabPane("Style / Theme"):
                        yield Label("Ratio Preset", classes="small-label")
                        yield Select(
                            [
                                ("1:1", "1:1"),
                                ("4:3", "4:3"),
                                ("16:9", "16:9"),
                                ("A4 portrait", "A4 portrait"),
                                ("A4 landscape", "A4 landscape"),
                            ],
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
        table = self.query_one("#results_table", DataTable)
        table.add_columns(
            "Filename",
            "Group",
            "Hardness (N)",
            "Cohesiveness",
            "Springiness",
            "Resilience",
            "Chewiness",
            "Adhesiveness",
            "Modulus (kPa)",
        )

        self._refresh_directory()
        self._update_figure_preview()
        self._log("App started.")

    def _log(self, message: str) -> None:
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        self.query_one("#log_stream", RichLog).write(f"[{timestamp}] {message}")

    def _set_status(self, text: str) -> None:
        self.query_one("#status-msg", Static).update(text)

    def _set_buttons_disabled(self, disabled: bool) -> None:
        for button_id in [
            "#btn_analyze",
            "#btn_export_tables",
            "#btn_export_plots",
            "#btn_export_all",
            "#btn_refresh",
        ]:
            self.query_one(button_id, Button).disabled = disabled
        if disabled:
            self.query_one("#btn_group_up", Button).disabled = True
            self.query_one("#btn_group_down", Button).disabled = True
        else:
            self._update_group_order_buttons()

    def _resolve_directory(self) -> Path:
        raw = self.query_one("#input_dir", Input).value.strip() or "."
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (self.base_dir / path).resolve()
        return path

    def _refresh_directory(self) -> None:
        try:
            directory = self._resolve_directory()
            if not directory.exists() or not directory.is_dir():
                self.file_records = []
                self.selected_file_index = None
                self._render_file_list()
                self._update_color_group_select()
                self._set_status(f"Invalid directory: {directory}")
                self._log(f"Invalid directory: {directory}")
                return

            files = sorted(
                [
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in {".csv", ".tra"}
                ],
                key=lambda item: item.name.lower(),
            )

            self.file_records = [
                {
                    "filename": path.name,
                    "path": str(path),
                    "group": infer_group_from_filename(path.name),
                }
                for path in files
            ]

            self.selected_file_index = None
            self._sync_group_order_from_records()
            self._render_file_list()
            self._update_color_group_select()

            if self.file_records:
                self._set_status(f"Loaded {len(self.file_records)} files from {directory}")
                self._log(f"Directory refreshed: {directory} ({len(self.file_records)} files)")
            else:
                self._set_status(f"No .csv/.tra files found in {directory}")
                self._log(f"Directory refreshed: {directory} (no compatible files)")
        except Exception as exc:
            self.file_records = []
            self.selected_file_index = None
            self._sync_group_order_from_records()
            self._render_file_list()
            self._update_color_group_select()
            self._set_status(f"Refresh failed: {exc}")
            self._log(f"Refresh failed: {exc}")

    def _render_file_list(self, selected_idx: int | None = None) -> None:
        list_view = self.query_one("#file_list", OptionList)
        list_view.clear_options()
        if not self.file_records:
            self.selected_file_index = None
            self.query_one("#input_group", Input).value = ""
            self.query_one("#selected_file_info", Static).update("Selected: none")
            list_view.add_option("No files detected")
            self._update_group_summary()
            return

        options: list[str] = []
        for idx, record in enumerate(self.file_records):
            group = record["group"]
            options.append(f"{idx + 1:02d} | {group:<18} | {record['filename']}")
        list_view.add_options(options)
        if selected_idx is None:
            selected_idx = 0
        selected_idx = max(0, min(selected_idx, len(self.file_records) - 1))
        list_view.highlighted = selected_idx
        self._set_selected_file(selected_idx, update_status=False)
        self._update_group_summary()

    def _set_selected_file(self, idx: int, update_status: bool = True) -> None:
        if idx < 0 or idx >= len(self.file_records):
            return
        self.selected_file_index = idx
        group = self.file_records[idx]["group"]
        filename = self.file_records[idx]["filename"]
        self.query_one("#input_group", Input).value = group
        self.query_one("#selected_file_info", Static).update(f"Selected: {filename} -> {group}")
        if update_status:
            self._set_status(f"Selected {filename}")

    def _update_group_summary(self) -> None:
        summary_widget = self.query_one("#group_summary", Static)
        if not self.file_records:
            summary_widget.update("Groups: none")
            return

        counts: dict[str, int] = {}
        for record in self.file_records:
            group = record["group"].strip() or "UNASSIGNED"
            counts[group] = counts.get(group, 0) + 1

        ordered_group_names = [group for group in self.group_order if group in counts]
        ordered_group_names.extend([group for group in counts.keys() if group not in ordered_group_names])
        ordered = [(group, counts[group]) for group in ordered_group_names]
        pieces = [f"{name}({count})" for name, count in ordered]
        preview = ", ".join(pieces[:6])
        if len(pieces) > 6:
            preview += f", +{len(pieces) - 6} more"
        summary_widget.update(f"Groups: {preview}")

    def _sync_group_order_from_records(self) -> None:
        groups_in_files: list[str] = []
        seen: set[str] = set()
        for record in self.file_records:
            group = record["group"].strip()
            if not group or group in seen:
                continue
            seen.add(group)
            groups_in_files.append(group)

        self.group_order = [group for group in self.group_order if group in seen]
        for group in groups_in_files:
            if group not in self.group_order:
                self.group_order.append(group)

        self.plot_style.ensure_group_colors(self.group_order)
        self._reorder_existing_stats_results()
        self._render_group_order_list()

    def _render_group_order_list(self) -> None:
        list_widget = self.query_one("#group_order_list", OptionList)
        list_widget.clear_options()

        if not self.group_order:
            self.selected_group_order_index = None
            list_widget.add_option("No groups")
            self._update_group_order_buttons()
            return

        options = [f"{idx + 1:02d}. {group}" for idx, group in enumerate(self.group_order)]
        list_widget.add_options(options)

        if self.selected_group_order_index is None:
            self.selected_group_order_index = 0
        self.selected_group_order_index = max(0, min(self.selected_group_order_index, len(self.group_order) - 1))
        list_widget.highlighted = self.selected_group_order_index
        self._update_group_order_buttons()

    def _update_group_order_buttons(self) -> None:
        up = self.query_one("#btn_group_up", Button)
        down = self.query_one("#btn_group_down", Button)
        idx = self.selected_group_order_index

        if idx is None or not self.group_order:
            up.disabled = True
            down.disabled = True
            return

        up.disabled = idx <= 0
        down.disabled = idx >= len(self.group_order) - 1

    def _reorder_existing_stats_results(self) -> None:
        if not self.stats_results:
            return
        for metric, result in self.stats_results.items():
            summary = result.get("summary_df")
            info = result.get("test_info", {})
            if summary is None or summary.empty:
                continue
            group_col = info.get("group_col", "Group")
            if group_col not in summary.columns:
                continue
            order_map = {group: idx for idx, group in enumerate(self.group_order)}
            summary = summary.copy()
            summary["_order"] = summary[group_col].astype(str).map(order_map).fillna(10_000)
            summary = summary.sort_values("_order", ascending=True).drop(columns=["_order"]).reset_index(drop=True)
            result["summary_df"] = summary
            result.setdefault("test_info", {})["group_order"] = self.group_order.copy()

    def _parse_batch_terms(self, raw_text: str) -> list[str]:
        terms = [term.strip().lower() for term in raw_text.split(",")]
        return [term for term in terms if term]

    def _filename_matches_terms(self, filename: str, terms: list[str]) -> bool:
        if not terms:
            return True

        lower_name = filename.lower()
        for term in terms:
            # For alnum terms, use token boundaries so n1 does not match n10.
            if term.isalnum():
                pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
                if re.search(pattern, lower_name):
                    return True
            else:
                if term in lower_name:
                    return True

        return False

    def _update_color_group_select(self) -> None:
        groups = [group for group in self.group_order if group.strip()]
        select = self.query_one("#select_color_group", Select)
        if not groups:
            select.set_options([("No groups", "__none__")])
            select.value = "__none__"
            return

        self.plot_style.ensure_group_colors(groups)
        options = [(group, group) for group in groups]
        select.set_options(options)
        if select.value not in groups:
            select.value = groups[0]
            self._sync_color_inputs_for_group(groups[0])

    def _sync_color_inputs_for_group(self, group: str) -> None:
        base_color = self.plot_style.get_color(group, "Force (N)")
        self.query_one("#input_force_hex", Input).value = self.plot_style.group_force_colors.get(group, base_color)
        self.query_one("#input_defo_hex", Input).value = self.plot_style.group_deformation_colors.get(group, base_color)
        self.query_one("#input_stress_hex", Input).value = self.plot_style.group_stress_colors.get(group, base_color)

    def _collect_tpa_config(self) -> TPAConfig:
        def as_float(widget_id: str, default: float) -> float:
            text = self.query_one(widget_id, Input).value.strip()
            try:
                return float(text)
            except ValueError:
                return default

        def as_int(widget_id: str, default: int) -> int:
            text = self.query_one(widget_id, Input).value.strip()
            try:
                return int(float(text))
            except ValueError:
                return default

        return TPAConfig(
            sample_height_mm=as_float("#input_height", 10.0),
            contact_area_mm2=as_float("#input_area", 100.0),
            baseline_points=max(as_int("#input_baseline_points", 10), 1),
            trigger_force_n=as_float("#input_trigger", 0.05),
            peak_prominence_n=as_float("#input_prominence", 0.5),
            peak_distance_pts=max(as_int("#input_peak_distance", 200), 1),
            modulus_strain_min_pct=as_float("#input_mod_min", 10.0),
            modulus_strain_max_pct=as_float("#input_mod_max", 30.0),
        )

    def _collect_figure_config(self) -> FigureConfig:
        ratio = str(self.query_one("#select_ratio", Select).value)

        def parse_float_or_none(widget_id: str) -> float | None:
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

        dpi = int(max(72, min(dpi, 1200)))

        return FigureConfig(ratio_preset=ratio, width_in=width, height_in=height, dpi=dpi)

    def _update_figure_preview(self) -> None:
        fig_cfg = self._collect_figure_config()
        width, height = fig_cfg.resolve_size(default=(10.0, 7.5))
        self.query_one("#fig-preview", Static).update(f"Effective Size: {width:.2f} x {height:.2f} in @ {fig_cfg.dpi} DPI")

    def _collect_graph_spec_from_ui(self) -> GraphSpec | None:
        x_var = str(self.query_one("#select_x_var", Select).value)
        y_text = self.query_one("#input_y_vars", Input).value.strip()
        y_vars = [token.strip() for token in y_text.split(",") if token.strip()]

        valid_y = [
            label
            for label in y_vars
            if label in VARIABLE_REGISTRY and VARIABLE_REGISTRY[label]["kind"] == "y"
        ]

        if not valid_y:
            self._set_status("No valid Y variables. Use labels like 'Force (N)'.")
            return None

        title = self.query_one("#input_graph_title", Input).value.strip() or "Custom Graph"
        mode = str(self.query_one("#select_graph_mode", Select).value)
        curve_mode = str(self.query_one("#select_curve_mode", Select).value)
        band_mode = str(self.query_one("#select_band_mode", Select).value)

        return GraphSpec(
            title=title,
            x_col=x_var,
            y_cols=valid_y,
            mode=mode,
            enabled=True,
            curve_mode=curve_mode,
            band_mode=band_mode,
        )

    def _render_graph_specs(self) -> None:
        if not self.graph_specs:
            self.query_one("#graph-spec-list", Static).update("(No custom graphs yet)")
            return

        lines = []
        for idx, spec in enumerate(self.graph_specs, start=1):
            y_joined = ", ".join(spec.y_cols)
            lines.append(
                f"{idx}. {spec.title}\n"
                f"   x={spec.x_col} | y={y_joined} | mode={spec.mode} | curve={spec.curve_mode}/{spec.band_mode}"
            )
        self.query_one("#graph-spec-list", Static).update("\n".join(lines))

    def _rebuild_results_table(self) -> None:
        table = self.query_one("#results_table", DataTable)
        table.clear(columns=False)

        if self.metrics_df.empty:
            return

        for _, row in self.metrics_df.iterrows():
            table.add_row(
                str(row.get("Filename", "")),
                str(row.get("Group", "")),
                str(row.get("Hardness (N)", "")),
                str(row.get("Cohesiveness", "")),
                str(row.get("Springiness", "")),
                str(row.get("Resilience", "")),
                str(row.get("Chewiness", "")),
                str(row.get("Adhesiveness", "")),
                str(row.get("Modulus (kPa)", "")),
            )

    @on(Button.Pressed, "#btn_refresh")
    def handle_refresh(self) -> None:
        self._refresh_directory()

    @on(OptionList.OptionSelected, "#file_list")
    def handle_file_selected(self, event: OptionList.OptionSelected) -> None:
        idx = int(event.option_index)
        if not self.file_records:
            return

        self._set_selected_file(idx, update_status=True)

    @on(OptionList.OptionHighlighted, "#file_list")
    def handle_file_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if not self.file_records:
            return
        idx = int(event.option_index)
        self._set_selected_file(idx, update_status=False)

    @on(OptionList.OptionHighlighted, "#group_order_list")
    def handle_group_order_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if not self.group_order:
            self.selected_group_order_index = None
        else:
            idx = int(event.option_index)
            if 0 <= idx < len(self.group_order):
                self.selected_group_order_index = idx
        self._update_group_order_buttons()

    @on(Button.Pressed, "#btn_group_up")
    def handle_group_up(self) -> None:
        idx = self.selected_group_order_index
        if idx is None or idx <= 0 or idx >= len(self.group_order):
            return
        self.group_order[idx - 1], self.group_order[idx] = self.group_order[idx], self.group_order[idx - 1]
        self.selected_group_order_index = idx - 1
        self._render_group_order_list()
        self._update_color_group_select()
        self._update_group_summary()
        self._reorder_existing_stats_results()
        self._set_status("Moved selected group up.")

    @on(Button.Pressed, "#btn_group_down")
    def handle_group_down(self) -> None:
        idx = self.selected_group_order_index
        if idx is None or idx < 0 or idx >= len(self.group_order) - 1:
            return
        self.group_order[idx + 1], self.group_order[idx] = self.group_order[idx], self.group_order[idx + 1]
        self.selected_group_order_index = idx + 1
        self._render_group_order_list()
        self._update_color_group_select()
        self._update_group_summary()
        self._reorder_existing_stats_results()
        self._set_status("Moved selected group down.")

    @on(Button.Pressed, "#btn_assign_group")
    def handle_assign_group(self) -> None:
        if self.selected_file_index is None:
            list_view = self.query_one("#file_list", OptionList)
            if self.file_records and list_view.highlighted is not None:
                highlighted_idx = int(list_view.highlighted)
                if 0 <= highlighted_idx < len(self.file_records):
                    self._set_selected_file(highlighted_idx, update_status=False)

        if self.selected_file_index is None:
            self._set_status("Select a file first.")
            return

        group = self.query_one("#input_group", Input).value.strip()
        if not group:
            self._set_status("Group name cannot be empty.")
            return

        current_idx = int(self.selected_file_index)
        self.file_records[current_idx]["group"] = group
        next_idx = min(current_idx + 1, len(self.file_records) - 1)
        self._sync_group_order_from_records()
        self._render_file_list(selected_idx=next_idx)
        self._update_color_group_select()
        self._set_status(f"Assigned '{group}' to {self.file_records[current_idx]['filename']}.")

    @on(Button.Pressed, "#btn_assign_matching")
    def handle_assign_matching(self) -> None:
        if not self.file_records:
            self._set_status("No files loaded.")
            return

        group = self.query_one("#input_group", Input).value.strip()
        if not group:
            self._set_status("Group name cannot be empty.")
            return

        raw_terms = self.query_one("#input_group_filter", Input).value.strip()
        terms = self._parse_batch_terms(raw_terms)
        matched_indices: list[int] = []

        for idx, record in enumerate(self.file_records):
            filename = record["filename"]
            if self._filename_matches_terms(filename, terms):
                record["group"] = group
                matched_indices.append(idx)

        if not matched_indices:
            self._set_status("No filenames matched the provided terms.")
            return

        self._sync_group_order_from_records()
        self._render_file_list(selected_idx=matched_indices[0])
        self._update_color_group_select()
        scope = (
            f"{len(matched_indices)} files (all)"
            if not terms
            else f"{len(matched_indices)} files matching {', '.join(terms)}"
        )
        self._set_status(f"Assigned group '{group}' to {scope}.")

    @on(Select.Changed, "#select_color_group")
    def handle_color_group_changed(self, event: Select.Changed) -> None:
        group = str(event.value)
        if group == "__none__":
            return
        self._sync_color_inputs_for_group(group)

    @on(Button.Pressed)
    def handle_parameter_info_click(self, event: Button.Pressed) -> None:
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
        group = str(self.query_one("#select_color_group", Select).value)
        if group == "__none__":
            self._set_status("No group selected for color update.")
            return

        force_hex = self.query_one("#input_force_hex", Input).value.strip().upper()
        defo_hex = self.query_one("#input_defo_hex", Input).value.strip().upper()
        stress_hex = self.query_one("#input_stress_hex", Input).value.strip().upper()

        if not all(_is_hex_color(value) for value in [force_hex, defo_hex, stress_hex]):
            self._set_status("Invalid hex color. Use #RRGGBB format.")
            return

        self.plot_style.group_force_colors[group] = force_hex
        self.plot_style.group_deformation_colors[group] = defo_hex
        self.plot_style.group_stress_colors[group] = stress_hex
        self._set_status(f"Applied custom colors to group '{group}'.")

    @on(Button.Pressed, "#btn_reset_palette")
    def reset_palette(self) -> None:
        self.plot_style = PlotStyleConfig()
        self.plot_style.ensure_group_colors(self.group_order)
        group = str(self.query_one("#select_color_group", Select).value)
        if group != "__none__":
            self._sync_color_inputs_for_group(group)
        self._set_status("Reset color overrides to auto palette.")

    @on(Button.Pressed, "#btn_add_graph")
    def add_graph_spec(self) -> None:
        spec = self._collect_graph_spec_from_ui()
        if spec is None:
            return

        self.graph_specs.append(spec)
        self._render_graph_specs()
        self._set_status(f"Added graph spec: {spec.title}")

    @on(Button.Pressed, "#btn_clear_graphs")
    def clear_graph_specs(self) -> None:
        self.graph_specs.clear()
        self._render_graph_specs()
        self._set_status("Cleared custom graph specs.")

    @on(Input.Changed, "#input_width")
    @on(Input.Changed, "#input_height_fig")
    @on(Input.Changed, "#input_dpi")
    @on(Select.Changed, "#select_ratio")
    def handle_figure_inputs_changed(self) -> None:
        self._update_figure_preview()

    @on(Button.Pressed, "#btn_analyze")
    def trigger_analysis(self) -> None:
        if not self.file_records:
            self._set_status("No files loaded.")
            return

        self._sync_group_order_from_records()
        config = self._collect_tpa_config()
        stats_mode = str(self.query_one("#select_stats_mode", Select).value)
        records = [record.copy() for record in self.file_records]
        group_order = self.group_order.copy()
        self.run_analysis_worker(config, stats_mode, records, group_order)

    @work(thread=True, exclusive=True)
    def run_analysis_worker(
        self,
        config: TPAConfig,
        stats_mode: str,
        records: list[dict[str, str]],
        group_order: list[str],
    ) -> None:
        self.call_from_thread(self._set_buttons_disabled, True)
        self.call_from_thread(self._set_status, "Running analysis...")
        self.call_from_thread(self._log, f"Analysis started with config: {asdict(config)}")

        metric_rows: list[dict[str, Any]] = []
        traces: list[pd.DataFrame] = []
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

                if "Error" in result:
                    failures.append(f"{filename}: {result['Error']}")
                    continue

                metrics_row = {
                    "Filename": filename,
                    "Group": group,
                    "Hardness (N)": result.get("Hardness (N)"),
                    "Cohesiveness": result.get("Cohesiveness"),
                    "Springiness": result.get("Springiness"),
                    "Resilience": result.get("Resilience"),
                    "Chewiness": result.get("Chewiness"),
                    "Adhesiveness": result.get("Adhesiveness"),
                    "Modulus (kPa)": result.get("Modulus (kPa)"),
                }
                metric_rows.append(metrics_row)

                trace = result.get("Trace Data")
                if isinstance(trace, pd.DataFrame) and not trace.empty:
                    traces.append(trace)

                for warning in result.get("Warnings", []):
                    warnings.append(f"{filename}: {warning}")
            except Exception as exc:
                failures.append(f"{filename}: {exc}")

        metrics_df = pd.DataFrame(metric_rows)
        trace_df = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()

        stats_results: dict[str, dict[str, Any]] = {}
        if not metrics_df.empty:
            for metric in [
                "Hardness (N)",
                "Cohesiveness",
                "Springiness",
                "Resilience",
                "Chewiness",
                "Adhesiveness",
                "Modulus (kPa)",
            ]:
                if metric not in metrics_df.columns:
                    continue
                metric_frame = metrics_df[["Group", metric]].dropna()
                if metric_frame.empty:
                    continue
                try:
                    stats_results[metric] = run_statistics(
                        metric_frame,
                        group_col="Group",
                        metric_col=metric,
                        alpha=0.05,
                        mode=stats_mode,
                        group_order=group_order,
                    )
                except Exception as exc:
                    warnings.append(f"Stats failed for {metric}: {exc}")

        self.call_from_thread(
            self._apply_analysis_results,
            metrics_df,
            trace_df,
            stats_results,
            warnings,
            failures,
        )
        self.call_from_thread(self._set_buttons_disabled, False)

    def _apply_analysis_results(
        self,
        metrics_df: pd.DataFrame,
        trace_df: pd.DataFrame,
        stats_results: dict[str, dict[str, Any]],
        warnings: list[str],
        failures: list[str],
    ) -> None:
        self.metrics_df = metrics_df
        self.trace_df = trace_df
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

        self._set_status(
            f"Analysis done. Valid files: {len(metrics_df)} | Stats metrics: {len(stats_results)} | Failures: {len(failures)}"
        )
        self._log("Analysis completed.")

    def _build_stats_exports(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary_frames: list[pd.DataFrame] = []
        pairwise_frames: list[pd.DataFrame] = []

        for metric, result in self.stats_results.items():
            summary = result["summary_df"].copy()
            summary.insert(0, "Metric", metric)
            summary_frames.append(summary)

            pairwise = result["pairwise_df"].copy()
            if not pairwise.empty:
                pairwise_frames.append(pairwise)

        summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
        pairwise_df = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
        return summary_df, pairwise_df

    def _current_export_root(self, base_name: str) -> Path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        root = Path(base_name) / stamp
        root.mkdir(parents=True, exist_ok=True)
        return root

    @on(Button.Pressed, "#btn_export_tables")
    def trigger_export_tables(self) -> None:
        if self.metrics_df.empty:
            self._set_status("No analysis results to export.")
            return

        metrics_df = self.metrics_df.copy()
        stats_results = self.stats_results.copy()
        self.export_tables_worker(metrics_df, stats_results)

    @work(thread=True, exclusive=True)
    def export_tables_worker(self, metrics_df: pd.DataFrame, stats_results: dict[str, dict[str, Any]]) -> None:
        self.call_from_thread(self._set_buttons_disabled, True)
        try:
            export_root = self._current_export_root("exports")
            metrics_path = export_root / "tpa_results_summary.csv"
            metrics_df.to_csv(metrics_path, index=False)

            summary_frames: list[pd.DataFrame] = []
            pairwise_frames: list[pd.DataFrame] = []
            for metric, result in stats_results.items():
                summary = result["summary_df"].copy()
                summary.insert(0, "Metric", metric)
                summary_frames.append(summary)
                pairwise = result["pairwise_df"].copy()
                if not pairwise.empty:
                    pairwise_frames.append(pairwise)

            summary_path = export_root / "tpa_group_stats.csv"
            pairwise_path = export_root / "tpa_pairwise_stats.csv"

            if summary_frames:
                pd.concat(summary_frames, ignore_index=True).to_csv(summary_path, index=False)
            else:
                pd.DataFrame().to_csv(summary_path, index=False)

            if pairwise_frames:
                pd.concat(pairwise_frames, ignore_index=True).to_csv(pairwise_path, index=False)
            else:
                pd.DataFrame().to_csv(pairwise_path, index=False)

            self.call_from_thread(self._set_status, f"Tables exported to {export_root}")
            self.call_from_thread(self._log, f"Tables exported: {export_root}")
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Table export failed: {exc}")
            self.call_from_thread(self._log, f"Table export failed: {exc}")
        finally:
            self.call_from_thread(self._set_buttons_disabled, False)

    @on(Button.Pressed, "#btn_export_plots")
    def trigger_export_plots(self) -> None:
        if self.trace_df.empty:
            self._set_status("No trace data to plot. Run analysis first.")
            return

        trace_df = self.trace_df.copy()
        stats_results = self.stats_results.copy()
        graph_specs = [GraphSpec(**asdict(spec)) for spec in self.graph_specs]
        style = PlotStyleConfig(
            group_force_colors=self.plot_style.group_force_colors.copy(),
            group_deformation_colors=self.plot_style.group_deformation_colors.copy(),
            group_stress_colors=self.plot_style.group_stress_colors.copy(),
            palette_name=self.plot_style.palette_name,
            replicate_alpha=self.plot_style.replicate_alpha,
            replicate_linewidth=self.plot_style.replicate_linewidth,
            mean_linewidth=self.plot_style.mean_linewidth,
        )
        group_order = self.group_order.copy()
        style.ensure_group_colors(group_order)
        fig_cfg = self._collect_figure_config()
        overlay_mode = str(self.query_one("#select_overlay_mode", Select).value)
        band_mode = str(self.query_one("#select_band_mode", Select).value)

        self.export_plots_worker(trace_df, stats_results, graph_specs, style, fig_cfg, overlay_mode, band_mode, group_order)

    @work(thread=True, exclusive=True)
    def export_plots_worker(
        self,
        trace_df: pd.DataFrame,
        stats_results: dict[str, dict[str, Any]],
        graph_specs: list[GraphSpec],
        style: PlotStyleConfig,
        fig_cfg: FigureConfig,
        overlay_mode: str,
        band_mode: str,
        group_order: list[str],
    ) -> None:
        self.call_from_thread(self._set_buttons_disabled, True)
        try:
            output_root = self._current_export_root("output_plots")

            # Baseline default stack plot.
            default_stack_path = output_root / "default_stack.png"
            plot_trace_stack(
                trace_df,
                spec={"curve_mode": overlay_mode if overlay_mode != "individual" else "individual", "band_mode": band_mode},
                style=style,
                output_path=default_stack_path,
                figure_config=fig_cfg,
            )

            # Grouped metric bars.
            if stats_results:
                grouped_bars_path = output_root / "grouped_metrics.png"
                plot_grouped_metrics(stats_results, style=style, output_path=grouped_bars_path, figure_config=fig_cfg)

            # Group-level overlay traces.
            overlay_dir = output_root / "overlays"
            overlay_spec = {
                "mode": overlay_mode,
                "band_mode": band_mode,
                "x_col": "Aligned Time (s)",
                "y_cols": ["Force (N)", "Deformation (mm)"],
                "group_order": group_order,
            }
            plot_overlay_traces(trace_df, overlay_spec=overlay_spec, style=style, output_dir=overlay_dir, figure_config=fig_cfg)

            # Custom graph specs.
            custom_dir = output_root / "custom"
            custom_payload = plot_custom_graphs(
                trace_df,
                graph_specs=graph_specs,
                style=style,
                output_dir=custom_dir,
                figure_config=fig_cfg,
            )
            for warning in custom_payload.get("warnings", []):
                self.call_from_thread(self._log, f"Plot warning: {warning}")

            self.call_from_thread(self._set_status, f"Plots exported to {output_root}")
            self.call_from_thread(self._log, f"Plots exported: {output_root}")
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Plot export failed: {exc}")
            self.call_from_thread(self._log, f"Plot export failed: {exc}")
        finally:
            self.call_from_thread(self._set_buttons_disabled, False)

    @on(Button.Pressed, "#btn_export_all")
    def trigger_export_all(self) -> None:
        if self.metrics_df.empty:
            self._set_status("No analysis results. Run analysis first.")
            return

        metrics_df = self.metrics_df.copy()
        trace_df = self.trace_df.copy()
        stats_results = self.stats_results.copy()
        graph_specs = [GraphSpec(**asdict(spec)) for spec in self.graph_specs]
        style = PlotStyleConfig(
            group_force_colors=self.plot_style.group_force_colors.copy(),
            group_deformation_colors=self.plot_style.group_deformation_colors.copy(),
            group_stress_colors=self.plot_style.group_stress_colors.copy(),
            palette_name=self.plot_style.palette_name,
            replicate_alpha=self.plot_style.replicate_alpha,
            replicate_linewidth=self.plot_style.replicate_linewidth,
            mean_linewidth=self.plot_style.mean_linewidth,
        )
        group_order = self.group_order.copy()
        style.ensure_group_colors(group_order)
        fig_cfg = self._collect_figure_config()
        overlay_mode = str(self.query_one("#select_overlay_mode", Select).value)
        band_mode = str(self.query_one("#select_band_mode", Select).value)

        self.export_all_worker(
            metrics_df,
            trace_df,
            stats_results,
            graph_specs,
            style,
            fig_cfg,
            overlay_mode,
            band_mode,
            group_order,
        )

    @work(thread=True, exclusive=True)
    def export_all_worker(
        self,
        metrics_df: pd.DataFrame,
        trace_df: pd.DataFrame,
        stats_results: dict[str, dict[str, Any]],
        graph_specs: list[GraphSpec],
        style: PlotStyleConfig,
        fig_cfg: FigureConfig,
        overlay_mode: str,
        band_mode: str,
        group_order: list[str],
    ) -> None:
        self.call_from_thread(self._set_buttons_disabled, True)
        try:
            root = self._current_export_root("exports")
            metrics_df.to_csv(root / "tpa_results_summary.csv", index=False)

            summary_frames: list[pd.DataFrame] = []
            pairwise_frames: list[pd.DataFrame] = []
            for metric, result in stats_results.items():
                summary = result["summary_df"].copy()
                summary.insert(0, "Metric", metric)
                summary_frames.append(summary)
                pairwise = result["pairwise_df"].copy()
                if not pairwise.empty:
                    pairwise_frames.append(pairwise)

            if summary_frames:
                pd.concat(summary_frames, ignore_index=True).to_csv(root / "tpa_group_stats.csv", index=False)
            else:
                pd.DataFrame().to_csv(root / "tpa_group_stats.csv", index=False)

            if pairwise_frames:
                pd.concat(pairwise_frames, ignore_index=True).to_csv(root / "tpa_pairwise_stats.csv", index=False)
            else:
                pd.DataFrame().to_csv(root / "tpa_pairwise_stats.csv", index=False)

            plot_root = root / "plots"
            plot_root.mkdir(parents=True, exist_ok=True)

            plot_trace_stack(
                trace_df,
                spec={"curve_mode": overlay_mode if overlay_mode != "individual" else "individual", "band_mode": band_mode},
                style=style,
                output_path=plot_root / "default_stack.png",
                figure_config=fig_cfg,
            )

            if stats_results:
                plot_grouped_metrics(
                    stats_results,
                    style=style,
                    output_path=plot_root / "grouped_metrics.png",
                    figure_config=fig_cfg,
                )

            plot_overlay_traces(
                trace_df,
                overlay_spec={
                    "mode": overlay_mode,
                    "band_mode": band_mode,
                    "x_col": "Aligned Time (s)",
                    "y_cols": ["Force (N)", "Deformation (mm)"],
                    "group_order": group_order,
                },
                style=style,
                output_dir=plot_root / "overlays",
                figure_config=fig_cfg,
            )

            custom_payload = plot_custom_graphs(
                trace_df,
                graph_specs=graph_specs,
                style=style,
                output_dir=plot_root / "custom",
                figure_config=fig_cfg,
            )
            for warning in custom_payload.get("warnings", []):
                self.call_from_thread(self._log, f"Plot warning: {warning}")

            self.call_from_thread(self._set_status, f"Export all completed: {root}")
            self.call_from_thread(self._log, f"Export all completed: {root}")
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Export all failed: {exc}")
            self.call_from_thread(self._log, f"Export all failed: {exc}")
        finally:
            self.call_from_thread(self._set_buttons_disabled, False)


if __name__ == "__main__":
    app = TPAAnalyzerApp()
    app.run()
