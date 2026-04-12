"""Export helpers shared by UI actions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tpa_analyzer.core.models import CustomGraphSpec, FigureConfig, GraphSpec, PlotStyleConfig
from tpa_analyzer.plotting.engine import (
    export_qc_report,
    plot_custom_graphs,
    plot_grouped_metrics,
    plot_overlay_traces,
    plot_trace_stack,
)


def format_stats_note(test_info: dict[str, Any]) -> str:
    """Build a compact human-readable statistics note for exported tables."""
    decision = str(test_info.get("decision", "unknown"))
    reason = str(test_info.get("decision_reason", ""))
    global_test = str(test_info.get("global_test", ""))
    alpha = test_info.get("alpha", "")
    global_p_raw = test_info.get("global_p")
    try:
        global_p_text = f"{float(global_p_raw):.4g}"
    except (TypeError, ValueError):
        global_p_text = "NA"
    return (
        f"Decision={decision}; Global={global_test} (p={global_p_text}); "
        f"Pairwise significant when adjusted p < {alpha}. Reason: {reason}"
    )


def build_stats_exports(
    stats_results: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build flattened summary and pairwise statistics export tables."""
    summary_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []

    for metric, result in stats_results.items():
        summary = result.get("summary_df", pd.DataFrame()).copy()
        test_info = result.get("test_info", {})
        note = format_stats_note(test_info)
        if not summary.empty:
            summary.insert(0, "Metric", metric)
            summary["Stats Mode"] = str(test_info.get("mode", ""))
            summary["Stats Decision"] = str(test_info.get("decision", ""))
            summary["Global Test"] = str(test_info.get("global_test", ""))
            summary["Global P"] = test_info.get("global_p")
            summary["Alpha"] = test_info.get("alpha")
            summary["Decision Reason"] = str(test_info.get("decision_reason", ""))
            summary["Stats Note"] = note
            summary_frames.append(summary)

        pairwise = result.get("pairwise_df", pd.DataFrame()).copy()
        if not pairwise.empty:
            pairwise["Stats Mode"] = str(test_info.get("mode", ""))
            pairwise["Stats Decision"] = str(test_info.get("decision", ""))
            pairwise["Global Test"] = str(test_info.get("global_test", ""))
            pairwise["Global P"] = test_info.get("global_p")
            pairwise["Alpha"] = test_info.get("alpha")
            pairwise["Decision Reason"] = str(test_info.get("decision_reason", ""))
            pairwise["Stats Note"] = note
            pairwise_frames.append(pairwise)

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    pairwise_df = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
    return summary_df, pairwise_df


def current_export_root(base_name: str) -> Path:
    """Create and return a timestamped export root."""
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root = Path(base_name) / stamp
    root.mkdir(parents=True, exist_ok=True)
    return root


def export_tables_bundle(
    root: Path,
    metrics_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    stats_results: dict[str, dict[str, Any]],
) -> None:
    """Write metrics, QC, and statistics tables to ``root``."""
    metrics_df.to_csv(root / "tpa_results_summary.csv", index=False)
    (qc_df if not qc_df.empty else pd.DataFrame()).to_csv(root / "tpa_qc_summary.csv", index=False)
    summary_df, pairwise_df = build_stats_exports(stats_results)
    (summary_df if not summary_df.empty else pd.DataFrame()).to_csv(root / "tpa_group_stats.csv", index=False)
    (pairwise_df if not pairwise_df.empty else pd.DataFrame()).to_csv(root / "tpa_pairwise_stats.csv", index=False)


def export_plot_bundle(
    root: Path,
    trace_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    stats_results: dict[str, dict[str, Any]],
    graph_specs: list[GraphSpec | CustomGraphSpec | dict[str, Any]],
    style: PlotStyleConfig,
    fig_cfg: FigureConfig,
    overlay_mode: str,
    band_mode: str,
    group_order: list[str],
    include_plots_dir: bool,
) -> list[str]:
    """Write all plot exports and return collected warnings."""
    plot_root = root / "plots" if include_plots_dir else root
    plot_root.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    try:
        plot_trace_stack(
            trace_df,
            spec={"curve_mode": overlay_mode if overlay_mode != "individual" else "individual", "band_mode": band_mode, "group_order": group_order},
            style=style,
            output_path=plot_root / "default_stack.png",
            figure_config=fig_cfg,
        )
    except Exception as exc:
        warnings.append(f"Default trace stack skipped: {exc}")

    if stats_results:
        plot_grouped_metrics(
            stats_results,
            style=style,
            output_path=plot_root / "grouped_metrics.png",
            figure_config=fig_cfg,
        )

    try:
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
    except Exception as exc:
        warnings.append(f"Trace overlays skipped: {exc}")

    custom_payload = plot_custom_graphs(
        trace_df=trace_df,
        metrics_df=metrics_df,
        qc_df=qc_df,
        graph_specs=graph_specs,
        style=style,
        output_dir=plot_root / "custom",
        figure_config=fig_cfg,
        group_order=group_order,
        stats_by_metric=stats_results,
    )
    warnings.extend([f"Plot warning: {warning}" for warning in custom_payload.get("warnings", [])])

    qc_payload = export_qc_report(
        trace_df=trace_df,
        qc_df=qc_df,
        output_dir=root / "qc_report" if include_plots_dir else plot_root / "qc_report",
        figure_config=fig_cfg,
    )
    warnings.extend([f"QC warning: {warning}" for warning in qc_payload.get("warnings", [])])
    return warnings
