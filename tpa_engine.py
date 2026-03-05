from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class TPAConfig:
    sample_height_mm: float = 10.0
    contact_area_mm2: float = 100.0
    baseline_points: int = 10
    trigger_force_n: float = 0.05
    peak_prominence_n: float = 0.5
    peak_distance_pts: int = 200
    modulus_strain_min_pct: float = 10.0
    modulus_strain_max_pct: float = 30.0


def infer_group_from_filename(filename: str) -> str:
    """Infer a default group by removing trailing replicate tokens from a filename stem."""
    stem = Path(filename).stem.strip()
    patterns = (
        r"([_-]?(rep|r)\d+)$",
        r"([_-]?\d+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, stem, re.IGNORECASE)
        if match:
            candidate = stem[: match.start()].rstrip("_-")
            if candidate:
                return candidate
    return stem


def _detect_header_line(lines: list[str]) -> tuple[int, str]:
    delimiters = [",", ";", "\t"]

    for idx, line in enumerate(lines[:80]):
        lower_line = line.lower()
        if (
            ("standardkraft" in lower_line or "force" in lower_line)
            and ("dehnung" in lower_line or "deformation" in lower_line)
            and ("pr" in lower_line or "time" in lower_line)
        ):
            delimiter = max(delimiters, key=line.count)
            return idx, delimiter

    return 2, ","


def parse_zwick_data(filepath: str | Path) -> pd.DataFrame:
    """Parse Zwick CSV/TRA files and return normalized numeric Time/Force/Deformation columns."""
    filepath = Path(filepath)

    with filepath.open("r", encoding="latin1", errors="ignore") as handle:
        lines = handle.readlines()

    header_idx, delimiter = _detect_header_line(lines)

    df = pd.read_csv(
        filepath,
        sep=delimiter,
        skiprows=header_idx,
        encoding="latin1",
        on_bad_lines="skip",
        engine="python",
    )

    # Normalize raw headers before mapping.
    cleaned_columns = [str(col).strip().strip('"') for col in df.columns]
    df.columns = cleaned_columns

    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_lower = col.lower()
        if "pr" in col_lower and ("zeit" in col_lower or "time" in col_lower):
            rename_map[col] = "Time"
        elif "standardkraft" in col_lower or "force" in col_lower:
            rename_map[col] = "Force"
        elif "dehnung" in col_lower or "deformation" in col_lower:
            rename_map[col] = "Deformation"

    df = df.rename(columns=rename_map)

    required = ["Time", "Force", "Deformation"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in file: {filepath.name}")

    # Convert to numeric and strip out unit lines / footer artifacts.
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)
    df = df[required]

    if df.empty:
        raise ValueError(f"No valid numeric data found in file: {filepath.name}")

    return df


def _find_crossing_start(force: np.ndarray, peak_idx: int, threshold: float) -> int:
    idx = peak_idx
    while idx > 0 and force[idx] > threshold:
        idx -= 1
    return idx


def _find_crossing_end(force: np.ndarray, peak_idx: int, threshold: float) -> int:
    idx = peak_idx
    n = len(force)
    while idx < n - 1 and force[idx] > threshold:
        idx += 1
    return idx


def _integrate_segment(time_vals: np.ndarray, signal_vals: np.ndarray, positive: bool) -> float:
    if len(time_vals) < 2:
        return 0.0

    mask = signal_vals > 0 if positive else signal_vals < 0
    if mask.sum() < 2:
        return 0.0

    return float(np.trapezoid(signal_vals[mask], time_vals[mask]))


def _detect_two_peaks(force: np.ndarray, config: TPAConfig) -> np.ndarray:
    attempts = [
        (config.trigger_force_n, config.peak_prominence_n),
        (config.trigger_force_n * 0.75, config.peak_prominence_n * 0.75),
        (config.trigger_force_n * 0.5, config.peak_prominence_n * 0.5),
        (0.0, config.peak_prominence_n * 0.25),
    ]

    for height, prominence in attempts:
        peaks, _ = find_peaks(
            force,
            height=max(height, 0.0),
            distance=max(int(config.peak_distance_pts), 1),
            prominence=max(prominence, 0.0),
        )
        if len(peaks) >= 2:
            return np.sort(peaks)[:2]

    return np.array([], dtype=int)


def calculate_tpa(
    df: pd.DataFrame,
    config: TPAConfig | None = None,
    file_id: str = "",
    group: str = "",
) -> dict[str, Any]:
    """Calculate core TPA metrics and return metrics plus trace-ready data for plotting."""
    if config is None:
        config = TPAConfig()

    if df.empty:
        return {"Error": "Input dataframe is empty."}

    work_df = df.copy()
    warnings: list[str] = []

    raw_force = work_df["Force"].to_numpy(dtype=float).copy()
    baseline_window = max(min(len(work_df), int(config.baseline_points)), 1)
    baseline = float(work_df["Force"].iloc[:baseline_window].mean())
    work_df["Force"] = work_df["Force"] - baseline

    force = work_df["Force"].to_numpy(dtype=float)
    time_vals = work_df["Time"].to_numpy(dtype=float)
    deformation = work_df["Deformation"].to_numpy(dtype=float)

    peaks = _detect_two_peaks(force, config)
    if len(peaks) < 2:
        return {"Error": "Could not detect two distinct compression cycles.", "Warnings": warnings}

    peak1_idx = int(peaks[0])
    peak2_idx = int(peaks[1])

    threshold = max(float(config.trigger_force_n), 0.01)
    bite1_start = _find_crossing_start(force, peak1_idx, threshold)
    bite1_end = _find_crossing_end(force, peak1_idx, threshold)

    if bite1_end >= peak2_idx:
        bite1_end = max(peak1_idx + 1, peak2_idx - 1)
        warnings.append("Adjusted first cycle endpoint because cycles overlapped.")

    bite2_start = _find_crossing_start(force, peak2_idx, threshold)
    if bite2_start <= bite1_end:
        bite2_start = min(max(bite1_end + 1, peak2_idx - 1), len(work_df) - 2)
        warnings.append("Adjusted second cycle start due to threshold crossing overlap.")

    bite2_end = _find_crossing_end(force, peak2_idx, threshold)
    if bite2_end <= bite2_start:
        bite2_end = len(work_df) - 1
        warnings.append("Second cycle end not found cleanly; using file end.")

    b1_slice = slice(bite1_start, bite1_end + 1)
    b2_slice = slice(bite2_start, bite2_end + 1)
    b1_up_slice = slice(bite1_start, peak1_idx + 1)
    b1_down_slice = slice(peak1_idx, bite1_end + 1)
    adh_slice = slice(bite1_end, bite2_start + 1)

    area1 = _integrate_segment(time_vals[b1_slice], force[b1_slice], positive=True)
    area2 = _integrate_segment(time_vals[b2_slice], force[b2_slice], positive=True)
    area1_up = _integrate_segment(time_vals[b1_up_slice], force[b1_up_slice], positive=True)
    area1_down = _integrate_segment(time_vals[b1_down_slice], force[b1_down_slice], positive=True)
    adhesiveness = _integrate_segment(time_vals[adh_slice], force[adh_slice], positive=False)

    hardness = float(force[peak1_idx])
    cohesiveness = float(area2 / area1) if area1 != 0 else 0.0
    resilience = float(area1_down / area1_up) if area1_up != 0 else 0.0

    time_delta_b1 = float(time_vals[peak1_idx] - time_vals[bite1_start])
    time_delta_b2 = float(time_vals[peak2_idx] - time_vals[bite2_start])
    springiness = float(time_delta_b2 / time_delta_b1) if time_delta_b1 != 0 else 0.0
    chewiness = float(hardness * cohesiveness * springiness)

    if config.sample_height_mm <= 0:
        true_strain_pct = np.full(len(work_df), np.nan)
        warnings.append("Sample height must be > 0; true strain disabled.")
    else:
        deformation_zero = deformation[bite1_start]
        true_strain_pct = ((deformation - deformation_zero) / config.sample_height_mm) * 100.0

    if config.contact_area_mm2 <= 0:
        true_stress_kpa = np.full(len(work_df), np.nan)
        warnings.append("Contact area must be > 0; true stress disabled.")
    else:
        true_stress_kpa = (force / config.contact_area_mm2) * 1000.0

    modulus_kpa = float("nan")
    if config.modulus_strain_min_pct >= config.modulus_strain_max_pct:
        warnings.append("Invalid modulus strain window; expected min < max.")
    else:
        comp_mask = np.zeros(len(work_df), dtype=bool)
        comp_mask[bite1_start : peak1_idx + 1] = True

        strain_mask = (
            (true_strain_pct >= config.modulus_strain_min_pct)
            & (true_strain_pct <= config.modulus_strain_max_pct)
        )
        fit_mask = comp_mask & strain_mask & np.isfinite(true_stress_kpa) & np.isfinite(true_strain_pct)

        if fit_mask.sum() >= 2:
            x_vals = true_strain_pct[fit_mask] / 100.0
            y_vals = true_stress_kpa[fit_mask]
            modulus_kpa = float(np.polyfit(x_vals, y_vals, 1)[0])
        else:
            warnings.append("Insufficient points for modulus fit in selected strain window.")

    aligned_time = time_vals - time_vals[bite1_start]

    cycle = np.zeros(len(work_df), dtype=int)
    cycle[bite1_start : bite1_end + 1] = 1
    cycle[bite2_start : bite2_end + 1] = 2

    trace_df = pd.DataFrame(
        {
            "Time (s)": time_vals,
            "Aligned Time (s)": aligned_time,
            "Force (N)": raw_force,
            "Force Corrected (N)": force,
            "Deformation (mm)": deformation,
            "True Stress (kPa)": true_stress_kpa,
            "True Strain (%)": true_strain_pct,
            "Cycle": cycle,
            "File": file_id,
            "Group": group,
        }
    )

    metrics = {
        "Hardness (N)": round(hardness, 3),
        "Cohesiveness": round(cohesiveness, 3),
        "Springiness": round(springiness, 3),
        "Resilience": round(resilience, 3),
        "Chewiness": round(chewiness, 3),
        "Adhesiveness": round(adhesiveness, 3),
        "Modulus (kPa)": round(modulus_kpa, 3) if np.isfinite(modulus_kpa) else float("nan"),
    }

    result: dict[str, Any] = dict(metrics)
    result["Trace Data"] = trace_df
    result["Cycle Markers"] = {
        "Peak1 Index": peak1_idx,
        "Peak2 Index": peak2_idx,
        "Bite1 Start": bite1_start,
        "Bite1 End": bite1_end,
        "Bite2 Start": bite2_start,
        "Bite2 End": bite2_end,
    }
    result["Warnings"] = warnings

    return result


def generate_plots(df: pd.DataFrame, filename: str, output_dir: str = "output_plots") -> str:
    """Backward-compatible per-file raw-trace export used by earlier app versions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(df["Time"], df["Force"], color="#2ca02c", linewidth=2)
    ax1.set_ylabel("Force (N)")
    ax1.set_title(f"TPA Raw Traces: {filename}")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(df["Time"], df["Deformation"], color="#1f77b4", linewidth=2)
    ax2.set_ylabel("Deformation (mm)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    save_path = output_path / f"{Path(filename).stem}_plot.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)
