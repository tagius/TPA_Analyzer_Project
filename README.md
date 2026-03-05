# Plant-Based TPA Analyzer (TUI)

Terminal UI for Double Compression (TPA) analysis of Zwick exports (`.csv` / `.tra`) with grouped statistics, QC reporting, and publication-ready plotting.

## Setup

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run the app:
   - `python app.py`

## What's New

- Robust Zwick parser:
  - Auto-detects delimiter (comma/semicolon/tab) and header row.
  - Accepts mixed English/German headers (`Force` / `Standardkraft`, `Deformation` / `Dehnung`, `Time` / `Zeit`).
- Smarter cycle detection:
  - Two-peak detection with fallback thresholds/prominence attempts.
  - Baseline force correction and overlap safeguards between cycle windows.
- Expanded TPA metrics:
  - Hardness, Cohesiveness, Springiness, Resilience, Chewiness, Adhesiveness.
  - True stress/true strain traces and modulus fit within a configurable strain window.
- Built-in QC package:
  - Per-file cycle markers, areas, control parameters, and warnings.
  - Annotated QC figures plus interpretation guide export.
- Statistics engine upgrades:
  - `auto`, `parametric`, and `nonparametric` modes.
  - Auto mode uses Shapiro + Levene checks.
  - Tests include Welch t-test, Mann-Whitney U, ANOVA + Tukey, and Kruskal-Wallis + Dunn (BH).
  - Compact-letter group significance display for summaries.
- Plot Studio:
  - Custom graph builder (panel/overlay layouts).
  - Curve modes: individual replicates, mean+band, or both.
  - Band type: SD or 95% CI.
  - Group overlays using aligned time.
  - Figure presets (`1:1`, `4:3`, `16:9`, `A4`) plus custom width/height and DPI.
- Grouping and style controls:
  - Auto group inference from filename.
  - Manual assignment, batch assignment by filename terms, and group order controls.
  - Stable per-group palette with per-group hex overrides.
- Session persistence:
  - Auto-saves working state to `.tpa_analyzer_session.json` in the active data directory.
  - Restores grouping, parameters, plot specs, and style settings on refresh.

## Workflow

1. Set a directory containing `.csv` / `.tra` files.
2. Review inferred groups and adjust manually or in batch.
3. Tune analysis parameters (trigger, prominence, peak distance, modulus window, etc.).
4. Run analysis.
5. Build custom plots if needed.
6. Export tables, plots, or both.

## Exports

### `Export Tables`
Creates `exports/<timestamp>/` with:
- `tpa_results_summary.csv`
- `tpa_qc_summary.csv`
- `tpa_group_stats.csv`
- `tpa_pairwise_stats.csv`

### `Export Plots`
Creates `output_plots/<timestamp>/` with:
- `default_stack.png`
- `grouped_metrics.png` (when stats are available)
- `overlays/overlay_<group>.png`
- `custom/custom_<index>_<title>.png`
- `qc_report/`:
  - `qc_summary.csv`
  - `qc_control_parameters.csv`
  - `qc_markers_and_areas.csv`
  - `files/*_qc.png`
  - `QC_REPORT_INTERPRETATION.md`

### `Export All`
Creates `exports/<timestamp>/` with all table exports plus:
- `plots/default_stack.png`
- `plots/grouped_metrics.png` (when stats are available)
- `plots/overlays/overlay_<group>.png`
- `plots/custom/custom_<index>_<title>.png`
- `qc_report/*` (same QC package as above)

## Dependencies

- `pandas`, `numpy`, `scipy`
- `matplotlib`, `seaborn`
- `textual`
- `pingouin`

## Build macOS + Windows executables (GitHub Actions)

The workflow at `.github/workflows/build-binaries.yml` produces:
- `tpa-analyzer-macos-app.zip` containing `TPA Analyzer.app` (macOS)
- `tpa-analyzer.exe` (Windows)
- Uses platform-specific icons from `assets/`:
  - `tpa-analyzer-icon.icns` for the macOS app bundle
  - `tpa-analyzer-icon.ico` for the Windows executable

Steps:
1. Push this repository to GitHub.
2. Open **Actions** -> **Build Binaries**.
3. Click **Run workflow** (or push a tag like `v1.0.0`).
4. Download artifacts:
   - `tpa-analyzer-macOS` (contains `tpa-analyzer-macos-app.zip` -> `TPA Analyzer.app`)
   - `tpa-analyzer-Windows` (contains `tpa-analyzer.exe`)

Troubleshooting:
- If a binary fails with `ModuleNotFoundError` for `textual.*` modules, rebuild with the current workflow (it includes `--collect-submodules textual --collect-data textual`).
