# TPA Analyzer

Responsive Textual TUI for Double Compression / Texture Profile Analysis (TPA) of Zwick exports (`.csv` / `.tra`), with grouped statistics, QC reporting, and flexible custom plotting.

## Highlights

- Packaged `src/` layout with separated analysis, stats, plotting, config, core, and UI modules
- Responsive Textual layout for wide, medium, and narrow terminal sizes
- Unified custom plot builder for trace variables and calculated metrics
- Multi-select `x` and `y` graph specs with auto expansion per selected `x`
- All calculated metrics available for plotting and grouped export:
  - Hardness
  - Cohesiveness
  - Springiness
  - Resilience
  - Chewiness
  - Adhesiveness
  - Modulus
- Environment-backed runtime defaults
- Pytest, Ruff, `uv`, and GitHub Actions CI support

## Project Structure

```text
src/tpa_analyzer/
  analysis/    Zwick parsing and TPA calculations
  config/      env-backed settings and logging
  core/        constants, models, errors, export/session helpers
  plotting/    trace, metric, QC, and custom graph rendering
  stats/       grouped hypothesis testing
  ui/          Textual app and responsive layout helpers
tests/         pytest suite
app.py         thin compatibility entrypoint
```

## Development

Install dependencies with `uv`:

```bash
uv sync --all-groups
```

Run the app:

```bash
uv run python app.py
```

Run tests:

```bash
uv run pytest
```

Run Ruff:

```bash
uv run ruff check .
```

## Configuration

Environment variables:

- `TPA_ANALYZER_LOG_LEVEL`
- `TPA_ANALYZER_DEFAULT_DATA_DIR`
- `TPA_ANALYZER_EXPORT_ROOT`
- `TPA_ANALYZER_PLOTS_ROOT`
- `TPA_ANALYZER_SESSION_AUTOSAVE`
- `TPA_ANALYZER_DEBUG`

## Export Outputs

`Export Tables` writes:

- `tpa_results_summary.csv`
- `tpa_qc_summary.csv`
- `tpa_group_stats.csv`
- `tpa_pairwise_stats.csv`

`Export Plots` writes:

- `default_stack.png`
- `grouped_metrics.png`
- `overlays/`
- `custom/`
- `qc_report/`

`Export All` writes tables plus plots under one timestamped export root.

## CI

Two GitHub Actions workflows are expected:

- `quality.yml` for Ruff, pytest, and packaging smoke checks on pushes and pull requests
- `build-binaries.yml` for release artifact builds on tags or manual dispatch
