"""Tests for parsing and TPA analysis helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tpa_analyzer.analysis.tpa import TPAConfig, calculate_tpa, parse_zwick_data
from tpa_analyzer.core.errors import AnalysisError, DataParseError


def _synthetic_trace() -> pd.DataFrame:
    """Build a synthetic two-peak trace suitable for TPA tests."""
    time_vals = np.linspace(0.0, 20.0, 1600)
    peak_1 = 5.0 * np.exp(-((time_vals - 4.0) ** 2) / 0.7)
    peak_2 = 4.0 * np.exp(-((time_vals - 12.0) ** 2) / 0.9)
    adhesive = -0.35 * np.exp(-((time_vals - 8.0) ** 2) / 0.18)
    force = peak_1 + peak_2 + adhesive
    deformation = np.linspace(0.0, 8.0, len(time_vals))
    return pd.DataFrame({"Time": time_vals, "Force": force, "Deformation": deformation})


def test_parse_zwick_data_detects_semicolon_header(tmp_path: Path) -> None:
    """The parser should detect the header row and semicolon delimiter."""
    data_path = tmp_path / "sample.csv"
    data_path.write_text(
        "meta line\nsecond meta line\nPruefzeit;Standardkraft;Dehnung\n0;0;0\n1;2;4\n",
        encoding="latin1",
    )
    result = parse_zwick_data(data_path)
    assert list(result.columns) == ["Time", "Force", "Deformation"]
    assert len(result) == 2


def test_parse_zwick_data_requires_expected_columns(tmp_path: Path) -> None:
    """The parser should reject files without the required columns."""
    data_path = tmp_path / "bad.csv"
    data_path.write_text("Header\nA,B,C\n1,2,3\n", encoding="latin1")
    with pytest.raises(DataParseError):
        parse_zwick_data(data_path)


def test_calculate_tpa_returns_metrics_and_trace_data() -> None:
    """A valid two-peak trace should produce metrics and trace payloads."""
    result = calculate_tpa(
        _synthetic_trace(), config=TPAConfig(), file_id="sample.csv", group="ctrl"
    )
    assert result["Hardness (N)"] > 0
    assert result["Springiness"] > 0
    assert isinstance(result["Trace Data"], pd.DataFrame)
    assert not result["Trace Data"].empty
    assert "QC Summary" in result


def test_calculate_tpa_rejects_empty_input() -> None:
    """Empty traces should raise a domain analysis error."""
    with pytest.raises(AnalysisError):
        calculate_tpa(pd.DataFrame(columns=["Time", "Force", "Deformation"]))
