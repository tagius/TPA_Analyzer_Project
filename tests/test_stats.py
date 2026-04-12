"""Tests for grouped statistical analysis."""

from __future__ import annotations

import pandas as pd

from tpa_analyzer.stats.engine import run_statistics


def test_run_statistics_two_group_parametric_mode() -> None:
    """Two-group parametric mode should return summary and pairwise results."""
    frame = pd.DataFrame(
        {
            "Group": ["A", "A", "A", "B", "B", "B"],
            "Hardness (N)": [4.8, 5.0, 5.1, 6.2, 6.3, 6.5],
        }
    )
    result = run_statistics(frame, group_col="Group", metric_col="Hardness (N)", mode="parametric")
    assert not result["summary_df"].empty
    assert not result["pairwise_df"].empty
    assert result["test_info"]["decision"] == "parametric"


def test_run_statistics_three_group_nonparametric_mode() -> None:
    """Three-group nonparametric mode should build compact-letter output."""
    frame = pd.DataFrame(
        {
            "Group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "Chewiness": [1.0, 1.1, 1.2, 2.4, 2.6, 2.7, 4.2, 4.3, 4.4],
        }
    )
    result = run_statistics(frame, group_col="Group", metric_col="Chewiness", mode="nonparametric")
    assert "Significance" in result["summary_df"].columns
    assert result["test_info"]["decision"] == "nonparametric"
