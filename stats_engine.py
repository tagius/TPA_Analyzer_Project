from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats


@dataclass
class StatsDecision:
    parametric: bool
    reason: str


def _letter_code(index: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    result = ""
    idx = index
    while True:
        result = letters[idx % 26] + result
        idx = idx // 26 - 1
        if idx < 0:
            break
    return result


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _row_float(row: pd.Series, keys: list[str], default: float = np.nan) -> float:
    """Return the first finite float found for one of the candidate column names."""
    for key in keys:
        if key not in row.index:
            continue
        value = row[key]
        if pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _choose_parametric(
    grouped: dict[str, np.ndarray],
    alpha: float,
    mode: str,
) -> StatsDecision:
    mode = mode.lower().strip()
    if mode == "parametric":
        return StatsDecision(parametric=True, reason="Forced parametric mode")
    if mode == "nonparametric":
        return StatsDecision(parametric=False, reason="Forced nonparametric mode")

    normality_ok = True
    for group_name, values in grouped.items():
        finite = values[np.isfinite(values)]
        if len(finite) >= 3:
            try:
                p_value = stats.shapiro(finite).pvalue
                if p_value < alpha:
                    normality_ok = False
                    return StatsDecision(
                        parametric=False,
                        reason=f"Shapiro failed for group '{group_name}' (p={p_value:.4g})",
                    )
            except Exception:
                normality_ok = False
                return StatsDecision(
                    parametric=False,
                    reason=f"Shapiro test failed for group '{group_name}'",
                )

    groups_for_levene = [vals[np.isfinite(vals)] for vals in grouped.values() if len(vals[np.isfinite(vals)]) >= 2]
    if len(groups_for_levene) >= 2:
        try:
            levene_p = stats.levene(*groups_for_levene).pvalue
            if levene_p < alpha:
                return StatsDecision(
                    parametric=False,
                    reason=f"Levene failed (p={levene_p:.4g})",
                )
        except Exception:
            return StatsDecision(parametric=False, reason="Levene test failed")

    if normality_ok:
        return StatsDecision(parametric=True, reason="Assumptions passed in auto mode")
    return StatsDecision(parametric=False, reason="Assumptions did not pass")


def _build_letters(
    group_order: list[str],
    significant_pairs: set[tuple[str, str]],
) -> dict[str, str]:
    letter_sets: list[set[str]] = []
    letters_by_group: dict[str, list[str]] = {group: [] for group in group_order}

    def compatible(group: str, assigned_groups: set[str]) -> bool:
        return all(_pair_key(group, other) not in significant_pairs for other in assigned_groups)

    for group in group_order:
        assigned_here = False
        for idx, group_set in enumerate(letter_sets):
            if compatible(group, group_set):
                group_set.add(group)
                letters_by_group[group].append(_letter_code(idx))
                assigned_here = True
        if not assigned_here:
            new_idx = len(letter_sets)
            letter_sets.append({group})
            letters_by_group[group].append(_letter_code(new_idx))

    # Ensure every non-significant pair shares at least one letter.
    for i, group_a in enumerate(group_order):
        for group_b in group_order[i + 1 :]:
            if _pair_key(group_a, group_b) in significant_pairs:
                continue
            shared = set(letters_by_group[group_a]).intersection(letters_by_group[group_b])
            if shared:
                continue

            merged = False
            for idx, group_set in enumerate(letter_sets):
                if compatible(group_a, group_set) and compatible(group_b, group_set):
                    if group_a not in group_set:
                        group_set.add(group_a)
                        letters_by_group[group_a].append(_letter_code(idx))
                    if group_b not in group_set:
                        group_set.add(group_b)
                        letters_by_group[group_b].append(_letter_code(idx))
                    merged = True
                    break

            if not merged:
                new_idx = len(letter_sets)
                letter_sets.append({group_a, group_b})
                code = _letter_code(new_idx)
                letters_by_group[group_a].append(code)
                letters_by_group[group_b].append(code)

    final_letters: dict[str, str] = {}
    for group, values in letters_by_group.items():
        unique = sorted(set(values), key=lambda item: (len(item), item))
        final_letters[group] = "".join(unique) if unique else "a"

    return final_letters


def run_statistics(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    alpha: float = 0.05,
    mode: str = "auto",
    group_order: list[str] | None = None,
) -> dict[str, Any]:
    """Run group-level summary and hypothesis tests for one metric column."""
    work = df[[group_col, metric_col]].copy()
    work = work.dropna(subset=[group_col, metric_col])
    work[group_col] = work[group_col].astype(str).str.strip()
    work = work[work[group_col] != ""]

    if work.empty:
        raise ValueError(f"No valid values available for metric '{metric_col}'.")

    grouped_series = {
        group: values.to_numpy(dtype=float)
        for group, values in work.groupby(group_col)[metric_col]
    }

    if group_order:
        order_from_user = [str(group).strip() for group in group_order if str(group).strip()]
        group_order_resolved = [group for group in order_from_user if group in grouped_series]
        for group in grouped_series:
            if group not in group_order_resolved:
                group_order_resolved.append(group)
    else:
        group_order_resolved = (
            work.groupby(group_col)[metric_col]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

    summary_df = (
        work.groupby(group_col)[metric_col]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "N", "mean": "Mean", "std": "SD"})
        .reset_index()
    )

    summary_df["SD"] = summary_df["SD"].fillna(0.0)
    summary_df["Mean±SD"] = summary_df.apply(lambda row: f"{row['Mean']:.3f} ± {row['SD']:.3f}", axis=1)

    pairwise_rows: list[dict[str, Any]] = []
    significant_pairs: set[tuple[str, str]] = set()
    n_groups = len(grouped_series)

    decision = _choose_parametric(grouped_series, alpha=alpha, mode=mode)
    global_test = "Summary only"
    global_p = np.nan

    if n_groups == 1:
        letters_map = {group_order_resolved[0]: "a"}
    elif n_groups == 2:
        g1, g2 = group_order_resolved[0], group_order_resolved[1]
        v1 = grouped_series[g1]
        v2 = grouped_series[g2]

        if decision.parametric:
            statistic, p_value = stats.ttest_ind(v1, v2, equal_var=False, nan_policy="omit")
            test_name = "Welch t-test"
        else:
            statistic, p_value = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            test_name = "Mann-Whitney U"

        global_test = test_name
        global_p = float(p_value)
        is_significant = bool(p_value < alpha)
        if is_significant:
            significant_pairs.add(_pair_key(g1, g2))

        pairwise_rows.append(
            {
                "Metric": metric_col,
                "Group1": g1,
                "Group2": g2,
                "Test": test_name,
                "Statistic": float(statistic),
                "P_raw": float(p_value),
                "P_adj": float(p_value),
                "Significant": is_significant,
            }
        )

        letters_map = _build_letters(group_order_resolved, significant_pairs)
    else:
        values = [grouped_series[group] for group in group_order_resolved]
        if decision.parametric:
            statistic, p_value = stats.f_oneway(*values)
            global_test = "One-way ANOVA + Tukey HSD"
            global_p = float(p_value)

            posthoc = pg.pairwise_tukey(data=work, dv=metric_col, between=group_col)
            for _, row in posthoc.iterrows():
                group_a = str(row["A"])
                group_b = str(row["B"])
                p_adj = _row_float(row, ["p_tukey", "p-tukey", "pval"])
                is_significant = bool(np.isfinite(p_adj) and p_adj < alpha)
                if is_significant:
                    significant_pairs.add(_pair_key(group_a, group_b))

                pairwise_rows.append(
                    {
                        "Metric": metric_col,
                        "Group1": group_a,
                        "Group2": group_b,
                        "Test": "Tukey HSD",
                        "Statistic": _row_float(row, ["T", "t"]),
                        "P_raw": p_adj,
                        "P_adj": p_adj,
                        "Significant": is_significant,
                    }
                )
        else:
            statistic, p_value = stats.kruskal(*values)
            global_test = "Kruskal-Wallis + Dunn (BH)"
            global_p = float(p_value)

            posthoc = pg.pairwise_tests(
                data=work,
                dv=metric_col,
                between=group_col,
                parametric=False,
                padjust="fdr_bh",
            )
            for _, row in posthoc.iterrows():
                group_a = str(row["A"])
                group_b = str(row["B"])
                p_raw = _row_float(row, ["p_unc", "p-unc"])
                p_adj = _row_float(row, ["p_corr", "p-corr"], default=p_raw)
                is_significant = bool(np.isfinite(p_adj) and p_adj < alpha)
                if is_significant:
                    significant_pairs.add(_pair_key(group_a, group_b))

                pairwise_rows.append(
                    {
                        "Metric": metric_col,
                        "Group1": group_a,
                        "Group2": group_b,
                        "Test": "Dunn (BH)",
                        "Statistic": _row_float(row, ["U_val", "U-val"]),
                        "P_raw": p_raw,
                        "P_adj": p_adj,
                        "Significant": is_significant,
                    }
                )

        letters_map = _build_letters(group_order_resolved, significant_pairs)

    summary_df["Significance"] = summary_df[group_col].map(letters_map).fillna("a")
    summary_df["_order"] = summary_df[group_col].map({group: idx for idx, group in enumerate(group_order_resolved)})
    summary_df = summary_df.sort_values("_order", ascending=True).drop(columns=["_order"]).reset_index(drop=True)

    pairwise_df = pd.DataFrame(pairwise_rows)

    return {
        "summary_df": summary_df,
        "pairwise_df": pairwise_df,
        "test_info": {
            "metric": metric_col,
            "group_col": group_col,
            "mode": mode,
            "group_order": group_order_resolved,
            "decision": "parametric" if decision.parametric else "nonparametric",
            "decision_reason": decision.reason,
            "global_test": global_test,
            "global_p": global_p,
            "alpha": alpha,
        },
        "letters_map": letters_map,
    }
