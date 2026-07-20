#!/usr/bin/env python3
"""Audit, summarize, and plot the consolidated v3 experiment campaign."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robust_mckp.plotting_style import (  # noqa: E402
    COLORS as STYLE_COLORS,
    apply_top_journal_style,
    figure_size,
    panel_label,
    polish_axis,
    save_figure,
)


METHOD_LABEL = {
    "hullround": "HullRound",
    "theta_bnb": r"$\theta$-B\&B",
    "theta_enum_highs": r"$\theta$-enumeration + HiGHS",
    "scip": "SCIP compact MILP",
    "highs": "HiGHS compact MILP",
}
FAMILY_LABEL = {
    "dense_frontier": "Dense frontier",
    "correlated_risk": "Correlated risk",
    "near_tie": "Near tie",
    "many_breakpoints": "Many breakpoints",
}
COLORS = {
    "hullround": "#009E73",
    "theta_bnb": "#0072B2",
    "theta_enum_highs": "#56B4E9",
    "scip": "#E69F00",
    "highs": "#CC79A7",
}
METHOD_MARKER = {
    "hullround": "o",
    "theta_bnb": "o",
    "theta_enum_highs": "^",
    "scip": "s",
    "highs": "D",
}
METHOD_LINESTYLE = {
    "hullround": "-",
    "theta_bnb": "-",
    "theta_enum_highs": "--",
    "scip": "-.",
    "highs": ":",
}


def rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def number(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def finite(values: Iterable[float]) -> List[float]:
    return [float(x) for x in values if math.isfinite(float(x))]


def med(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(statistics.median(vals)) if vals else float("nan")


def quant(values: Iterable[float], q: float) -> float:
    vals = finite(values)
    return float(np.quantile(vals, q)) if vals else float("nan")


def truth(row: Dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() == "true"


def group(rows_: Sequence[Dict[str, str]], key: str) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows_:
        out[row[key]].append(row)
    return out


def reference_objectives(hard: Sequence[Dict[str, str]]) -> Dict[str, float]:
    refs: Dict[str, List[float]] = defaultdict(list)
    for row in hard:
        if truth(row, "certified_optimal") and math.isfinite(number(row, "objective")):
            refs[row["instance_id"]].append(number(row, "objective"))
    out: Dict[str, float] = {}
    for key, values in refs.items():
        scale = max(1.0, max(abs(v) for v in values))
        if max(values) - min(values) > 1e-6 * scale:
            raise AssertionError(f"certified methods disagree on {key}: {values}")
        out[key] = float(statistics.median(values))
    return out


def audit(
    hard: Sequence[Dict[str, str]],
    anytime: Sequence[Dict[str, str]],
    failures: Sequence[Dict[str, str]],
    policy: Sequence[Dict[str, str]],
    stress: Sequence[Dict[str, str]],
    exact: Sequence[Dict[str, str]],
    app_config: Dict[str, object],
) -> List[str]:
    if not hard or not anytime or not failures:
        raise AssertionError("one or more v3 evidence files are empty")
    if any(row["status"] == "error" for row in hard + anytime):
        bad = [row["run_key"] for row in hard + anytime if row["status"] == "error"]
        raise AssertionError(f"benchmark error rows: {bad[:8]}")
    invalid = [row["run_key"] for row in hard + anytime if math.isfinite(number(row, "objective")) and not truth(row, "valid_certificate")]
    if invalid:
        raise AssertionError(f"finite incumbents without valid robust certificates: {invalid[:8]}")
    refs = reference_objectives(hard)
    if not refs:
        raise AssertionError("no certified hard-benchmark reference objectives")
    for row in hard:
        if truth(row, "certified_optimal") and row["instance_id"] in refs:
            if abs(number(row, "objective") - refs[row["instance_id"]]) > 1e-6 * max(1.0, abs(refs[row["instance_id"]])):
                raise AssertionError(f"certified objective mismatch: {row['run_key']}")
    if any(not truth(row, "failure_detected") for row in failures):
        missed = [row["case_id"] for row in failures if not truth(row, "failure_detected")]
        raise AssertionError(f"certificate-failure cases not exposed: {missed[:8]}")
    app_seeds = int(app_config["seeds"])
    app_gammas = [int(x) for x in app_config["resolved_gamma_values"]]
    protocols = [str(x) for x in app_config["stress_protocols"]]
    expected_policy = app_seeds * 3 * len(app_gammas)
    if len(policy) != expected_policy:
        raise AssertionError(f"application policy rows: expected {expected_policy}, found {len(policy)}")
    if len(stress) != expected_policy * len(protocols):
        raise AssertionError(f"application stress rows: expected {expected_policy * len(protocols)}, found {len(stress)}")
    if len(exact) != app_seeds * 4:
        raise AssertionError(f"application exact-subset rows: expected {app_seeds * 4}, found {len(exact)}")
    if len({row["seed"] for row in policy}) != app_seeds:
        raise AssertionError("application portfolio count does not match configuration")
    if sorted({row["protocol"] for row in stress}) != sorted(protocols):
        raise AssertionError("application stress protocols do not match configuration")
    matching_robust = [
        row for row in policy
        if row["method"] == "HullRound" and row["policy_gamma"] == row["eval_gamma"]
    ]
    if any(number(row, "robust_certificate") < -1e-6 for row in matching_robust):
        raise AssertionError("invalid matching-budget application policy")
    if any(
        math.isfinite(number(row, "objective")) and number(row, "robust_certificate") < -1e-6
        for row in exact
    ):
        raise AssertionError("invalid exact-subset application incumbent")
    shortcuts = sorted({row["shortcut"] for row in failures})
    return [
        "V3 EXPERIMENT AUDIT: PASS",
        f"hard rows: {len(hard)}",
        f"hard instances: {len({r['instance_id'] for r in hard})}",
        f"certified reference instances: {len(refs)}",
        f"anytime rows: {len(anytime)}",
        f"certificate-failure cases: {len(failures)}",
        f"application portfolios: {app_seeds}",
        f"application policy rows: {len(policy)}",
        f"application stress rows: {len(stress)}",
        f"application exact-subset rows: {len(exact)}",
        f"unsafe shortcuts tested: {', '.join(shortcuts)}",
        "finite-incumbent robust certificate failures: 0",
        "certified cross-method objective mismatches: 0",
        "undetected unsafe-shortcut failures: 0",
        "invalid matching-budget application policies: 0",
        "invalid exact-subset application incumbents: 0",
    ]


def hard_summary(hard: Sequence[Dict[str, str]], refs: Dict[str, float]) -> Dict[str, object]:
    by_method = group(hard, "method")
    methods: Dict[str, Dict[str, object]] = {}
    for method, data in by_method.items():
        certified = sum(truth(row, "certified_optimal") for row in data)
        gaps = []
        for row in data:
            ref = refs.get(row["instance_id"])
            obj = number(row, "objective")
            if ref is not None and math.isfinite(obj):
                gaps.append(max(0.0, (ref - obj) / max(1.0, abs(ref))))
        methods[method] = {
            "rows": len(data),
            "certified": certified,
            "valid_incumbents": sum(truth(row, "valid_certificate") for row in data),
            "median_runtime": med(number(row, "runtime_seconds") for row in data),
            "p90_runtime": quant((number(row, "runtime_seconds") for row in data), 0.90),
            "median_primal_gap": med(gaps),
            "p90_primal_gap": quant(gaps, 0.90),
            "median_reported_gap": med(number(row, "relative_gap") for row in data),
            "median_overrun": med(number(row, "budget_overrun_ratio") for row in data),
        }
    hr_gaps = []
    for row in by_method.get("hullround", []):
        ref = refs.get(row["instance_id"])
        if ref is not None:
            hr_gaps.append(max(0.0, (ref - number(row, "objective")) / max(1.0, abs(ref))))
    return {
        "instances": len({row["instance_id"] for row in hard}),
        "rows": len(hard),
        "families": sorted({row["family"] for row in hard}),
        "n_values": sorted({int(row["n"]) for row in hard}),
        "reference_instances": len(refs),
        "methods": methods,
        "hullround_gap_median": med(hr_gaps),
        "hullround_gap_p95": quant(hr_gaps, 0.95),
        "hullround_gap_max": max(hr_gaps) if hr_gaps else float("nan"),
    }


def anytime_summary(anytime: Sequence[Dict[str, str]]) -> Dict[str, object]:
    cells: Dict[str, Dict[str, object]] = {}
    for (method, budget), data in _group_pairs(anytime, "method", "time_limit_seconds").items():
        gaps = [number(row, "relative_gap") for row in data]
        cells[f"{method}|{budget}"] = {
            "method": method,
            "budget": float(budget),
            "rows": len(data),
            "certified": sum(truth(row, "certified_optimal") for row in data),
            "valid_incumbents": sum(truth(row, "valid_certificate") for row in data),
            "median_gap": med(gaps),
            "p90_gap": quant(gaps, 0.90),
            "median_runtime": med(number(row, "runtime_seconds") for row in data),
            "median_overrun": med(number(row, "budget_overrun_ratio") for row in data),
        }
    return {"rows": len(anytime), "cells": cells}


def _group_pairs(data: Sequence[Dict[str, str]], a: str, b: str) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    out: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in data:
        out[(row[a], row[b])].append(row)
    return out


def failure_summary(failures: Sequence[Dict[str, str]]) -> Dict[str, object]:
    shortcuts: Dict[str, Dict[str, object]] = {}
    for shortcut, data in group(failures, "shortcut").items():
        errors = [abs(number(row, "objective_error")) for row in data if math.isfinite(number(row, "objective_error"))]
        shortcuts[shortcut] = {
            "cases": len(data),
            "detected": sum(truth(row, "failure_detected") for row in data),
            "detection_rate": sum(truth(row, "failure_detected") for row in data) / len(data),
            "median_absolute_objective_error": med(errors),
        }
    return {"cases": len(failures), "shortcuts": shortcuts}


def application_summary(
    policy: Sequence[Dict[str, str]],
    stress: Sequence[Dict[str, str]],
    exact: Sequence[Dict[str, str]],
    config: Dict[str, object],
) -> Dict[str, object]:
    n = int(config["n"])
    target_gamma = int(math.floor(math.sqrt(n)))
    policy_cells: Dict[str, Dict[str, object]] = {}
    for method in ("nominal", "HullRound", "box"):
        data = [row for row in policy if row["method"] == method and int(row["eval_gamma"]) == target_gamma]
        if method == "HullRound":
            data = [row for row in data if int(row["policy_gamma"]) == target_gamma]
        policy_cells[method] = {
            "rows": len(data),
            "median_revenue_ratio": med(number(row, "revenue_ratio") for row in data),
            "median_certificate": med(number(row, "robust_certificate") for row in data),
            "median_share_changed": med(number(row, "share_changed_vs_nominal") for row in data),
        }
    stress_cells: Dict[str, Dict[str, object]] = {}
    for method in ("nominal", "HullRound", "box"):
        for protocol in config["stress_protocols"]:
            data = [
                row for row in stress
                if row["method"] == method
                and int(row["eval_gamma"]) == target_gamma
                and row["protocol"] == protocol
            ]
            if method == "HullRound":
                data = [row for row in data if int(row["policy_gamma"]) == target_gamma]
            stress_cells[f"{method}|{protocol}"] = {
                "rows": len(data),
                "median_violation_probability": med(number(row, "violation_probability") for row in data),
                "median_mean_shortfall": med(number(row, "mean_margin_shortfall") for row in data),
                "median_q05_margin": med(number(row, "q05_margin") for row in data),
            }
    exact_bnb = [row for row in exact if row["method"] == "Global theta B&B"]
    hr_exact = [row for row in exact if row["method"] == "HullRound"]
    return {
        "portfolios": int(config["seeds"]),
        "n": n,
        "m": int(config["m"]),
        "stress_scenarios": int(config["stress_scenarios"]),
        "protocols": list(config["stress_protocols"]),
        "target_gamma": target_gamma,
        "policy_cells": policy_cells,
        "stress_cells": stress_cells,
        "exact_rows": len(exact),
        "exact_certified": sum(row["status"] == "optimal" for row in exact_bnb),
        "exact_total": len(exact_bnb),
        "hullround_exact_gap_median": med(number(row, "gap_to_exact") for row in hr_exact),
        "hullround_exact_gap_p95": quant((number(row, "gap_to_exact") for row in hr_exact), 0.95),
    }


def fmt(x: object, digits: int = 4) -> str:
    if isinstance(x, int):
        return str(x)
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "--"
    return f"{value:.{digits}f}" if math.isfinite(value) else "--"


def pct(x: object, digits: int = 1) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "--"
    return f"{100.0 * value:.{digits}f}\\%" if math.isfinite(value) else "--"


def write_tables(
    out: Path,
    hard: Dict[str, object],
    anytime: Dict[str, object],
    failures: Dict[str, object],
    application: Dict[str, object],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    hard_lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r"Method & Rows & Opt. & Valid inc. & Median s & p90 primal gap & Median reported gap \\", r"\midrule",
    ]
    for method in ("hullround", "theta_bnb", "theta_enum_highs", "scip", "highs"):
        data = hard["methods"].get(method)
        if not data:
            continue
        hard_lines.append(
            f"{METHOD_LABEL[method]} & {data['rows']} & {data['certified']} & {data['valid_incumbents']} & "
            f"{fmt(data['median_runtime'], 3)} & {pct(data['p90_primal_gap'], 3)} & {pct(data['median_reported_gap'], 3)} \\\\"
        )
    hard_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "hard_benchmark_summary.tex").write_text("\n".join(hard_lines) + "\n", encoding="utf-8")

    anytime_lines = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
        r"Method & Budget s & Rows & Opt. & Valid inc. & Median gap & p90 gap & Median s & Overrun \\", r"\midrule",
    ]
    cells = sorted(anytime["cells"].values(), key=lambda x: (float(x["budget"]), str(x["method"])))
    for data in cells:
        anytime_lines.append(
            f"{METHOD_LABEL[data['method']]} & {fmt(data['budget'], 1)} & {data['rows']} & {data['certified']} & "
            f"{data['valid_incumbents']} & {pct(data['median_gap'], 3)} & {pct(data['p90_gap'], 3)} & "
            f"{fmt(data['median_runtime'], 3)} & {fmt(data['median_overrun'], 2)}$\\times$ \\\\"
        )
    anytime_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "anytime_frontier_summary.tex").write_text("\n".join(anytime_lines) + "\n", encoding="utf-8")

    failure_lines = [
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Unsafe shortcut & Cases & Exposed & Exposure rate & Median absolute objective error \\", r"\midrule",
    ]
    labels = {
        "theta_zero_only": r"Check only $\theta=0$",
        "lp_hull_as_integer_filter": "Use LP hull as integer filter",
        "gap_before_all_theta_bounds": r"Report gap before all $\theta$ bounds",
        "no_original_robust_recheck": "Skip original robust recheck",
    }
    for shortcut, data in failures["shortcuts"].items():
        failure_lines.append(
            f"{labels.get(shortcut, shortcut)} & {data['cases']} & {data['detected']} & "
            f"{pct(data['detection_rate'], 1)} & {fmt(data['median_absolute_objective_error'], 3)} \\\\"
        )
    failure_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "certificate_failure_summary.tex").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    protocol_labels = {
        "iid": "IID bounded",
        "common_factor": "Common factor",
        "segment_block": "Segment block",
        "heavy_tail": "Heavy tail",
        "undercalibrated": "Under-calibrated",
        "cross_price_substitution": "Cross-price sensitivity",
    }
    app_lines = [
        r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
        r"Policy & Stress protocol & Portfolios & Median violation & Median mean shortfall \\", r"\midrule",
    ]
    method_labels = {"nominal": "Nominal", "HullRound": "HullRound", "box": "Box robust"}
    for method in ("nominal", "HullRound", "box"):
        for protocol in application["protocols"]:
            data = application["stress_cells"][f"{method}|{protocol}"]
            app_lines.append(
                f"{method_labels[method]} & {protocol_labels.get(protocol, protocol)} & {data['rows']} & "
                f"{pct(data['median_violation_probability'], 2)} & {fmt(data['median_mean_shortfall'], 2)} \\\\"
            )
        if method != "box":
            app_lines.append(r"\addlinespace")
    app_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "application_stress_summary.tex").write_text("\n".join(app_lines) + "\n", encoding="utf-8")


def write_macros(
    path: Path,
    hard: Dict[str, object],
    anytime: Dict[str, object],
    failures: Dict[str, object],
    application: Dict[str, object],
) -> None:
    methods = hard["methods"]
    largest_budget = max(float(cell["budget"]) for cell in anytime["cells"].values())
    def cell(method: str) -> Dict[str, object]:
        matches = [
            value for value in anytime["cells"].values()
            if value["method"] == method and abs(float(value["budget"]) - largest_budget) < 1e-12
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one anytime cell for {method} at {largest_budget}")
        return matches[0]
    app_hr = application["policy_cells"]["HullRound"]
    commands = {
        "VThreeHardInstances": hard["instances"],
        "VThreeHardRows": hard["rows"],
        "VThreeHardReferenceInstances": hard["reference_instances"],
        "VThreeHardMaxN": max(hard["n_values"]),
        "VThreeFailureCases": failures["cases"],
        "VThreeFailureExposureRate": pct(sum(x["detected"] for x in failures["shortcuts"].values()) / failures["cases"], 1),
        "VThreeHullGapMedian": pct(hard["hullround_gap_median"], 4),
        "VThreeHullGapPtfive": pct(hard["hullround_gap_p95"], 4),
        "VThreeHullGapMax": pct(hard["hullround_gap_max"], 4),
        "VThreeBNBCertified": methods["theta_bnb"]["certified"],
        "VThreeSCIPCertified": methods["scip"]["certified"],
        "VThreeHiGHSCertified": methods["highs"]["certified"],
        "VThreeThetaEnumCertified": methods["theta_enum_highs"]["certified"],
        "VThreeAnytimeBudget": fmt(largest_budget, 1),
        "VThreeAnytimeBNBGap": pct(cell("theta_bnb")["median_gap"], 3),
        "VThreeAnytimeSCIPGap": pct(cell("scip")["median_gap"], 3),
        "VThreeAnytimeHiGHSGap": pct(cell("highs")["median_gap"], 3),
        "VThreeAppPortfolios": application["portfolios"],
        "VThreeAppProducts": application["n"],
        "VThreeAppScenarios": application["stress_scenarios"],
        "VThreeAppProtocols": len(application["protocols"]),
        "VThreeAppGamma": application["target_gamma"],
        "VThreeAppRevenueRetention": pct(app_hr["median_revenue_ratio"], 2),
        "VThreeAppShareChanged": pct(app_hr["median_share_changed"], 2),
        "VThreeAppExactCertified": application["exact_certified"],
        "VThreeAppExactTotal": application["exact_total"],
        "VThreeAppHullGapPtfive": pct(application["hullround_exact_gap_p95"], 4),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in commands.items()) + "\n", encoding="utf-8")


def plot_evidence(path: Path, hard_rows: Sequence[Dict[str, str]], anytime_rows: Sequence[Dict[str, str]], failure_rows: Sequence[Dict[str, str]], refs: Dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    apply_top_journal_style()
    fig, axs = plt.subplots(2, 2, figsize=figure_size("double"))
    methods = ["theta_bnb", "theta_enum_highs", "scip", "highs"]
    ns = sorted({int(row["n"]) for row in hard_rows})
    for method in methods:
        shares = []
        for n in ns:
            data = [row for row in hard_rows if row["method"] == method and int(row["n"]) == n]
            shares.append(sum(truth(row, "certified_optimal") for row in data) / len(data) if data else float("nan"))
        axs[0, 0].plot(
            ns,
            shares,
            marker=METHOD_MARKER[method],
            linestyle=METHOD_LINESTYLE[method],
            color=COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=METHOD_LABEL[method],
        )
    axs[0, 0].set_ylim(-0.03, 1.03)
    axs[0, 0].set_xticks(ns)
    axs[0, 0].set_xlabel("Number of classes $n$")
    axs[0, 0].set_ylabel("Certified-optimal share")
    axs[0, 0].legend(fontsize=6.15, loc="center right")
    polish_axis(axs[0, 0], minor_y=True)
    panel_label(axs[0, 0], "(a)")

    for method in ("theta_bnb", "scip", "highs"):
        budgets = sorted({number(row, "time_limit_seconds") for row in anytime_rows if row["method"] == method})
        mids = [med(number(row, "relative_gap") for row in anytime_rows if row["method"] == method and number(row, "time_limit_seconds") == b) for b in budgets]
        p90 = [quant((number(row, "relative_gap") for row in anytime_rows if row["method"] == method and number(row, "time_limit_seconds") == b), 0.90) for b in budgets]
        axs[0, 1].plot(
            budgets,
            mids,
            marker=METHOD_MARKER[method],
            color=COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=METHOD_LABEL[method],
        )
        axs[0, 1].plot(budgets, p90, linestyle=":", color=COLORS[method], alpha=0.82)
    axs[0, 1].set_xscale("log")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlabel("Nominal time budget (seconds)")
    axs[0, 1].set_ylabel("Relative gap")
    polish_axis(axs[0, 1], grid_axis="both")
    panel_label(axs[0, 1], "(b)")

    shortcuts = sorted({row["shortcut"] for row in failure_rows})
    detected = [sum(truth(row, "failure_detected") for row in failure_rows if row["shortcut"] == s) for s in shortcuts]
    totals = [sum(row["shortcut"] == s for row in failure_rows) for s in shortcuts]
    rates = [d / total for d, total in zip(detected, totals)]
    short_labels = ["All-$\theta$ bounds", "LP-hull filter", "Robust recheck", "$\theta=0$ only"]
    y = np.arange(len(shortcuts), dtype=float)
    axs[1, 0].hlines(y, 0, rates, color="0.70", linewidth=1.2)
    axs[1, 0].scatter(rates, y, s=25, color=STYLE_COLORS["red"], marker="D", zorder=3)
    for yi, rate, count, total in zip(y, rates, detected, totals):
        axs[1, 0].annotate(
            f"{count}/{total}",
            xy=(rate, yi),
            xytext=(-7, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=6.4,
            color="0.20",
        )
    axs[1, 0].set_xlim(0, 1.06)
    axs[1, 0].set_xticks([0, 0.5, 1.0])
    axs[1, 0].set_yticks(y, short_labels)
    axs[1, 0].invert_yaxis()
    axs[1, 0].set_xlabel("Unsafe cases exposed")
    polish_axis(axs[1, 0], grid_axis="x")
    panel_label(axs[1, 0], "(c)")

    families = list(FAMILY_LABEL)
    gap_groups = []
    for family in families:
        values = []
        for row in hard_rows:
            if row["method"] != "hullround" or row["family"] != family or row["instance_id"] not in refs:
                continue
            ref = refs[row["instance_id"]]
            values.append(10_000.0 * max(0.0, (ref - number(row, "objective")) / max(1.0, abs(ref))))
        gap_groups.append(values)
    box = axs[1, 1].boxplot(
        gap_groups,
        showfliers=False,
        patch_artist=True,
        widths=0.48,
        boxprops={"facecolor": STYLE_COLORS["pale_blue"], "edgecolor": COLORS["theta_bnb"]},
        whiskerprops={"color": "0.30"},
        capprops={"color": "0.30"},
        medianprops={"color": "black", "linewidth": 1.25},
    )
    del box
    for idx, values in enumerate(gap_groups, start=1):
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.zeros(len(values))
        axs[1, 1].scatter(
            idx + jitter,
            values,
            s=8,
            facecolor="white",
            edgecolor=COLORS["theta_bnb"],
            linewidth=0.45,
            alpha=0.68,
            zorder=3,
        )
        axs[1, 1].text(idx, 1.015, f"$N={len(values)}$", transform=axs[1, 1].get_xaxis_transform(), ha="center", va="bottom", fontsize=5.9, color="0.30", clip_on=False)
    axs[1, 1].set_yscale("symlog", linthresh=0.01)
    axs[1, 1].set_xticks(range(1, len(families) + 1), [FAMILY_LABEL[f].replace(" ", "\n") for f in families])
    axs[1, 1].set_ylabel("HullRound gap (basis points)")
    axs[1, 1].axhline(0, color="0.35", linewidth=0.7, zorder=1)
    polish_axis(axs[1, 1])
    panel_label(axs[1, 1], "(d)")
    fig.subplots_adjust(wspace=0.34, hspace=0.43)
    save_figure(fig, path, {"source": "v3 consolidated hard, anytime, and certificate-failure CSVs", "panels": 4})


def plot_application(
    path: Path,
    policy_rows: Sequence[Dict[str, str]],
    stress_rows: Sequence[Dict[str, str]],
    exact_rows: Sequence[Dict[str, str]],
    config: Dict[str, object],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    apply_top_journal_style()
    fig, axs = plt.subplots(2, 2, figsize=figure_size("double"))
    gammas = sorted(int(x) for x in config["resolved_gamma_values"])
    hr_policy = [
        row for row in policy_rows
        if row["method"] == "HullRound" and row["policy_gamma"] == row["eval_gamma"]
    ]
    revenue = [med(number(row, "revenue_ratio") for row in hr_policy if int(row["eval_gamma"]) == gamma) for gamma in gammas]
    revenue_q1 = [quant((number(row, "revenue_ratio") for row in hr_policy if int(row["eval_gamma"]) == gamma), 0.25) for gamma in gammas]
    revenue_q3 = [quant((number(row, "revenue_ratio") for row in hr_policy if int(row["eval_gamma"]) == gamma), 0.75) for gamma in gammas]
    axs[0, 0].plot(gammas, revenue, marker="o", color=STYLE_COLORS["blue"], markeredgecolor="white", markeredgewidth=0.45)
    axs[0, 0].fill_between(gammas, revenue_q1, revenue_q3, color=STYLE_COLORS["sky"], alpha=0.24, linewidth=0)
    axs[0, 0].set_xlabel("Uncertainty budget $\\Gamma$")
    axs[0, 0].set_ylabel("Revenue ratio")
    axs[0, 0].legend(
        handles=[
            Line2D([], [], color=STYLE_COLORS["blue"], marker="o", label="Median"),
            Patch(facecolor=STYLE_COLORS["sky"], alpha=0.24, edgecolor="none", label="Interquartile range"),
        ],
        loc="lower left",
        fontsize=6.2,
    )
    polish_axis(axs[0, 0], minor_y=True)
    panel_label(axs[0, 0], "(a)")

    protocol_labels = {
        "iid": "IID", "common_factor": "Common factor", "segment_block": "Segment block",
        "heavy_tail": "Heavy tail", "undercalibrated": "Under-calibrated",
        "cross_price_substitution": "Cross-price",
    }
    protocol_colors = [STYLE_COLORS[k] for k in ("blue", "orange", "green", "purple", "red", "sky")]
    protocol_markers = ["o", "s", "^", "D", "v", "P"]
    protocol_lines = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 1))]
    for protocol, color, marker, linestyle in zip(config["stress_protocols"], protocol_colors, protocol_markers, protocol_lines):
        values = [
            med(
                number(row, "violation_probability") for row in stress_rows
                if row["method"] == "HullRound"
                and row["policy_gamma"] == row["eval_gamma"]
                and int(row["eval_gamma"]) == gamma
                and row["protocol"] == protocol
            )
            for gamma in gammas
        ]
        axs[0, 1].plot(
            gammas,
            values,
            marker=marker,
            linestyle=linestyle,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=protocol_labels[str(protocol)],
        )
    axs[0, 1].set_xlabel("Uncertainty budget $\\Gamma$")
    axs[0, 1].set_ylabel("Median violation probability")
    axs[0, 1].legend(fontsize=5.8, ncol=2, loc="upper right")
    polish_axis(axs[0, 1], minor_y=True)
    panel_label(axs[0, 1], "(b)")

    target = int(math.floor(math.sqrt(int(config["n"]))))
    methods = ("nominal", "HullRound", "box")
    method_labels = ("Nominal", "HullRound", "Box robust")
    y = np.arange(len(config["stress_protocols"]), dtype=float)
    offsets = (-0.18, 0.0, 0.18)
    method_colors = ("0.25", STYLE_COLORS["blue"], STYLE_COLORS["green"])
    method_markers = ("s", "o", "D")
    for idx, (method, label) in enumerate(zip(methods, method_labels)):
        vals = []
        for protocol in config["stress_protocols"]:
            data = [
                row for row in stress_rows
                if row["method"] == method and int(row["eval_gamma"]) == target and row["protocol"] == protocol
            ]
            if method == "HullRound":
                data = [row for row in data if int(row["policy_gamma"]) == target]
            vals.append(med(number(row, "violation_probability") for row in data))
        axs[1, 0].scatter(
            vals,
            y + offsets[idx],
            s=25,
            marker=method_markers[idx],
            facecolor=method_colors[idx] if idx != 1 else "white",
            edgecolor=method_colors[idx],
            linewidth=0.9,
            label=label,
            zorder=3,
        )
    axs[1, 0].set_yticks(
        y,
        [protocol_labels[str(p)] for p in config["stress_protocols"]],
        fontsize=6.1,
    )
    axs[1, 0].invert_yaxis()
    axs[1, 0].set_xlim(-0.025, 0.82)
    axs[1, 0].set_xlabel(f"Median violation probability at $\\Gamma={target}$")
    axs[1, 0].axvline(0, color="0.45", linewidth=0.7)
    axs[1, 0].legend(fontsize=6.1, loc="lower right")
    polish_axis(axs[1, 0], grid_axis="x")
    panel_label(axs[1, 0], "(c)")

    exact_gammas = sorted({int(row["gamma"]) for row in exact_rows if row["method"] == "HullRound"})
    gap_groups = [
        [10_000 * number(row, "gap_to_exact") for row in exact_rows if row["method"] == "HullRound" and int(row["gamma"]) == gamma]
        for gamma in exact_gammas
    ]
    axs[1, 1].boxplot(
        gap_groups,
        showfliers=False,
        patch_artist=True,
        widths=0.42,
        boxprops={"facecolor": STYLE_COLORS["pale_blue"], "edgecolor": STYLE_COLORS["blue"]},
        whiskerprops={"color": "0.30"},
        capprops={"color": "0.30"},
        medianprops={"color": "black", "linewidth": 1.25},
    )
    for idx, values in enumerate(gap_groups, start=1):
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(len(values))
        axs[1, 1].scatter(
            idx + jitter,
            values,
            s=9,
            facecolor="white",
            edgecolor=STYLE_COLORS["blue"],
            linewidth=0.45,
            alpha=0.72,
            zorder=3,
        )
        axs[1, 1].text(idx, 1.015, f"$N={len(values)}$", transform=axs[1, 1].get_xaxis_transform(), ha="center", va="bottom", fontsize=6.0, color="0.30", clip_on=False)
    axs[1, 1].set_xticks(range(1, len(exact_gammas) + 1), [str(gamma) for gamma in exact_gammas])
    axs[1, 1].set_xlabel("Exact-subset uncertainty budget $\\Gamma$")
    axs[1, 1].set_ylabel("HullRound gap (basis points)")
    axs[1, 1].axhline(0, color="0.45", linewidth=0.7)
    polish_axis(axs[1, 1])
    panel_label(axs[1, 1], "(d)")
    fig.subplots_adjust(wspace=0.35, hspace=0.44)
    save_figure(fig, path, {"source": "v3 semi-synthetic policy, stress, and exact-subset CSVs", "panels": 4})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/v3_experiments_20260718")
    parser.add_argument("--table-dir", default="paper_versions/v3/tables/v3_experiments")
    parser.add_argument("--figure-dir", default="paper_versions/v3/figures/v3_experiments")
    parser.add_argument("--macro-file", default="paper_versions/v3/auto/v3_experiment_numbers.tex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    hard_rows = rows(input_dir / "hard_benchmark.csv")
    anytime_rows = rows(input_dir / "anytime_frontier.csv")
    failure_rows = rows(input_dir / "certificate_failure_audit.csv")
    app_dir = input_dir / "application"
    policy_rows = rows(app_dir / "pathC_policy_results.csv")
    stress_rows = rows(app_dir / "pathC_stress_results.csv")
    exact_rows = rows(app_dir / "pathC_exact_subset_results.csv")
    app_config = json.loads((app_dir / "pathC_config.json").read_text(encoding="utf-8"))
    audit_lines = audit(hard_rows, anytime_rows, failure_rows, policy_rows, stress_rows, exact_rows, app_config)
    refs = reference_objectives(hard_rows)
    hard = hard_summary(hard_rows, refs)
    anytime = anytime_summary(anytime_rows)
    failures = failure_summary(failure_rows)
    application = application_summary(policy_rows, stress_rows, exact_rows, app_config)
    summary = {"hard": hard, "anytime": anytime, "failures": failures, "application": application}
    (input_dir / "audit.txt").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    (input_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tables(Path(args.table_dir), hard, anytime, failures, application)
    write_macros(Path(args.macro_file), hard, anytime, failures, application)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_evidence(figure_dir / "v3_experimental_evidence.pdf", hard_rows, anytime_rows, failure_rows, refs)
    plot_application(figure_dir / "v3_application_evidence.pdf", policy_rows, stress_rows, exact_rows, app_config)
    print("\n".join(audit_lines))


if __name__ == "__main__":
    main()
