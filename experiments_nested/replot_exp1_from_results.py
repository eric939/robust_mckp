"""Recreate nested Exp1 figures from saved CSV results (no solver rerun)."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


GAMMA_REGIMES = [
    ("0", lambda n: 0),
    ("sqrt(n)", lambda n: int(math.floor(math.sqrt(n)))),
    ("0.1n", lambda n: int(math.floor(0.1 * n))),
]

REGIME_DISPLAY = {
    "0": r"$\Gamma = 0$",
    "sqrt(n)": r"$\Gamma = \lfloor\sqrt{n}\rfloor$",
    "0.1n": r"$\Gamma = \lfloor 0.1n \rfloor$",
}


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, default="results_nested/exp1_gap.csv")
    parser.add_argument("--output-dir", type=str, default="figs_nested")
    parser.add_argument(
        "--anchor",
        type=str,
        choices=["median", "mean", "both"],
        default="median",
        help="Horizontal anchor line in panel (b).",
    )
    return parser.parse_args()


def _to_float(row: Dict[str, str], key: str) -> float:
    val = row.get(key, "")
    if val is None:
        return float("nan")
    sval = str(val).strip()
    if sval == "":
        return float("nan")
    try:
        return float(sval)
    except Exception:
        return float("nan")


def _load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n = int(float(r["n"]))
            gamma = int(float(r["gamma"]))
            gamma_regime = str(r["gamma_regime"])
            l_rd = _to_float(r, "l_rd")
            bound = _to_float(r, "bound")
            gap_lp_rd = _to_float(r, "gap_lp_rd")
            ratio = _to_float(r, "ratio_lrd_bound")
            n_gap_lp = _to_float(r, "n_gap_lp")

            if not np.isfinite(ratio):
                if bound > 0:
                    ratio = l_rd / bound
                elif abs(l_rd) <= 1e-12:
                    ratio = 0.0
            if not np.isfinite(n_gap_lp) and np.isfinite(gap_lp_rd):
                n_gap_lp = n * gap_lp_rd

            rows.append(
                {
                    "n": n,
                    "gamma": gamma,
                    "gamma_regime": gamma_regime,
                    "l_rd": l_rd,
                    "bound": bound,
                    "gap_lp_rd": gap_lp_rd,
                    "ratio_lrd_bound": ratio,
                    "n_gap_lp": n_gap_lp,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    rows = _load_rows(input_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {input_csv}")

    plt = try_import_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib not available")

    colors = {"0": "#1f77b4", "sqrt(n)": "#ff7f0e", "0.1n": "#2ca02c"}
    markers = {"0": "o", "sqrt(n)": "s", "0.1n": "D"}

    # Panel (a): ratio L_rd / bound vs n
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for regime_label, _ in GAMMA_REGIMES:
        regime_rows = sorted([r for r in rows if r["gamma_regime"] == regime_label], key=lambda r: int(r["n"]))
        if not regime_rows:
            continue
        ns = np.array([r["n"] for r in regime_rows], dtype=float)
        ratio = np.array([r["ratio_lrd_bound"] for r in regime_rows], dtype=float)
        ax.plot(
            ns,
            ratio,
            marker=markers.get(regime_label, "o"),
            linestyle="-",
            color=colors.get(regime_label, "tab:blue"),
            markersize=6,
            label=REGIME_DISPLAY.get(regime_label, regime_label),
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8, label=r"Bound threshold $=1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$L_{\mathrm{rd}} / \Delta V_{\max}^{\theta}$")
    ax.set_title(r"Normalized round-down loss vs $n$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(output_dir / "gap_loss_vs_n.pdf")
    plt.close(fig)

    # Panel (b): n * Gap_LP vs n + anchor line(s)
    fig2, ax2 = plt.subplots(figsize=(5.1, 4.2))
    for regime_label, _ in GAMMA_REGIMES:
        pts = sorted([r for r in rows if r["gamma_regime"] == regime_label], key=lambda r: int(r["n"]))
        if not pts:
            continue
        ns = np.array([r["n"] for r in pts], dtype=float)
        scaled = np.array([r["n_gap_lp"] for r in pts], dtype=float)
        ax2.plot(
            ns,
            scaled,
            color=colors.get(regime_label, "tab:blue"),
            marker=markers.get(regime_label, "o"),
            linewidth=1.5,
            markersize=6,
            alpha=0.9,
            label=REGIME_DISPLAY.get(regime_label, regime_label),
        )

    all_scaled = np.array([float(r["n_gap_lp"]) for r in rows], dtype=float)
    all_scaled = all_scaled[np.isfinite(all_scaled)]
    if all_scaled.size:
        if args.anchor in {"median", "both"}:
            y_med = float(np.median(all_scaled))
            ax2.axhline(
                y_med,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=r"Median of $n \times \mathrm{Gap}_{\mathrm{LP}}$",
            )
        if args.anchor in {"mean", "both"}:
            y_mean = float(np.mean(all_scaled))
            ax2.axhline(
                y_mean,
                color="gray",
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
                label=r"Mean of $n \times \mathrm{Gap}_{\mathrm{LP}}$",
            )

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$n$")
    ax2.set_ylabel(r"$n \times \mathrm{Gap}_{\mathrm{LP}}$")
    ax2.set_title(r"Scaled relative gap vs $n$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(output_dir / "loss_vs_bound.pdf")
    plt.close(fig2)

    print(f"Replotted Exp1 figures from {input_csv}")
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()

