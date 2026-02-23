"""Recreate nested Exp3 figures from saved CSV outputs (no solver rerun)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results_nested")
    parser.add_argument("--output-dir", type=str, default="figs_nested")
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(results_dir / "exp3_risk_frontier.csv")
    heatmap_rows = _read_csv(results_dir / "exp3_tightness_heatmap.csv")

    if not rows:
        raise RuntimeError(f"No rows in {results_dir / 'exp3_risk_frontier.csv'}")

    # Reuse plot style and plotting functions from the experiment script.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from exp3_risk_frontier import apply_pub_style, plot_frontier, plot_heatmap, plot_tail_margin

    n_values = sorted({int(float(r["n"])) for r in rows})
    if len(n_values) != 1:
        raise RuntimeError(f"Expected one n in Exp3 CSV, found: {n_values}")
    n = n_values[0]

    alpha_values = sorted({round(float(r["alpha"]), 2) for r in rows})

    apply_pub_style()
    plot_frontier(rows, alpha_values, n, output_dir)
    plot_tail_margin(rows, alpha_values, n, output_dir)
    plot_heatmap(heatmap_rows, n, output_dir)

    print(f"Replotted Exp3 figures from {results_dir}/")
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()

