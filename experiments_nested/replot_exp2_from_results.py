"""Recreate nested Exp2 figures from saved CSV outputs (no solver rerun)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict


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
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    # Import plotting helpers from the experiment script itself.
    from exp2_scalability import apply_pub_style, plot_hull_boxplot, plot_runtime, plot_runtime_breakdown

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_rows = _read_csv(results_dir / "exp2_runtime.csv")
    hull_rows_theta0 = _read_csv(results_dir / "exp2_hull_sizes_theta0.csv")

    # Infer plot axes values from saved files.
    n_values = sorted({int(float(r["n"])) for r in runtime_rows})
    m_values_runtime = sorted({int(float(r["m"])) for r in runtime_rows})
    m_values_hull = sorted({int(float(r["m"])) for r in hull_rows_theta0})
    n_hull = int(sorted({int(float(r["n"])) for r in hull_rows_theta0})[-1])

    apply_pub_style()
    plot_runtime(runtime_rows, m_values_runtime, n_values, output_dir)
    plot_hull_boxplot(hull_rows_theta0, m_values_hull, n_hull, output_dir)
    plot_runtime_breakdown(runtime_rows, n_values, output_dir)

    print(f"Replotted Exp2 figures from {results_dir}/")
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()

