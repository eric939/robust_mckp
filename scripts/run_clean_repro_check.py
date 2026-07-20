#!/usr/bin/env python3
"""Run a bounded code-only reproducibility check."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "clean_repro_check"
SUMMARY = OUT_DIR / "clean_repro_check_summary.txt"

KEY_SCRIPTS = [
    "scripts/run_v3_experiments.py",
    "scripts/build_v3_experiment_evidence.py",
    "scripts/benchmark_solvers.py",
    "scripts/run_pathC_data_calibration.py",
    "scripts/run_pathC_semisynthetic_application.py",
    "scripts/run_parametric_sweep_ablation.py",
    "scripts/run_parametric_sweep_benchmarks.py",
    "scripts/run_segment_local_budget_experiments.py",
    "scripts/run_true_clean_room_check.py",
]


def log(message: str) -> None:
    print(message, flush=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    log("$ " + " ".join(cmd))
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)
    if proc.stdout:
        for line in proc.stdout.rstrip().splitlines():
            log("  " + line)
    if proc.stderr:
        for line in proc.stderr.rstrip().splitlines():
            log("  [stderr] " + line)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run bounded smoke checks.")
    args = parser.parse_args()
    if not args.quick:
        parser.error("Only --quick mode is currently implemented.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("Clean code-only reproducibility check\n", encoding="utf-8")

    try:
        log("1. Import check")
        run([sys.executable, "-c", "import robust_mckp; print('robust_mckp import ok')"])

        log("2. Unit tests")
        run([sys.executable, "-m", "pytest", "-q"])

        log("3. Py-compile experiment scripts")
        run([sys.executable, "-m", "py_compile", *KEY_SCRIPTS])

        smoke = "results/clean_repro_check/v3_smoke"
        log("4. Run consolidated certificate-failure smoke")
        run(
            [
                sys.executable,
                "scripts/run_v3_experiments.py", "failures", "--output-dir", smoke,
                "--failure-replicates", "3",
            ]
        )

        log("5. Run consolidated hard-instance smoke")
        run(
            [
                sys.executable, "scripts/run_v3_experiments.py", "hard", "--output-dir", smoke,
                "--families", "dense_frontier", "--n-values", "12", "--m", "6",
                "--gamma-modes", "sqrt", "--seeds", "1", "--time-limit", "0.25",
            ]
        )

        log("6. Run consolidated anytime smoke")
        run(
            [
                sys.executable, "scripts/run_v3_experiments.py", "anytime", "--output-dir", smoke,
                "--anytime-families", "dense_frontier", "--anytime-n-values", "12", "--m", "6",
                "--anytime-budgets", "0.1", "--anytime-seeds", "1",
            ]
        )

        log("7. Generate synthetic application calibration")
        run(
            [
                sys.executable, "scripts/run_pathC_data_calibration.py", "--source", "synthetic_only",
                "--output-dir", "results/pathC/calibration",
            ]
        )

        log("8. Run consolidated application smoke")
        run(
            [
                sys.executable, "scripts/run_v3_experiments.py", "application", "--output-dir", smoke,
                "--application-seeds", "1", "--application-n", "30", "--application-m", "6",
                "--application-scenarios", "100", "--application-exact-n", "12",
                "--application-exact-time-limit", "1",
            ]
        )

        log("9. Audit and build consolidated evidence")
        run(
            [
                sys.executable, "scripts/build_v3_experiment_evidence.py", "--input-dir", smoke,
                "--table-dir", f"{smoke}/tables", "--figure-dir", f"{smoke}/figures",
                "--macro-file", f"{smoke}/v3_numbers.tex",
            ]
        )

        log("RESULT clean reproducibility check passed")
        return 0
    except Exception as exc:
        log(f"RESULT clean reproducibility check failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
