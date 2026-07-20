#!/usr/bin/env python3
"""Preregistered validation and scaling campaign for the compressed oracle."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research.compressed_interval_oracle import CompressedThetaIntervalOracle  # noqa: E402
from research.novelty_go_no_go import (  # noqa: E402
    ThetaIntervalOracle,
    _gamma_for,
    _write_csv,
    full_theta_lp_scan,
)
from research.structural_feasibility_study import (  # noqa: E402
    adaptive_clique_interval_bound,
    adaptive_interval_bound,
    bounded_theta_clique_lp,
)
from research.benchmark_instances import build_benchmark_instance  # noqa: E402


FAMILIES = ("dense_frontier", "correlated_risk", "near_tie", "many_breakpoints")

# Fixed before the confirmatory scaling run.  Passing all gates confirms a
# competitive bounding contribution, not superiority of the complete integer
# solver or publication-level novelty by itself.
PREREGISTERED_GATES = {
    "adaptive_tolerance_rate_min": 1.0,
    "large_case_geomean_speedup_min": 2.0,
    "large_case_win_rate_min": 0.75,
    "families_with_median_speedup_gt_one_min": 3,
    "root_dominance_rate_min": 0.75,
    "dense_identity_max_absolute_error": 2e-6,
}


def _median_timed(
    function: Callable[[], dict],
    repeats: int,
) -> tuple[float, dict, list[float]]:
    times: list[float] = []
    results: list[dict] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
        results.append(result)
    index = int(np.argsort(np.asarray(times))[len(times) // 2])
    return float(times[index]), results[index], times


def _compressed_root(instance) -> dict:
    oracle = CompressedThetaIntervalOracle(instance)
    result = oracle.bound(0, len(oracle.thetas) - 1)
    return {
        "bound": result.upper_bound,
        "evaluations": result.evaluations,
        "preprocessing_seconds": oracle.preprocessing_seconds,
    }


def _clique_root(instance) -> dict:
    thetas = CompressedThetaIntervalOracle(instance).thetas
    bound, seconds, theta = bounded_theta_clique_lp(
        instance, float(thetas[0]), float(thetas[-1])
    )
    return {"bound": bound, "lp_seconds": seconds, "theta": theta}


def _compressed_adaptive(instance, time_limit: float) -> dict:
    oracle = CompressedThetaIntervalOracle(instance)
    result = adaptive_interval_bound(
        instance,
        oracle,
        relative_tolerance=1e-6,
        time_limit=time_limit,
    )
    result["preprocessing_seconds"] = oracle.preprocessing_seconds
    return result


def _clique_adaptive(instance, time_limit: float) -> dict:
    return adaptive_clique_interval_bound(
        instance,
        relative_tolerance=1e-6,
        time_limit=time_limit,
    )


def run_validation(output_dir: Path, seeds: Sequence[int]) -> dict:
    rows: list[dict] = []
    max_error = 0.0
    min_validity_slack = float("inf")
    for family, seed in itertools.product(FAMILIES, seeds):
        instance = build_benchmark_instance(family, 90, 6, _gamma_for(90, "sqrt"), seed)
        dense = ThetaIntervalOracle(instance)
        compressed = CompressedThetaIntervalOracle(instance)
        lambda_values = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
        local_error = 0.0
        for lambda_value in lambda_values:
            expected = dense.values_at_lambda(lambda_value, 0, len(dense.thetas) - 1)
            actual = compressed.values_at_lambda(
                lambda_value, 0, len(compressed.thetas) - 1
            )
            finite = np.isfinite(expected) & np.isfinite(actual)
            if np.any(finite):
                local_error = max(
                    local_error, float(np.max(np.abs(expected[finite] - actual[finite])))
                )
        dense_bound = dense.bound(0, len(dense.thetas) - 1)
        compressed_bound = compressed.bound(0, len(compressed.thetas) - 1)
        scan, scan_seconds, _ = full_theta_lp_scan(instance)
        validity_slack = compressed_bound.upper_bound - scan
        if validity_slack < -2e-6:
            raise AssertionError(f"compressed bound invalid on {instance.name}: {validity_slack}")
        max_error = max(max_error, local_error, abs(dense_bound.upper_bound - compressed_bound.upper_bound))
        min_validity_slack = min(min_validity_slack, validity_slack)
        rows.append(
            {
                "instance": instance.name,
                "family": family,
                "seed": seed,
                "theta_count": len(dense.thetas),
                "max_lambda_value_error": local_error,
                "dense_bound": dense_bound.upper_bound,
                "compressed_bound": compressed_bound.upper_bound,
                "bound_absolute_error": abs(dense_bound.upper_bound - compressed_bound.upper_bound),
                "full_scan_bound": scan,
                "compressed_validity_slack": validity_slack,
                "full_scan_seconds": scan_seconds,
            }
        )
        _write_csv(output_dir / "validation.csv", rows)
    summary = {
        "instances": len(rows),
        "max_dense_identity_absolute_error": max_error,
        "minimum_validity_slack": min_validity_slack,
        "identity_gate_pass": max_error
        <= PREREGISTERED_GATES["dense_identity_max_absolute_error"],
        "validity_gate_pass": min_validity_slack >= -2e-6,
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_scaling(
    output_dir: Path,
    sizes: Sequence[int],
    seeds: Sequence[int],
    repeats: int,
    time_limit: float,
) -> dict:
    rows: list[dict] = []
    for family, n, seed in itertools.product(FAMILIES, sizes, seeds):
        instance = build_benchmark_instance(family, n, 6, _gamma_for(n, "sqrt"), seed)

        # Alternate the first method by seed to reduce systematic thermal/order bias.
        if seed % 2:
            clique_root_time, clique_root, _ = _median_timed(
                lambda: _clique_root(instance), repeats
            )
            compressed_root_time, compressed_root, _ = _median_timed(
                lambda: _compressed_root(instance), repeats
            )
            clique_time, clique, _ = _median_timed(
                lambda: _clique_adaptive(instance, time_limit), repeats
            )
            compressed_time, compressed, _ = _median_timed(
                lambda: _compressed_adaptive(instance, time_limit), repeats
            )
        else:
            compressed_root_time, compressed_root, _ = _median_timed(
                lambda: _compressed_root(instance), repeats
            )
            clique_root_time, clique_root, _ = _median_timed(
                lambda: _clique_root(instance), repeats
            )
            compressed_time, compressed, _ = _median_timed(
                lambda: _compressed_adaptive(instance, time_limit), repeats
            )
            clique_time, clique, _ = _median_timed(
                lambda: _clique_adaptive(instance, time_limit), repeats
            )

        scale = max(1.0, abs(float(clique_root["bound"])))
        rows.append(
            {
                "instance": instance.name,
                "family": family,
                "n": n,
                "seed": seed,
                "theta_count": len(CompressedThetaIntervalOracle(instance).thetas),
                "repeats": repeats,
                "compressed_root_seconds": compressed_root_time,
                "clique_root_seconds": clique_root_time,
                "root_speedup": clique_root_time / max(compressed_root_time, 1e-12),
                "compressed_root_bound": compressed_root["bound"],
                "clique_root_bound": clique_root["bound"],
                "compressed_root_improvement_pct": 100.0
                * (float(clique_root["bound"]) - float(compressed_root["bound"]))
                / scale,
                "compressed_adaptive_seconds": compressed_time,
                "clique_adaptive_seconds": clique_time,
                "adaptive_speedup": clique_time / max(compressed_time, 1e-12),
                "compressed_status": compressed["status"],
                "clique_status": clique["status"],
                "compressed_relative_gap": compressed["relative_gap"],
                "clique_relative_gap": clique["relative_gap"],
                "compressed_splits": compressed["interval_splits"],
                "clique_splits": clique["interval_splits"],
                "compressed_theta_lp_evaluations": compressed["theta_lp_evaluations"],
                "clique_theta_lp_evaluations": clique["theta_lp_evaluations"],
            }
        )
        _write_csv(output_dir / "scaling.csv", rows)

    large = [row for row in rows if int(row["n"]) >= 720]
    speedups = np.asarray([float(row["adaptive_speedup"]) for row in large])
    root_dominance = np.asarray(
        [
            float(row["compressed_root_bound"])
            <= float(row["clique_root_bound"])
            + 1e-7 * max(1.0, abs(float(row["clique_root_bound"])))
            for row in large
        ],
        dtype=float,
    )
    tolerance = np.asarray(
        [
            float(row["compressed_relative_gap"]) <= 1e-6
            and float(row["clique_relative_gap"]) <= 1e-6
            for row in large
        ],
        dtype=float,
    )
    family_medians = {
        family: statistics.median(
            float(row["adaptive_speedup"])
            for row in large
            if row["family"] == family
        )
        for family in FAMILIES
    }
    metrics = {
        "adaptive_tolerance_rate": float(np.mean(tolerance)),
        "large_case_geomean_speedup": float(np.exp(np.mean(np.log(speedups)))),
        "large_case_median_speedup": float(np.median(speedups)),
        "large_case_win_rate": float(np.mean(speedups > 1.0)),
        "families_with_median_speedup_gt_one": sum(
            value > 1.0 for value in family_medians.values()
        ),
        "root_dominance_rate": float(np.mean(root_dominance)),
        "family_median_adaptive_speedup": family_medians,
    }
    gates = {
        "adaptive_tolerance": metrics["adaptive_tolerance_rate"]
        >= PREREGISTERED_GATES["adaptive_tolerance_rate_min"],
        "geomean_speedup": metrics["large_case_geomean_speedup"]
        >= PREREGISTERED_GATES["large_case_geomean_speedup_min"],
        "win_rate": metrics["large_case_win_rate"]
        >= PREREGISTERED_GATES["large_case_win_rate_min"],
        "family_breadth": metrics["families_with_median_speedup_gt_one"]
        >= PREREGISTERED_GATES["families_with_median_speedup_gt_one_min"],
        "root_dominance": metrics["root_dominance_rate"]
        >= PREREGISTERED_GATES["root_dominance_rate_min"],
    }
    summary = {
        "instances": len(rows),
        "large_instances": len(large),
        "sizes": list(sizes),
        "seeds": list(seeds),
        "preregistered_gates": PREREGISTERED_GATES,
        "metrics": metrics,
        "gates": gates,
        "positive_confirmation": all(gates.values()),
    }
    (output_dir / "scaling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def write_environment(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        },
        "preregistered_gates": PREREGISTERED_GATES,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "scale", "all"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "positive_confirmation_20260719",
    )
    parser.add_argument("--sizes", default="360,720,1440")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    write_environment(output_dir, args)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    result: dict[str, object] = {}
    if args.command in {"validate", "all"}:
        result["validation"] = run_validation(output_dir, seeds)
    if args.command in {"scale", "all"}:
        sizes = [int(value) for value in args.sizes.split(",") if value]
        result["scaling"] = run_scaling(
            output_dir,
            sizes=sizes,
            seeds=seeds,
            repeats=args.repeats,
            time_limit=args.time_limit,
        )
    (output_dir / "study_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
