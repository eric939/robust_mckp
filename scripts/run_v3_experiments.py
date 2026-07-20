#!/usr/bin/env python3
"""Run the consolidated v3 experimental campaign.

The campaign has three deliberately separate layers:

1. hard benchmark instances with binding capacity and low dominance;
2. matched anytime runs at several wall-clock budgets; and
3. a certificate-failure audit for shortcuts ruled out by the paper.

All outputs are append-safe CSV files under one versioned directory.  The
semi-synthetic application remains in ``run_pathC_semisynthetic_application``
because it has its own calibration inputs, but the ``all`` command invokes it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments_nested"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robust_mckp import GlobalThetaBNBConfig, Option, PricingInstance, solve  # noqa: E402
from robust_mckp.certificate import compute_certificate  # noqa: E402
from robust_mckp.exact_bnb import (  # noqa: E402
    brute_force_global_robust,
    build_full_theta_candidates,
    solve_global_theta_bnb,
)
from scripts.benchmark_solvers import (  # noqa: E402
    solve_full_robust_highs,
    solve_full_robust_scip,
    solve_theta_enum_highs,
)


DEFAULT_OUT = ROOT / "results" / "v3_experiments_20260718"
HARD_FAMILIES = ("dense_frontier", "correlated_risk", "near_tie", "many_breakpoints")
# Run the compact HiGHS model before the repeated fixed-theta SciPy/HiGHS
# calls.  HiGHS 1.14 can retain numerical state after a long enumeration in
# the same Python process, so this ordering keeps the matched baseline clean.
METHODS = ("hullround", "theta_bnb", "highs", "theta_enum_highs", "scip")
ANYTIME_METHODS = ("theta_bnb", "scip", "highs")

BENCHMARK_FIELDS = [
    "run_key", "instance_id", "family", "n", "m", "gamma", "gamma_mode", "seed",
    "method", "time_limit_seconds", "status", "certified_optimal", "objective",
    "upper_bound", "relative_gap", "robust_certificate", "valid_certificate",
    "runtime_seconds", "budget_overrun_ratio", "nodes_explored", "theta_count",
    "theta_pruned", "full_theta_set", "instance_options", "notes",
]

FAILURE_FIELDS = [
    "case_id", "shortcut", "replicate", "correct_status", "correct_objective",
    "unsafe_status", "unsafe_objective", "objective_error", "failure_detected", "details",
]


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _append(path: Path, fields: Sequence[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def _completed(path: Path, key: str) -> set[str]:
    return {row[key] for row in _csv_rows(path)}


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _rng(family: str, n: int, m: int, gamma: int, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"v3|{family}|{n}|{m}|{gamma}|{seed}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16) % (2**32))


def gamma_for(n: int, mode: str) -> int:
    if mode == "sqrt":
        return int(math.floor(math.sqrt(n)))
    if mode == "twenty_percent":
        return max(1, int(math.floor(0.20 * n)))
    if mode == "ten_percent":
        return max(1, int(math.floor(0.10 * n)))
    raise ValueError(mode)


def build_hard_instance(family: str, n: int, m: int, gamma: int, seed: int) -> PricingInstance:
    """Create capacity-binding instances rather than loose all-upgrade cases."""
    if family not in HARD_FAMILIES:
        raise ValueError(f"unknown v3 family: {family}")
    rng = _rng(family, n, m, gamma, seed)
    max_cost = 14.0
    base_margin = {
        "dense_frontier": 4.6,
        "correlated_risk": 4.1,
        "near_tie": 3.9,
        "many_breakpoints": 4.4,
    }[family]
    deviation_grid = np.linspace(0.20, 3.20, 25)
    items: List[List[Option]] = []
    for i in range(n):
        base_value = 8.0 + rng.uniform(0.0, 0.4)
        group: List[Option] = [Option(value=base_value, margin=base_margin, uncertainty=0.0)]
        costs = np.linspace(0.8, max_cost, max(1, m - 1))
        for k, cost in enumerate(costs, start=1):
            if family == "dense_frontier":
                value = base_value + 8.0 * math.sqrt(cost) + rng.normal(0.0, 0.35)
                deviation = float(deviation_grid[(7 * i + 3 * k + seed) % len(deviation_grid)])
            elif family == "correlated_risk":
                value = base_value + 3.15 * cost + rng.normal(0.0, 0.75)
                raw = 0.30 + 0.19 * cost + rng.normal(0.0, 0.08)
                deviation = float(deviation_grid[int(np.argmin(np.abs(deviation_grid - raw)))])
            elif family == "near_tie":
                value = base_value + 2.25 * cost + rng.normal(0.0, 1.65)
                deviation = float(deviation_grid[(5 * i + k + seed) % len(deviation_grid)])
            else:
                value = base_value + 7.5 * math.sqrt(cost) + rng.normal(0.0, 0.45)
                # Unique deviations make root-bound initialization itself a
                # meaningful part of the end-to-end workload.
                deviation = float(0.15 + 0.003 * (i * max(1, m - 1) + k) + 1e-6 * seed)
            margin = base_margin - cost + rng.normal(0.0, 0.08)
            group.append(Option(value=max(0.0, float(value)), margin=float(margin), uncertainty=deviation))
        items.append(group)
    return PricingInstance(items=items, gamma=int(gamma), name=f"{family}_n{n}_m{m}_g{gamma}_s{seed}")


def _objective(instance: PricingInstance, selection: Sequence[int]) -> float:
    return float(sum(instance.items[i][int(j)].value for i, j in enumerate(selection)))


def _blank(instance: PricingInstance, family: str, n: int, m: int, gamma: int, mode: str, seed: int, method: str, limit: float) -> Dict[str, object]:
    instance_id = f"{family}|{n}|{m}|{mode}|{gamma}|{seed}"
    return {
        "run_key": f"{instance_id}|{method}|{limit:g}",
        "instance_id": instance_id,
        "family": family,
        "n": n,
        "m": m,
        "gamma": gamma,
        "gamma_mode": mode,
        "seed": seed,
        "method": method,
        "time_limit_seconds": limit,
        "status": "error",
        "certified_optimal": False,
        "objective": float("nan"),
        "upper_bound": float("nan"),
        "relative_gap": float("nan"),
        "robust_certificate": float("nan"),
        "valid_certificate": False,
        "runtime_seconds": float("nan"),
        "budget_overrun_ratio": float("nan"),
        "nodes_explored": 0,
        "theta_count": len(build_full_theta_candidates(instance)),
        "theta_pruned": 0,
        "full_theta_set": method in {"theta_bnb", "theta_enum_highs"},
        "instance_options": instance.n_options,
        "notes": "",
    }


def run_method(instance: PricingInstance, family: str, n: int, m: int, gamma: int, mode: str, seed: int, method: str, limit: float) -> Dict[str, object]:
    row = _blank(instance, family, n, m, gamma, mode, seed, method, limit)
    t0 = time.perf_counter()
    try:
        if method == "hullround":
            sol = solve(instance, upgrade_completion=True)
            cert = compute_certificate(instance, sol.selections) if sol.selections else float("nan")
            row.update(
                status="feasible" if sol.is_feasible else "infeasible",
                objective=float(sol.objective),
                robust_certificate=float(cert),
                valid_certificate=bool(sol.is_feasible and cert >= -1e-8),
                notes="certified-feasible primal method; no optimality claim",
            )
        elif method == "theta_bnb":
            result = solve_global_theta_bnb(
                instance,
                GlobalThetaBNBConfig(
                    time_limit_seconds=float(limit),
                    node_limit=250_000,
                    theta_order="lp_bound_desc",
                    use_caches=True,
                    use_objective_cutoff=True,
                    use_fast_residual_lp_bound=True,
                ),
            )
            row.update(
                status=result.status,
                certified_optimal=result.status == "optimal",
                objective=float(result.objective_value),
                upper_bound=float(result.upper_bound),
                relative_gap=float(result.relative_gap),
                robust_certificate=float(result.robust_certificate),
                valid_certificate=bool(result.validation_flags.get("robust_certificate_feasible", False)),
                nodes_explored=result.total_nodes_explored,
                theta_count=result.theta_count_total,
                theta_pruned=result.theta_count_pruned_by_bound,
                notes=result.message,
            )
        elif method == "theta_enum_highs":
            result = solve_theta_enum_highs(
                instance,
                time_limit_per_theta=max(0.05, float(limit)),
                max_thetas=350,
            )
            objective = float(result.get("objective", float("nan")))
            cert = float(result.get("certificate_value", float("nan")))
            certified = bool(result.get("certified", False)) and str(result.get("status", "")).upper() == "OPTIMAL"
            row.update(
                status=str(result.get("status", "unknown")).lower(),
                certified_optimal=certified,
                objective=objective,
                upper_bound=objective if certified else float("nan"),
                relative_gap=0.0 if certified else float("nan"),
                robust_certificate=cert,
                valid_certificate=math.isfinite(cert) and cert >= -1e-7,
                theta_count=int(result.get("theta_count", row["theta_count"])),
                notes=str(result.get("failed_status", "")),
            )
        elif method == "scip":
            result = solve_full_robust_scip(instance, time_limit=float(limit), threads=1)
            objective = float(result.get("objective", float("nan")))
            cert = float(result.get("certificate_value", float("nan")))
            row.update(
                status=str(result.get("status", "unknown")).lower(),
                certified_optimal=bool(result.get("certified", False)),
                objective=objective,
                upper_bound=float(result.get("best_bound", float("nan"))),
                relative_gap=float(result.get("mip_gap", float("nan"))),
                robust_certificate=cert,
                valid_certificate=math.isfinite(cert) and cert >= -1e-7,
            )
        elif method == "highs":
            result = solve_full_robust_highs(instance, time_limit=float(limit), threads=1)
            objective = float(result.get("objective", float("nan")))
            cert = float(result.get("certificate_value", float("nan")))
            row.update(
                status=str(result.get("status", "unknown")).lower(),
                certified_optimal=bool(result.get("certified", False)),
                objective=objective,
                upper_bound=float(result.get("best_bound", float("nan"))),
                relative_gap=float(result.get("mip_gap", float("nan"))),
                robust_certificate=cert,
                valid_certificate=math.isfinite(cert) and cert >= -1e-7,
                notes=str(result.get("message", "")),
            )
        else:
            raise ValueError(method)
    except Exception as exc:
        row["status"] = "error"
        row["notes"] = f"{type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - t0
    row["runtime_seconds"] = runtime
    row["budget_overrun_ratio"] = runtime / limit if limit > 0 and method != "hullround" else float("nan")
    return row


def write_environment(out_dir: Path, args: argparse.Namespace) -> None:
    packages: Dict[str, str] = {}
    for package in ("numpy", "scipy", "highspy", "pyscipopt", "matplotlib"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    payload = {
        "campaign": "v3 consolidated experiments",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "packages": packages,
        "thread_environment": {k: os.environ.get(k, "unset") for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "arguments": vars(args),
        "resolved_design": {
            "hard": {
                "families": [x.strip() for x in args.families.split(",") if x.strip()],
                "n_values": _parse_ints(args.n_values),
                "m": args.m,
                "gamma_modes": [x.strip() for x in args.gamma_modes.split(",") if x.strip()],
                "methods": [x.strip() for x in args.methods.split(",") if x.strip()],
                "seeds": list(range(args.seed_start, args.seed_start + args.seeds)),
                "time_limit_seconds": args.time_limit,
            },
            "anytime": {
                "families": [x.strip() for x in args.anytime_families.split(",") if x.strip()],
                "n_values": _parse_ints(args.anytime_n_values),
                "budgets_seconds": _parse_floats(args.anytime_budgets),
                "methods": [x.strip() for x in args.anytime_methods.split(",") if x.strip()],
                "seeds": list(range(args.seed_start, args.seed_start + args.anytime_seeds)),
            },
            "certificate_failure_replicates_per_shortcut": args.failure_replicates,
            "application": {
                "portfolios": args.application_seeds,
                "n": args.application_n,
                "m": args.application_m,
                "stress_scenarios": args.application_scenarios,
                "exact_subset_n": args.application_exact_n,
                "exact_time_limit_seconds": args.application_exact_time_limit,
            },
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "environment.json"
    stages: Dict[str, object] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            stages.update(previous.get("stage_runs", {}))
            previous_command = previous.get("arguments", {}).get("command")
            if previous_command:
                stages[str(previous_command)] = {
                    "created": previous.get("created"),
                    "arguments": previous.get("arguments"),
                }
        except (OSError, ValueError, TypeError):
            pass
    application_config = out_dir / "application" / "pathC_config.json"
    if application_config.exists() and "application" not in stages:
        try:
            stages["application"] = {
                "source": str(application_config),
                "arguments": json.loads(application_config.read_text(encoding="utf-8")),
            }
        except (OSError, ValueError, TypeError):
            pass
    stages[str(args.command)] = {"created": payload["created"], "arguments": payload["arguments"]}
    payload["stage_runs"] = stages
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_hard(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    path = out / "hard_benchmark.csv"
    done = _completed(path, "run_key") if args.resume else set()
    if path.exists() and not args.resume:
        path.unlink()
    families = [x.strip() for x in args.families.split(",") if x.strip()]
    ns = _parse_ints(args.n_values)
    modes = [x.strip() for x in args.gamma_modes.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    counter = 0
    for family in families:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for n in ns:
                for mode in modes:
                    gamma = gamma_for(n, mode)
                    instance = build_hard_instance(family, n, args.m, gamma, seed)
                    counter += 1
                    print(f"[hard {counter}] {family} n={n} m={args.m} gamma={gamma} seed={seed}", flush=True)
                    for method in methods:
                        limit = args.time_limit
                        run_key = f"{family}|{n}|{args.m}|{mode}|{gamma}|{seed}|{method}|{limit:g}"
                        if run_key in done:
                            print(f"  {method}: existing", flush=True)
                            continue
                        print(f"  {method}", flush=True)
                        _append(path, BENCHMARK_FIELDS, run_method(instance, family, n, args.m, gamma, mode, seed, method, limit))


def run_anytime(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    path = out / "anytime_frontier.csv"
    done = _completed(path, "run_key") if args.resume else set()
    if path.exists() and not args.resume:
        path.unlink()
    families = [x.strip() for x in args.anytime_families.split(",") if x.strip()]
    ns = _parse_ints(args.anytime_n_values)
    budgets = _parse_floats(args.anytime_budgets)
    methods = [x.strip() for x in args.anytime_methods.split(",") if x.strip()]
    counter = 0
    for family in families:
        for seed in range(args.seed_start, args.seed_start + args.anytime_seeds):
            for n in ns:
                gamma = gamma_for(n, "twenty_percent")
                instance = build_hard_instance(family, n, args.m, gamma, seed)
                counter += 1
                print(f"[anytime {counter}] {family} n={n} gamma={gamma} seed={seed}", flush=True)
                for budget in budgets:
                    for method in methods:
                        run_key = f"{family}|{n}|{args.m}|twenty_percent|{gamma}|{seed}|{method}|{budget:g}"
                        if run_key in done:
                            continue
                        print(f"  budget={budget:g}s {method}", flush=True)
                        _append(path, BENCHMARK_FIELDS, run_method(instance, family, n, args.m, gamma, "twenty_percent", seed, method, budget))


def _theta_zero_residual(instance: PricingInstance, selection: Sequence[int]) -> float:
    return float(sum(instance.items[i][int(j)].margin - abs(instance.items[i][int(j)].uncertainty) for i, j in enumerate(selection)))


def _failure_row(case_id: str, shortcut: str, replicate: int, correct_status: str, correct_objective: float, unsafe_status: str, unsafe_objective: float, details: str) -> Dict[str, object]:
    err = correct_objective - unsafe_objective if math.isfinite(unsafe_objective) else float("inf")
    return {
        "case_id": case_id,
        "shortcut": shortcut,
        "replicate": replicate,
        "correct_status": correct_status,
        "correct_objective": correct_objective,
        "unsafe_status": unsafe_status,
        "unsafe_objective": unsafe_objective,
        "objective_error": err,
        "failure_detected": (unsafe_status != correct_status) or (not math.isfinite(unsafe_objective)) or abs(err) > 1e-8,
        "details": details,
    }


def run_failures(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    path = out / "certificate_failure_audit.csv"
    if path.exists():
        path.unlink()
    for rep in range(args.failure_replicates):
        # Unsafe shortcut 1: checking theta=0 only.
        n = 2 + rep % 5
        deviation = 1.0 + 0.01 * rep
        margin = (deviation + 0.5) / n
        inst = PricingInstance(items=[[Option(1.0 + 0.01 * i, margin, deviation)] for i in range(n)], gamma=1)
        exact = brute_force_global_robust(inst)
        unsafe_feasible = _theta_zero_residual(inst, [0] * n) >= -1e-9
        unsafe_obj = _objective(inst, [0] * n) if unsafe_feasible else float("nan")
        _append(path, FAILURE_FIELDS, _failure_row(
            f"reduced_theta_{rep}", "theta_zero_only", rep, exact.status, exact.objective_value,
            "optimal" if unsafe_feasible else "infeasible", unsafe_obj,
            "A selected-option breakpoint restores feasibility; theta=0 alone rejects it.",
        ))

        # Unsafe shortcut 2: treating LP upper-hull vertices as the integer set.
        capacity = 4.0 + 0.1 * rep
        middle = 8.0 + 0.02 * rep
        high = 2.0 * middle + 2.0
        inst2 = PricingInstance(
            items=[
                [
                    Option(0.0, capacity, 0.0),
                    Option(middle, 0.0, 0.0),
                    Option(high, -capacity, 0.0),
                ]
            ],
            gamma=0,
        )
        exact2 = brute_force_global_robust(inst2)
        unsafe2 = 0.0  # the high endpoint is infeasible and the middle point is below the LP hull
        _append(path, FAILURE_FIELDS, _failure_row(
            f"hull_filter_{rep}", "lp_hull_as_integer_filter", rep, exact2.status, exact2.objective_value,
            "optimal", unsafe2, "The below-hull middle option is the integer optimum at the binding capacity.",
        ))

        # Unsafe shortcut 3: reporting a zero gap before every theta has a bound.
        value = 10.0 + 0.1 * rep
        inst3 = PricingInstance(
            items=[
                [Option(0.0, 0.5, 0.0), Option(value, 0.75, 1.0)],
                [Option(0.0, 0.5, 0.0), Option(value + 0.5, 0.75, 1.0)],
            ],
            gamma=1,
        )
        exact3 = brute_force_global_robust(inst3)
        naive_theta_zero_objective = value + 0.5
        _append(path, FAILURE_FIELDS, _failure_row(
            f"missing_bound_{rep}", "gap_before_all_theta_bounds", rep, exact3.status, exact3.objective_value,
            "optimal", naive_theta_zero_objective,
            "The visited theta=0 row appears closed, but theta=1 contains a better feasible solution.",
        ))

        # Unsafe shortcut 4: accepting an incumbent without the original robust check.
        inst4 = build_hard_instance("correlated_risk", 12, 8, 3, 50_000 + rep)
        high_selection = [max(range(len(group)), key=lambda j: group[j].value) for group in inst4.items]
        unsafe4 = _objective(inst4, high_selection)
        cert4 = compute_certificate(inst4, high_selection)
        hr4 = solve(inst4)
        correct4 = float(hr4.objective)
        unsafe_status = "feasible" if cert4 >= -1e-8 else "infeasible_incumbent_accepted"
        _append(path, FAILURE_FIELDS, _failure_row(
            f"unchecked_incumbent_{rep}", "no_original_robust_recheck", rep, "feasible", correct4,
            unsafe_status, unsafe4,
            f"Direct robust residual of highest-value selection: {cert4:.6g}.",
        ))


def run_application(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_pathC_semisynthetic_application.py"),
        "--calibration-dir", str(ROOT / "results" / "pathC" / "calibration"),
        "--output-dir", str(Path(args.output_dir) / "application"),
        "--seeds", str(args.application_seeds),
        "--n", str(args.application_n),
        "--m", str(args.application_m),
        "--gamma-grid", "0,5,sqrt,0.1n,0.25n,n",
        "--stress-scenarios", str(args.application_scenarios),
        "--run-exact-small-subset",
        "--exact-small-subset-n", str(args.application_exact_n),
        "--exact-time-limit", str(args.application_exact_time_limit),
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("hard", "anytime", "failures", "application", "all"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--families", default=",".join(HARD_FAMILIES))
    parser.add_argument("--n-values", default="50,100,180")
    parser.add_argument("--m", type=int, default=12)
    parser.add_argument("--gamma-modes", default="sqrt,twenty_percent")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=20260720)
    parser.add_argument("--time-limit", type=float, default=3.0)
    parser.add_argument("--anytime-families", default="dense_frontier,correlated_risk,near_tie")
    parser.add_argument("--anytime-n-values", default="100,180")
    parser.add_argument("--anytime-budgets", default="0.1,1,3")
    parser.add_argument("--anytime-methods", default=",".join(ANYTIME_METHODS))
    parser.add_argument("--anytime-seeds", type=int, default=3)
    parser.add_argument("--failure-replicates", type=int, default=50)
    parser.add_argument("--application-seeds", type=int, default=30)
    parser.add_argument("--application-n", type=int, default=180)
    parser.add_argument("--application-m", type=int, default=12)
    parser.add_argument("--application-scenarios", type=int, default=20_000)
    parser.add_argument("--application-exact-n", type=int, default=50)
    parser.add_argument("--application-exact-time-limit", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    write_environment(out, args)
    commands = ("failures", "hard", "anytime", "application") if args.command == "all" else (args.command,)
    for command in commands:
        {"hard": run_hard, "anytime": run_anytime, "failures": run_failures, "application": run_application}[command](args)


if __name__ == "__main__":
    main()
