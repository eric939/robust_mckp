# Reproducibility Guide

This repository tracks solver and experiment code only. Generated result files,
figures, tables, manuscript builds, and submission packages are intentionally
left out of git so a clone starts from code and regenerates outputs locally.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[experiments,validation,dev]"
```

Optional baselines use PySCIPOpt/SCIP and HiGHS when available. The core solver
and smoke checks do not require commercial solvers.

## Bounded Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_clean_repro_check.py --quick
```

The bounded check verifies imports and tests, py-compiles the experiment
scripts, runs a small publication benchmark, creates synthetic Path C
calibration data, and runs a small semi-synthetic pricing experiment. Outputs
are written under `results/clean_repro_check/`.

For a clean-copy check:

```bash
.venv/bin/python scripts/run_true_clean_room_check.py --work-dir /tmp/robust_mckp_clean_check --quick
```

This copies the repository to a temporary directory, installs it in a fresh
virtual environment, and runs the same bounded checks there.

## Experiment Entry Points

Synthetic robust-MCKP smoke benchmark:

```bash
.venv/bin/python scripts/run_publication_benchmarks.py --smoke --output-dir results/publication_benchmarks_smoke
```

Publishable synthetic/retail experiment bundle:

```bash
.venv/bin/python scripts/run_publishable_experiments.py --smoke
.venv/bin/python scripts/run_solver_benchmarks.py --smoke
.venv/bin/python scripts/plot_publishable_results.py
```

Semi-synthetic pricing application:

```bash
.venv/bin/python scripts/run_pathC_data_calibration.py --source synthetic_only --output-dir results/pathC/calibration
.venv/bin/python scripts/run_pathC_semisynthetic_application.py --calibration-dir results/pathC/calibration --output-dir results/pathC/semisynthetic_application_smoke --seeds 1 --n 60 --m 8 --stress-scenarios 200 --gamma-grid 0,sqrt,n --run-exact-small-subset
```

Diagnostics:

```bash
.venv/bin/python scripts/run_branching_strategy_diagnostics.py
.venv/bin/python scripts/run_tight_capacity_diagnostics.py
```

Most scripts accept `--output-dir` or related path options. Use those options to
keep new outputs separate from previous local runs.
