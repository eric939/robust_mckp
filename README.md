# robust_mckp

Reference solver and reproducibility code for Gamma-robust multiple-choice
knapsack problems (MCKP), with experiments for the accompanying discrete
pricing study.

This GitHub repository intentionally contains only code needed to install the
solver, run tests, and regenerate computational experiments. Manuscript source,
submission packages, generated figures, generated tables, benchmark CSVs, and
local review notes are not tracked here.

## Contents

- `src/robust_mckp/`: solver package, including HullRound and exact
  theta-enumerated branch-and-bound.
- `experiments_nested/`: synthetic and nested robust-MCKP experiment drivers.
- `experiments_case_retail/`: semi-synthetic pricing experiment driver.
- `scripts/`: benchmark, plotting, summarization, and reproducibility scripts.
- `tests/`: public API, certificate, and exact-solver tests.

Generated outputs are written under `results/`, `paper/`, or local output
directories and are ignored by git.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[experiments,validation,dev]"
```

Minimal package-only install from GitHub:

```bash
python3 -m pip install "git+https://github.com/eric939/robust_mckp.git"
```

## Quick Check

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_clean_repro_check.py --quick
```

The clean check runs tests, py-compiles the experiment scripts, and executes
small smoke experiments into `results/clean_repro_check/`.

## Reproducing Experiments

Smoke runs:

```bash
.venv/bin/python scripts/run_publication_benchmarks.py --smoke --output-dir results/publication_benchmarks_smoke
.venv/bin/python scripts/run_publishable_experiments.py --smoke
.venv/bin/python scripts/run_solver_benchmarks.py --smoke
.venv/bin/python scripts/run_pathC_data_calibration.py --source synthetic_only --output-dir results/pathC/calibration
.venv/bin/python scripts/run_pathC_semisynthetic_application.py --calibration-dir results/pathC/calibration --output-dir results/pathC/semisynthetic_application_smoke --seeds 1 --n 60 --m 8 --stress-scenarios 200 --gamma-grid 0,sqrt,n --run-exact-small-subset
```

Larger benchmark campaigns use the same scripts without `--smoke` and with the
desired grid, time-limit, and output-directory options. Optional exact-solver
baselines use PySCIPOpt/SCIP and HiGHS when available.

## Citation

Please cite the software and accompanying manuscript. A machine-readable
citation file is included in `CITATION.cff`.

```bibtex
@misc{shao2026robustmckp,
  title = {robust_mckp: Structure-exploiting algorithms for Gamma-robust MCKP},
  author = {Shao, Eric},
  year = {2026},
  howpublished = {\url{https://github.com/eric939/robust_mckp}}
}
```

## License

MIT. See `LICENSE`.
