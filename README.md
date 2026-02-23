# robust-mckp

Reference implementation of a robust hull-greedy solver for discrete pricing optimization under budgeted demand uncertainty, plus reproducible experiment scripts for the accompanying paper.

This repository contains:

- A pip-installable Python package: `robust-mckp` (import: `robust_mckp`)
- Reproduction scripts for the paper’s numerical experiments (`experiments_nested/`)
- A stylized retail case study (`experiments_case_retail/`)

Legacy exploratory scripts in `experiments/` are **not** part of the paper reproduction workflow.

## What Problem Is Solved?

For each item group \(i\), choose one discrete option \(j\) with:

$$
\begin{aligned}
v_{ij} &= \omega_i x_{ij}\hat g_i(x_{ij}) \\
s_{ij} &= \omega_i(x_{ij}-\Delta a_i)\hat g_i(x_{ij}) \\
t_{ij} &= \omega_i(x_{ij}-\Delta a_i)\delta_i(x_{ij})
\end{aligned}
$$

and solve

$$
\max \sum_i v_{i,j(i)}
\quad\text{s.t.}\quad
\sum_i s_{i,j(i)} - \beta(x,\Gamma) \ge 0,
$$

where \(\beta(x,\Gamma)\) is the sum of the \(\Gamma\) largest values among \(\{|t_{i,j(i)}|\}\).

The solver implements the paper’s Robust Hull-Greedy algorithm:

1. Enumerate candidate \(\theta\) breakpoints.
2. Solve each fixed-\(\theta\) LP relaxation exactly via upper-hull filtering + greedy filling.
3. Recover a discrete solution by one-item rounding (with repair/completion heuristics).
4. Certify exact robust feasibility using the original \(s,t\).

## Installation

### Package use (minimal)

```bash
pip install robust-mckp
```

### Local development / paper reproduction (recommended)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install matplotlib tqdm
```

Optional (only needed for MILP / LP cross-validation in some experiments):

```bash
python -m pip install scipy
```

Notes:

- Use `python` (or `./.venv/bin/python`) after activating `.venv`.
- Do **not** force `/opt/homebrew/bin/python3` if you want the `.venv` packages.

## Quickstart

### Low-level API (precomputed \(v,s,t\))

```python
from robust_mckp import PricingInstance, Option, solve

instance = PricingInstance(
    items=[
        [Option(value=5.0, margin=1.0, uncertainty=0.2),
         Option(value=6.0, margin=0.5, uncertainty=0.4)],
        [Option(value=4.0, margin=1.0, uncertainty=0.3),
         Option(value=7.0, margin=0.2, uncertainty=0.5)],
    ],
    gamma=1,
)

solution = solve(instance)
print(solution.objective)
print(solution.selections)
print(solution.is_feasible)
print(solution.theta)
print(solution.certificate_value)
```

### High-level API (raw pricing inputs)

```python
from robust_mckp import from_pricing_data, solve

instance = from_pricing_data(
    reference_prices=[100, 120],
    weights=[1.0, 0.8],
    price_menus=[[90, 100, 110], [108, 120, 132]],
    demands=[[0.8, 0.7, 0.6], [0.9, 0.8, 0.7]],
    uncertainties=[[0.1, 0.15, 0.2], [0.12, 0.16, 0.2]],
    margin_target=0.9,
    tolerances=[0.15, 0.15],
    gamma=1,
)

solution = solve(instance)
```

## Public API

- `Option(value, margin, uncertainty, price=None)`
- `PricingInstance(items, gamma, name=None)`
- `Solution(...)`
- `from_pricing_data(...) -> PricingInstance`
- `filter_admissible_options(...) -> PricingInstance`
- `solve(instance: PricingInstance, *, upgrade_completion=True) -> Solution`

`Solution` includes (among others):

- `selections`
- `objective`
- `lp_value`
- `gap_to_lp`
- `certificate_value`
- `theta`
- `elapsed`
- `is_feasible`
- `metadata` (instrumentation/debug info)

## Algorithm Notes (Implementation)

This implementation includes exact practical speedups for \(\theta\)-enumeration:

- Reduced \(\theta\)-candidate set via strong dominance in \((s, v, |t|)\)
- Incremental sweep for baseline maximizers and capacity updates across sorted breakpoints

The final robust feasibility check is always done with the exact certificate using original \(s,t\).

## Complexity

- Per \(\theta\): upper-hull filtering + LP greedy solve is dominated by \(O(nm\log m)\) hull work
- Candidate set size is at most \(O(nm)\)
- Worst-case total: \(O(n^2 m^2 \log m)\)

In practice, hull compression and \(\theta\)-candidate reduction significantly reduce runtime.

## Repository Layout

- `src/robust_mckp/` — solver package
- `tests/` — unit and integration tests
- `examples/` — basic usage examples
- `experiments_nested/` — **paper reproduction** (numerical results, nested-prefix design)
- `experiments_case_retail/` — **paper reproduction** (stylized retail case study)
- `experiments/` — legacy exploratory scripts (not used for paper reproduction)

## Reproducing the Paper Experiments

### 1) Nested numerical experiments (`experiments_nested/`)

These correspond to the paper’s main numerical section (integrality gap, scalability, frontier, summary table) using the deterministic nested-prefix design.

Run from repository root:

```bash
python experiments_nested/exp1_integrality_gap.py --output-dir figs --results-dir results_nested
python experiments_nested/exp2_scalability.py --output-dir figs --results-dir results_nested
python experiments_nested/exp3_risk_frontier.py --output-dir figs --results-dir results_nested
python experiments_nested/exp4_summary_table.py --results-dir results_nested --tables-dir tables_nested
```

Optional flags:

- `--fast` for smoke checks
- `--enable-milp` for MILP benchmarks (requires `scipy`)
- `--global-milp` (Exp1) for exhaustive \(\theta\)-MILP on small instances
- `--validate-lp` (Exp2) for LP cross-validation against HiGHS (`scipy`)

### 2) Replot from saved results (no solver rerun)

```bash
python experiments_nested/replot_exp1_from_results.py --input-csv results_nested/exp1_gap.csv --output-dir figs
python experiments_nested/replot_exp2_from_results.py --results-dir results_nested --output-dir figs
python experiments_nested/replot_exp3_from_results.py --results-dir results_nested --output-dir figs
```

### 3) Stylized retail case study (`experiments_case_retail/`)

This produces the figures and CSVs for the retail category pricing application.

```bash
python experiments_case_retail/case_retail_pricing.py --output-dir figs --results-dir results_case
```

Fast smoke run:

```bash
python experiments_case_retail/case_retail_pricing.py --fast --scenarios 2000
```

### Expected Figure Outputs (drop-in names)

Nested experiments:

- `figs/gap_loss_vs_n.pdf`
- `figs/loss_vs_bound.pdf`
- `figs/runtime_vs_n.pdf`
- `figs/hull_sizes.pdf`
- `figs/runtime_breakdown.pdf`
- `figs/risk_frontier.pdf`
- `figs/tail_margin.pdf`
- `figs/tightness_heatmap.pdf`

Retail case study:

- `figs/case_frontier.pdf`
- `figs/case_margin_dist.pdf`

## Running Tests

```bash
python -m pytest -q
```

If `pytest` is missing:

```bash
python -m pip install pytest
```

## Citation

If you use this code, please cite both:

- The accompanying pricing paper (this repository’s linked manuscript)
- Bertsimas & Sim (2004) for the \(\Gamma\)-budget robust optimization model

```bibtex
@article{bertsimas2004price,
  title={The Price of Robustness},
  author={Bertsimas, Dimitris and Sim, Melvyn},
  journal={Operations Research},
  volume={52},
  number={1},
  pages={35--53},
  year={2004}
}
```

## License

MIT (see `LICENSE`).

