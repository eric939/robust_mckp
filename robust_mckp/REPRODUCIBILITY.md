# Reproducibility Guide

This repository is organized so the generated outputs can be dropped directly into the manuscript
_Robust Discrete Pricing Optimization via Multiple-Choice Knapsack Reductions_.

## Recommended workflow

Create an environment and install all artifact dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[experiments,validation,dev]"
```

Run the fast smoke artifact:

```bash
make paper-fast
```

Run the full manuscript artifact:

```bash
make paper-all
```

Run the auxiliary validation and benchmark checks cited in the manuscript:

```bash
make paper-validate
```

## Output mapping

The commands above produce the exact output paths expected by the manuscript.

### Main numerical section

- `paper/exp1/gap_loss_vs_n.pdf`
  Used for the left panel of `\Cref{fig:gap_decay}`.
- `paper/exp1/loss_vs_bound.pdf`
  Used for the right panel of `\Cref{fig:gap_decay}`.
- `paper/exp2/runtime_vs_n.pdf`
  Used for panel (a) of `\Cref{fig:scalability}`.
- `paper/exp2/hull_sizes.pdf`
  Used for panel (b) of `\Cref{fig:scalability}`.
- `paper/exp2/runtime_breakdown.pdf`
  Used for panel (c) of `\Cref{fig:scalability}`.
- `paper/exp3/risk_frontier.pdf`
  Used for panel (a) of `\Cref{fig:risk_frontier}`.
- `paper/exp3/tail_margin.pdf`
  Used for panel (b) of `\Cref{fig:risk_frontier}`.
- `paper/exp3/tightness_heatmap.pdf`
  Used for panel (c) of `\Cref{fig:risk_frontier}`.
- `paper/tables_nested/numerics_summary.tex`
  Standalone LaTeX tabular for `\Cref{tab:numerics_summary}`.
- `paper/tables_nested/numerics_summary_rows.tex`
  Table rows only, if you prefer to embed them in an existing table environment.

### Retail case study

- `paper/retail pricing/case_frontier.pdf`
  Used for panel (a) of `\Cref{fig:case_study}`.
- `paper/retail pricing/case_margin_dist.pdf`
  Used for panel (b) of `\Cref{fig:case_study}`.

## Script-level commands

If you do not want to use the `Makefile`, run:

```bash
python experiments_nested/exp1_integrality_gap.py --output-dir paper/exp1 --results-dir paper/results_nested
python experiments_nested/exp2_scalability.py --output-dir paper/exp2 --results-dir paper/results_nested
python experiments_nested/exp3_risk_frontier.py --output-dir paper/exp3 --results-dir paper/results_nested
python experiments_nested/exp4_summary_table.py --results-dir paper/results_nested --tables-dir paper/tables_nested
python experiments_case_retail/case_retail_pricing.py --output-dir "paper/retail pricing" --results-dir paper/results_case
```

Auxiliary validation and benchmark commands:

```bash
python experiments_nested/exp1_integrality_gap.py --enable-milp --global-milp --output-dir paper/exp1 --results-dir paper/results_nested
python experiments_nested/exp2_scalability.py --validate-lp --output-dir paper/exp2 --results-dir paper/results_nested
python experiments_nested/exp4_summary_table.py --enable-milp --results-dir paper/results_nested --tables-dir paper/tables_nested
```

Helpful optional flags:

- `--fast`
  Reduced grids for smoke checks.
- `--enable-milp`
  Enables SciPy-based fixed-\(\theta\) MILP benchmarks where implemented.
- `--global-milp`
  Exhaustive \(\theta\)-MILP benchmark in Experiment 1 on small instances.
- `--validate-lp`
  LP cross-validation against HiGHS in Experiment 2.

## Determinism

- The nested experiments use a deterministic master seed of `42` by default.
- The nested-prefix construction matches the manuscript protocol: every larger instance extends the smaller one by adding items, rather than resampling the full portfolio.
- The case study also defaults to seed `42`.

## Verification

Run the solver smoke tests before or after reproducing figures:

```bash
make test
```

The tests cover:

- public low-level API solve/certificate consistency
- admissible-menu filtering in the preprocessing layer
- parity between the optimized solver and the naive reference implementation on a fixed small instance
