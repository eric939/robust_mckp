# robust_mckp

Reference implementation and reproducibility artifact for the manuscript
_Robust Discrete Pricing Optimization via Multiple-Choice Knapsack Reductions_.

The repository contains:

- `src/robust_mckp/`: the solver package
- `experiments_nested/`: scripts for the main numerical results
- `experiments_case_retail/`: the stylized retail case study
- `tests/`: smoke tests for the public API and solver/reference parity

## Scope

For each item group \(i\), select one discrete option \(j\) with

$$
\begin{aligned}
v_{ij} &= \omega_i x_{ij}\hat g_i(x_{ij}), \\
s_{ij} &= \omega_i(x_{ij}-\Delta a_i)\hat g_i(x_{ij}), \\
t_{ij} &= \omega_i(x_{ij}-\Delta a_i)\delta_i(x_{ij}),
\end{aligned}
$$

and solve

$$
\max \sum_i v_{i,j(i)}
\quad \text{s.t.} \quad
\sum_i s_{i,j(i)} - \beta(x,\Gamma) \ge 0,
$$

where \(\beta(x,\Gamma)\) is the sum of the \(\Gamma\) largest values in
\(\{|t_{i,j(i)}|\}\).

The implemented algorithm follows the manuscript:

1. Enumerate candidate \(\theta\) breakpoints.
2. Solve each fixed-\(\theta\) LP relaxation exactly via upper-hull filtering and greedy filling.
3. Recover a discrete solution by feasibility-preserving rounding, with optional repair/completion.
4. Certify exact robust feasibility using the original \(s,t\) coefficients.

## Installation

Minimal install from GitHub:

```bash
python -m pip install "git+https://github.com/eric939/robust_mckp.git"
```

Local development and paper reproduction:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[experiments,validation,dev]"
```

Extras:

- `experiments`: plotting and progress bars
- `validation`: SciPy-based LP and MILP cross-checks used by some scripts
- `dev`: pytest

## Quickstart

Low-level API:

```python
from robust_mckp import Option, PricingInstance, solve

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
print(solution.objective, solution.selections, solution.certificate_value)
```

High-level API from raw pricing inputs:

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

Public API:

- `Option(value, margin, uncertainty, price=None)`
- `PricingInstance(items, gamma, name=None)`
- `Solution(...)`
- `from_pricing_data(...)`
- `filter_admissible_options(...)`
- `solve(instance, *, upgrade_completion=True)`

## Reproducing the Manuscript

The repository now supports output paths that match the manuscript figure and table includes exactly.

Fast smoke reproduction:

```bash
make paper-fast
```

Full reproduction:

```bash
make paper-all
```

Validation and benchmark checks referenced in the manuscript:

```bash
make paper-validate
```

These targets write outputs to:

- `paper/exp1/`
- `paper/exp2/`
- `paper/exp3/`
- `paper/retail pricing/`
- `paper/results_nested/`
- `paper/results_case/`
- `paper/tables_nested/`

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the section-by-section mapping to the manuscript.

If you prefer running scripts directly:

```bash
python experiments_nested/exp1_integrality_gap.py --output-dir paper/exp1 --results-dir paper/results_nested
python experiments_nested/exp2_scalability.py --output-dir paper/exp2 --results-dir paper/results_nested
python experiments_nested/exp3_risk_frontier.py --output-dir paper/exp3 --results-dir paper/results_nested
python experiments_nested/exp4_summary_table.py --results-dir paper/results_nested --tables-dir paper/tables_nested
python experiments_case_retail/case_retail_pricing.py --output-dir "paper/retail pricing" --results-dir paper/results_case
```

For the auxiliary benchmark statements in the manuscript:

```bash
python experiments_nested/exp1_integrality_gap.py --enable-milp --global-milp --output-dir paper/exp1 --results-dir paper/results_nested
python experiments_nested/exp2_scalability.py --validate-lp --output-dir paper/exp2 --results-dir paper/results_nested
python experiments_nested/exp4_summary_table.py --enable-milp --results-dir paper/results_nested --tables-dir paper/tables_nested
```

## Testing

Run the smoke tests with:

```bash
make test
```

or

```bash
python -m pytest -q
```

## Repository Notes

- The optimized solver includes two exact speedups: strong dominance reduction in \((s, v, |t|)\) and an incremental \(\theta\)-sweep for baseline updates.
- The exact certificate is always evaluated on the final discrete selection.
- Default experiment seeds and parameter grids match the manuscript text.
- The generated nested summary table follows the manuscript's published column layout; auxiliary MILP-gap data remain available in the CSV output.

## Citation

Please cite the repository and the accompanying manuscript. A machine-readable citation file is included at [`CITATION.cff`](CITATION.cff).

Software citation:

```bibtex
@misc{shao2026robustmckp,
  title = {robust\_mckp: Reproducible code for robust discrete pricing via MCKP reductions},
  author = {Shao, Eric},
  year = {2026},
  howpublished = {\url{https://github.com/eric939/robust_mckp}}
}
```

Manuscript citation:

```bibtex
@misc{shao2026robustpricing,
  title = {Robust Discrete Pricing Optimization via Multiple-Choice Knapsack Reductions},
  author = {Shao, Eric},
  year = {2026},
  note = {Preprint}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
