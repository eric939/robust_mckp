# robust-mckp

Robust Hull-Greedy algorithm for discrete pricing optimization under budgeted demand uncertainty.

## Problem Formulation
We have item groups indexed by \(i=1,\dots,n\). Each group has discrete options \(j\) with

$$
\begin{aligned}
 v_{i,j} &= \omega_i\, x_{i,j}\, \hat g_i(x_{i,j}) \\
 s_{i,j} &= \omega_i\, (x_{i,j} - \Delta a_i)\, \hat g_i(x_{i,j}) \\
 t_{i,j} &= \omega_i\, (x_{i,j} - \Delta a_i)\, \delta_i(x_{i,j}).
\end{aligned}
$$

The robust pricing problem is

$$
\max \sum_i v_{i,j(i)}
\quad\text{s.t.}\quad
\sum_i s_{i,j(i)} - \beta(x,\Gamma) \ge 0,
$$

where \(\beta(x,\Gamma)\) is the sum of the \(\Gamma\) largest values among \(|t_{i,j(i)}|\). Admissible options are those with prices in
\([(1-\sigma_i)a_i,(1+\sigma_i)a_i]\).

## Install
```bash
pip install robust-mckp
```

## Quickstart
```python
from robust_mckp import PricingInstance, Option, solve

instance = PricingInstance(
    items=[
        [Option(5.0, 1.0, 0.2), Option(6.0, 0.5, 0.4)],
        [Option(4.0, 1.0, 0.3), Option(7.0, 0.2, 0.5)],
    ],
    gamma=1,
)

solution = solve(instance)
print(solution.objective, solution.selections, solution.is_feasible)
```

High-level construction from raw pricing data:
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
- `Option(value: float, margin: float, uncertainty: float, price: Optional[float])`
- `PricingInstance(items: List[List[Option]], gamma: int)`
- `Solution(selections: List[int], objective: float, lp_value: float, gap_to_lp: float, certificate_value: float, theta: float, elapsed: float, is_feasible: bool)`
- `from_pricing_data(...) -> PricingInstance`
- `filter_admissible_options(instance, reference_prices, tolerances) -> PricingInstance`
- `solve(instance: PricingInstance) -> Solution`

## Algorithm Summary
1. Enumerate candidate \(\theta\in \{0\} \cup \{|t_{i,j}|\}\).
2. For each \(\theta\), form modified margins \(s^\theta_{i,j}=s_{i,j}-\max(0,|t_{i,j}|-\theta)\).
3. Baseline-slack transform and build upper hulls of \((c_{i,j}^\theta, v_{i,j})\).
4. Solve LP by greedy filling of hull segments.
5. Round to a discrete solution (round-down; optional round-up + repair; optional upgrade completion) and certify feasibility using the original \(s,t\).

**Complexity.** Per \(\theta\), hull filtering is \(O(n m \log m)\). With up to \(O(nm)\) candidate \(\theta\) values, worst-case complexity is \(O(n^2 m^2 \log m)\).

## Citation
```bibtex
@article{bertsimas2004price,
  title={The Price of Robustness},
  author={Bertsimas, Dimitris and Sim, Melvyn},
  journal={Operations Research},
  year={2004}
}
```
