# V4 literature and novelty audit

Audit date: 19 July 2026. Scope: exact and relaxation algorithms for globally budgeted robust binary optimization, robust knapsack/MCKP, fixed-threshold MCKP LP algorithms, and adjacent robust-MCKP models. Searches used title/abstract/full-text queries for combinations of *robust*, *multiple-choice knapsack*, *budgeted uncertainty*, *threshold*, *Lagrangian*, *parametric*, *envelope*, and *simultaneous*. The conclusion below is necessarily a documented “to the best of our knowledge” assessment, not proof that no unpublished or unindexed result exists.

A final date-sensitive sweep for 2025--2026 work also returned the author's separate March 2026 v3 preprint, *Robust Discrete Pricing Optimization via Multiple-Choice Knapsack Reductions*, because of adjacent robust-MCKP terminology.  It addresses a different research question and does not contain or anticipate the v4 all-threshold group-envelope oracle, minimax dominance theorem, or protocol-fixed evidence.  It is therefore neither a predecessor nor a source of the v4 novelty claim.  The sweep found no external paper deriving the v4 oracle.  Recent work on locally budgeted uncertainty, robust selection with information discovery, and two-stage robust optimization likewise uses different uncertainty structures or decision models and does not address the precise simultaneous-evaluation task audited here.

## Verdict

The broad problem is not novel. The defensible contribution is narrow and algorithmic:

> For an exactly-one, single-constraint MCKP under one global Bertsimas–Sim budget, cancellation of the fixed-threshold baseline converts the Lagrangian dual at a fixed multiplier into per-group maxima of one constant and one common-slope line on each local deviation interval. Prefix/suffix maxima and range accumulation then evaluate that multiplier simultaneously at all global thresholds in `O(B + K log(B+1))` time and `O(B + K)` working storage, where `K` is the option count and `B` the number of distinct thresholds.

The resulting interval bound is valid by weak duality and exact on singleton intervals after multiplier minimization. I found no prior source deriving this cancellation, the two-envelope representation, or the simultaneous all-threshold evaluation algorithm. The claim should therefore be phrased as “to our knowledge” and tied to the precise model and task above.

The final revision adds a formal comparison with the closest bounded-threshold
group-clique relaxation. The epigraph dual of the exact minimax envelope bound
mixes fixed-threshold group-simplex solutions under one aggregate capacity
row. Aggregating that mixture gives a feasible point of the group-clique LP
with the same objective, proving that the exact minimax envelope bound is no
larger. This is an objective-bound dominance result, not a claim that the paper
introduces a stronger robust formulation.

## Non-equivalence matrix

| Literature stream | Established result | Why it does not subsume v4 |
|---|---|---|
| Bertsimas & Sim (2003, 2004) | Cardinality-budget robust discrete optimization; scalar protection threshold; a finite family of modified nominal problems; compact robust counterpart. | Supplies the threshold reduction. It does not jointly evaluate the LP/Lagrangian bounds of all robust MCKP thresholds through exactly-one group envelopes. |
| Álvarez-Miranda, Ljubić & Toth (2013); Lee & Kwon (2014) | Filtering/reducing the set of nominal subproblems needed by the Bertsimas–Sim algorithm. | Reduces how many threshold-indexed nominal problems are solved; does not derive simultaneous evaluation of a multiplier over all thresholds. |
| Hansknecht, Richter & Stiller (2018) | Divide-and-conquer threshold search for robust shortest paths, generalized to robust binary linear optimization. | Prunes threshold solves using monotonicity; does not exploit MCKP exactly-one menus or replace interval LPs by group-envelope range accumulation. |
| Atamtürk (2006); Fischetti & Monaci (2012); Joung & Park (2021); Joung, Oh & Lee (2023) | Strong/compact formulations, cuts, submodularity, and comparisons of robust-knapsack LP relaxations. | Strengthens or reformulates the robust polyhedron. V4 computes a particular interval relaxation faster; it does not claim a new robust formulation or stronger fixed-threshold LP. |
| Büsing, Gersing & Koster (2023) | Strong bilinear formulation, bounded-threshold linearizations, clique strengthening, threshold branching, and improved divide-and-conquer for robust binary optimization. | This is the closest algorithmic comparator. V4 specializes to exactly-one groups and avoids solving its bounded-threshold clique LP by evaluating a different valid Lagrangian interval bound directly. |
| Zemel (1980); Dyer (1984); Sinha & Zoltners (1979); modern MCKP survey (Szkaliczki, 2025) | Fast, including linear-time, algorithms for one fixed MCKP LP; Lagrangian and hull/greedy structure are classical. | V4 does not claim a faster solver for one MCKP LP. It amortizes one multiplier evaluation across a complete robust-threshold family. |
| Monaci, Pferschy & Serafini (2013) | Exact methods and approximation results for the standard robust knapsack problem. | No exactly-one group structure and no simultaneous MCKP threshold envelope. |
| Caserta & Voß (2019) | Robust multiple-choice multidimensional knapsack under ellipsoidal/covariance uncertainty, with conic/linear reformulations and a matheuristic. | Different uncertainty set, multiple-resource geometry, and computational target. |

## Claims that v4 must not make

- The Bertsimas–Sim threshold representation, scalar dualization, or breakpoint candidate set is new.
- Lagrangian relaxation or a fast solution method for a single MCKP LP is new.
- Bounded-threshold branching, clique/GUB aggregation, divide-and-conquer threshold search, or robust-knapsack strong formulations are new.
- The compressed interval bound is always equal to the maximum of the fixed-threshold LPs before splitting. It is a minimax upper bound and may be looser on nonsingleton intervals.
- The method is a universally faster exact integer solver. The demonstrated result concerns certification of the complete threshold-disjunctive LP relaxation.
- Absence from the reviewed literature is absolute proof of novelty.

## Primary sources

1. Bertsimas, D. & Sim, M. (2003), “Robust Discrete Optimization and Network Flows,” *Mathematical Programming* 98, 49–71. DOI: 10.1007/s10107-003-0396-4.
2. Bertsimas, D. & Sim, M. (2004), “The Price of Robustness,” *Operations Research* 52, 35–53. DOI: 10.1287/opre.1030.0065.
3. Álvarez-Miranda, E., Ljubić, I. & Toth, P. (2013), “A Note on the Bertsimas & Sim Algorithm for Robust Combinatorial Optimization Problems,” *4OR* 11, 349–360. DOI: 10.1007/s10288-013-0231-6.
4. Lee, T. & Kwon, C. (2014), “A Short Note on the Robust Combinatorial Optimization Problems with Cardinality Constrained Uncertainty,” *4OR* 12, 373–378. DOI: 10.1007/s10288-014-0270-7.
5. Hansknecht, C., Richter, A. & Stiller, S. (2018), “Fast Robust Shortest Path Computations,” *OASIcs ATMOS* 65, 5:1–5:21. DOI: 10.4230/OASIcs.ATMOS.2018.5.
6. Atamtürk, A. (2006), “Strong Formulations of Robust Mixed 0–1 Programming,” *Mathematical Programming* 108, 235–250. DOI: 10.1007/s10107-006-0709-5.
7. Büsing, C., Gersing, T. & Koster, A. M. C. A. (2023), “A Branch and Bound Algorithm for Robust Binary Optimization with Budget Uncertainty,” *Mathematical Programming Computation* 15, 269–326. DOI: 10.1007/s12532-022-00232-2.
8. Zemel, E. (1980), “The Linear Multiple Choice Knapsack Problem,” *Operations Research* 28, 1412–1423. DOI: 10.1287/opre.28.6.1412.
9. Dyer, M. E. (1984), “An O(n) Algorithm for the Multiple-Choice Knapsack Linear Program,” *Mathematical Programming* 29, 57–63. DOI: 10.1007/BF02591729.
10. Szkaliczki, T. (2025), “Solution Methods for the Multiple-Choice Knapsack Problem and Their Applications,” *Mathematics* 13, 1097. DOI: 10.3390/math13071097.
11. Monaci, M., Pferschy, U. & Serafini, P. (2013), “Exact Solution of the Robust Knapsack Problem,” *Computers & Operations Research* 40, 2625–2631. DOI: 10.1016/j.cor.2013.05.005.
12. Caserta, M. & Voß, S. (2019), “The Robust Multiple-Choice Multidimensional Knapsack Problem,” *Omega* 86, 16–27. DOI: 10.1016/j.omega.2018.06.014.
13. Joung, S. & Park, K. (2021), “Robust Mixed 0–1 Programming and Submodularity,” *INFORMS Journal on Optimization* 3, 183–199. DOI: 10.1287/ijoo.2019.0042.
14. Joung, S., Oh, S. & Lee, K. (2023), “Comparative Analysis of Linear Programming Relaxations for the Robust Knapsack Problem,” *Annals of Operations Research* 323, 65–78. DOI: 10.1007/s10479-022-05161-w.

## Publication recommendation

Proceed with v4 as an independent focused oracle paper. The manuscript proves
the cancellation, complexity, and exact-minimax dominance results; compares against the sparsely
assembled bounded-threshold group-clique formulation; and distinguishes the
protocol-fixed confirmatory evidence from the post-confirmatory UCI-calibrated
and exact-integration panels. A top general OR journal remains a stretch
because the separate exact audit does not show universal integer-solver
improvement. A top specialized optimization or computational-optimization
journal is the strongest fit.
