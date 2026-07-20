Dear Editors:

Please consider the manuscript “Simultaneous Group-Envelope Bounds for
Γ-Robust Multiple-Choice Knapsack Problems” as a Focused Technical submission
to *Operations Research*.

The paper studies a specific computational bottleneck in robust discrete
optimization. Under cardinality-budget uncertainty, a binary model reduces to
a family of nominal problems indexed by deviation thresholds. For a
multiple-choice knapsack problem with exactly-one groups, we show that
Lagrangian dualization cancels the threshold-dependent group baselines. This
exposes two simple group envelopes and permits simultaneous evaluation of one
multiplier across the complete threshold family. The resulting minimax
interval certificate is valid, exact on feasible singleton intervals, and
provably no weaker than the bounded-threshold group-clique relaxation. The
paper also gives an adaptive certification algorithm with explicit global
bound invariants.

The computational study is designed around the paper’s mechanism rather than
one aggregate benchmark. Separate panels test algebraic identity, kernel
scaling, identical-interval oracle behavior, end-to-end certification,
robustness to altered budgets and menu widths, stress scaling through 5,760
groups, and application-derived coefficient realism. In the protocol-fixed
60-instance primary comparison, both methods reach the same prescribed
tolerance; the proposed certificate wins all paired timings and has a 2.46-fold
geometric-mean speedup, with a 95% design-stratified bootstrap interval of
2.29 to 2.65. A separate exact-integration audit is deliberately reported as a
scope boundary: integer subproblem work can dominate, and the paper makes no
claim of universal superiority over compact mixed-integer optimization.

For transparency, a public predecessor is:

Zi Yuan Eric Shao, “Robust Discrete Pricing Optimization via Multiple-Choice
Knapsack Reductions,” arXiv:2603.18653 (March 2026).

The submitted manuscript is a focused successor rather than a duplicate
version. It removes the predecessor’s broad pricing and rounding claims and
does not present the classical Bertsimas–Sim threshold reduction or the
fixed-MCKP relaxation as new. Its central contributions—the simultaneous
group-envelope evaluator, the exact minimax dominance theorem, the adaptive
certificate invariant, the protocol-fixed sparse-comparator study, and the
scoped exact-integration audit—are new to this version. The public preprint is
disclosed here so that the editorial office can evaluate overlap directly
while preserving the journal’s review procedure.

The manuscript and electronic companion include data-and-code statements. A
public reproducibility repository contains the serialized protocol, raw timing
records, environment records, tests, source hashes, and generators for every
reported numerical artifact. A separate identity-scanned archive is available
for anonymous review.

The manuscript is not under review elsewhere, has not appeared in archival
journal form, and presents original work. The author declares no relevant
financial conflict of interest.

Thank you for your consideration.

Sincerely,

Eric Shao

Department of Mathematics, ETH Zürich

ershao@student.ethz.ch
