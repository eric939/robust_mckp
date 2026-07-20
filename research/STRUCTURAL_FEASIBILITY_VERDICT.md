# Structural feasibility verdict after the second go/no-go campaign

Date: 19 July 2026

## Verdict

**NO-GO for the initial exact-method angle explored for the independent v4
project.** The candidate contributions tested in this campaign do not clear
novelty and competitiveness simultaneously. This is a negative result supported
by exact polyhedral enumeration, exact separation, solver-controlled end-to-end
tests, scaling to 720 groups, and a direct implementation of the strongest
transferable bounded-threshold clique baseline from Büsing, Gersing, and Koster
(2023). It is not an assessment or proposed rewrite of the separate v3 paper.

The unchanged robust-MCKP problem can still support a publishable contribution,
but only conditionally: a new theorem and algorithm must exploit the GUB/menu
structure to compute or separate the full threshold-disjunctive relaxation more
efficiently than the known RP4/bounded-z/clique approaches. The experiments do
not confirm such a result yet. Without that theorem, the algorithmic project
should stop rather than move into manuscript rewriting.

## Gate-by-gate decision

| Gate | Required evidence | Result | Decision |
|---|---|---:|---|
| New polyhedral structure | A materially new facet class beyond basic conflicts and known lifted GUB/polymatroid inequalities | 400/409 nontrivial exact facets were non-conflict; 62.6% used coefficients above one, but no new class or separator was established | **FAIL** |
| Useful exact separation | Material closure of the residual root gap on the model objective | Exact basic-conflict separation found a cut on 25% of 36 cases; median cuts = 0 and median closure = 0 | **FAIL** |
| Solver impact | Better certification, runtime, or nodes than plain SCIP | Plain/enhanced certification both 88.9%; median 1.214 s vs 1.326 s; speed ratio 0.996 | **FAIL** |
| Strong relaxation | Clear strength beyond the compact and prior-art bounded-z relaxations | Lagrangian interval bound was tighter than the group-clique root on 75% of 36 cases and was essentially at the full disjunctive value at the median | **PASS on strength only** |
| End-to-end exact method | Material improvement over the existing threshold solver and SCIP | Earlier campaign: 87.5% certification for both interval and threshold methods; 0.332 s vs 0.286 s; SCIP certified 100% | **FAIL** |
| Scaling | At least 5x end-to-end acceleration robustly across families | Adaptive oracle achieved 5x on 50% of 12 scaling cases and 5.36x at n=720 only on many-breakpoints; median over the 12 cases was 3.60x | **FAIL** |
| State-of-the-art interval baseline | Beat bounded-z branching with exact-one group cliques | Both methods met 1e-6 tolerance on all 36 cases. Overall median was 0.0715 s (ours) vs 0.0540 s (clique). On many-breakpoints: 0.2587 s vs 0.0152 s | **FAIL** |
| Reproducibility/validity | Independent limiting-case tests and repository regression tests | Exact rational cddlib facets, SCIP controls, singleton/root equivalence tests, and 93 passing tests | **PASS** |

## Numerical evidence

### Exact hull and facet campaign

Twenty-four full-dimensional integer instances (four or five groups, three
options per group) produced 677 exact rational facets through cddlib:

- 409 were nontrivial;
- only 9 were basic minimal-conflict facets;
- 400 were non-conflict facets;
- all 409 nontrivial facets cut both the compact and the complete
  threshold-disjunctive LP;
- 62.59% of the disjunctive-cutting facets had a coefficient larger than one;
- median facet support was seven variables.

This conclusively rejects basic group conflicts as a hull description. It does
not establish novelty: the observed weighted rows occupy the territory of
classical lifted GUB covers and robust polymatroid inequalities. A paper cannot
call them new until a formal non-equivalence theorem is proved.

### Exact conflict separation and SCIP

The exact separator was run on 36 instances across four adversarial families
and n in {30, 60, 90}. A SCIP-certified reference was available for 88.9% within
the 15-second limit.

- Median compact gap: 0.5566%.
- Median full-disjunctive gap: 0.0459%.
- Median exact-conflict gap: 0.5566%.
- Separator found at least one cut on 25% of instances; median cut count: zero.
- Plain SCIP median: 1.214 s; cut-enhanced SCIP median: 1.326 s.
- Certification rate: 88.9% for both.

Exact separation removes the earlier concern that the weak result was merely a
bad heuristic separator. The basic conflict family itself is not useful enough.

### Scaling and the decisive prior-art control

The Lagrangian interval oracle was compared with exhaustive threshold LP scans
through n=720. On the unique-threshold many-breakpoint family it was consistently
about 5.1--5.6x faster and evaluated below 1% of thresholds. This initially looked
promising.

The decisive control specializes the bounded-z clique formulation of Büsing,
Gersing, and Koster to each exactly-one option group. It was verified to equal:

1. the fixed-threshold LP on singleton intervals; and
2. the repository's compact group LP on the full threshold interval.

On 36 cases (four families, n in {60, 90, 180}, three seeds), both adaptive
methods reached relative gap 1e-6 on every case:

| Family | Median ours (s) | Median clique (s) | Clique time / our time | Ours / full scan | Clique / full scan |
|---|---:|---:|---:|---:|---:|
| correlated_risk | 0.0719 | 0.0583 | 0.804 | 0.72x | 0.85x |
| dense_frontier | 0.0373 | 0.0580 | 1.584 | 2.03x | 1.29x |
| many_breakpoints | 0.2587 | 0.0152 | 0.0586 | 5.17x | 87.33x |
| near_tie | 0.0366 | 0.0654 | 1.804 | 1.90x | 1.04x |

Our method wins modestly on dense-frontier and near-tie cases, but loses on
correlated-risk and loses catastrophically on the family designed to justify
threshold compression. Across all 36 cases, the median times are 0.0715 s for
our method and 0.0540 s for the clique baseline. Thus the apparent scaling result
does not survive comparison with the relevant state of the art.

### Relaxation strength versus prior art

Across 36 n={30,60,90} instances:

- the group-clique bounded-z LP dominated the ungrouped bounded-z LP on 100%;
- the Lagrangian interval bound was strictly tighter than the group-clique root
  on 75%;
- median excess over the complete threshold-disjunctive LP was 1.0049% for the
  ungrouped formulation, 0.5753% for the clique formulation, and approximately
  1.2e-8% for the Lagrangian bound.

This is the one scientifically valuable result. It identifies a potentially
interesting bound, but a strong bound is not a publishable algorithm when a known
baseline reaches the same stopping criterion faster.

## Literature cross-check and novelty consequence

- Atamtürk (2006) already gives strong polynomial-size robust formulations and
  the RP4 full-disjunction construction when the nominal formulation is tight.
- Joung and Park (2021) already give extended polymatroid inequalities, polynomial
  separation, and a convex-hull result for a closely related robust continuous
  0-1 set.
- Büsing, Gersing, and Koster (2023) already combine bounded thresholds, strong
  formulations, clique strengthening, and threshold branching; they also show
  that the machinery largely transfers to uncertain constraints. Their DnC+
  and branch-and-bound methods are therefore the correct state-of-the-art
  comparators, not exhaustive threshold scanning.

Consequently, none of these statements is defensible as a headline novelty:

- “we branch on threshold intervals”;
- “we avoid enumerating all thresholds”;
- “we exploit mutually exclusive options with clique constraints”;
- “we add minimal infeasible group conflicts”; or
- “we use the complete threshold disjunction.”

## The only remaining credible path

Keep the core robust-MCKP problem, but impose one final theory-first gate before
any manuscript rewrite:

1. Prove exactly what relaxation the Lagrangian oracle computes. The target is
   equivalence to, or strict dominance over, the RP4/full-threshold relaxation
   after projection for the MCKP/GUB structure.
2. Reduce its current many-breakpoint preprocessing cost. A publishable target is
   an O(N log N) or similarly sharp support algorithm whose end-to-end time beats
   the clique interval baseline, including construction time.
3. Formally classify the 400 non-conflict facets. Continue only if at least one
   infinite family is not implied by known lifted GUB cover or extended
   polymatroid inequalities and admits practical separation.
4. Re-run against the authors' BnB/DnC+ implementation or a faithful
   uncertain-constraint port, then require a preregistered win: at least 20--30%
   lower shifted-geometric-mean time, no lower certification rate, and gains on
   at least three of four families.

If step 1 produces no new theorem, or step 3 classifies all facets as known, the
top-journal methodological goal is not feasible with the unchanged contribution
space. At that point the rational choices are an application/empirical paper with
new data and decisions, a reproducibility/software paper, or a different robust
uncertainty model whose structure has not already been covered.

## Reproducibility record

- `research/novelty_go_no_go.py`: first campaign.
- `research/structural_feasibility_study.py`: exact facets, exact separator,
  strong formulations, scaling, and clique comparison.
- `tests/test_novelty_go_no_go.py` and
  `tests/test_structural_feasibility_study.py`: validity/regression tests.
- The historical scratch result directories are intentionally omitted from the
  clean v4 release. The durable design and numerical findings are retained in
  this verdict; the executable campaign sources and regression tests remain.

Final regression command: `PYTHONPATH=. .venv/bin/pytest -q` (93 tests passed).
