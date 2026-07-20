# Revision History and Scientific Guardrails

This file replaces earlier transient worklogs, mock-review reports, claim
checklists, and final-gate notes. It is intentionally short and should record
only durable facts needed by future maintainers.

## Version 4 — Current Manuscript

Date: 2026-07-20.

**v4 is the current paper.** Its canonical source is
`paper_versions/v4/main_v4.tex`. It is an independent strategic pivot, not a
revision, successor, or replacement of v3. Its focused algorithmic
contribution is simultaneous Lagrangian evaluation over the complete
robust-MCKP threshold family using exactly-one group envelopes.

Durable v4 changes:

- derives the cancellation and two-envelope representation, the
  prefix/suffix range algorithm, and its `O(B + K log(B + 1))` time and
  `O(B + K)` working-storage bounds;
- proves through an epigraph-dual mapping that the exact minimax envelope bound
  is no larger than the bounded-threshold group-clique LP;
- proves interval-bound validity, feasible-singleton equality with the
  fixed-threshold LP, and the adaptive certificate invariant;
- embeds either interval bound in an exact threshold search and reports a
  separate 12-instance audit against enumeration and compact SCIP, preserving
  the negative many-breakpoints boundary;
- explicitly excludes the classical Bertsimas–Sim reduction, fixed-MCKP LP
  algorithms, filtering, branching, and clique strengthening from the novelty
  claim;
- validates the algebra on 40 irregular held-out instances and compares against
  a sparsely assembled bounded-threshold group-clique LP under a serialized
  paired protocol;
- reports 60/60 primary timing wins and complete tolerance attainment, plus
  common-trace, robustness, stress, and post-confirmatory UCI-calibrated panels;
  and
- releases the protocol, raw repetitions, instance-level results, environment
  records, aggregate public-data calibration, generated paper inputs, tests,
  and a hash-verification command.

V4's finite-menu pricing specialization is only a motivating example and a
source of application-derived coefficient scales. It does not establish a
manuscript lineage or shared publication claim with the separate v3 project.

### Mathematical and release audit

Date: 2026-07-20.

- Corrected the adaptive certificate statement so exhaustive threshold
  evaluation removes obsolete interval records before asserting
  `UB = LB = M`.
- Made initialization robust when the root endpoints and midpoint are
  infeasible, and added regressions for both a hidden feasible anchor and a
  globally infeasible threshold family.
- Added the bounded-threshold group-clique validity derivation, interval-wide
  dominance tests, and explicit distinctions between real-arithmetic bounds,
  solver-tolerance certificates, ragged theoretical storage, and the released
  padded implementation.
- Reconciled the UCI data statement with the released protocol: the source has
  541,909 records, while calibration uses the first 200,000; 192,451 retained
  observations form 2,549 SKU aggregates.
- Rebuilt all public and blind PDFs after 142 tests, 40 fresh irregular-case
  validations, and the release hash/row-count/protocol checks passed.
- Converted the abstract and introduction to OPRE Focused Technical format,
  moved all proof material into the main paper, removed version metadata from
  the blind build, added subject classifications, and repaired the end-float
  prior-art table.

## Version 3 — Separate Manuscript

Date: 2026-07-18.

Its canonical source is `paper_versions/v3/main_v3.tex`; v2 remains unchanged
as an earlier snapshot. The v3 revision implemented the following novelty
corrections:

- attributes full-breakpoint enumeration to Bertsimas--Sim and robust-knapsack
  antecedents;
- attributes upper-hull and exact-MCKP primitives to the classical MCKP
  literature;
- positions the contribution as an integrated, certificate-preserving solver
  framework rather than a new robust decomposition or state-of-the-art solver;
- maps the direct MILP and independent enumeration baselines to their prior-art
  counterparts;
- promotes the negative computational results and semi-synthetic application
  limitations into the abstract, contribution statement, experiments, and
  conclusion; and
- expands the robust optimization, robust knapsack, exact MCKP, and robust
  pricing references identified by the novelty review.

### V3 experiment upgrade and repository consolidation

Date: 2026-07-18.

- Replaced the loose-capacity primary benchmark with 120 binding-capacity
  instances (600 matched rows) across four hard families through (n=180).
- Added a 162-row matched anytime frontier with realized-runtime overrun
  reporting and valid incumbent--upper-bound gaps.
- Added a 200-case adversarial audit covering reduced-threshold checking,
  LP-hull integer filtering, incomplete global bounds, and skipped original
  robust rechecks; all witnesses expose the unsafe shortcut.
- Expanded the semi-synthetic study to 30 portfolios, 20,000 paired stress
  draws per cell, six protocols, and 60 certified exact subset solves.
- Consolidated generation and evidence audit in `run_v3_experiments.py` and
  `build_v3_experiment_evidence.py`; removed superseded benchmark, plotting,
  summarization, result, and package snapshots.
- The final evidence audit reports zero finite-incumbent certificate failures,
  zero certified cross-method objective mismatches, and zero missed unsafe
  cases.

### V3 editorial audit and submission trim

Date: 2026-07-18.

- Narrowed the main publication angle to the two strongest contributions:
  robust-feasible HullRound certificates and exact limited-run gap semantics.
- Moved the correct but empirically slower parametric-sweep theorem,
  validation table, and maintenance evidence to the electronic companion.
- Consolidated repeated prior-art disclaimers, routine branch-and-bound
  mechanics, an unused classical greedy special case, and fragmented
  application subsections without removing any headline theorem, limitation,
  or experiment.
- Reduced the public and blind main papers from 32 to 30 physical pages;
  references now begin on page 27 and the end-float benchmark table occupies
  page 30, for about 28 journal-counted pages.  The expanded companion is 9
  pages and the executive summary remains 1 page.
- Recompiled all public/blind artifacts with embedded fonts and no undefined
  references, citation failures, or overfull/underfull boxes.

## Current Paper Positioning

The v4 manuscript is framed around one narrow contribution for finite-menu,
integer-budget \(\Gamma\)-robust MCKP: simultaneous evaluation of a fixed
Lagrange multiplier over the complete threshold grid using exactly-one group
envelopes and range accumulation. The adaptive procedure certifies the maximum
fixed-threshold LP value, not the robust integer optimum. The classical
threshold reduction, fixed-MCKP algorithms, filtering, branching, clique
strengthening, and a universal solver-superiority claim remain outside the
novelty claim.

Discrete pricing is a motivating specialization and supplies a
post-confirmatory, UCI-calibrated semi-synthetic coefficient panel. It is not
transaction-level validation, causal demand estimation, or evidence of
commercial pricing performance.

## Non-Negotiable Correctness Rules

- The threshold set is the complete original deviation set plus zero; reduced
  sets cannot support the paper's complete-family claim.
- Infeasible thresholds are excluded using the exact group-baseline capacity
  test before defining the target maximum.
- Every interval record must upper-bound every unevaluated feasible threshold
  it covers; discarded bounds remain in the global upper bound until dominated
  or made obsolete by exhaustive evaluation.
- Finite multiplier search may weaken an interval bound but must never be
  described as exact minimization except on independently verified feasible
  singletons.
- The released stopping test is a scaled floating-point solver-tolerance
  certificate, not an interval-arithmetic proof.
- The theorem's `O(B + K)` storage bound assumes ragged group arrays; the
  released padded implementation is `O(B + n m_max + K)` in general.
- The implemented theory assumes integer
  \(\Gamma\in\{0,\ldots,n\}\); fractional uncertainty budgets require an
  explicit extension and new validation.

## Adversarial Journal-Compliance Pass

Date: 2026-07-14.

Changes:

- Recast the abstract opening in accessible, text-only language while
  retaining every technical result and limitation; the abstract remains well
  below the 200-word limit.
- Replaced the blind omission of the prior working paper with an anonymous
  citation and an explicit statement of the additional certification,
  limited-gap, sweep, and computational contributions.
- Replaced nonstandard subject classifications with three categories from the
  official OR/MS classification scheme and accompanying descriptive phrases.
- Corrected the submission documentation: the one-page executive summary is
  optional for the Optimization area, not required.
- Added a strictly anonymous code/data supplement and a separate anonymous
  upload bundle so the mixed public/blind archive cannot be uploaded by
  mistake.
- Verified the principal figures in grayscale as well as color; line styles,
  point symbols, and contrast remain distinguishable under print conversion.

Validation:

- The anonymous code/data supplement passes the full 82-test suite and all
  458 submission-campaign checks from inside the packaged directory.
- Static citation/label checks find no duplicate labels, missing references,
  missing bibliography entries, uncited bibliography entries, or stale
  placeholders.
- The final compile, numerical, test, anonymity, font, packaging, and visual
  gates all passed after this rebuild.

## Final Tightening and Formatting Pass

Date: 2026-07-14.

Changes:

- Removed about 500 words (roughly 4% of the manuscript source) without
  removing any theorem, proof obligation, numerical result, limitation, or
  reproducibility statement.
- Consolidated repeated scope caveats, certificate descriptions, numerical
  transitions, and application qualifications; shortened figure captions and
  the conclusion.
- Reduced the optional one-page executive summary from 391 to 300 words.
- Applied ragged-right formatting to narrow companion table columns, removing
  the remaining underfull-box warnings.
- Reduced journal-counted main-paper length from 30 to 29 pages excluding
  references; the electronic companion remains 8 pages and the executive
  summary remains 1 page.

Validation:

- Public/blind main and companion PDFs and the executive summary compile
  without undefined references, undefined citations, overfull or underfull
  boxes, or TeX errors.
- All 458 submission-campaign checks, all 110 legacy claim checks, and all
  reported-number checks pass; the earlier full 82-test run remains current
  because this pass changed only manuscript and submission-document prose.
- All PDF fonts are embedded; strict blind-identity and stale-string scans
  pass.

## Latest Experimental Revision Pass

Date: 2026-07-14.

Changes:

- Replaced the incrementally assembled exact benchmark with a clean four-seed
  campaign: 96 instances and 480 matched rows across six families and five
  methods under uniform 10-second limits. The final rerun is explicitly
  single-threaded and uses end-to-end timing, including preprocessing or MILP
  model translation and solve.
- Added an independent five-seed performance campaign: 240 instances and 1,200
  matched rows over four non-tight families, three problem sizes, two menu
  widths, and two uncertainty-budget regimes. Optimized exact B&B, SCIP, and
  HiGHS certify all 240 rows; the uncached baseline certifies 236, and all 236
  jointly certified four-solver objectives agree.
- Expanded the semi-synthetic application from 8 to 20 portfolios and from
  2,500 to 10,000 stress draws per policy and protocol; paired policy
  comparisons now use common random numbers.
- Expanded the application exact subset to 40 certified reference rows.
- Replaced the principal empirical graphics with embedded-font,
  colorblind-safe vector figures showing replications, Wilson intervals,
  interquartile ranges, certification status, performance profiles, primal
  gaps, and paired runtime ratios. The only retained raster layers are four
  454-ppi heatmap tiles in one companion diagnostic; all text and axes remain
  vector.
- Removed redundant small-solver and empirical-synthesis figures/tables and corrected the
  companion build so tables remain in their cited C.* and D.* sections rather
  than being renumbered by main-paper end-float processing.
- Added a one-page nontechnical executive summary as optional submission and
  cover-letter support.
- Revised the numerical claims to state the clean rerun's negative as well as
  positive evidence: the optimized exact configuration has no material median
  runtime advantage, SCIP/HiGHS are faster median baselines, and tight capacity
  contains all 12 limited exact rows.

Validation:

- All 458 submission-campaign checks and all 110 legacy claim checks passed.
- All reported-number checks and the full 82-test suite passed.
- Public/blind main and companion PDFs compile without undefined references,
  undefined citations, overfull boxes, or TeX errors.
- Visual checks passed for the new vector figures, benchmark tables, companion
  diagnostics, end-float pages, and executive summary; all PDF fonts are
  embedded and strict blind identity scans passed.
- The main paper has 30 countable pages excluding references; the electronic
  companion has 8 pages, and the executive summary has 1 page.

## Previous Revision Pass

Date: 2026-07-13.

Changes:

- Compressed the manuscript from a 40-page combined draft into a 30-page
  journal-oriented main paper plus a 12-page electronic companion.
- Shortened the title, abstract, introduction, related-work positioning,
  numerical discussion, application, conclusion, and end matter.
- Merged redundant contribution, regime, algorithm, and application material;
  retained the definitions and qualifications needed for the proofs and
  certificates.
- Moved proof detail, implementation validation, supplementary tables, and
  secondary figures to the electronic companion.
- Added public/blind Operations Research wrappers with 11-point type, 1.5
  spacing, and one-inch margins.

Validation:

- All 110 manuscript-to-results claim checks passed.
- All reported-number checks passed.
- Public/blind main and companion PDFs compiled without undefined references,
  undefined citations, overfull boxes, or TeX errors.
- Strict identity scans of both blind PDFs passed.
- The main paper contains 30 countable pages (29 body pages plus one table
  page) and a one-page reference list; the electronic companion is 12 pages.

## Earlier Revision Pass

Date: 2026-05-22.

Changes:

- Strengthened the manuscript novelty map and global exactness/gap theorem.
- Updated the exact global solver to initialize root LP upper-bound records for
  every \(\theta\) before reporting finite global gaps.
- Added tests for the per-\(\theta\) upper-bound-record invariant under limited
  global runs.
- Added HullRound diagnostics for \(\Delta V_{\max}^{\theta}\), realized
  round-down loss, and scale-normalized certificate strength.
- Added `delta_v_max_over_lp_ub` to publishable experiment CSV generation.
- Clarified the split between public GitHub code and the exact local
  submission artifact snapshot.
- Synced rebuilt public/blind manuscript source and PDFs into
  `submission_ready/`.

Validation commands run:

```bash
.venv/bin/python -m pytest -q
cd paper_versions/v2 && tectonic --keep-logs main_v2.tex
cd paper_versions/v2 && tectonic --keep-logs main_v2_blind.tex
.venv/bin/python scripts/run_clean_repro_check.py --quick
.venv/bin/python scripts/run_solver_benchmarks.py --smoke --exact-only --theta-time-limit 2 --full-time-limit 2 --fixed-time-limit 2
```

Observed results:

- Full test suite passed: 82 tests.
- Public and blind manuscript builds passed.
- Public PDF text grep found no stale section/reference wording checked in
  `SUBMISSION.md`.
- Blind PDF strict identity regex found no author, institution, GitHub, or
  arXiv self-identification leaks.
- Clean reproducibility smoke check passed.
- Open-solver smoke completed; SCIP and HiGHS were available, Gurobi and CPLEX
  were unavailable.

## Remaining Scientific Limits

- Numerical claims must remain tied to the dated submission campaign and its
  recorded configuration; rerun the campaign before changing them.
- SCIP/HiGHS remain strong median-runtime baselines in the tested environment.
  The paper should not claim general runtime dominance over mature MILP
  solvers.
- Tight-capacity rows are the main exact-solver bottleneck.
- The semi-synthetic pricing application is own-price separable and controlled
  by construction; cross-price substitution and transaction-level validation
  remain future work.
