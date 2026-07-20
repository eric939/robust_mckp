# Revision History and Scientific Guardrails

This branch records only durable facts for the independent v4 publication
project. The separate v3 manuscript, experiments, and history remain on the
`v3` branch; v4 is neither a revision of v3 nor a replacement for it.

## Version 4 — Current Manuscript

Date: 2026-07-20.

The canonical source is `paper_versions/v4/main_v4.tex`. The focused
algorithmic contribution is simultaneous Lagrangian evaluation over the
complete robust-MCKP threshold family using exactly-one group envelopes.

Durable contributions and boundaries:

- The cancellation and two-envelope representation yields a prefix/suffix
  range algorithm with `O(B + K log(B + 1))` time and `O(B + K)` theoretical
  working storage for ragged group arrays.
- An epigraph-dual mapping proves that the exact minimax envelope bound is no
  larger than the bounded-threshold group-clique LP.
- Interval-bound validity, feasible-singleton equality with the
  fixed-threshold LP, and the adaptive certificate invariant are proved.
- The interval bounds can be embedded in an exact threshold search; a separate
  12-instance audit checks enumeration and compact SCIP and preserves the
  negative many-breakpoints boundary.
- The classical Bertsimas--Sim reduction, fixed-MCKP LP algorithms, filtering,
  branching, clique strengthening, and universal solver-superiority claims
  are explicitly outside the novelty claim.
- Algebraic validation uses 40 irregular held-out instances. The released
  campaign also contains primary, common-trace, robustness, stress, exact
  integration, and post-confirmatory UCI-calibrated panels.
- The finite-menu pricing specialization is a motivating example and a source
  of application-derived coefficient scales. It is not transaction-level
  validation, causal demand estimation, or evidence of commercial pricing
  performance.

## Mathematical and Release Audit

Date: 2026-07-20.

- Corrected the adaptive certificate statement so exhaustive threshold
  evaluation removes obsolete interval records before asserting
  `UB = LB = M`.
- Made initialization robust when root endpoints and the midpoint are
  infeasible, with regressions for a hidden feasible anchor and a globally
  infeasible threshold family.
- Added the bounded-threshold group-clique validity derivation,
  interval-dominance tests, and explicit distinctions between real-arithmetic
  bounds, solver-tolerance certificates, ragged theoretical storage, and the
  released padded implementation.
- Reconciled the UCI data statement: the source has 541,909 records; calibration
  uses the first 200,000, of which 192,451 retained observations form 2,549 SKU
  aggregates.
- Rebuilt public and blind release artifacts after the numerical, test,
  protocol, row-count, and hash checks passed.

## Non-Negotiable Correctness Rules

- The threshold set is the complete original deviation set plus zero. Reduced
  sets cannot support the complete-family claim.
- Exclude infeasible thresholds using the exact group-baseline capacity test
  before defining the target maximum.
- Every active interval record must upper-bound every unevaluated feasible
  threshold it covers. Discarded bounds remain in the global upper bound until
  dominated or made obsolete by exhaustive evaluation.
- Finite multiplier search may weaken an interval bound. It must not be called
  exact minimization except on independently verified feasible singletons.
- The released stopping test is a scaled floating-point solver-tolerance
  certificate, not an interval-arithmetic proof.
- The theorem's `O(B + K)` storage bound assumes ragged group arrays; the
  released padded implementation is `O(B + n m_max + K)` in general.
- The implemented theory assumes integer `Gamma` in `{0, ..., n}`. Fractional
  uncertainty budgets require an explicit extension and new validation.

## Remaining Scientific Limits

- Numerical claims are tied to the dated frozen campaign and its recorded
  environment and protocol. Regenerate the manifest after any source or
  evidence change.
- The timing study supports the reported simultaneous-evaluation advantage in
  the tested regime, not universal dominance over every alternative solver.
- The many-breakpoints family remains a documented boundary case for exact
  integration.
- The UCI-calibrated application is post-confirmatory and semi-synthetic; its
  role is coefficient-scale realism, not external validation of decisions.
