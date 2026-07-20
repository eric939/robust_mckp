# V4 experiment audit log

Date: 20 July 2026. The final evidence directory is `results/v4_publication_20260720_final`.

## Protocol-fixed design

Before the final confirmatory phases, `v4_publication_campaign.py protocol` wrote the complete design and gates to `protocol.json`. Its SHA-256 digest is `20fd417c6765573b98c88b648abb8af74775e5d0f7d0796aa8a0097853f211ce`. Instance families, sizes, seeds, repetitions, timing order, single-thread rule, tolerance, statistical unit, family--size-cell bootstrap, and go/no-go thresholds were not changed during the final run.

An earlier sparse-comparator run is preserved in `results/v4_publication_20260720`. It produced 60/60 timing wins and a 95% CI above one but failed the predeclared 2x geometric-mean gate (1.79x observed). Profiling then identified redundant reconstruction in the shared fixed-threshold LP routine and repeated evaluation of the compressed oracle's fixed multiplier grid. Those algorithm-preserving inefficiencies were removed, new parity tests were added, and a new protocol was serialized before the complete final rerun. The failed run is not used in manuscript tables.

## Tests before final runs

- Exact trace identity against the dense oracle.
- Direct-formula identity for the cancellation equation.
- Equality of compressed and dense interval optimization.
- Validity against complete fixed-threshold LP scans.
- Singleton equality with the fixed-threshold LP.
- Deterministic instance-stratified bootstrap and instance-level aggregation.
- Certified solver retry behavior.
- Retention of tolerance-pruned interval bounds in the final certificate.
- Direct fixed-LP handling of singleton comparator intervals.
- Sparse CSR assembly of both clique-LP constraint matrices.
- Exact endpoint handling for repeated and near-repeated deviations.
- Separation of zero from arbitrarily small positive multipliers.
- Equality of the reusable direct-hull fixed-MCKP LP routine with the independent reference solver.
- Reuse of fixed multiplier traces across adaptive child intervals.
- Deterministic family--size-cell bootstrap and repeat-block timing diagnostics.

## Corrections found during the audit

1. **Ambiguous HiGHS status.** One wide-menu comparator LP returned a feasible primal with model status “unknown.” A feasible solution to this maximization LP is not a safe upper bound. The implementation now retries dual simplex and interior point and accepts only optimizer-certified optimality or infeasibility. The unchanged robustness panel was restarted.
2. **Discarded-bound accounting.** Both adaptive wrappers originally removed intervals once they were within tolerance but did not retain their last upper bounds in the reported terminal certificate. Pruning choices and attained fixed-threshold values were unaffected, but some `exact` labels were too strong. Both wrappers now carry the maximum discarded bound until the lower bound dominates it. A regression test constructs the failure mode. Every affected panel was rerun.
3. **Singleton fairness.** The comparator wrapper solved a bounded-threshold LP and a fixed-threshold LP when a child was already a singleton; the compressed wrapper solved only the fixed LP. The redundant comparator solve was removed, a regression test was added, and every end-to-end timing panel was rerun from the beginning.
4. **Dense comparator assembly.** The clique comparator used dense matrices despite a sparse formulation. Both equality and inequality matrices now use CSR storage; all headline experiments were rerun. The earlier dense-baseline speedups are superseded.
5. **Search-policy mismatch.** The wrappers used oracle-specific fourth candidates. Both now evaluate the same endpoints and midpoint, leaving the interval bound as the sole method difference.
6. **Inference and repeat selection.** Bootstrap resampling now preserves every family--size cell. The representative numerical result is paired with the median runtime after all repeated bounds and statuses pass consistency checks. Raw outputs include bounds, work counts, execution order, sparse nonzeros, and CSR bytes.
7. **Fixed-LP overhead and trace reuse.** Both methods now share a direct group-hull fixed-MCKP LP routine validated against the independent reference at every threshold in 40 held-out instances. The compressed oracle caches its constant-size multiplier grid across child intervals, preserving `O(B+K)` storage.

No gate, seed, family, instance size, timing rule, or unsuccessful instance was changed or excluded during the final protocol-fixed run. Only that complete run feeds the confirmatory manuscript artifacts. A separate nine-instance UCI-calibrated panel was added afterward and is labeled post-confirmatory. The generated evidence manifest hashes the final source files and every CSV/JSON evidence file.
