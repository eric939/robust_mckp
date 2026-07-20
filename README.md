# robust_mckp

Reference implementation and reproducibility artifact for **“Simultaneous
Group-Envelope Bounds for Γ-Robust Multiple-Choice Knapsack Problems.”**
Manuscript v4 (July 2026) is the canonical working paper. Version 3 is retained
only as the predecessor from which the focused v4 question was separated.

The v4 contribution is an all-threshold Lagrangian evaluation algorithm for
exactly-one groups. For `K` options and `B` robust-deviation thresholds, it
evaluates a multiplier in `O(B + K log(B + 1))` time and `O(B + K)` working
storage. The paper combines this oracle with a valid adaptive certificate for
the maximum fixed-threshold MCKP LP value and proves that the exact minimax
bound is no weaker than the bounded-threshold group-clique LP. A released exact
interval-search integration validates global gap accounting without claiming a
universally superior integer solver.

## Repository map

- `research/compressed_interval_oracle.py`: proposed group-envelope oracle.
- `research/bound_dominance.py`: exact epigraph LP used to validate the formal
  dominance theorem.
- `research/integrated_exact_solver.py`: exact integer threshold-interval
  search using either the envelope or clique bound.
- `research/exact_integration_campaign.py`: controlled exact-solver audit.
- `research/v4_publication_campaign.py`: frozen validation and experiment
  protocol, gates, generators, and statistical summaries.
- `research/generate_v4_publication_artifacts.py`: tables, figure, macros, and
  hash manifest used by the manuscript.
- `research/LITERATURE_NOVELTY_AUDIT_V4.md`: source-by-source novelty audit.
- `tests/test_compressed_interval_oracle.py` and
  `tests/test_v4_publication_campaign.py`: v4 algebra and protocol tests.
- `paper_versions/v4/`: canonical manuscript source and generated inputs.
- `results/v4_publication_20260720_final/`: released instance-level results,
  raw timing repetitions, protocol, environments, summaries, and public-data
  calibration aggregates.
- `src/robust_mckp/`: predecessor solver components reused by the experiments.

Submission-ready manuscript PDFs are checked in under `output/pdf/`; their
source files and build instructions are listed in `SUBMISSION.md`.

## Install and verify

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[experiments,validation,dev]"
make v4-verify PYTHON=.venv/bin/python
```

`v4-verify` runs the test suite, checks every released evidence hash, regenerates
the manuscript tables/macros/figure in a temporary directory, and compares the
generated text artifacts with the checked-in versions. It does not rerun the
long timing campaign.

To rerun the complete frozen campaign and rebuild its paper artifacts:

```bash
make v4-reproduce PYTHON=.venv/bin/python
```

The full command uses the released UCI-derived aggregate calibration, never the
raw transaction file. See `REPRODUCIBILITY.md` for the protocol, run-time scope,
data provenance, and separate manuscript-build command.

## Headline released evidence

- 40/40 algebraic and complete-scan validation cases pass.
- 60/60 primary instances reach scaled tolerance `1e-6`; the proposed method
  wins 60/60 paired timings with geometric-mean speedup 2.456 (95% stratified
  bootstrap CI [2.290, 2.649]).
- 24/24 common-trace cases favor the proposed oracle, with geometric-mean
  speedup 3.272 and no weaker matched interval bound.
- All 36 robustness cases and all eight stress cases reach tolerance; the
  stress geometric-mean speedup is 5.766 through 5,760 groups.
- The post-confirmatory, 200,000-record UCI-calibrated semi-synthetic panel wins
  9/9 timings, with geometric-mean speedup 1.799 (95% bootstrap interval
  [1.527, 2.186]).
- In the separate 12-instance exact audit, envelope and clique interval search
  each certify 10 cases, enumeration certifies 11, and compact SCIP certifies
  all 12; certified objectives agree within `1.14e-13`.

These are instance-level paired results on the recorded single-threaded
environment, not universal performance claims.

## Citation and license

Please cite `CITATION.cff` and the accompanying manuscript. Licensed under the
MIT License; see `LICENSE`.
