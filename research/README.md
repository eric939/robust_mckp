# v4 research and publication pipeline

This directory contains the current executable research and audit trail for the
focused v4 contribution. Superseded exploratory verdicts and campaign drivers
were removed from the publication branch so they cannot be mistaken for
released evidence.

## Canonical v4 components

- `compressed_interval_oracle.py`: simultaneous exactly-one group-envelope
  evaluation.
- `benchmark_instances.py`: deterministic v4 benchmark families with an
  explicit v4 seed namespace.
- `v4_publication_campaign.py`: serialized validation, ablation, comparator,
  primary, robustness, stress, application-derived, and published-coefficient
  experiments.
- `generate_v4_publication_artifacts.py`: manuscript macros, tables, figure,
  and evidence hash manifest.
- `LITERATURE_NOVELTY_AUDIT_V4.md`: source-by-source novelty assessment and
  claim boundary.
- `V4_EXPERIMENT_AUDIT_LOG.md`: experimental audit trail.

Run the released-artifact verification from the repository root:

```bash
make v4-verify PYTHON=.venv/bin/python
```

Run the full serialized protocol into new output directories:

```bash
make v4-reproduce PYTHON=.venv/bin/python
```

`novelty_go_no_go.py` and `structural_feasibility_study.py` now provide shared
oracle and adaptive-certificate components used by the released campaign and
tests. Only files hashed by `paper_versions/v4/auto/v4_evidence_manifest.json`
and records in `results/v4_publication_20260721_certified_final/` support the
manuscript's numerical claims.
