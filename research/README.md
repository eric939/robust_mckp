# v4 research and publication pipeline

This directory records the research path from the failed initial novelty tests
to the focused v4 contribution.

## Canonical v4 components

- `compressed_interval_oracle.py`: simultaneous exactly-one group-envelope
  evaluation.
- `benchmark_instances.py`: deterministic v4 benchmark families with a frozen
  seed namespace.
- `v4_publication_campaign.py`: frozen validation, ablation, comparator,
  confirmatory, robustness, stress, and application-derived experiments.
- `generate_v4_publication_artifacts.py`: manuscript macros, tables, figure,
  and evidence hash manifest.
- `LITERATURE_NOVELTY_AUDIT_V4.md`: source-by-source novelty assessment and
  claim boundary.
- `V4_EXPERIMENT_AUDIT_LOG.md`: experimental audit trail.

Run the released-artifact verification from the repository root:

```bash
make v4-verify PYTHON=.venv/bin/python
```

Run the full frozen protocol into new output directories:

```bash
make v4-reproduce PYTHON=.venv/bin/python
```

The older `novelty_go_no_go.py`, `structural_feasibility_study.py`, and related
verdict files document negative or intermediate research stages. They are
provenance, not manuscript evidence unless cited by the v4 manifest. Their
historical scratch result directories are intentionally excluded from the
clean v4 release; the durable designs and findings remain in the verdicts and
executable source.
