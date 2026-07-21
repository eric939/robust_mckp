# Manuscript v4 -- simultaneous group-envelope bounds

Version 4 is an independent strategic pivot organized around simultaneous
Lagrangian evaluation over the full robust-MCKP threshold family. It is not a
revision, successor, or replacement of v3. Its central contribution, theorem
set, computational protocol, and publication claim stand on their own; exact
integer integration appears only as a scoped audit of the oracle's downstream
effect.

Canonical source: `main_v4.tex`.

Packaged PDFs are available under `output/pdf/`, including the full manuscript,
OPRE main article, blinded article, electronic companions, and one-page
executive summary.

From the repository root, build the public/blind paper variants with:

```bash
make v4-package
```

Build the identity-scanned anonymous review supplement with:

```bash
make v4-anonymous-package PYTHON=.venv/bin/python
```

The numerical claims are generated from `results/v4_publication_20260721_certified_final` by
`research/generate_v4_publication_artifacts.py`. That directory contains the
serialized protocol, its digest, separate environment records, instance-level and
raw timing CSVs, JSON gate summaries, and the evidence used in every table and
figure. The source-by-source novelty assessment is in
`research/LITERATURE_NOVELTY_AUDIT_V4.md`.
