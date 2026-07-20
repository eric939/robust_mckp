# Manuscript v4 -- simultaneous group-envelope bounds

Version 4 is a focused manuscript based on v3. It preserves v3 unchanged and
repositions the paper around simultaneous Lagrangian evaluation over the full
robust-MCKP threshold family. The broad v3 pricing and rounding story is
removed; exact integer integration is retained only as a scoped audit of the
oracle's downstream effect.

Canonical source: `main_v4.tex`.

Packaged PDFs are available under `output/pdf/`, including the full manuscript,
OPRE main article, blinded article, electronic companions, and one-page
executive summary.

From the repository root, build the public/blind paper variants with:

```bash
make v4-package
```

The numerical claims are generated from `results/v4_publication_20260720_final` by
`research/generate_v4_publication_artifacts.py`. That directory contains the
serialized protocol, its digest, separate environment records, instance-level and
raw timing CSVs, JSON gate summaries, and the evidence used in every table and
figure. The source-by-source novelty assessment is in
`research/LITERATURE_NOVELTY_AUDIT_V4.md`.
