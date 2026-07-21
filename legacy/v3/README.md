# V3 archive

V3 is the May 2026 certifying full-breakpoint research line, distinct from the
new V4 group-envelope pivot.

`code/robust_mckp-v3-code-2e2829e.tar.gz` is a complete `git archive` of commit
`2e2829e55243b846e5944965a839eb9b28dd5e9c`. It preserves the package, exact
theta-enumerated branch-and-bound, parametric sweep, local-budget extension,
solver baselines, publication experiment drivers, tests, and reproducibility
and submission records.

The same commit is retained by branch `v3` and annotated tag
`legacy-v3-code-20260522`. See `manuscript/RECOVERY_STATUS.md` for the precise
recovery provenance of the manuscript files that were stored only in an
ignored local directory.

`manuscript/original/robust_mckp_v3_full_20260522.pdf` is the recovered
complete public V3 paper. It is 39 letter-size pages and carries the final V3
title, author identity, embedded creation time, figures, tables, appendices,
and bibliography. Its embedded creation time is 44 seconds after the final V3
code commit, and its SHA-256 digest is recorded in `legacy/MANIFEST.sha256`.
This is the canonical V3 manuscript artifact.

The `manuscript/reconstructed/` directory contains a seven-page, fully
typeset reconstruction of the V3 scientific record. Its title page and PDF
metadata state that it was reconstructed in July 2026 from the exact V2/arXiv
source and final V3 code and documentation. It predates recovery of the
complete PDF and is retained only as a provenance-labeled recovery artifact.
A compatible Tectonic build is:

```bash
tectonic -X compile robust_mckp_v3_legacy_reconstruction.tex
```

Run that command inside `legacy/v3/manuscript/reconstructed/`. The checked-in
PDF is the visually audited archival build.
