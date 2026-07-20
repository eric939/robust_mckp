# Submission and Public Artifact Manifest

**Current-version flag (2026-07-20):** `paper_versions/v4/` is canonical.
Version 3 is a separate manuscript and is not part of the v4 submission.

## Journal builds

The main source is `paper_versions/v4/main_v4.tex`. The wrappers select public
or blind metadata and main-paper or electronic-companion content:

- `main_v4_opre.tex`: public main manuscript;
- `main_v4_opre_blind.tex`: anonymous main manuscript;
- `main_v4_ec.tex`: public electronic companion;
- `main_v4_ec_blind.tex`: anonymous electronic companion; and
- `executive_summary_opre.tex`: optional one-page editorial summary.

Build and package all six PDFs with:

```bash
make v4-package
```

Build the identity-scanned anonymous supplement with:

```bash
make v4-anonymous-package PYTHON=.venv/bin/python
```

The same target also builds `main_v4.pdf`, the convenient combined
main-plus-appendix version; it is not a separate journal upload.

The current local builds are 16 pages for each OPRE main-paper variant, four
pages for each companion, 15 pages for the combined reading version, and one
page for the executive summary. The summary is cover-letter support and should
be uploaded only if the journal permits it.

The OPRE wrapper is prepared as a Focused Technical submission: the abstract
is text-forward, the introduction contains no equations or mathematical
notation, and all mathematical proofs appear in the main paper. The electronic
companion is limited to comparator implementation, exact-integration audit,
statistical protocol, generators, and reproducibility detail.

For anonymous review, upload only the blind main manuscript, blind electronic
companion, and the generated anonymous supplement if the journal accepts
review artifacts. Inspect the journal-generated merged proof before final
submission. Do not upload the public PDFs, public TeX source, `pyproject.toml`,
or `CITATION.cff` in an anonymous submission.

## Public GitHub artifact

Repository: <https://github.com/eric939/robust_mckp>

The v4 branch/package must expose:

- `research/compressed_interval_oracle.py`;
- `research/bound_dominance.py`, `research/integrated_exact_solver.py`, and
  `research/exact_integration_campaign.py`;
- the complete frozen campaign and artifact generator under `research/`;
- all v4-specific tests;
- `paper_versions/v4/` source, generated TeX inputs, vector figure, and evidence
  manifest;
- `results/v4_publication_20260720_final/`, including raw timing repetitions,
  instance-level records, summaries, protocol, environments, and UCI-derived
  aggregates; and
- current `README.md`, `REPRODUCIBILITY.md`, `SUBMISSION.md`, `CITATION.cff`,
  and `REVISION_HISTORY.md`.

Submission-ready article PDFs are checked in under `output/pdf/` and can also
be attached to a tagged GitHub release or deposited with the journal artifact.
Raw UCI Online Retail transactions are not redistributed.

## Pre-submission checks

```bash
make v4-verify PYTHON=.venv/bin/python
make v4-package
make v4-anonymous-package PYTHON=.venv/bin/python
```

Then check the compiled files:

```bash
pdftotext paper_versions/v4/main_v4_opre.pdf /tmp/main_v4_public.txt
pdftotext paper_versions/v4/main_v4_opre_blind.pdf /tmp/main_v4_blind.txt
rg -ni "TODO|PLACEHOLDER|Version 3|main_v3|results/v3" /tmp/main_v4_public.txt
rg -ni "\\b(Eric|Shao|ershao)\\b|ETH Zürich|github.com/eric939|robust_mckp" /tmp/main_v4_blind.txt
rg -ni "undefined references|undefined citation|multiply defined|missing file" \
  paper_versions/v4/main_v4_*.log
```

All three searches should return no actionable hit. Bibliographic references
to a prior working paper, if retained, must follow the target journal's
double-blind self-citation policy.

Before upload, create an immutable `v4` release tag from the exact submitted
commit and record that tag or archive DOI in the submission form. Branch names
alone are mutable and are not an archival identifier.

## Independence and disclosure rule

The supplied `paper_versions/v4/cover_letter_opre.md` presents v4 as an
independent strategic pivot with its own research question, novelty claim,
theorems, evidence, and submission package. Do not describe v4 as a revision,
successor, replacement, or superseding version of v3. If a journal form asks
for other public manuscripts with adjacent terminology, identify v3 as a
separate project and state that it does not contain or anticipate the v4
group-envelope evaluator, minimax dominance theorem, or protocol-fixed study.
Do not imply that the classical Bertsimas–Sim threshold reduction or
fixed-MCKP LP geometry is new.

## Data and claim policy

The public UCI panel is post-confirmatory and semi-synthetic. Its calibration
uses a nonrandom 200,000-record prefix to set coefficient scales; it does not
estimate causal demand or validate commercial pricing outcomes. Runtime claims
are tied to the released single-threaded environment and comparator
implementation. The principal theoretical claim is simultaneous evaluation and
valid certification of the fixed-threshold LP family together with formal
dominance over the group-clique interval LP. The exact integration validates
global gap accounting but does not support universal integer-solver
superiority.
