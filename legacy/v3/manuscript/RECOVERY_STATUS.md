# V3 manuscript recovery status

The final V3 code and scientific records are preserved exactly. The V3
manuscript files themselves are not present in Git history.

The V3 branch's `SUBMISSION.md` records that the local artifact workspace was
expected to contain:

- `paper_versions/v2/main_v2.tex`;
- `paper_versions/v2/main_v2_blind.tex`;
- `submission_ready/main_v2_public.pdf`;
- `submission_ready/main_v2_blind_submission.pdf`;
- generated `auto/`, `tables/`, `figures/`, and result summaries.

Those paths were ignored by the V3 `.gitignore`, so no commit contains them.
They were subsequently removed from the local working tree during cleanup.
Searches of reachable and unreachable Git objects, GitHub branches, tags,
releases and workflow artifacts, macOS Spotlight and Trash, and local Time
Machine snapshots found no exact copy.

Available exact reconstruction inputs are:

1. the public arXiv v1 PDF and source preserved under `legacy/v2/`;
2. the complete V3 code snapshot in `legacy/v3/code/`;
3. V3's `README.md`, `REVISION_HISTORY.md`, `SUBMISSION.md`, tests, and
   experiment drivers inside that snapshot; and
4. the public `v3` Git branch and immutable V3 tag.

A provenance-labeled legacy reconstruction was created in July 2026 at
`reconstructed/robust_mckp_v3_legacy_reconstruction.tex`, with the matching
PDF beside it. The reconstruction states the V3 model, full-breakpoint
decomposition, hull-rounding certificate, global-gap invariant, exact sweep,
implementation record, and evidence limits supported by the surviving source.
Its title page and metadata explicitly say that it is not the deleted original.

An externally recovered original should still be added under `original/`,
accompanied by its source location, recovery date, and SHA-256 checksum. It
must not replace or be conflated with the reconstruction.
