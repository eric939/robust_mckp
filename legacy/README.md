# V2 and V3 legacy archive

This directory preserves the recoverable pre-v4 research lines independently
of mutable working branches and ignored local build directories. Legacy files
must not be rewritten, regenerated in place, or removed by workspace cleanup.

## Version map

| Line | Code anchor | Preserved payload | Provenance status |
| --- | --- | --- | --- |
| V2 | `baddc9e30049da0ba2a59b466b87b45bd166ef65` | Complete Git code snapshot plus the exact arXiv v1 PDF and source package for arXiv:2603.18653 | Exact public March 2026 artifacts |
| V3 | `2e2829e55243b846e5944965a839eb9b28dd5e9c` | Complete final V3 Git code snapshot plus the recovered 39-page public manuscript PDF | Exact code and exact branch-aligned May 2026 paper; later reconstruction retained separately |

V2 denotes the March 2026 arXiv-era paper and code state. V3 denotes the May
2026 certifying full-breakpoint research program preserved on the `v3` branch.
The V3 repository documentation continued to call its ignored manuscript file
`main_v2.tex`; filenames alone therefore do not define the research version.

## Independent anchors

The code snapshots are protected three ways:

1. self-contained `tar.gz` exports in this directory;
2. Git branches `v2` and `v3`; and
3. annotated tags `legacy-v2-code-20260318` and
   `legacy-v3-code-20260522`.

`MANIFEST.sha256` authenticates every archived payload. The extracted arXiv
source is retained beside the original source tarball for immediate inspection.

The V3 manuscript working directory was ignored by Git and was removed before
this archive was created. On 21 July 2026, its complete 39-page public PDF was
recovered from a surviving Apple Mail saved-attachment cache. The PDF's
embedded creation time is 22 May 2026 at 21:47:50 CEST, 44 seconds after the
final V3 code commit. Its title and content agree with the contemporaneous V3
repository and Codex build records. The exact byte stream is preserved under
`v3/manuscript/original/`; see `v3/manuscript/RECOVERY_STATUS.md` for the full
provenance analysis. The seven-page July reconstruction remains under
`v3/manuscript/reconstructed/` as a clearly labeled historical recovery
artifact, not as the canonical paper.
