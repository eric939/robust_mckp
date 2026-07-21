from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_manifest_payloads_match() -> None:
    manifest = ROOT / "legacy" / "MANIFEST.sha256"
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        payload = ROOT / relative.strip()
        assert payload.is_file(), relative
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == expected


def test_code_archives_are_complete_git_exports() -> None:
    expected = {
        ROOT / "legacy/v2/code/robust_mckp-v2-code-baddc9e.tar.gz":
            "robust_mckp-v2-code-baddc9e/README.md",
        ROOT / "legacy/v3/code/robust_mckp-v3-code-2e2829e.tar.gz":
            "robust_mckp-v3-code-2e2829e/REVISION_HISTORY.md",
    }
    for archive, required_member in expected.items():
        with tarfile.open(archive, "r:gz") as bundle:
            assert required_member in bundle.getnames()


def test_cleanup_never_targets_legacy_paper_versions() -> None:
    cleanup = (ROOT / "scripts/clean_workspace.py").read_text()
    assert 'ROOT / "paper_versions" / "v2"' not in cleanup
    assert 'ROOT / "paper_versions" / "v3"' not in cleanup
    assert 'ROOT / "legacy"' not in cleanup
