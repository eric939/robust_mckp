#!/usr/bin/env python3
"""Preview or remove only ignored, reproducible workspace debris."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIPPED_ROOTS = {ROOT / ".git", ROOT / ".venv", ROOT / "data_cache"}
TRANSIENT_DIRECTORIES = (
    ROOT / ".pytest_cache",
    ROOT / "tmp",
    ROOT / "src" / "robust_mckp.egg-info",
    ROOT / "output" / "anonymous",
    ROOT / "paper_versions" / "v2",
    ROOT / "paper_versions" / "v3",
    ROOT / "paper_versions" / "v4" / "build_ec",
    ROOT / "paper_versions" / "v4" / "build_ec_blind",
    ROOT / "paper_versions" / "v4" / "build_full",
    ROOT / "paper_versions" / "v4" / "build_opre",
    ROOT / "paper_versions" / "v4" / "build_opre_blind",
    ROOT / "paper_versions" / "v4" / "build_summary",
    ROOT / "paper_versions" / "v4" / "figures" / "final_algorithmic_strengthening",
    ROOT / "paper_versions" / "v4" / "figures" / "top_journal",
    ROOT / "paper_versions" / "v4" / "figures" / "v3_experiments",
    ROOT / "paper_versions" / "v4" / "tables" / "final_algorithmic_strengthening",
    ROOT / "paper_versions" / "v4" / "tables" / "submission_upgrade",
    ROOT / "paper_versions" / "v4" / "tables" / "v3_experiments",
)
LEGACY_V4_FILES = (
    ROOT / "paper_versions" / "v4" / "auto" / "auto_numbers.tex",
    ROOT / "paper_versions" / "v4" / "auto" / "extended_auto_numbers.tex",
    ROOT / "paper_versions" / "v4" / "auto" / "submission_performance_numbers.tex",
    ROOT / "paper_versions" / "v4" / "auto" / "submission_upgrade_numbers.tex",
    ROOT / "paper_versions" / "v4" / "auto" / "v3_experiment_numbers.tex",
    ROOT / "paper_versions" / "v4" / "auto" / "v4_experiment_numbers.tex",
    ROOT / "paper_versions" / "v4" / "tables" / "extended_ablation_table.tex",
    ROOT / "paper_versions" / "v4" / "tables" / "scalability_table.tex",
    ROOT / "paper_versions" / "v4" / "tables" / "v4_confirmatory_table.tex",
    ROOT / "paper_versions" / "v4" / "main_v4_opre.ttt",
    ROOT / "paper_versions" / "v4" / "main_v4_opre_blind.ttt",
)
LATEX_SUFFIXES = (".aux", ".log", ".out", ".toc", ".xdv", ".synctex.gz")


def tracked_files() -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {
        (ROOT / item.decode("utf-8")).resolve()
        for item in result.stdout.split(b"\0")
        if item
    }


def skipped(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in SKIPPED_ROOTS)


def candidates(tracked: set[Path]) -> list[Path]:
    found: set[Path] = set()
    results_root = ROOT / "results"
    result_directories = (
        tuple(
            path
            for path in results_root.iterdir()
            if path.is_dir() and path.name != "v4_publication_20260721_certified_final"
        )
        if results_root.is_dir()
        else ()
    )
    for path in (*TRANSIENT_DIRECTORIES, *result_directories, *LEGACY_V4_FILES):
        resolved = path.resolve()
        contains_tracked_file = any(
            item == resolved or resolved in item.parents for item in tracked
        )
        if path.exists() and not contains_tracked_file:
            found.add(path)
    for path in ROOT.rglob(".DS_Store"):
        if not skipped(path) and path.resolve() not in tracked:
            found.add(path)
    for path in ROOT.rglob("__pycache__"):
        if not skipped(path) and not any(path.resolve() in item.parents for item in tracked):
            found.add(path)
    paper = ROOT / "paper_versions" / "v4"
    if paper.is_dir():
        for path in paper.iterdir():
            if not path.is_file() or path.resolve() in tracked:
                continue
            if path.suffix == ".pdf" or any(path.name.endswith(suffix) for suffix in LATEX_SUFFIXES):
                found.add(path)
    return sorted(found, key=lambda path: (len(path.parts), path.as_posix()), reverse=True)


def size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the displayed removals")
    args = parser.parse_args()
    paths = candidates(tracked_files())
    total = sum(size(path) for path in paths)
    action = "REMOVE" if args.apply else "WOULD REMOVE"
    for path in paths:
        print(f"{action}: {path.relative_to(ROOT)}")
    print(f"{action}: {len(paths)} paths, {total / (1024**2):.1f} MiB")
    if args.apply:
        for path in paths:
            if path.exists():
                remove(path)


if __name__ == "__main__":
    main()
