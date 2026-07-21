#!/usr/bin/env python3
"""Run every serialized v4 publication phase without modifying the driver."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    "validate",
    "kernel",
    "trace",
    "primary",
    "robustness",
    "stress",
    "application",
    "external",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "v4_reproduction",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=ROOT / "results" / "v4_publication_20260721_certified_final" / "uci_calibration",
    )
    parser.add_argument(
        "--external-archive",
        type=Path,
        default=ROOT / "data_cache" / "RobustKnapsack.zip",
    )
    args = parser.parse_args()
    driver = ROOT / "research" / "v4_publication_campaign.py"
    for phase in PHASES:
        command = [
            sys.executable,
            str(driver),
            phase,
            "--output-dir",
            str(args.output_dir),
        ]
        if phase == "application":
            command.extend(["--calibration-dir", str(args.calibration_dir)])
        if phase == "external":
            command.extend(["--external-archive", str(args.external_archive)])
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
