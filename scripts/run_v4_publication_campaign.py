#!/usr/bin/env python3
"""Run every frozen v4 publication phase without modifying the recorded driver."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASES = ("validate", "kernel", "trace", "primary", "robustness", "stress", "application")


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
        default=ROOT / "results" / "v4_publication_20260720_final" / "uci_calibration",
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
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
