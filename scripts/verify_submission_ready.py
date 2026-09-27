#!/usr/bin/env python3
"""Run the local IEEE Access submission gate checks."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "ieee-access"
MANIFEST = DIST / "Beyond_Proximity_IEEE_Access_submission_manifest.json"


def run(label: str, command: list[str]) -> None:
    print(f"==> {label}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    python = sys.executable
    run("pytest", [python, "-B", "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider"])
    run("ieee compliance", [python, "-B", "scripts/check_ieee_access_paper.py"])
    run("package", [python, "-B", "scripts/package_ieee_access.py"])
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))
    print("Submission gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
