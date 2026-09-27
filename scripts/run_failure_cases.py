#!/usr/bin/env python3
"""Run curated room-disambiguation failure cases and export results for the paper."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="default")
    parser.add_argument("--out", default="outputs/failure_cases/failure_cases_default.json")
    args = parser.parse_args()

    from backend.query.failure_cases import run_failure_suite

    result = run_failure_suite(args.scene)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    s = result["summary"]
    print(f"Cases run: {result['cases_run']}")
    print(f"Graph room correct: {s['graph_room_correct']}")
    print(f"Flat room correct:  {s['flat_room_correct']}")
    print(f"Flat wrong room:    {s['flat_wrong_room_count']}")
    print(f"Graph wins room:    {s['graph_wins_room_count']}")
    print(f"Wrote {out_path}")

    print("\nPer-case:")
    for r in result["results"]:
        c = r["case"]
        print(
            f"  {c['query'][:45]:45s} "
            f"flat={r['flat']['view_id']}({r['flat']['classification']}) "
            f"graph={r['graph']['view_id']}({r['graph']['classification']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
