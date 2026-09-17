#!/usr/bin/env python3
"""Merge the two parallel same-target LM arms into the paired report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lm_context import analyze_four_conditions


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--ncp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = rows(args.native) + rows(args.ncp)
    report = analyze_four_conditions(json.loads(args.manifest.read_text()), records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": report["status"], "pairs": report["pairs"]}))


if __name__ == "__main__":
    main()
