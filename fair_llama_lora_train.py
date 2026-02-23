#!/usr/bin/env python3
"""
Compatibility entrypoint for fair 8B LoRA training.
Delegates to 2026-02-22/scripts/run_llama8b_fair_suite.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parent / "2026-02-22" / "scripts" / "run_llama8b_fair_suite.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing target script: {target}")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
