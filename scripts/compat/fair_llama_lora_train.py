#!/usr/bin/env python3
"""
Compatibility entrypoint for fair 8B LoRA training.
Delegates to scripts/run_llama8b_fair_suite.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    scripts_dir = Path(__file__).resolve().parents[1]
    target = scripts_dir / "run_llama8b_fair_suite.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing target script: {target}")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
