#!/usr/bin/env python3
"""Build a curated anonymous NeurIPS supplement archive.

This intentionally does not archive the repository root.  It copies only the
paper source, public EVQ-Cosh library code, primary reproduction entrypoints,
tests, curated data snapshots, and public-facing docs, then scans the staged
tree for common identity and server-path leaks before writing a zip file.
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evq-cosh-neurips2026-supplement.zip"
STAGE = ROOT / "build_supplement"

ALLOWLIST = [
    "README.md",
    "requirements.txt",
    "requirements-lock.txt",
    "pytest.ini",
    ".github/workflows/smoke.yml",
    "docs/overview/REPRODUCE.md",
    "docs/overview/DATA_PREPARATION.md",
    "docs/overview/PAPER_CLAIMS_MAP.md",
    "data/curated",
    "paper/main.tex",
    "paper/neurips_2026.sty",
    "paper/README.md",
    "paper/compile_aidemo.sh",
    "paper/sections",
    "paper/appendix",
    "paper/tables",
    "paper/figs",
    "paper/refs",
    "scripts/__init__.py",
    "scripts/lib",
    "scripts/core_text_phases/__init__.py",
    "scripts/core_text_phases/run_evq_sweep.py",
    "scripts/core_text_phases/phase14c_multiscale_evq_yarn.py",
    "scripts/core_text_phases/phase16_formula_optimality_sweep.py",
    "scripts/core_text_phases/phase21b_quality_eval_clean.py",
    "scripts/figures",
    "scripts/supporting_eval/__init__.py",
    "scripts/supporting_eval/eval_passkey_scratch.py",
    "tests",
]

EXCLUDE_NAMES = {
    "__pycache__",
    ".DS_Store",
    "unused",
}

EXCLUDE_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".log",
    ".npz",
    ".out",
    ".pyc",
    ".synctex.gz",
}

LEAK_PATTERNS = re.compile(
    rb"misaya|yanghej|hejaz|sshpass|seetacloud|connect\.bjb|connect\.west|"
    rb"@hejazfs|/Users/|/root/autodl-tmp|wandb\.ai",
    re.IGNORECASE,
)


def should_skip(path: Path) -> bool:
    if path.name in EXCLUDE_NAMES:
        return True
    text = path.name
    return any(text.endswith(suffix) for suffix in EXCLUDE_SUFFIXES)


def copy_item(rel: str, stage: Path) -> None:
    src = ROOT / rel
    if not src.exists():
        raise FileNotFoundError(f"allowlisted path does not exist: {rel}")
    dst = stage / rel
    if src.is_dir():
        shutil.copytree(src, dst, ignore=lambda _d, names: [n for n in names if should_skip(Path(n))])
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def scan_for_leaks(stage: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(p for p in stage.rglob("*") if p.is_file()):
        data = path.read_bytes()
        if LEAK_PATTERNS.search(data):
            hits.append(str(path.relative_to(stage)))
    return hits


def write_zip(stage: Path, output: Path) -> None:
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(stage))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-stage", action="store_true")
    args = parser.parse_args()

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for rel in ALLOWLIST:
        copy_item(rel, STAGE)

    hits = scan_for_leaks(STAGE)
    if hits:
        for hit in hits:
            print(f"LEAK-CHECK-FAIL {hit}")
        raise SystemExit("Supplement leak check failed; archive not written.")

    write_zip(STAGE, args.output)
    print(f"Wrote {args.output}")

    if not args.keep_stage:
        shutil.rmtree(STAGE)


if __name__ == "__main__":
    main()
