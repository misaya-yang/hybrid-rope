#!/usr/bin/env python3
"""
Generate a one-glance repository file taxonomy report.

This tool is non-destructive:
- It only scans and classifies.
- It writes a markdown index and optional move suggestions.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple


TOP_LEVEL_CATEGORY: Dict[str, str] = {
    "docs": "documentation",
    "scripts": "execution_entrypoints",
    "rope": "core_algorithm",
    "results": "curated_results",
    "paper_exports": "paper_assets",
    "artifacts": "manifests_and_small_runtime_assets",
    "archives": "historical_snapshots",
    "data": "datasets_or_links",
    "experiments": "one_off_or_non_core",
    "knowledge_base": "research_notes",
    "outputs": "temporary_outputs",
    "tools": "ops_tooling",
    "eval": "evaluation_helpers",
    "handoff": "dated_handoff",
    "neurips_plan": "planning",
    "sigmoid_rope_experiments": "subproject_sigmoid",
    "tmp_phase4_compare": "temporary_workspace",
}

ALLOWED_ROOT_FILES = {
    "README.md",
    "AI_HANDOFF.md",
    ".gitignore",
    "AGENTS.md",
    "train.py",
}


def classify_top_level(entry: Path) -> Tuple[str, str]:
    if entry.is_dir():
        category = TOP_LEVEL_CATEGORY.get(entry.name, "unclassified_directory")
        status = "ok" if category != "unclassified_directory" else "review"
        return category, status
    category = "root_file"
    status = "ok" if entry.name in ALLOWED_ROOT_FILES else "review"
    return category, status


def render_markdown(root: Path) -> str:
    rows: List[Tuple[str, str, str, str]] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if entry.name == ".git":
            continue
        category, status = classify_top_level(entry)
        rows.append((entry.name, "dir" if entry.is_dir() else "file", category, status))

    docs_dir = root / "docs"
    docs_rows: List[Tuple[str, str]] = []
    if docs_dir.exists():
        for f in sorted(docs_dir.glob("*.md")):
            docs_rows.append((f.name, "paper-facing root doc"))

    lines: List[str] = []
    lines.append("# Repo File Index")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Root: `{root.as_posix()}`")
    lines.append("")
    lines.append("## Top-level Classification")
    lines.append("")
    lines.append("| Name | Type | Category | Status |")
    lines.append("|---|---|---|---|")
    for name, typ, category, status in rows:
        lines.append(f"| `{name}` | `{typ}` | `{category}` | `{status}` |")
    lines.append("")
    lines.append("## Docs Root Files")
    lines.append("")
    lines.append("| File | Role |")
    lines.append("|---|---|")
    if docs_rows:
        for name, role in docs_rows:
            lines.append(f"| `docs/{name}` | `{role}` |")
    else:
        lines.append("| (none) | (none) |")
    lines.append("")
    lines.append("## Review Actions")
    lines.append("")
    lines.append("- `status=review` entries should be moved, archived, or documented in `docs/exp/EXPERIMENT_INVENTORY.md`.")
    lines.append("- This report is non-destructive and safe to run repeatedly.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repository file taxonomy report.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("docs/exp/REPO_FILE_INDEX.md"))
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(root), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
