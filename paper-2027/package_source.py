#!/usr/bin/env python3
"""Package the active manuscript and its exact TeX dependencies."""
from __future__ import annotations

import hashlib
import ast
import re
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER / "exponent-allocation-source.zip"


def runtime_sources() -> dict[str, bytes]:
    """Bundle existing frozen-evaluation entrypoints and their local imports."""
    repo = PAPER / 'runtime' if (PAPER / 'runtime/experiments').is_dir() else PAPER.parent
    roots = [
        "experiments/llama3_60dir_20260911/prepare_planb_panel.py",
        "experiments/nongeometric_screen/prepare_long_sources.py",
        "experiments/fixed_rope_three_interfaces_20260913/prepare_llama_ppl46.py",
        "experiments/fixed_rope_three_interfaces_20260913/prepare_olmo_ppl46.py",
        "experiments/fixed_rope_three_interfaces_20260913/tables.py",
        "experiments/olmo_recovery_20260912/recovery_v2_eval.py",
        "experiments/fixed_rope_three_interfaces_20260913/tailspline_llama_classic_report.py",
        "experiments/fixed_rope_three_interfaces_20260913/tailspline_olmo_classic_report.py",
    ]
    pending = [repo / name for name in roots]
    files: set[Path] = set()

    def enqueue(parts: list[str]) -> None:
        base = repo.joinpath(*parts)
        for candidate in [base.with_suffix('.py'), base / '__init__.py']:
            if candidate.is_file() and candidate not in files:
                pending.append(candidate)

    while pending:
        path = pending.pop()
        if path in files:
            continue
        files.add(path)
        parent = list(path.relative_to(repo).parts[:-1])
        for count in range(1, len(parent) + 1):
            init = repo.joinpath(*parent[:count], '__init__.py')
            if init.is_file() and init not in files:
                pending.append(init)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for item in node.names:
                    enqueue(item.name.split('.'))
            elif isinstance(node, ast.ImportFrom):
                prefix = parent[:len(parent) - node.level + 1] if node.level else []
                parts = prefix + (node.module.split('.') if node.module else [])
                enqueue(parts)
                for item in node.names:
                    if item.name != '*':
                        enqueue(parts + [item.name])
    return {'runtime/' + p.relative_to(repo).as_posix(): p.read_bytes() for p in sorted(files)}


def source_files() -> set[Path]:
    files: set[Path] = set()

    def add(path: Path) -> None:
        path = path.resolve()
        if not path.is_relative_to(PAPER) or not path.is_file():
            raise ValueError(f"Missing or external manuscript dependency: {path.name}")
        if path in files:
            return
        files.add(path)
        if path.suffix != ".tex":
            return
        content = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())
        for name in re.findall(r"\\(?:input|include)\{([^}]+)\}", content):
            add(PAPER / (name if Path(name).suffix else name + ".tex"))
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content):
            add(PAPER / "figs" / (name if Path(name).suffix else name + ".pdf"))
        for name in re.findall(r"\\bibliography\{([^}]+)\}", content):
            for part in name.split(","):
                add(PAPER / (part + ".bib"))

    add(PAPER / "main.tex")
    for name in ["main.pdf", "main.bbl", "compile.sh", "package_source.py", "SUPPLEMENT_README.md",
                 "runtime/README.md", "figs/fig_method_overview.svg",
                 "figs/allocation_design.py", "figs/make_allocation_value.py", "figs/allocation_value_inputs.json",
                 "figs/verify_interval_design.py", "figs/interval_development_inputs.json",
                 "figs/make_m4_tradeoff.py", "figs/m4_tradeoff_inputs.json", "figs/m4_tradeoff_points.csv",
                 "figs/verify_explicit_geometry.py", "figs/explicit_geometry_examples.json",
                 "figs/make_exponent_revision_figures.py", "figs/figure_inputs.json",
                 "figs/profile_diagnostic_inputs.json", "figs/verify_profile_diagnostics.py",
                 "figs/make_story_figures.py", "figs/story_figure_inputs.json",
                 "figs/finite_window_geometry.json", "figs/verify_recovered_assets.py",
                 "figs/recovered_asset_inputs.json", "figs/verify_routing_schedule.py",
                 "figs/routing_protocol_receipts.json", "figs/exponent_revision_source_receipt.json", "figs/llama_temporal_summary.json", "figs/recorded_runtime_identities.json"]:
        add(PAPER / name)
    for pattern in ["*.sty", "*.bst"]:
        for path in PAPER.glob(pattern):
            add(path)
    return files


def main() -> None:
    files = sorted(source_files())
    manifest = []
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(PAPER).as_posix()
            content = path.read_bytes()
            if path.suffix in {".tex", ".bib"} and re.search(rb"/Users/|/root/|sshpass", content):
                raise ValueError(f"Private path in manuscript source: {relative}")
            archive.writestr(relative, content)
            manifest.append(f"{hashlib.sha256(content).hexdigest()}  {relative}")
        runtime = runtime_sources()
        for relative, content in runtime.items():
            archive.writestr(relative, content)
            manifest.append(f"{hashlib.sha256(content).hexdigest()}  {relative}")
        archive.writestr("SHA256SUMS", "\n".join(manifest) + "\n")
        archive.writestr("README.txt", (
            "Beyond the Base: Exponent Allocation in RoPE\n\n"
            "This archive contains the complete active TeX source, bibliography,\n"
            "local style files, plotted figures, and compiled manuscript PDF.\n"
            "Unzip into an empty directory and run: bash compile.sh\n"
            "Requirements: pdflatex with bibtex, or Tectonic.\n\n"
            "The recorded-result figures and tables can be regenerated:\n"
            "  python3 figs/make_exponent_revision_figures.py\n"
            "  python3 figs/make_story_figures.py\n"
            "  python3 figs/make_m4_tradeoff.py\n"
            "  python3 figs/make_allocation_value.py\n"
            "  python3 figs/allocation_design.py\n"
            "This uses the bundled figs/figure_inputs.json, with original-source\n"
            "SHA256 values and the 778 natural-QA row scores (no prompt/output text).\n"
            "Python requirements: NumPy and Matplotlib. Regeneration performs no\n"
            "model execution. The remaining historical figures are supplied as PDF.\n\n"
            "Verify the two explicit finite-frequency examples:\n"
            "  python3 figs/verify_explicit_geometry.py\n"
            "  python3 figs/verify_interval_design.py\n"
            "  python3 figs/verify_profile_diagnostics.py\n"
            "  python3 figs/verify_recovered_assets.py\n"
            "  python3 figs/verify_routing_schedule.py\n\n"
            "The appendix contains the mathematical derivations and experimental\n"
            "protocols. Frozen experiment entrypoints and local imports are in\n"
            "runtime/. See runtime/README.md for execution instructions. Model\n"
            "checkpoints and raw experiment streams are maintained separately.\n"
        ))
    print(f"Packaged {len(files)} manuscript files and {len(runtime)} runtime files: {OUTPUT.name} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
