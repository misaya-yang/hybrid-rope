#!/usr/bin/env python3
"""Package the active manuscript and its exact TeX dependencies."""
from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER / "exponent-allocation-source.zip"


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
    for name in ["main.pdf", "main.bbl", "compile.sh", "package_source.py",
                 "figs/verify_explicit_geometry.py", "figs/explicit_geometry_examples.json",
                 "figs/make_exponent_revision_figures.py", "figs/figure_inputs.json",
                 "figs/exponent_revision_source_receipt.json", "figs/llama_temporal_summary.json", "figs/recorded_runtime_identities.json"]:
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
        archive.writestr("SHA256SUMS", "\n".join(manifest) + "\n")
        archive.writestr("README.txt", (
            "Beyond the Base: Exponent Allocation in RoPE\n\n"
            "This archive contains the complete active TeX source, bibliography,\n"
            "local style files, plotted figures, and compiled manuscript PDF.\n"
            "Unzip into an empty directory and run: bash compile.sh\n"
            "Requirements: a TeX installation with pdflatex and bibtex.\n\n"
            "Five revised figures and four result tables can be regenerated:\n"
            "  python3 figs/make_exponent_revision_figures.py\n"
            "This uses the bundled figs/figure_inputs.json, with original-source\n"
            "SHA256 values and the 778 natural-QA row scores (no prompt/output text).\n"
            "Python requirements: NumPy and Matplotlib. Regeneration performs no\n"
            "model execution. The remaining historical figures are supplied as PDF.\n\n"
            "Verify the two explicit finite-frequency examples:\n"
            "  python3 figs/verify_explicit_geometry.py\n\n"
            "The appendix contains the mathematical derivations and experimental\n"
            "protocols. This is a manuscript source archive; model checkpoints\n"
            "and raw experiment streams are maintained separately.\n"
        ))
    print(f"Packaged {len(files)} manuscript files: {OUTPUT.name} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
