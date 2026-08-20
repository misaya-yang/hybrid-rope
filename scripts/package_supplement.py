#!/usr/bin/env python3
"""Build a curated anonymous reviewer supplement archive.

This intentionally does not archive the repository root.  It copies only the
paper source, public EVQ-Cosh library code, primary reproduction entrypoints,
tests, curated data snapshots, and public-facing docs, then scans the staged
tree for common identity and server-path leaks before writing a zip file.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = {
    "neurips2026": ROOT / "evq-cosh-neurips2026-supplement.zip",
    "iclr2027": ROOT / "rope-spectral-budget-iclr2027-supplement.zip",
}
STAGE = ROOT / "build_supplement"

NEURIPS2026_ALLOWLIST = [
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
    "scripts/core_text_phases/run_gqa_evq_experiment.py",
    "scripts/core_text_phases/eval_dsr.py",
    "scripts/core_text_phases/phase14c_multiscale_evq_yarn.py",
    "scripts/core_text_phases/phase16_formula_optimality_sweep.py",
    "scripts/core_text_phases/export_phase16_manifest.py",
    "scripts/core_text_phases/phase18_base_generalization_sweep.py",
    "scripts/core_text_phases/phase21b_quality_eval_clean.py",
    "scripts/figures",
    "scripts/supporting_eval/__init__.py",
    "scripts/supporting_eval/eval_passkey_scratch.py",
    "tests",
]

ICLR2027_ALLOWLIST = [
    "README.md",
    "requirements.txt",
    "requirements-lock.txt",
    "pytest.ini",
    ".github/workflows/smoke.yml",
    "docs/overview/REPRODUCE.md",
    "docs/overview/DATA_PREPARATION.md",
    "docs/overview/PAPER_CLAIMS_MAP.md",
    "data/curated",
    "paper-2027/main.tex",
    "paper-2027/iclr2027_conference.sty",
    "paper-2027/iclr2027_conference.bst",
    "paper-2027/natbib.sty",
    "paper-2027/fancyhdr.sty",
    "paper-2027/algorithm.sty",
    "paper-2027/algorithmic.sty",
    "paper-2027/math_commands.tex",
    "paper-2027/sections",
    "paper-2027/appendix",
    "paper-2027/tables",
    "paper-2027/refs",
    "paper-2027/figs/fig_method_overview.pdf",
    "paper-2027/figs/fig_identification.pdf",
    "paper-2027/figs/make_fig_method_overview.py",
    "paper-2027/figs/make_fig_identification.py",
    "scripts/__init__.py",
    "scripts/lib",
    "scripts/analysis/full_rope_collision_audit.py",
    "scripts/analysis/attention_fisher_50m_probe.py",
    "scripts/analysis/base_only_50m_control.py",
    "scripts/analysis/ruler_family_bootstrap.py",
    "scripts/core_text_phases/__init__.py",
    "scripts/core_text_phases/run_evq_sweep.py",
    "scripts/core_text_phases/eval_dsr.py",
    "scripts/core_text_phases/phase16_formula_optimality_sweep.py",
    "scripts/core_text_phases/phase16_exact_range_factorial_m4.py",
    "scripts/supporting_eval/__init__.py",
    "scripts/supporting_eval/eval_passkey_scratch.py",
    "tests/test_rope_core.py",
    "tests/test_fmrope_125m_l256_500m.py",
    "experiments/native_rope_evq_150m/model.py",
    "rebuttal/rebuttal_0723/experiments/geo_rope_contract.py",
    "rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/__init__.py",
    "rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/prepare.py",
    "rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/protocol.py",
    "rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/run_experiment.py",
    "rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json",
    "rebuttal/rebuttal_0723/theory_results/llama8b_matched_ruler_mix_20260726.json",
    "rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/metrics.json",
    "rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/ruler13_native_examples.jsonl",
    "rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/ruler13_evq_examples.jsonl",
]

PROFILES = {
    "neurips2026": NEURIPS2026_ALLOWLIST,
    "iclr2027": ICLR2027_ALLOWLIST,
}

EXCLUDE_NAMES = {
    "__pycache__",
    ".DS_Store",
    "test_paper_experiment_workspace.py",
    "test_rebuttal_evidence_bundle.py",
    "unused",
}

EXCLUDE_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".log",
    ".npz",
    ".out",
    ".pt",
    ".bin",
    ".pyc",
    ".safetensors",
    ".synctex.gz",
}

LEAK_PATTERNS = re.compile(
    rb"misaya|yanghej|hejaz|sshpass|seetacloud|connect\.bjb|connect\.west|"
    rb"@hejazfs|@privaterelay|/" + rb"Users/|/root/autodl-tmp|wandb\.ai|"
    rb"AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|"
    rb"ghp_[A-Za-z0-9]{20,}|BEGIN OPENSSH PRIVATE KEY|BEGIN RSA PRIVATE KEY",
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


def scan_for_trace_only(stage: Path) -> list[str]:
    """Reject quarantined evidence even if its filename changes."""
    hits: list[str] = []
    for path in sorted(stage.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("provenance_status") == "trace-only":
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
    parser.add_argument("--profile", choices=PROFILES, default="neurips2026")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-stage", action="store_true")
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUTS[args.profile]

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for rel in PROFILES[args.profile]:
        copy_item(rel, STAGE)

    hits = scan_for_leaks(STAGE)
    if hits:
        for hit in hits:
            print(f"LEAK-CHECK-FAIL {hit}")
        raise SystemExit("Supplement leak check failed; archive not written.")

    trace_only_hits = scan_for_trace_only(STAGE)
    if trace_only_hits:
        for hit in trace_only_hits:
            print(f"TRACE-ONLY-CHECK-FAIL {hit}")
        raise SystemExit("Trace-only evidence must not enter the reviewer supplement.")

    write_zip(STAGE, output)
    print(f"Wrote {output}")

    if not args.keep_stage:
        shutil.rmtree(STAGE)


if __name__ == "__main__":
    main()
