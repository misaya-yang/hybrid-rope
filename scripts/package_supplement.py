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
    "requirements.txt",
    "requirements-lock.txt",
    "pytest.ini",
    "data/curated",
    "paper-2027/main.tex",
    "paper-2027/compile.sh",
    "paper-2027/build.mk",
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
    "paper-2027/figs/fig_8b_causal_source_use.pdf",
    "paper-2027/figs/fig_evidence_overview.pdf",
    "paper-2027/figs/fig_frozen_fixed_support.pdf",
    "paper-2027/figs/fig_frequency_geometry.pdf",
    "paper-2027/figs/fig_exact_range_control.pdf",
    "paper-2027/figs/fig_olmo_scale_crossover.pdf",
    "paper-2027/figs/fig_spectral_budget_scaling.pdf",
    "paper-2027/figs/make_fig_8b_causal_source_use.py",
    "paper-2027/figs/make_fig_evidence_overview.py",
    "paper-2027/figs/make_fig_frozen_fixed_support.py",
    "paper-2027/figs/make_fig_frequency_geometry.py",
    "paper-2027/figs/make_fig_exact_range_control.py",
    "paper-2027/figs/make_fig_olmo_scale_crossover.py",
    "paper-2027/figs/make_fig_spectral_budget_scaling.py",
    "scripts/__init__.py",
    "scripts/lib",
    "analysis/full_rope_audit/verify_core.py",
    "analysis/full_rope_audit/verify_small_models.py",
    "scripts/analysis/full_rope_collision_audit.py",
    "scripts/analysis/third_axis_ceiling.py",
    "scripts/analysis/attention_fisher_50m_probe.py",
    "scripts/analysis/base_only_50m_control.py",
    "scripts/analysis/export_uniqueness_budgeted_tables.py",
    "scripts/analysis/rope_transport/__init__.py",
    "scripts/analysis/rope_transport/same_support_controls.py",
    "scripts/analysis/ruler_family_bootstrap.py",
    "scripts/core_text_phases/__init__.py",
    "scripts/core_text_phases/run_evq_sweep.py",
    "scripts/core_text_phases/phase16_formula_optimality_sweep.py",
    "scripts/core_text_phases/phase16_exact_range_factorial_m4.py",
    "scripts/supporting_eval/__init__.py",
    "scripts/supporting_eval/eval_passkey_scratch.py",
    "tests/test_rope_core.py",
    "tests/test_fmrope_125m_l256_500m.py",
    "tests/test_same_support_rope_controls.py",
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

PROFILE_EXCLUDE_NAMES = {
    "iclr2027": {
        "a4_supporting_experiments.tex",
        "table_evq_ramp.tex",
        "primary1_evq_yarn_10pct_raw.json",
        "mla_channel_count_125m_pilot.json",
        "table2_evq_yarn_454m_passkey_10pct.json",
        "phase11b_125m_l256_3seed.json",
        "quality_454m_full_eval.json",
    },
}

PROFILE_RENAMED_FILES = {
    "iclr2027": {
        "paper-2027/SUPPLEMENT_README.md": "README.md",
        "paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json":
            "data/curated/exact_range_151m_3seed_result.json",
    },
}
# Backward-compatible name used by existing supplement contract tests.
ALLOWLIST = NEURIPS2026_ALLOWLIST

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


def should_skip(path: Path, extra_exclude_names: set[str] | frozenset[str] = frozenset()) -> bool:
    if path.name in EXCLUDE_NAMES or path.name in extra_exclude_names:
        return True
    text = path.name
    return any(text.endswith(suffix) for suffix in EXCLUDE_SUFFIXES)


def copy_item(
    rel: str,
    stage: Path,
    extra_exclude_names: set[str] | frozenset[str] = frozenset(),
) -> None:
    src = ROOT / rel
    if not src.exists():
        raise FileNotFoundError(f"allowlisted path does not exist: {rel}")
    dst = stage / rel
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            ignore=lambda _d, names: [
                name
                for name in names
                if should_skip(Path(name), extra_exclude_names)
            ],
        )
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_renamed_file(src_rel: str, dst_rel: str, stage: Path) -> None:
    src = ROOT / src_rel
    if not src.is_file():
        raise FileNotFoundError(f"renamed supplement file does not exist: {src_rel}")
    dst = stage / dst_rel
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

    profile_excludes = PROFILE_EXCLUDE_NAMES.get(args.profile, frozenset())
    for rel in PROFILES[args.profile]:
        copy_item(rel, STAGE, profile_excludes)
    for src_rel, dst_rel in PROFILE_RENAMED_FILES.get(args.profile, {}).items():
        copy_renamed_file(src_rel, dst_rel, STAGE)

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
