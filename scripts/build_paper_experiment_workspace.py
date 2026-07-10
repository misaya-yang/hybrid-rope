#!/usr/bin/env python3
"""Build the tracked, manifest-driven paper experiment code workspace."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "paper_experiments"
CODE_DIR = WORKSPACE / "code"
MANIFEST_PATH = WORKSPACE / "MANIFEST.json"


FAMILIES = {
    "shared_rope_and_training": [
        "scripts/__init__.py",
        "scripts/train.py",
        "scripts/lib/__init__.py",
        "scripts/lib/rope/__init__.py",
        "scripts/lib/rope/attn_hist.py",
        "scripts/lib/rope/inject.py",
        "scripts/lib/rope/learnable_evq.py",
        "scripts/lib/rope/schedules.py",
        "scripts/core_text_phases/__init__.py",
        "scripts/core_text_phases/run_evq_sweep.py",
    ],
    "primary_i_evq_yarn": [
        "scripts/core_text_phases/phase14c_multiscale_evq_yarn.py",
        "scripts/supporting_eval/__init__.py",
        "scripts/supporting_eval/eval_passkey_scratch.py",
    ],
    "primary_ii_pe_dominant": [
        "scripts/core_text_phases/eval_pe_baselines.py",
        "scripts/core_text_phases/phase11_L256_extrap.py",
        "scripts/core_text_phases/phase11_yarn_eval.py",
        "scripts/core_text_phases/phase11b_125m_dape.py",
        "scripts/core_text_phases/phase11c_454m_scaling.py",
    ],
    "primary_iii_mla": [
        "scripts/core_text_phases/audit_rope_checkpoint.py",
        "scripts/core_text_phases/eval_extended_3seeds.py",
        "scripts/core_text_phases/eval_super_extrap.py",
        "scripts/core_text_phases/gqa_patch.py",
        "scripts/core_text_phases/make_artifact_manifest.py",
        "scripts/core_text_phases/mla_patch.py",
        "scripts/core_text_phases/run_gqa_evq_experiment.py",
        "scripts/core_text_phases/yarn_finetune_eval.py",
    ],
    "theory_and_mechanism": [
        "scripts/core_text_phases/eval_dsr.py",
        "scripts/core_text_phases/evq_analysis.py",
        "scripts/core_text_phases/export_phase16_manifest.py",
        "scripts/core_text_phases/phase16_formula_optimality_sweep.py",
        "scripts/analysis/compute_eta_vp.py",
        "scripts/analysis/exp_tau_theory_verify.py",
        "scripts/analysis/tau_direct_optimization.py",
        "scripts/analysis/tau_exact_derivation.py",
        "scripts/analysis/tau_position_discrimination.py",
        "scripts/analysis/tau_scaling_analysis.py",
        "scripts/analysis/verify_c_coll.py",
        "scripts/analysis/verify_softmax_transport.py",
        "scripts/analysis/verify_stiffness_and_regime.py",
        "scripts/m4_max_36gb/theory_numerical_verification.py",
        "scripts/theory_B_floor_higher_order.py",
        "scripts/verify_tau_unified.py",
    ],
    "supporting_text": [
        "scripts/core_text_phases/eval_longbench_nll.py",
        "scripts/core_text_phases/phase8d_scaling_law.py",
        "scripts/core_text_phases/phase8f_multi_seed.py",
        "scripts/core_text_phases/phase15_750m_2k_to_4k_continue_ckpt_eval.py",
        "scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py",
        "scripts/core_text_phases/phase17c_extended_eval.py",
        "scripts/core_text_phases/phase17c_resume_evq.py",
        "scripts/core_text_phases/phase18_base_generalization_sweep.py",
        "scripts/core_text_phases/phase21b_quality_eval_clean.py",
        "scripts/core_text_phases/visualize_attention_distance.py",
        "scripts/supporting_eval/eval_longbench.py",
        "scripts/supporting_eval/eval_multi_needle.py",
        "scripts/supporting_eval/eval_niah_heatmap.py",
        "scripts/supporting_eval/eval_niah_recall.py",
        "scripts/text_eval/eval_454m_multilength.py",
        "scripts/text_eval/llama3_continued_pretrain.py",
    ],
    "supporting_video_dit": [
        "scripts/video_temporal/eval_32f_indist.py",
        "scripts/video_temporal/eval_perframe_accuracy.py",
        "scripts/video_temporal/eval_riflex.py",
        "scripts/video_temporal/eval_temporal_precision.py",
        "scripts/video_temporal/run_dit_temporal.py",
        "scripts/video_temporal/run_perframe_accuracy.py",
        "scripts/video_temporal/run_phase23_fvd_verify.py",
        "scripts/video_temporal/run_video_temporal.py",
        "scripts/video_temporal/run_video_temporal_allocation_sweep.py",
        "scripts/video_temporal/theory_predicts_experiment.py",
        "scripts/video_temporal/theory_predicts_v2.py",
        "scripts/video_temporal/video_dit.py",
    ],
    "supporting_lora_8b": [
        "experiments/lora_evq_v2/compare_results.py",
        "experiments/lora_evq_v2/download_model_data.py",
        "experiments/lora_evq_v2/dryrun_validate.py",
        "experiments/lora_evq_v2/eval_evq_lora.py",
        "experiments/lora_evq_v2/eval_positional_ppl.py",
        "experiments/lora_evq_v2/eval_ruler.py",
        "experiments/lora_evq_v2/run.sh",
        "experiments/lora_evq_v2/test_evq_yarn.py",
        "experiments/lora_evq_v2/train_evq_lora.py",
        "experiments/lora_evq_v2/validate_checkpoint_artifact.py",
    ],
    "data_preparation": [
        "scripts/data_prep/prepare_8k_mixed_500m.py",
        "scripts/data_prep/prepare_longbench_local_data.py",
        "scripts/data_prep/prepare_moving_mnist_video.py",
        "scripts/data_prep/prepare_oscillating_mnist_video.py",
        "scripts/data_prep/tokenize_synth.py",
        "scripts/text_eval/prepare_training_data.py",
    ],
    "paper_figures": [
        "scripts/figures/build_fig5_downstream_qa.sh",
        "scripts/figures/build_fig6_tau_rank.sh",
        "scripts/figures/fig0_main_story.py",
        "scripts/figures/fig1_neurips.py",
        "scripts/figures/fig2_evq_yarn_orthogonality.py",
        "scripts/figures/fig3_pe_dominant_scaling.py",
        "scripts/figures/fig5_downstream_qa_nll.tex",
        "scripts/figures/fig6_tau_rank_readable.tex",
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reset_code_tree() -> None:
    if not CODE_DIR.exists():
        CODE_DIR.mkdir(parents=True)
        return
    unexpected = [path for path in CODE_DIR.rglob("*") if path.is_file() and not path.is_symlink()]
    if unexpected:
        raise RuntimeError(f"refusing to remove non-link workspace file: {unexpected[0]}")
    for path in sorted(CODE_DIR.rglob("*"), reverse=True):
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def build() -> dict:
    reset_code_tree()
    memberships: dict[str, list[str]] = {}
    for family, sources in FAMILIES.items():
        for source in sources:
            memberships.setdefault(source, []).append(family)

    files = []
    for source in sorted(memberships):
        source_path = ROOT / source
        if not source_path.is_file():
            raise FileNotFoundError(source)
        link_path = CODE_DIR / source
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(os.path.relpath(source_path, link_path.parent))
        files.append(
            {
                "source": source,
                "workspace_path": str(link_path.relative_to(WORKSPACE)),
                "families": sorted(memberships[source]),
                "sha256": sha256(source_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "layout": "repo-relative symbolic links to canonical tracked sources",
        "families": {name: sorted(paths) for name, paths in FAMILIES.items()},
        "files": files,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    result = build()
    print(f"paper experiment workspace: {len(result['files'])} files")
