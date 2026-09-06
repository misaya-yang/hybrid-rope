#!/usr/bin/env python3
"""Extended M4 budget/range/score matrix for target-free allocation.

The existing Phase16 runner remains the owner of model, data, checkpoint and
MPS mechanics.  This file only supplies the frozen spectral-score ablations and
the matrix/report wrapper; it does not modify the historical sweep scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import phase16_phase_isotropy_50m_m4 as base  # noqa: E402


RAW_ROOT = REPO_ROOT / "results/theory/phase_allocation_budget_matrix_m4_20260824"
FINAL_REPORT = (
    REPO_ROOT
    / "paper-2027/research/attention-aware-retrofit/results/"
    / "PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md"
)
FINAL_RECEIPT = (
    REPO_ROOT
    / "paper-2027/research/attention-aware-retrofit/evidence/"
    / "PHASE_ALLOCATION_M4_EXTENDED_RECEIPT_20260824.json"
)

STAGE_A = {
    "name": "stageA_base256_50m",
    "base": 256.0,
    "train_tokens": 50_331_648,
    "arms": ("FMRoPE", "anchored EVQ-Cosh", "phase-isotropy", "pair-volume", "min-eigenvalue"),
}
STAGE_B = {
    "name": "stageB_base500k_25m",
    "base": 500_000.0,
    "train_tokens": 25_165_824,
    "arms": ("FMRoPE", "anchored EVQ-Cosh", "phase-isotropy", "min-eigenvalue"),
}


@dataclass
class MatrixSpec(base.p.RunSpec):
    arm: str
    base_value: float
    stage_name: str


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def score_table(score: str, head_dim: int, train_length: int, omega_max: float, omega_min: float) -> torch.Tensor:
    """Endpoint-inclusive q^(1/3) quantiles for a fixed spectral score q."""
    if head_dim != base.HEAD_DIM or train_length != base.TRAIN_LENGTH:
        raise ValueError("matrix is frozen to d_head=64 and L_train=256")
    if score not in {"phase-isotropy", "pair-volume", "min-eigenvalue"}:
        raise ValueError(f"unknown score {score!r}")
    phi = torch.linspace(0.0, 1.0, 65_537, dtype=torch.float64)
    span = math.log(float(omega_max) / float(omega_min))
    omega = float(omega_max) * torch.exp(-span * phi)
    chi = base.causal_chi_closed_form(train_length, 2.0 * omega).abs().clamp(0.0, 1.0)
    lam_minus = (1.0 - chi) / 2.0
    lam_plus = (1.0 + chi) / 2.0
    if score == "phase-isotropy":
        q = (lam_minus / lam_plus).clamp_min(0.0)
    elif score == "pair-volume":
        q = (4.0 * lam_minus * lam_plus).clamp_min(0.0)
    else:
        q = lam_minus.clamp_min(0.0)
    density = q.pow(1.0 / 3.0)
    dx = 1.0 / float(phi.numel() - 1)
    cdf = torch.zeros_like(phi)
    cdf[1:] = torch.cumsum((density[:-1] + density[1:]) * (0.5 * dx), dim=0)
    if not bool(torch.isfinite(cdf).all()) or float(cdf[-1]) <= 0.0:
        raise RuntimeError(f"{score}: invalid density CDF")
    cdf = cdf / cdf[-1].clone()
    if not bool((torch.diff(cdf) > 0).all()):
        raise RuntimeError(f"{score}: CDF is not strictly increasing")
    quantiles = torch.linspace(0.0, 1.0, base.K, dtype=torch.float64)
    right = torch.searchsorted(cdf, quantiles, right=True).clamp(1, cdf.numel() - 1)
    left = right - 1
    frac = (quantiles - cdf[left]) / (cdf[right] - cdf[left])
    phi_nodes = phi[left] + frac * (phi[right] - phi[left])
    phi_nodes[0], phi_nodes[-1] = 0.0, 1.0
    out = float(omega_max) * torch.exp(-span * phi_nodes)
    out[0], out[-1] = float(omega_max), float(omega_min)
    return out.float().contiguous()


def build_table(arm: str, base_value: float) -> torch.Tensor:
    reference = base.std_geo_inv_freq(base.HEAD_DIM, base_value, dtype=torch.float64)
    if arm == "FMRoPE":
        out = reference.float()
    elif arm == "anchored EVQ-Cosh":
        phi = base.evq_cosh_phi(base.K, tau=base.COSH_TAU, midpoint=True, dtype=torch.float64)
        shape = (phi - phi[0]) / (phi[-1] - phi[0])
        span = math.log(float(reference[0]) / float(reference[-1]))
        out = (float(reference[0]) * torch.exp(-span * shape)).float()
    else:
        out = score_table(arm, base.HEAD_DIM, base.TRAIN_LENGTH, float(reference[0]), float(reference[-1]))
    out[0], out[-1] = reference[0].float(), reference[-1].float()
    if not bool(torch.isfinite(out).all()) or not bool(torch.all(out[:-1] > out[1:])):
        raise AssertionError(f"{arm}/base={base_value}: invalid frequency table")
    if not torch.equal(out[[0, -1]], reference.float()[[0, -1]]):
        raise AssertionError(f"{arm}/base={base_value}: endpoint mismatch")
    return out.contiguous()


def validate_tables(stages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "PASS", "tables": {}, "identity_max_abs_error": 0.0}
    d = torch.arange(base.TRAIN_LENGTH, dtype=torch.float64)
    weights = (base.TRAIN_LENGTH - d) / (base.TRAIN_LENGTH * (base.TRAIN_LENGTH + 1) / 2.0)
    for stage in stages:
        key = stage["name"]
        ref = build_table("FMRoPE", stage["base"])
        out["tables"][key] = {}
        for arm in stage["arms"]:
            table = build_table(arm, stage["base"])
            max_error = 0.0
            for omega in table.double():
                x = torch.stack((torch.cos(omega * d), torch.sin(omega * d)), dim=1)
                gram = (x * weights[:, None]).T @ x
                lhs = 4.0 * torch.linalg.det(gram)
                chi = (weights * torch.exp(torch.complex(torch.zeros_like(d), 2.0 * omega * d))).sum()
                max_error = max(max_error, float((lhs - (1.0 - chi.abs().square())).abs()))
            out["identity_max_abs_error"] = max(out["identity_max_abs_error"], max_error)
            out["tables"][key][arm] = {
                "base": stage["base"],
                "endpoint_exact": bool(torch.equal(table[[0, -1]], ref[[0, -1]])),
                "strict_monotone": bool(torch.all(table[:-1] > table[1:])),
                "float32_sha256": tensor_sha256(table),
                "normalized_nodes": (-torch.log(table.double() / table[0].double()) / math.log(float(table[0] / table[-1]))).tolist(),
            }
    if out["identity_max_abs_error"] > 2e-10:
        raise AssertionError(f"determinant identity failed: {out['identity_max_abs_error']}")
    return out


def build_specs(stage: dict[str, Any]) -> list[MatrixSpec]:
    return [
        MatrixSpec(
            stage="phase_allocation_budget_matrix_m4",
            run_id=f"{stage['name']}_{arm.replace(' ', '_').replace('-', '_')}_seed{base.SEED}",
            seq_len=base.TRAIN_LENGTH,
            num_heads=8,
            head_dim=base.HEAD_DIM,
            tau=0.0 if arm != "anchored EVQ-Cosh" else base.COSH_TAU,
            theory_tau=0.0,
            seed=base.SEED,
            tier="50m",
            train_tokens=stage["train_tokens"],
            eval_lengths=list(base.EVAL_LENGTHS),
            passkey_lengths=[],
            passkey_trials=0,
            eval_chunks=base.EVAL_CHUNKS,
            dsr_eval_length=0,
            dsr_distances=[],
            dsr_trials=0,
            passkey_mix_ratio=0.0,
            arm=arm,
            base_value=stage["base"],
            stage_name=stage["name"],
        )
        for arm in stage["arms"]
    ]


def build_context(root: Path, train_tokens: int) -> base.p.SweepContext:
    ctx = base.build_context(root, train_tokens)
    ctx.inv_freq_builder = lambda spec: build_table(spec.arm, spec.base_value)
    return ctx


def matched_init(ctx: base.p.SweepContext, specs: Iterable[MatrixSpec]) -> dict[str, str]:
    cfg = base.p.build_cfg(ctx.base_cfg, base.TRAIN_LENGTH, 8)
    hashes: dict[str, str] = {}
    for spec in specs:
        base.p.set_seed(base.SEED)
        model = base.p.GPT(cfg, build_table(spec.arm, spec.base_value))
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
        hashes[spec.arm] = digest.hexdigest()
        del model
    if len(set(hashes.values())) != 1:
        raise AssertionError(f"matched initialization failed: {hashes}")
    return hashes


def protocol(ctx: base.p.SweepContext, specs: list[MatrixSpec], anchors: dict[str, Any], init_hashes: dict[str, str], table_receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "device": base.p.DEVICE,
        "dtype": str(base.p.DTYPE),
        "stage": specs[0].stage_name,
        "base": specs[0].base_value,
        "seed": base.SEED,
        "train_tokens": specs[0].train_tokens,
        "model": {"tier": "50m", "hidden_size": 512, "num_layers": 6, "num_heads": 8, "head_dim": 64, "K": 32},
        "batch": {"global": 256, "micro": 32, "grad_accum": 8},
        "optimizer": "existing Phase16 M4 AdamW schedule; no result-driven changes",
        "data": {
            "dataset": "local_wikitext",
            "train_prefix_tokens": int(ctx.train_flat.numel()),
            "val_prefix_tokens": int(ctx.val_flat.numel()),
            "train_prefix_sha256": base.sha256_tokens(ctx.train_flat),
            "val_prefix_sha256": base.sha256_tokens(ctx.val_flat),
            "row_order": "deterministic randint(seed*1000003 + global_micro_step) over fixed sequential train rows",
        },
        "anchors": anchors,
        "matched_initialization_sha256": init_hashes,
        "tables": table_receipt["tables"][specs[0].stage_name],
    }


def load_stage_results(root: Path, specs: Iterable[MatrixSpec]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for spec in specs:
        path = root / "runs" / spec.run_id / "result.json"
        if path.exists():
            row = base.p.read_json(path, {})
            row["arm"] = spec.arm
            out[spec.arm] = row
    return out


def weighted(nll: dict[str, Any], field: str) -> float | None:
    vals, weights = [], []
    for length in (512, 1024, 2048):
        row = nll.get(str(length))
        if not row:
            return None
        vals.append(float(row[field]))
        weights.append(math.log2(length / base.TRAIN_LENGTH + 1.0))
    return float(np.average(vals, weights=weights))


def summarize_stage(root: Path, specs: list[MatrixSpec], table_receipt: dict[str, Any], protocol_row: dict[str, Any]) -> dict[str, Any]:
    rows = load_stage_results(root, specs)
    baseline = rows.get("FMRoPE", {}).get("nll", {})
    summary: dict[str, Any] = {"stage": specs[0].stage_name, "base": specs[0].base_value, "train_tokens": specs[0].train_tokens, "arms": {}, "paired": {}, "weighted_ood_tail": {}, "protocol": protocol_row, "construction": table_receipt}
    for arm, row in rows.items():
        summary["arms"][arm] = row.get("nll", {})
        summary["weighted_ood_tail"][arm] = weighted(row.get("nll", {}), "tail_nll")
    for arm, row in summary["arms"].items():
        if arm == "FMRoPE":
            continue
        summary["paired"][arm] = {}
        for length in base.EVAL_LENGTHS:
            a, b = row.get(str(length), {}), baseline.get(str(length), {})
            summary["paired"][arm][str(length)] = {
                "tail_minus_FMRoPE": float(a["tail_nll"] - b["tail_nll"]) if a and b else None,
                "full_minus_FMRoPE": float(a["full_nll"] - b["full_nll"]) if a and b else None,
            }
    base_weight = summary["weighted_ood_tail"].get("FMRoPE")
    summary["weighted_gain_vs_FMRoPE"] = {
        arm: (base_weight - value if base_weight is not None and value is not None else None)
        for arm, value in summary["weighted_ood_tail"].items()
    }
    return summary


def write_report(stage_summaries: list[dict[str, Any]], construction: dict[str, Any]) -> dict[str, Any]:
    report = {"status": "COMPLETE_INTERNAL_PRELIMINARY" if all(s["arms"].keys() >= {"FMRoPE"} for s in stage_summaries) else "IN_PROGRESS", "canonical_verdict": "SCREEN_UNRESOLVED", "construction": construction, "stages": stage_summaries, "scope": "internal preliminary method screening; not manuscript evidence"}
    lines = ["# Extended target-free allocation M4 matrix (2026-08-24)", "", f"- Status: **{report['status']}**", f"- Canonical verdict: **{report['canonical_verdict']}**", "- Scope: internal preliminary research only; no manuscript claim.", "", "## Stage summaries", ""]
    for stage in stage_summaries:
        baseline = "FMRoPE" if stage["base"] == 256.0 else "Geo (raw key `FMRoPE`)"
        if stage["base"] != 256.0:
            lines += [f"### {stage['stage']} (base={stage['base']}, tokens/arm={stage['train_tokens']})", "", "The baseline is scientifically Geo; `FMRoPE` is retained only as the raw runner key.", ""]
        else:
            lines += [f"### {stage['stage']} (base={stage['base']}, tokens/arm={stage['train_tokens']})", ""]
        lines += [f"| Arm | weighted OOD tail NLL | gain vs {baseline} |", "|---|---:|---:|"]
        for arm, value in stage["weighted_ood_tail"].items():
            gain = stage["weighted_gain_vs_FMRoPE"].get(arm)
            display_arm = baseline if arm == "FMRoPE" else arm
            lines.append(f"| {display_arm} | {value} | {gain} |")
        lines += ["", "```json", json.dumps(stage["paired"], indent=2), "```", ""]
    lines += ["## Construction receipt", "", "```json", json.dumps(construction, indent=2), "```", "", "No candidate is promoted from this matrix without a separately frozen multi-seed protocol.", ""]
    FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return report


def write_receipt(report: dict[str, Any]) -> None:
    receipt = {
        "receipt_version": 1,
        "experiment": "phase_allocation_budget_matrix_m4_20260824",
        "status": report["status"],
        "canonical_verdict": report["canonical_verdict"],
        "scientific_identity_corrections": {
            "stageB_base500k_25m": {
                "raw_arm_key": "FMRoPE",
                "scientific_name": "Geo",
            }
        },
        "raw_root": "results/theory/phase_allocation_budget_matrix_m4_20260824/",
        "scope": report["scope"],
        "stages": report["stages"],
        "construction": report["construction"],
    }
    FINAL_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def run_stage(stage: dict[str, Any], construction: dict[str, Any]) -> dict[str, Any]:
    root = RAW_ROOT / stage["name"]
    root.mkdir(parents=True, exist_ok=True)
    specs = build_specs(stage)
    ctx = build_context(root, stage["train_tokens"])
    anchors = base.build_anchors(ctx.val_flat)
    init_hashes = matched_init(ctx, specs)
    table_receipt = {"tables": {stage["name"]: construction["tables"][stage["name"]]}, "identity_max_abs_error": construction["identity_max_abs_error"]}
    protocol_row = protocol(ctx, specs, anchors, init_hashes, table_receipt)
    base.p.atomic_write_json(root / "protocol_match.json", protocol_row)
    base.p.probe_micro_batch = base.safe_batch_plan
    base.p.safe_eval_ppl = base.fixed_anchor_nll
    base.p.safe_eval_passkey = lambda **_kwargs: {"details": {}, "summary": {}, "global": {}}
    base.p.safe_eval_dsr = lambda **_kwargs: (None, None)
    with base.p.SweepLock(root / "sweep.lock"):
        for spec in specs:
            try:
                base.run_one(ctx, spec, anchors)
            except Exception as exc:
                base.p.atomic_write_json(root / "runs" / spec.run_id / "failure.json", {"arm": spec.arm, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[anomaly] {stage['name']} {spec.arm}: {exc}", flush=True)
    summary = summarize_stage(root, specs, table_receipt, protocol_row)
    base.p.atomic_write_json(root / "reports" / "summary.json", summary)
    print(f"[stage-complete] {stage['name']}: {summary['weighted_gain_vs_FMRoPE']}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("A", "B", "all"), default="all")
    parser.add_argument("--mode", choices=("preflight", "run"), default="run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if base.p.DEVICE != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError(f"MPS is required; detected device={base.p.DEVICE}")
    stages = [STAGE_A] if args.stage == "A" else [STAGE_B] if args.stage == "B" else [STAGE_A, STAGE_B]
    construction = validate_tables(stages)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    base.p.atomic_write_json(RAW_ROOT / "construction_receipt.json", construction)
    if args.mode == "preflight":
        print(json.dumps(construction, indent=2))
        return
    summaries = []
    for stage in stages:
        summaries.append(run_stage(stage, construction))
        report = write_report(summaries, construction)
        write_receipt(report)
    print(f"[done] report={FINAL_REPORT}", flush=True)


if __name__ == "__main__":
    main()
