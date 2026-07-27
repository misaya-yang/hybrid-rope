#!/usr/bin/env python3
"""Exact-range pure-shape factorial using the existing Phase16 M4 harness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

import phase16_formula_optimality_sweep as p
from scripts.lib.rope.schedules import evq_cosh_phi


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "results"
    / "theory"
    / "phase16_exact_range_factorial_m4_20260724"
)
ARMS = (
    "native_geo",
    "anchored_cosh_m075",
    "anchored_cosh_rule",
    "anchored_cosh_m125",
    "anchored_exp_rule",
)
BASES = (500_000.0, 1_000_000.0)
SEQ_LENS = (256, 1024)
HEAD_COUNTS = (4, 8, 16)
SEEDS = (42, 137, 256)
EXTREME_ARMS = ("anchored_cosh_m050", "anchored_cosh_m150")


@dataclass
class ExactRangeSpec(p.RunSpec):
    rope_base: float
    arm: str
    tau_multiplier: float | None
    schedule_family: str
    save_milestones: bool = False


def normalized_native_grid(head_dim: int) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, head_dim // 2, dtype=torch.float64)


def normalized_cosh_grid(head_dim: int, tau: float) -> torch.Tensor:
    phi = evq_cosh_phi(
        head_dim // 2, tau=tau, midpoint=True, dtype=torch.float64
    )
    return (phi - phi[0]) / (phi[-1] - phi[0])


def matched_exponential_grid(target: torch.Tensor) -> tuple[torch.Tensor, float]:
    uniform = normalized_native_grid(target.numel() * 2)
    target_rms = torch.sqrt(torch.mean((target - uniform).square())).item()
    t = uniform

    def candidate(parameter: float) -> torch.Tensor:
        if parameter < 1e-10:
            return t
        return torch.expm1(parameter * t) / math.expm1(parameter)

    low, high = 0.0, 128.0
    if torch.sqrt(torch.mean((candidate(high) - uniform).square())).item() < target_rms:
        raise RuntimeError("exponential family cannot match Cosh deformation")
    for _ in range(80):
        middle = 0.5 * (low + high)
        rms = torch.sqrt(
            torch.mean((candidate(middle) - uniform).square())
        ).item()
        if rms < target_rms:
            low = middle
        else:
            high = middle
    parameter = 0.5 * (low + high)
    return candidate(parameter), parameter


def schedule_grid(spec: ExactRangeSpec) -> tuple[torch.Tensor, dict[str, Any]]:
    uniform = normalized_native_grid(spec.head_dim)
    if spec.arm == "native_geo":
        return uniform, {"family": "uniform", "tau": 0.0}

    tau = spec.theory_tau * float(spec.tau_multiplier or 1.0)
    cosh = normalized_cosh_grid(spec.head_dim, tau)
    if spec.arm == "anchored_exp_rule":
        grid, parameter = matched_exponential_grid(cosh)
        return grid, {
            "family": "exponential",
            "matched_cosh_tau": tau,
            "shape_parameter": parameter,
        }
    return cosh, {"family": "cosh", "tau": tau}


def build_inv_freq(spec: ExactRangeSpec) -> torch.Tensor:
    grid, _ = schedule_grid(spec)
    channels = spec.head_dim // 2
    log_span = (channels - 1) / channels * math.log(spec.rope_base)
    return torch.exp(-log_span * grid).float()


def build_specs(train_tokens: int, eval_chunks: int) -> list[ExactRangeSpec]:
    specs: list[ExactRangeSpec] = []
    hidden_size = int(p.TIER_CONFIGS["50m"]["hidden_size"])
    multipliers = {
        "native_geo": None,
        "anchored_cosh_m075": 0.75,
        "anchored_cosh_rule": 1.0,
        "anchored_cosh_m125": 1.25,
        "anchored_exp_rule": 1.0,
    }
    for base in BASES:
        base_tag = f"B{int(base // 1000)}K"
        for seq_len in SEQ_LENS:
            for num_heads in HEAD_COUNTS:
                head_dim = hidden_size // num_heads
                theory_tau = p.predicted_tau(head_dim, seq_len)
                for seed in SEEDS:
                    for arm in ARMS:
                        multiplier = multipliers[arm]
                        tau = 0.0 if multiplier is None else theory_tau * multiplier
                        specs.append(
                            ExactRangeSpec(
                                stage="exact_range_factorial",
                                run_id=(
                                    f"{base_tag}_L{seq_len}_H{num_heads}_Dh{head_dim}"
                                    f"_{arm}_seed{seed}"
                                ),
                                seq_len=seq_len,
                                num_heads=num_heads,
                                head_dim=head_dim,
                                tau=round(tau, 6),
                                theory_tau=round(theory_tau, 6),
                                seed=seed,
                                tier="50m",
                                train_tokens=train_tokens,
                                eval_lengths=p.extrapolation_lengths(seq_len),
                                passkey_lengths=[],
                                passkey_trials=0,
                                eval_chunks=eval_chunks,
                                dsr_eval_length=0,
                                dsr_distances=[],
                                dsr_trials=0,
                                passkey_mix_ratio=0.01,
                                rope_base=base,
                                arm=arm,
                                tau_multiplier=multiplier,
                                schedule_family=(
                                    "uniform"
                                    if arm == "native_geo"
                                    else "exponential"
                                    if arm == "anchored_exp_rule"
                                    else "cosh"
                                ),
                                save_milestones=arm
                                in {
                                    "native_geo",
                                    "anchored_cosh_rule",
                                    "anchored_exp_rule",
                                },
                            )
                        )
    return specs


def build_extreme_specs(train_tokens: int, eval_chunks: int) -> list[ExactRangeSpec]:
    """Two canonical-base extrema: min tau=(L1024,Dh32), max tau=(L256,Dh128)."""
    specs: list[ExactRangeSpec] = []
    for seq_len, num_heads in ((1024, 16), (256, 4)):
        head_dim = 512 // num_heads
        theory_tau = p.predicted_tau(head_dim, seq_len)
        for seed in SEEDS:
            for arm, multiplier in zip(EXTREME_ARMS, (0.5, 1.5), strict=True):
                specs.append(
                    ExactRangeSpec(
                        stage="exact_range_extreme",
                        run_id=(
                            f"B500K_L{seq_len}_H{num_heads}_Dh{head_dim}"
                            f"_{arm}_seed{seed}"
                        ),
                        seq_len=seq_len,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        tau=round(theory_tau * multiplier, 6),
                        theory_tau=round(theory_tau, 6),
                        seed=seed,
                        tier="50m",
                        train_tokens=train_tokens,
                        eval_lengths=p.extrapolation_lengths(seq_len),
                        passkey_lengths=[],
                        passkey_trials=0,
                        eval_chunks=eval_chunks,
                        dsr_eval_length=0,
                        dsr_distances=[],
                        dsr_trials=0,
                        passkey_mix_ratio=0.01,
                        rope_base=500_000.0,
                        arm=arm,
                        tau_multiplier=multiplier,
                        schedule_family="cosh",
                    )
                )
    return specs


def sha256_trainable(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_schedules(specs: Iterable[ExactRangeSpec]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    grouped: dict[tuple[float, int, int], list[ExactRangeSpec]] = defaultdict(list)
    for spec in specs:
        grouped[(spec.rope_base, spec.seq_len, spec.head_dim)].append(spec)
    for key, rows in grouped.items():
        unique = {row.arm: row for row in rows}
        invs = {arm: build_inv_freq(unique[arm]).double() for arm in ARMS}
        reference = invs["native_geo"]
        for arm, inv in invs.items():
            if not torch.all(inv[:-1] > inv[1:]):
                raise AssertionError(f"{key}/{arm}: inv_freq is not strictly decreasing")
            if not torch.equal(inv[[0, -1]], reference[[0, -1]]):
                raise AssertionError(f"{key}/{arm}: exact endpoint match failed")
        row = unique["anchored_cosh_rule"]
        grids = {arm: schedule_grid(unique[arm])[0] for arm in ARMS}
        _, exp_meta = schedule_grid(unique["anchored_exp_rule"])
        uniform = grids["native_geo"]
        target_rms = torch.sqrt(
            torch.mean((grids["anchored_cosh_rule"] - uniform).square())
        ).item()
        achieved_rms = torch.sqrt(
            torch.mean((grids["anchored_exp_rule"] - uniform).square())
        ).item()
        receipts.append(
            {
                "rope_base": key[0],
                "seq_len": key[1],
                "head_dim": key[2],
                "rule_tau": row.theory_tau,
                "endpoint_max": float(reference[0]),
                "endpoint_min": float(reference[-1]),
                "log_span": float(torch.log(reference[0] / reference[-1])),
                "exponential_parameter": exp_meta["shape_parameter"],
                "deformation_definition": (
                    "RMS normalized log-frequency node displacement from "
                    "uniform, including the two zero-displacement endpoints"
                ),
                "cosh_target_deformation_rms": target_rms,
                "exponential_achieved_deformation_rms": achieved_rms,
                "absolute_matching_error": abs(achieved_rms - target_rms),
                "normalized_log_frequency_nodes": {
                    arm: grid.tolist() for arm, grid in grids.items()
                },
                "inv_freq_vectors": {
                    arm: inv.tolist() for arm, inv in invs.items()
                },
                "inv_freq_hashes": {
                    arm: p.sha256_tensor(inv.float()) for arm, inv in invs.items()
                },
            }
        )
    return receipts


def build_context(args: argparse.Namespace) -> p.SweepContext:
    profile = p.SweepProfile(
        name="exact_range_factorial_m4",
        tier="50m",
        seq_lens=list(SEQ_LENS),
        head_counts=list(HEAD_COUNTS),
        seeds=list(SEEDS),
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        tau_multipliers=[0.75, 1.0, 1.25],
        passkey_mix_ratio=0.01,
        eval_chunks=args.eval_chunks,
        passkey_trials_pilot=0,
        passkey_trials_confirm=0,
        dsr_trials=0,
        top_k=1,
    )
    context_args = SimpleNamespace(
        work_root=str(args.work_root),
        dataset="local_wikitext",
    )
    ctx = p.build_sweep_context(context_args, profile)
    ctx.inv_freq_builder = build_inv_freq
    return ctx


def write_plan(
    root: Path, specs: list[ExactRangeSpec], receipts: list[dict[str, Any]]
) -> None:
    p.atomic_write_json(root / "plan.json", [asdict(spec) for spec in specs])
    p.atomic_write_json(root / "schedule_receipts.json", receipts)


def assert_matched_initialization(ctx: p.SweepContext, specs: list[ExactRangeSpec]) -> None:
    rows = [row for row in specs if row.seed == SEEDS[0]][: len(ARMS)]
    hashes = []
    for spec in rows:
        p.set_seed(spec.seed)
        cfg = p.build_cfg(ctx.base_cfg, spec.seq_len, spec.num_heads)
        model = p.GPT(cfg, build_inv_freq(spec))
        hashes.append(sha256_trainable(model))
        del model
    if len(set(hashes)) != 1:
        raise AssertionError("matched initialization failed across schedule arms")


def disable_auxiliary_eval() -> None:
    p.safe_eval_passkey = lambda **_kwargs: {"details": {}, "summary": {}, "global": {}}
    p.safe_eval_dsr = lambda **_kwargs: (None, None)


def safe_m4_batch_plan(
    _ctx: p.SweepContext,
    cfg: dict[str, Any],
    seq_len: int,
    num_heads: int,
    target_effective_batch: int,
) -> p.BatchPlan:
    safe_micro = {
        (256, 4): 32,
        (256, 8): 32,
        (256, 16): 32,
        (1024, 4): 4,
        (1024, 8): 2,
        (1024, 16): 2,
    }[(seq_len, num_heads)]
    grad_accum = math.ceil(target_effective_batch / safe_micro)
    return p.BatchPlan(
        config_key=p.config_key(seq_len, num_heads, int(cfg["head_dim"])),
        probed_micro_batch=safe_micro,
        target_effective_batch=target_effective_batch,
        micro_batch_size=safe_micro,
        grad_accum=grad_accum,
        effective_batch_size=safe_micro * grad_accum,
        tokens_per_step=safe_micro * grad_accum * seq_len,
        probe_candidates=[safe_micro],
    )


def run_specs(ctx: p.SweepContext, specs: list[ExactRangeSpec]) -> None:
    disable_auxiliary_eval()
    p.probe_micro_batch = safe_m4_batch_plan
    total = len(specs)
    for index, spec in enumerate(specs, 1):
        result_path = p.result_path_for(ctx, spec)
        if result_path.exists():
            print(f"[skip] {index}/{total} {spec.run_id}", flush=True)
            continue
        print(f"[run] {index}/{total} {spec.run_id}", flush=True)
        p.execute_run(ctx, spec)
        if p.STOP_REQUESTED:
            raise p.SweepError("stop requested")


def weighted_extrapolation_nll(row: dict[str, Any]) -> float | None:
    train_len = int(row["seq_len"])
    weighted, total = 0.0, 0.0
    for length_text, ppl in row.get("ppl", {}).items():
        length = int(length_text)
        if length <= train_len:
            continue
        weight = math.log2(length / train_len + 1.0)
        weighted += weight * math.log(float(ppl))
        total += weight
    return weighted / total if total else None


def old_phase16_summary() -> dict[str, Any]:
    path = REPO_ROOT / "data" / "curated" / "phase16_99run_manifest.csv"
    if not path.exists():
        return {"available": False}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {
        "available": True,
        "path": str(path.relative_to(REPO_ROOT)),
        "runs": len(rows),
        "comparison_boundary": (
            "descriptive only: the historical midpoint-Cosh sweep did not "
            "hold sampled endpoints/span fixed"
        ),
    }


def generate_report(root: Path, specs: list[ExactRangeSpec]) -> dict[str, Any]:
    completed = []
    for spec in specs:
        result = p.read_json(root / "runs" / spec.run_id / "result.json", {})
        if result:
            result["arm"] = spec.arm
            result["rope_base"] = spec.rope_base
            result["weighted_extrapolation_nll"] = weighted_extrapolation_nll(result)
            completed.append(result)

    paired: list[dict[str, Any]] = []
    groups: dict[tuple[float, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in completed:
        key = (
            float(row["rope_base"]),
            int(row["seq_len"]),
            int(row["head_dim"]),
            int(row["seed"]),
        )
        groups[key][row["arm"]] = row
    for key, arms in sorted(groups.items()):
        if not all(arm in arms for arm in ARMS):
            continue
        geo = arms["native_geo"]["weighted_extrapolation_nll"]
        cosh_values = {
            arm: arms[arm]["weighted_extrapolation_nll"]
            for arm in (
                "anchored_cosh_m075",
                "anchored_cosh_rule",
                "anchored_cosh_m125",
            )
        }
        best_cosh_arm = min(cosh_values, key=cosh_values.get)
        paired.append(
            {
                "rope_base": key[0],
                "seq_len": key[1],
                "head_dim": key[2],
                "seed": key[3],
                "rule_minus_geo_nll": cosh_values["anchored_cosh_rule"] - geo,
                "exp_minus_geo_nll": (
                    arms["anchored_exp_rule"]["weighted_extrapolation_nll"] - geo
                ),
                "cosh_rule_minus_exp_nll": (
                    cosh_values["anchored_cosh_rule"]
                    - arms["anchored_exp_rule"]["weighted_extrapolation_nll"]
                ),
                "cosh_rule_regret": (
                    cosh_values["anchored_cosh_rule"] - cosh_values[best_cosh_arm]
                ),
                "best_cosh_arm": best_cosh_arm,
            }
        )

    def mean(field: str) -> float | None:
        values = [float(row[field]) for row in paired]
        return float(np.mean(values)) if values else None

    summary = {
        "generated_at": p.now_ts(),
        "status": "complete" if len(completed) == len(specs) else "running",
        "planned_runs": len(specs),
        "completed_runs": len(completed),
        "claim_tier": "supporting/mechanistic; 50M local WikiText",
        "paired_complete_configs": len(paired),
        "aggregate": {
            "mean_rule_minus_geo_nll": mean("rule_minus_geo_nll"),
            "mean_exp_minus_geo_nll": mean("exp_minus_geo_nll"),
            "mean_cosh_rule_minus_exp_nll": mean("cosh_rule_minus_exp_nll"),
            "mean_cosh_rule_regret": mean("cosh_rule_regret"),
        },
        "paired": paired,
        "historical_phase16": old_phase16_summary(),
    }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    p.atomic_write_json(report_dir / "summary.json", summary)
    lines = [
        "# M4 exact-range pure-shape factorial",
        "",
        f"- Status: **{summary['status']}**",
        f"- Completed: {len(completed)}/{len(specs)} runs",
        f"- Paired complete cells: {len(paired)}/36",
        "- Evidence tier: supporting/mechanistic; 50M local WikiText",
        "- Control: identical sampled frequency extrema and log-span; only interior allocation changes",
        "",
        "## Aggregate paired NLL deltas",
        "",
        "| Contrast | Mean delta (negative is first arm better) |",
        "|---|---:|",
    ]
    for label, field in (
        ("Cosh rule - Geo", "mean_rule_minus_geo_nll"),
        ("Exponential rule - Geo", "mean_exp_minus_geo_nll"),
        ("Cosh rule - Exponential", "mean_cosh_rule_minus_exp_nll"),
        ("Cosh rule regret vs best preregistered Cosh", "mean_cosh_rule_regret"),
    ):
        value = summary["aggregate"][field]
        lines.append(f"| {label} | {value:.6f} |" if value is not None else f"| {label} | NA |")
    lines.extend(
        [
            "",
            "Historical 99-run comparisons are descriptive only because that sweep changed",
            "sampled endpoints/span together with the midpoint-Cosh interior shape.",
            "",
        ]
    )
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("plan", "self-test", "smoke", "run", "report"), default="run"
    )
    parser.add_argument("--work-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--train-tokens", type=int, default=8_388_608)
    parser.add_argument("--val-tokens", type=int, default=1_048_576)
    parser.add_argument("--eval-chunks", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_root = args.work_root.expanduser().resolve()
    if args.mode == "smoke":
        args.work_root = Path(f"{args.work_root}_smoke")
        args.train_tokens = min(args.train_tokens, 262_144)
        args.val_tokens = min(args.val_tokens, 262_144)
        args.eval_chunks = 1

    specs = build_specs(args.train_tokens, args.eval_chunks)
    receipts = validate_schedules(specs)
    args.work_root.mkdir(parents=True, exist_ok=True)
    write_plan(args.work_root, specs, receipts)
    if args.mode == "plan":
        print(f"[plan] {len(specs)} runs -> {args.work_root}")
        return

    if args.mode == "report":
        print(json.dumps(generate_report(args.work_root, specs), indent=2))
        return

    ctx = build_context(args)
    assert_matched_initialization(ctx, specs)
    if args.mode == "self-test":
        print("[self-test] exact endpoints, monotonicity, and matched initialization: OK")
        return
    if args.mode == "smoke":
        specs = [specs[0]]

    with p.SweepLock(args.work_root / "sweep.lock"):
        run_specs(ctx, specs)
        summary = generate_report(args.work_root, specs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
