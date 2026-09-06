#!/usr/bin/env python3
"""Target-free phase-isotropy screen on the existing 50M M4 harness.

This runner deliberately keeps the historical Phase16 files untouched.  It
reuses their model, deterministic row sampler, checkpointing, and MPS batch
plan, while adding only the new frequency construction and fixed-anchor NLL
receipt needed for the three-arm 50M screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import phase16_formula_optimality_sweep as p
from rebuttal.rebuttal_0723.experiments.geo_rope_contract import std_geo_inv_freq
from scripts.lib.rope.schedules import evq_cosh_phi


RAW_ROOT = REPO_ROOT / "results" / "theory" / "phase_isotropy_50m_m4_20260824"
REPORT_PATH = (
    REPO_ROOT
    / "paper-2027/research/attention-aware-retrofit/results/"
    / "PHASE_ISOTROPY_50M_M4_RESULT_20260824.md"
)
EVIDENCE_PATH = (
    REPO_ROOT
    / "paper-2027/research/attention-aware-retrofit/evidence/"
    / "PHASE_ISOTROPY_50M_M4_RECEIPT_20260824.json"
)

TRAIN_LENGTH = 256
HEAD_DIM = 64
K = HEAD_DIM // 2
SEED = 137
TRAIN_TOKENS = 8_388_608
VAL_TOKENS = 1_048_576
EVAL_LENGTHS = (256, 512, 1024, 2048)
EVAL_CHUNKS = 4
ANCHOR_SEED = 9_999
BASE = 256.0
COSH_TAU = 4.0

ARMS = ("FMRoPE", "anchored EVQ-Cosh", "phase-isotropy")


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def causal_chi_closed_form(length: int, t: torch.Tensor) -> torch.Tensor:
    """E[(exp(itd))] for p_L(d) proportional to L-d, evaluated on CPU."""
    if t.dtype != torch.float64:
        t = t.to(torch.float64)
    z = torch.exp(torch.complex(torch.zeros_like(t), t))
    den = 1.0 - z
    numerator = (
        float(length) * (1.0 - z**length) / den
        - z
        * (
            1.0
            - float(length) * z ** (length - 1)
            + float(length - 1) * z**length
        )
        / den.square()
    )
    out = numerator / (float(length) * float(length + 1) / 2.0)
    # The construction never evaluates t=0, but keep the identity exact for
    # the numerical self-check and future callers.
    small = t.abs() < 1e-10
    if bool(small.any()):
        out = out.clone()
        out[small] = 1.0 + 0.0j
    return out


def causal_chi_direct(length: int, t: torch.Tensor) -> torch.Tensor:
    d = torch.arange(length, dtype=torch.float64)
    weights = (float(length) - d) / (float(length) * float(length + 1) / 2.0)
    return (
        weights[None, :]
        * torch.exp(
            torch.complex(
                torch.zeros((t.numel(), length), dtype=torch.float64),
                t.reshape(-1, 1) * d.reshape(1, -1),
            )
        )
    ).sum(dim=1)


def phase_isotropy_inv_freq(
    head_dim: int,
    train_length: int,
    omega_max: float,
    omega_min: float,
) -> torch.Tensor:
    """Build endpoint-inclusive rho_iso quantiles from only the four inputs."""
    if head_dim // 2 != K:
        raise ValueError(f"screen requires K={K}, got head_dim={head_dim}")
    if train_length != TRAIN_LENGTH:
        raise ValueError(f"screen requires L_train={TRAIN_LENGTH}, got {train_length}")
    if not (0.0 < omega_min < omega_max):
        raise ValueError("frequency endpoints must satisfy 0 < omega_min < omega_max")

    # Fixed numerical quadrature resolution is an implementation detail, not a
    # selector: rho depends only on (K, L_train, omega_max, omega_min).
    phi = torch.linspace(0.0, 1.0, 65_537, dtype=torch.float64)
    span = math.log(float(omega_max) / float(omega_min))
    omega = float(omega_max) * torch.exp(-span * phi)
    chi = causal_chi_closed_form(train_length, 2.0 * omega)
    abs_chi = chi.abs().clamp(0.0, 1.0)
    isotropy = ((1.0 - abs_chi) / (1.0 + abs_chi)).clamp_min(0.0)
    density = isotropy.pow(1.0 / 3.0)
    dx = 1.0 / float(phi.numel() - 1)
    cdf = torch.zeros_like(phi)
    cdf[1:] = torch.cumsum((density[:-1] + density[1:]) * (0.5 * dx), dim=0)
    if not bool(torch.isfinite(cdf).all()) or float(cdf[-1]) <= 0.0:
        raise RuntimeError("phase-isotropy CDF is non-finite or empty")
    cdf = cdf / cdf[-1].clone()
    if not bool((torch.diff(cdf) > 0).all()):
        raise RuntimeError("phase-isotropy CDF is not strictly increasing")

    quantiles = torch.linspace(0.0, 1.0, K, dtype=torch.float64)
    right = torch.searchsorted(cdf, quantiles, right=True).clamp(1, cdf.numel() - 1)
    left = right - 1
    frac = (quantiles - cdf[left]) / (cdf[right] - cdf[left])
    phi_nodes = phi[left] + frac * (phi[right] - phi[left])
    phi_nodes[0], phi_nodes[-1] = 0.0, 1.0
    inv_freq = float(omega_max) * torch.exp(-span * phi_nodes)
    # Reuse the exact endpoint values supplied by the canonical FMRoPE arm.
    inv_freq[0] = float(omega_max)
    inv_freq[-1] = float(omega_min)
    return inv_freq.float().contiguous()


def anchored_cosh_inv_freq(head_dim: int, base: float, tau: float) -> torch.Tensor:
    phi = evq_cosh_phi(head_dim // 2, tau=tau, midpoint=True, dtype=torch.float64)
    shape = (phi - phi[0]) / (phi[-1] - phi[0])
    reference = std_geo_inv_freq(head_dim, base, dtype=torch.float64)
    span = math.log(float(reference[0]) / float(reference[-1]))
    out = (float(reference[0]) * torch.exp(-span * shape)).float()
    out[0], out[-1] = reference[0].float(), reference[-1].float()
    return out.contiguous()


def build_inv_freq(arm: str) -> torch.Tensor:
    reference = std_geo_inv_freq(HEAD_DIM, BASE, dtype=torch.float64)
    if arm == "FMRoPE":
        out = reference.float()
    elif arm == "anchored EVQ-Cosh":
        out = anchored_cosh_inv_freq(HEAD_DIM, BASE, COSH_TAU)
    elif arm == "phase-isotropy":
        out = phase_isotropy_inv_freq(
            HEAD_DIM,
            TRAIN_LENGTH,
            float(reference[0]),
            float(reference[-1]),
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")
    out = out.contiguous()
    if not torch.isfinite(out).all() or not torch.all(out[:-1] > out[1:]):
        raise AssertionError(f"{arm}: frequency tensor is not finite and strict decreasing")
    if not torch.equal(out[[0, -1]], reference.float()[[0, -1]]):
        raise AssertionError(f"{arm}: endpoint mismatch")
    return out


def validate_construction() -> dict[str, Any]:
    reference = build_inv_freq("FMRoPE").double()
    tensors = {arm: build_inv_freq(arm) for arm in ARMS}
    d = torch.arange(TRAIN_LENGTH, dtype=torch.float64)
    weights = (float(TRAIN_LENGTH) - d) / (TRAIN_LENGTH * (TRAIN_LENGTH + 1) / 2.0)
    determinant_errors: list[float] = []
    chi_errors: list[float] = []
    for arm, inv in tensors.items():
        for omega in inv.double():
            x = torch.stack((torch.cos(omega * d), torch.sin(omega * d)), dim=1)
            gram = (x * weights[:, None]).T @ x
            lhs = 4.0 * torch.linalg.det(gram)
            chi = (weights * torch.exp(torch.complex(torch.zeros_like(d), 2.0 * omega * d))).sum()
            rhs = 1.0 - chi.abs().square()
            determinant_errors.append(float((lhs - rhs).abs()))
            chi_errors.append(float((causal_chi_closed_form(TRAIN_LENGTH, (2.0 * omega).reshape(1))[0] - chi).abs()))
    if max(determinant_errors) > 2e-10 or max(chi_errors) > 2e-10:
        raise AssertionError("phase-isotropy determinant/characteristic identity failed")
    return {
        "status": "PASS",
        "inputs": {
            "K": K,
            "L_train": TRAIN_LENGTH,
            "omega_max": float(reference[0]),
            "omega_min": float(reference[-1]),
        },
        "formula": {
            "p_L": "(L-d)/(L(L+1)/2), d=0..L-1",
            "chi_L": "E_d exp(i t d)",
            "i_L": "(1-|chi_L(2 omega)|)/(1+|chi_L(2 omega)|)",
            "rho": "i_L(omega(phi))^(1/3), endpoint-inclusive inverse-CDF",
            "density_grid_points": 65_537,
        },
        "endpoint_exact": {
            arm: bool(torch.equal(value[[0, -1]], reference.float()[[0, -1]]))
            for arm, value in tensors.items()
        },
        "strict_monotone": {
            arm: bool(torch.all(value[:-1] > value[1:]))
            for arm, value in tensors.items()
        },
        "float32_sha256": {arm: tensor_sha256(value) for arm, value in tensors.items()},
        "max_abs_error_4det_identity": max(determinant_errors),
        "max_abs_error_chi_closed_form": max(chi_errors),
    }


@dataclass
class PhaseSpec(p.RunSpec):
    arm: str


def build_specs(train_tokens: int = TRAIN_TOKENS) -> list[PhaseSpec]:
    return [
        PhaseSpec(
            stage="phase_isotropy_50m_m4",
            run_id=f"{arm.replace(' ', '_').replace('-', '_')}_seed{SEED}",
            seq_len=TRAIN_LENGTH,
            num_heads=8,
            head_dim=HEAD_DIM,
            tau=0.0 if arm == "FMRoPE" else (COSH_TAU if arm == "anchored EVQ-Cosh" else 0.0),
            theory_tau=0.0,
            seed=SEED,
            tier="50m",
            train_tokens=train_tokens,
            eval_lengths=list(EVAL_LENGTHS),
            passkey_lengths=[],
            passkey_trials=0,
            eval_chunks=EVAL_CHUNKS,
            dsr_eval_length=0,
            dsr_distances=[],
            dsr_trials=0,
            passkey_mix_ratio=0.0,
            arm=arm,
        )
        for arm in ARMS
    ]


def build_context(root: Path, train_tokens: int) -> p.SweepContext:
    profile = p.SweepProfile(
        name="phase_isotropy_50m_m4",
        tier="50m",
        seq_lens=[TRAIN_LENGTH],
        head_counts=[8],
        seeds=[SEED],
        train_tokens=train_tokens,
        val_tokens=VAL_TOKENS,
        tau_multipliers=[1.0],
        passkey_mix_ratio=0.0,
        eval_chunks=EVAL_CHUNKS,
        passkey_trials_pilot=0,
        passkey_trials_confirm=0,
        dsr_trials=0,
        top_k=1,
    )
    args = SimpleNamespace(work_root=str(root), dataset="local_wikitext")
    ctx = p.build_sweep_context(args, profile)
    ctx.inv_freq_builder = lambda spec: build_inv_freq(spec.arm)
    return ctx


def sha256_tokens(tokens: torch.Tensor) -> str:
    return tensor_sha256(tokens)


def build_anchors(val_flat: torch.Tensor) -> dict[str, Any]:
    rng = np.random.RandomState(ANCHOR_SEED)
    offsets: dict[str, list[int]] = {}
    for length in EVAL_LENGTHS:
        max_start = int(val_flat.numel()) - int(length)
        if max_start <= 0:
            raise ValueError(f"validation prefix too short for L={length}")
        count = min(EVAL_CHUNKS, max(1, max_start // max(length, 1)))
        offsets[str(length)] = sorted(
            int(v) for v in rng.choice(max_start, size=count, replace=False)
        )
    return {
        "seed": ANCHOR_SEED,
        "eval_chunks": EVAL_CHUNKS,
        "val_prefix_tokens": int(val_flat.numel()),
        "offsets": offsets,
        "offsets_sha256": hashlib.sha256(
            json.dumps(offsets, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def assert_matched_initialization(ctx: p.SweepContext, specs: Iterable[PhaseSpec]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    cfg = p.build_cfg(ctx.base_cfg, TRAIN_LENGTH, 8)
    for spec in specs:
        p.set_seed(SEED)
        model = p.GPT(cfg, build_inv_freq(spec.arm))
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
        hashes[spec.arm] = digest.hexdigest()
        del model
    if len(set(hashes.values())) != 1:
        raise AssertionError(f"matched initialization failed: {hashes}")
    return hashes


def safe_batch_plan(
    _ctx: p.SweepContext,
    cfg: dict[str, Any],
    seq_len: int,
    num_heads: int,
    target_effective_batch: int,
) -> p.BatchPlan:
    if (seq_len, num_heads) != (TRAIN_LENGTH, 8):
        raise ValueError("unexpected batch-plan configuration")
    micro = 32
    accum = math.ceil(target_effective_batch / micro)
    return p.BatchPlan(
        config_key=p.config_key(seq_len, num_heads, int(cfg["head_dim"])),
        probed_micro_batch=micro,
        target_effective_batch=target_effective_batch,
        micro_batch_size=micro,
        grad_accum=accum,
        effective_batch_size=micro * accum,
        tokens_per_step=micro * accum * seq_len,
        probe_candidates=[micro],
    )


CURRENT_RUN: dict[str, Any] = {}


def fixed_anchor_nll(
    model: p.GPT,
    val_flat: torch.Tensor,
    eval_lengths: Iterable[int],
    _eval_chunks: int,
) -> dict[str, float]:
    anchors = CURRENT_RUN["anchors"]["offsets"]
    run_dir: Path = CURRENT_RUN["run_dir"]
    model.eval()
    model.extend_rope(max(EVAL_LENGTHS) + 64)
    rows: dict[str, Any] = {}
    for length in eval_lengths:
        full_losses: list[float] = []
        tail_losses: list[float] = []
        for offset in anchors[str(length)]:
            chunk = val_flat[offset : offset + length].unsqueeze(0).to(p.DEVICE)
            try:
                with torch.no_grad():
                    logits = model(chunk[:, :-1])
                    losses = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        chunk[:, 1:].reshape(-1),
                        reduction="none",
                    ).reshape(1, -1)
                full_losses.append(float(losses.mean().item()))
                tail_n = min(128, losses.shape[-1])
                tail_losses.append(float(losses[:, -tail_n:].mean().item()))
                del logits, losses
            finally:
                del chunk
                p.maybe_clear_device_cache()
        rows[str(length)] = {
            "offsets": anchors[str(length)],
            "full_nll": float(np.mean(full_losses)),
            "tail_nll": float(np.mean(tail_losses)),
            "full_ppl": float(math.exp(np.mean(full_losses))),
            "n": len(full_losses),
        }
    p.atomic_write_json(run_dir / "nll_eval.json", rows)
    return {key: round(value["full_ppl"], 6) for key, value in rows.items()}


class MPSMemoryProbe:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.samples: list[dict[str, int | None]] = []
        self.thread: threading.Thread | None = None

    @staticmethod
    def sample() -> dict[str, int | None]:
        current = driver = None
        if p.DEVICE == "mps" and hasattr(torch, "mps"):
            current_fn = getattr(torch.mps, "current_allocated_memory", None)
            driver_fn = getattr(torch.mps, "driver_allocated_memory", None)
            current = int(current_fn()) if current_fn else None
            driver = int(driver_fn()) if driver_fn else None
        return {"current_allocated_bytes": current, "driver_allocated_bytes": driver}

    def _loop(self) -> None:
        while not self.stop.is_set():
            self.samples.append(self.sample())
            self.stop.wait(0.5)

    def __enter__(self) -> "MPSMemoryProbe":
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.samples.append(self.sample())

    def receipt(self) -> dict[str, Any]:
        def peak(key: str) -> int | None:
            values = [int(row[key]) for row in self.samples if row.get(key) is not None]
            return max(values) if values else None

        return {
            "sample_count": len(self.samples),
            "peak_current_allocated_bytes": peak("current_allocated_bytes"),
            "peak_driver_allocated_bytes": peak("driver_allocated_bytes"),
            "peak_current_allocated_mib": (
                peak("current_allocated_bytes") / 2**20
                if peak("current_allocated_bytes") is not None
                else None
            ),
            "peak_driver_allocated_mib": (
                peak("driver_allocated_bytes") / 2**20
                if peak("driver_allocated_bytes") is not None
                else None
            ),
        }


def run_one(ctx: p.SweepContext, spec: PhaseSpec, anchors: dict[str, Any]) -> dict[str, Any]:
    run_dir = p.run_dir_for(ctx, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    CURRENT_RUN.update({"anchors": anchors, "run_dir": run_dir, "arm": spec.arm})
    started = time.time()
    with MPSMemoryProbe() as probe:
        result = p.execute_run(ctx, spec)
    memory = probe.receipt()
    result_path = p.result_path_for(ctx, spec)
    result = p.read_json(result_path, result)
    result["arm"] = spec.arm
    result["mps_memory"] = memory
    result["elapsed_sec_total"] = round(time.time() - started, 2)
    result["nll"] = p.read_json(run_dir / "nll_eval.json", {})
    p.atomic_write_json(result_path, result)
    p.atomic_write_json(run_dir / "mps_memory.json", memory)
    print(f"[complete] {spec.arm}: {result.get('nll', {})}  peak_mps_mib={memory.get('peak_driver_allocated_mib')}", flush=True)
    return result


def weighted_nll(nll: dict[str, Any], field: str) -> float | None:
    values = []
    weights = []
    for length in (512, 1024, 2048):
        row = nll.get(str(length))
        if not row:
            return None
        weight = math.log2(length / TRAIN_LENGTH + 1.0)
        values.append(float(row[field]))
        weights.append(weight)
    return float(np.average(values, weights=weights))


def load_results(root: Path, specs: Iterable[PhaseSpec]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for spec in specs:
        row = p.read_json(root / "runs" / spec.run_id / "result.json", {})
        if row:
            row["arm"] = spec.arm
            out[spec.arm] = row
    return out


def generate_report(root: Path, specs: list[PhaseSpec], construction: dict[str, Any], protocol: dict[str, Any], anomalies: list[str]) -> dict[str, Any]:
    results = load_results(root, specs)
    paired: dict[str, Any] = {}
    for arm, row in results.items():
        paired[arm] = row.get("nll", {})
    fm = paired.get("FMRoPE", {})
    phase = paired.get("phase-isotropy", {})
    cosh = paired.get("anchored EVQ-Cosh", {})
    deltas: dict[str, dict[str, float | None]] = {}
    for length in EVAL_LENGTHS:
        key = str(length)
        deltas[key] = {}
        for field in ("tail_nll", "full_nll"):
            base = fm.get(key, {}).get(field)
            for arm, data in (("anchored EVQ-Cosh", cosh), ("phase-isotropy", phase)):
                value = data.get(key, {}).get(field)
                deltas[key][f"{arm}_minus_FMRoPE_{field}"] = (
                    float(value) - float(base) if value is not None and base is not None else None
                )
    phase_gain_tail = (
        weighted_nll(fm, "tail_nll") - weighted_nll(phase, "tail_nll")
        if weighted_nll(fm, "tail_nll") is not None and weighted_nll(phase, "tail_nll") is not None
        else None
    )
    cosh_gain_tail = (
        weighted_nll(fm, "tail_nll") - weighted_nll(cosh, "tail_nll")
        if weighted_nll(fm, "tail_nll") is not None and weighted_nll(cosh, "tail_nll") is not None
        else None
    )
    retention = phase_gain_tail / cosh_gain_tail if cosh_gain_tail and cosh_gain_tail > 0 else None
    gate_checks = {
        "all_three_complete": len(results) == 3 and all(all(str(L) in paired.get(arm, {}) for L in EVAL_LENGTHS) for arm in ARMS),
        "phase_256_tail_delta_le_0.01": deltas.get("256", {}).get("phase-isotropy_minus_FMRoPE_tail_nll") is not None and deltas["256"]["phase-isotropy_minus_FMRoPE_tail_nll"] <= 0.01,
        "phase_ood_tail_all_better": all(
            deltas.get(str(L), {}).get("phase-isotropy_minus_FMRoPE_tail_nll") is not None
            and deltas[str(L)]["phase-isotropy_minus_FMRoPE_tail_nll"] < 0.0
            for L in (512, 1024, 2048)
        ),
        "phase_retains_80pct_cosh_gain_when_cosh_gain_positive": (
            True if cosh_gain_tail is None or cosh_gain_tail <= 0 else retention is not None and retention >= 0.80
        ),
    }
    passed = all(gate_checks.values()) and not anomalies
    raw_gate = "PROMISING_50M_SCREEN_ONLY" if passed else "FAILED_50M_GATE"
    verdict = (
        "SCREEN_UNRESOLVED"
        if not passed and (cosh_gain_tail is None or cosh_gain_tail <= 0)
        else raw_gate
    )
    report = {
        "status": verdict,
        "generated_raw_gate": raw_gate,
        "raw_root": "results/theory/phase_isotropy_50m_m4_20260824/",
        "construction": construction,
        "protocol": protocol,
        "arms": paired,
        "paired_deltas": deltas,
        "weighted_ood": {
            "metric": "log2(L/L_train+1)-weighted final-128-token NLL over 512/1024/2048",
            "FMRoPE": weighted_nll(fm, "tail_nll"),
            "anchored_EVQ_Cosh": weighted_nll(cosh, "tail_nll"),
            "phase_isotropy": weighted_nll(phase, "tail_nll"),
            "anchored_EVQ_Cosh_gain_vs_FMRoPE": cosh_gain_tail,
            "phase_isotropy_gain_vs_FMRoPE": phase_gain_tail,
            "phase_retention_of_cosh_gain": retention,
        },
        "gate_checks": gate_checks,
        "anomalies": anomalies,
        "run_time_sec": {arm: row.get("elapsed_sec_total") for arm, row in results.items()},
        "peak_mps_memory": {arm: row.get("mps_memory") for arm, row in results.items()},
    }
    lines = [
        "# Phase-isotropy 50M M4 screen (2026-08-24)",
        "",
        f"- Canonical verdict: **{verdict}**",
        f"- Generated raw gate string: `{raw_gate}`",
        "- Scope: internal method screening only; no manuscript claim.",
        "- Model: existing 50M M4 harness, `L_train=256`, `d_head=64`, `K=32`, seed `137`.",
        "- Main metric: final-128-token teacher-forced NLL; full NLL retained below.",
        "",
        "## NLL by length",
        "",
        "| Arm | 256 tail/full | 512 tail/full | 1024 tail/full | 2048 tail/full |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = paired.get(arm, {})
        cells = []
        for length in EVAL_LENGTHS:
            item = row.get(str(length), {})
            cells.append(
                f"{item.get('tail_nll', 'NA'):.6f}/{item.get('full_nll', 'NA'):.6f}"
                if item.get("tail_nll") is not None
                else "NA"
            )
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Paired deltas (negative means the named arm is better)",
        "",
        "```json",
        json.dumps(deltas, indent=2),
        "```",
        "",
        "## Weighted OOD and gate",
        "",
        f"- Weighted tail OOD FMRoPE: `{weighted_nll(fm, 'tail_nll')}`; anchored EVQ-Cosh gain: `{cosh_gain_tail}`; phase-isotropy gain: `{phase_gain_tail}`; retention: `{retention}`.",
        f"- Checks: `{json.dumps(gate_checks, sort_keys=True)}`",
        f"- Frequency float32 hashes: `{json.dumps(construction['float32_sha256'], sort_keys=True)}`",
        f"- Training/data/init/anchor matching: `{json.dumps(protocol, sort_keys=True)}`",
        f"- Runtime seconds: `{json.dumps(report['run_time_sec'], sort_keys=True)}`",
        f"- Peak MPS memory: `{json.dumps(report['peak_mps_memory'], sort_keys=True)}`",
        f"- Anomalies: `{json.dumps(anomalies)}`",
        "",
        "The finite one-seed screen is not a pooled effect, not a continuous optimum, and not evidence for deployment or paper promotion.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    p.atomic_write_json(root / "reports" / "summary.json", report)
    return report


def write_machine_receipt(root: Path, construction: dict[str, Any], protocol: dict[str, Any], report: dict[str, Any] | None = None) -> None:
    receipt = {
        "receipt_version": 1,
        "status": report.get("status") if report else "preflight",
        "generated_raw_gate": report.get("generated_raw_gate") if report else None,
        "experiment": "phase_isotropy_50m_m4_20260824",
        "raw_root": "results/theory/phase_isotropy_50m_m4_20260824/",
        "construction": construction,
        "protocol": protocol,
        "gate_checks": report.get("gate_checks") if report else None,
        "weighted_ood": report.get("weighted_ood") if report else None,
        "anomalies": report.get("anomalies", []) if report else [],
    }
    p.atomic_write_json(root / "receipt.json", receipt)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "smoke", "run", "report"), default="run")
    parser.add_argument("--work-root", type=Path, default=RAW_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if p.DEVICE != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError(f"MPS is required; detected device={p.DEVICE}")
    root = args.work_root.expanduser().resolve()
    construction = validate_construction()
    root.mkdir(parents=True, exist_ok=True)
    p.atomic_write_json(root / "construction_receipt.json", construction)
    specs = build_specs(TRAIN_TOKENS if args.mode != "smoke" else 262_144)
    if args.mode == "validate":
        write_machine_receipt(root, construction, {"status": "not_run"})
        print(json.dumps(construction, indent=2))
        return

    if args.mode == "smoke":
        specs = [specs[-1]]
        root = root / "smoke"
        root.mkdir(parents=True, exist_ok=True)

    ctx = build_context(root, specs[0].train_tokens)
    anchors = build_anchors(ctx.val_flat)
    init_hashes = assert_matched_initialization(ctx, build_specs(specs[0].train_tokens))
    protocol = {
        "device": p.DEVICE,
        "dtype": str(p.DTYPE),
        "seed": SEED,
        "train_tokens": specs[0].train_tokens,
        "global_batch_size": 256,
        "micro_batch_size": 32,
        "grad_accum": 8,
        "optimizer": "AdamW(lr=6e-4, betas=(0.9,0.95), weight_decay=0.1), cosine 0.1 floor, 5% warmup",
        "model": {"tier": "50m", "hidden_size": 512, "num_layers": 6, "num_heads": 8, "head_dim": HEAD_DIM, "K": K},
        "data": {
            "dataset": "local_wikitext",
            "train_prefix_tokens": int(ctx.train_flat.numel()),
            "val_prefix_tokens": int(ctx.val_flat.numel()),
            "train_prefix_sha256": sha256_tokens(ctx.train_flat),
            "val_prefix_sha256": sha256_tokens(ctx.val_flat),
            "row_order": "deterministic randint(seed*1000003 + global_micro_step) over fixed sequential train rows",
        },
        "anchors": anchors,
        "matched_initialization_sha256": init_hashes,
        "frequency_float32_sha256": construction["float32_sha256"],
    }
    p.atomic_write_json(root / "protocol_match.json", protocol)
    p.probe_micro_batch = safe_batch_plan
    p.safe_eval_ppl = fixed_anchor_nll
    p.safe_eval_passkey = lambda **_kwargs: {"details": {}, "summary": {}, "global": {}}
    p.safe_eval_dsr = lambda **_kwargs: (None, None)

    anomalies: list[str] = []
    results: list[dict[str, Any]] = []
    with p.SweepLock(root / "sweep.lock"):
        for spec in specs:
            try:
                results.append(run_one(ctx, spec, anchors))
            except Exception as exc:  # keep the final gate explicit if one arm fails
                anomalies.append(f"{spec.arm}: {type(exc).__name__}: {exc}")
                print(f"[anomaly] {anomalies[-1]}", flush=True)
        report = generate_report(root, specs, construction, protocol, anomalies)
    write_machine_receipt(root, construction, protocol, report)
    print(f"[done] {report['status']} report={REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
