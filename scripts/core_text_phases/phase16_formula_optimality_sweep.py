#!/usr/bin/env python3
"""Phase 16: local formula-optimality sweep harness.

This script tests whether the theory-predicted EVQ temperature

    tau*(L_train, d_head) = d_head / sqrt(L_train)

is empirically near-optimal under a broader sweep:

- multiple seeds
- multiple train lengths
- multiple head counts (with fixed hidden size, so d_head changes)
- resumable training
- structured progress / event logs
- automatic markdown + JSON report generation

The default `local_m4` profile is tuned for an Apple M4 Max workstation:
- tier: 50M backbone
- lengths: 256 / 512 / 1024
- heads: 4 / 8 / 16
- seeds: 42 / 137 / 256
- staged sweep: pilot seed over all taus, then multi-seed confirmation on top taus

Examples:
    conda activate aidemo
    python scripts/core_text_phases/phase16_formula_optimality_sweep.py --profile smoke
    python scripts/core_text_phases/phase16_formula_optimality_sweep.py --profile local_m4
    python scripts/core_text_phases/phase16_formula_optimality_sweep.py --mode report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Conservative MPS allocator defaults inferred from official PyTorch docs:
# keep the soft watermark below the hard watermark to reduce surprise OOMs.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.95")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.90")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SUPPORTING_DIR = SCRIPT_DIR.parent / "supporting_eval"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SUPPORTING_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORTING_DIR))

from run_evq_sweep import (  # noqa: E402
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    GPT,
    TIER_CONFIGS,
    evq_cosh_inv_freq,
    get_batch_from_data,
    load_data,
    load_val,
    set_seed,
)
from eval_dsr import compute_summary as compute_dsr_summary  # noqa: E402
from eval_dsr import eval_dsr_single_model  # noqa: E402
from eval_passkey_scratch import (  # noqa: E402
    MixedDataset,
    eval_passkey_nll_gap,
    sanity_check_tokenizer,
)


STOP_REQUESTED = False


PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "tier": "50m",
        "seq_lens": [256],
        "head_counts": [8],
        "seeds": [42],
        "train_tokens": 1_048_576,
        "val_tokens": 262_144,
        "tau_multipliers": [0.75, 1.0, 1.25],
        "passkey_mix_ratio": 0.01,
        "eval_chunks": 2,
        "passkey_trials_pilot": 4,
        "passkey_trials_confirm": 6,
        "dsr_trials": 4,
        "top_k": 1,
    },
    "local_m4": {
        "tier": "50m",
        "seq_lens": [256, 512, 1024],
        "head_counts": [4, 8, 16],
        "seeds": [42, 137, 256],
        "train_tokens": 8_388_608,
        "val_tokens": 1_048_576,
        "tau_multipliers": [0.75, 1.0, 1.25, 1.5],
        "passkey_mix_ratio": 0.01,
        "eval_chunks": 4,
        "passkey_trials_pilot": 8,
        "passkey_trials_confirm": 16,
        "dsr_trials": 8,
        "top_k": 2,
    },
    "full": {
        "tier": "50m",
        "seq_lens": [256, 512, 1024, 2048],
        "head_counts": [4, 8, 16],
        "seeds": [42, 137, 256, 314],
        "train_tokens": 12_582_912,
        "val_tokens": 2_097_152,
        "tau_multipliers": [0.7, 0.85, 1.0, 1.15, 1.35, 1.6],
        "passkey_mix_ratio": 0.01,
        "eval_chunks": 6,
        "passkey_trials_pilot": 10,
        "passkey_trials_confirm": 20,
        "dsr_trials": 10,
        "top_k": 2,
    },
}


def _handle_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[signal] received {signum}; will stop after the current safe point", flush=True)


signal.signal(signal.SIGINT, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)


class SweepError(RuntimeError):
    """Base harness failure."""


class TrainingOOMError(SweepError):
    """Raised when training still OOMs after local recovery."""


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def maybe_clear_device_cache() -> None:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False, default=str), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def sha256_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def run_cmd(cmd: Sequence[str], timeout: int = 30) -> str:
    out = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if out.returncode != 0:
        return out.stderr.strip() or out.stdout.strip()
    return out.stdout.strip()


def detect_conda_executable() -> str:
    direct = Path.home() / "miniconda3" / "bin" / "conda"
    if direct.exists():
        return str(direct)
    found = shutil.which("conda")
    return found or ""


def detect_active_conda_env(conda_exe: str) -> str:
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    if env_name:
        return env_name
    if not conda_exe:
        return ""
    raw = run_cmd([conda_exe, "env", "list", "--json"])
    try:
        data = json.loads(raw)
        active_prefix = data.get("default_prefix") or ""
        envs = data.get("envs", [])
        if active_prefix in envs:
            return Path(active_prefix).name
    except Exception:
        pass
    return ""


def detect_host_memory_gb() -> Optional[float]:
    if sys.platform == "darwin":
        raw = run_cmd(["sysctl", "-n", "hw.memsize"])
        try:
            return int(raw) / (1024 ** 3)
        except Exception:
            return None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    return int(parts[1]) * 1024 / (1024 ** 3)
        except Exception:
            return None
    return None


def with_retries(name: str, fn, attempts: int = 3, sleep_sec: float = 2.0):
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - runtime recovery
            last_exc = exc
            print(f"[retry] {name} failed on attempt {attempt}/{attempts}: {exc}", flush=True)
            maybe_clear_device_cache()
            if attempt < attempts:
                time.sleep(sleep_sec * attempt)
    raise SweepError(f"{name} failed after {attempts} attempts: {last_exc}")


def _repeat_trim(ids: List[int], max_tokens: int) -> torch.Tensor:
    if not ids:
        raise SweepError("Local text corpus is empty; cannot build token stream")
    out = list(ids)
    while len(out) < max_tokens:
        out.extend(ids[: max_tokens - len(out)])
    return torch.tensor(out[:max_tokens], dtype=torch.long)


def load_local_wikitext_tokens(tokenizer, split: str, max_tokens: int) -> torch.Tensor:
    root = Path.home() / ".cache" / "huggingface" / "datasets" / "wikitext" / "wikitext-2-raw-v1" / "0.0.0"
    if not root.exists():
        raise SweepError(f"Local wikitext cache not found at {root}")
    candidates = sorted(root.iterdir())
    if not candidates:
        raise SweepError(f"No local wikitext versions under {root}")
    arrow_path = candidates[-1] / f"wikitext-{split}.arrow"
    if not arrow_path.exists():
        raise SweepError(f"Expected local arrow file missing: {arrow_path}")
    ds = HFDataset.from_file(str(arrow_path))
    ids: List[int] = []
    for row in ds:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
    return _repeat_trim(ids, max_tokens)


def predicted_tau(head_dim: int, seq_len: int) -> float:
    return float(head_dim) / math.sqrt(float(seq_len))


def round_tau(value: float) -> float:
    return round(float(value), 2)


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def config_key(seq_len: int, num_heads: int, head_dim: int) -> str:
    return f"L{seq_len}_H{num_heads}_Dh{head_dim}"


def build_tau_grid(pred_tau: float, multipliers: Sequence[float]) -> List[float]:
    values = [0.0]
    for mult in multipliers:
        tau = pred_tau * float(mult)
        if tau > 1e-8:
            values.append(min(tau, 16.0))
    rounded = sorted({round_tau(v) for v in values})
    return rounded


def extrapolation_lengths(seq_len: int, hard_cap: int = 8192) -> List[int]:
    lengths = []
    for factor in (1, 2, 4, 8):
        value = seq_len * factor
        if value <= hard_cap:
            lengths.append(value)
    return lengths


def passkey_lengths(seq_len: int, hard_cap: int = 8192) -> List[int]:
    lengths = []
    for factor in (2, 4):
        value = seq_len * factor
        if value <= hard_cap:
            lengths.append(value)
    if not lengths:
        lengths.append(seq_len)
    return lengths


def dsr_distances(seq_len: int, eval_length: int) -> List[int]:
    ratios = [0.5, 1.0, 2.0, 3.0]
    out = []
    for ratio in ratios:
        dist = int(seq_len * ratio)
        if 0 < dist < eval_length:
            out.append(dist)
    return sorted(set(out))


def build_cfg(base_cfg: Dict[str, Any], seq_len: int, num_heads: int) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    hidden = int(cfg["hidden_size"])
    if hidden % num_heads != 0:
        raise ValueError(f"hidden_size={hidden} not divisible by num_heads={num_heads}")
    head_dim = hidden // num_heads
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    cfg["num_heads"] = num_heads
    cfg["head_dim"] = head_dim
    cfg["seq_len"] = seq_len
    cfg["max_position_embeddings"] = seq_len
    return cfg


def chunk_tokens(flat_tokens: torch.Tensor, seq_len: int, max_tokens: int) -> torch.Tensor:
    usable = min(int(flat_tokens.numel()), int(max_tokens))
    usable = (usable // seq_len) * seq_len
    if usable < seq_len:
        raise ValueError(
            f"Not enough flat tokens to build seq_len={seq_len}: have {flat_tokens.numel()}, need {seq_len}"
        )
    return flat_tokens[:usable].reshape(-1, seq_len).clone()


@dataclass
class SweepProfile:
    name: str
    tier: str
    seq_lens: List[int]
    head_counts: List[int]
    seeds: List[int]
    train_tokens: int
    val_tokens: int
    tau_multipliers: List[float]
    passkey_mix_ratio: float
    eval_chunks: int
    passkey_trials_pilot: int
    passkey_trials_confirm: int
    dsr_trials: int
    top_k: int


@dataclass
class RunSpec:
    stage: str
    run_id: str
    seq_len: int
    num_heads: int
    head_dim: int
    tau: float
    theory_tau: float
    seed: int
    tier: str
    train_tokens: int
    eval_lengths: List[int]
    passkey_lengths: List[int]
    passkey_trials: int
    eval_chunks: int
    dsr_eval_length: int
    dsr_distances: List[int]
    dsr_trials: int
    passkey_mix_ratio: float


@dataclass
class BatchPlan:
    config_key: str
    probed_micro_batch: int
    target_effective_batch: int
    micro_batch_size: int
    grad_accum: int
    effective_batch_size: int
    tokens_per_step: int
    probe_candidates: List[int] = field(default_factory=list)


@dataclass
class SweepContext:
    args: argparse.Namespace
    profile: SweepProfile
    root: Path
    runs_dir: Path
    cache_dir: Path
    reports_dir: Path
    events_path: Path
    env_snapshot: Dict[str, Any]
    tokenizer: Any
    train_flat: torch.Tensor
    val_flat: torch.Tensor
    filler_tokens: torch.Tensor
    base_cfg: Dict[str, Any]


class SweepLock:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "SweepLock":
        if self.path.exists():
            existing = read_json(self.path, {})
            pid = int(existing.get("pid", -1))
            if pid > 0:
                try:
                    os.kill(pid, 0)
                except OSError:
                    pass
                else:
                    raise SweepError(
                        f"sweep lock active at {self.path}; pid={pid} started={existing.get('started_at')}"
                    )
        atomic_write_json(
            self.path,
            {
                "pid": os.getpid(),
                "started_at": now_ts(),
                "host": platform.node(),
                "cwd": str(REPO_ROOT),
            },
        )
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if self.path.exists():
            self.path.unlink()


def load_profile(args: argparse.Namespace) -> SweepProfile:
    if args.profile not in PROFILE_PRESETS:
        raise ValueError(f"Unknown profile: {args.profile}")
    data = dict(PROFILE_PRESETS[args.profile])
    if args.tier:
        data["tier"] = args.tier
    if args.seq_lens:
        data["seq_lens"] = parse_csv_ints(args.seq_lens)
    if args.head_counts:
        data["head_counts"] = parse_csv_ints(args.head_counts)
    if args.seeds:
        data["seeds"] = parse_csv_ints(args.seeds)
    if args.train_tokens is not None:
        data["train_tokens"] = int(args.train_tokens)
    if args.val_tokens is not None:
        data["val_tokens"] = int(args.val_tokens)
    if args.tau_multipliers:
        data["tau_multipliers"] = parse_csv_floats(args.tau_multipliers)
    if args.passkey_mix_ratio is not None:
        data["passkey_mix_ratio"] = float(args.passkey_mix_ratio)
    if args.eval_chunks is not None:
        data["eval_chunks"] = int(args.eval_chunks)
    return SweepProfile(name=args.profile, **data)


def detect_environment_snapshot(profile: SweepProfile, root: Path) -> Dict[str, Any]:
    conda_exe = detect_conda_executable()
    active_env = detect_active_conda_env(conda_exe)
    host_mem_gb = detect_host_memory_gb()
    return {
        "timestamp": now_ts(),
        "python": sys.version,
        "platform": platform.platform(),
        "node": platform.node(),
        "repo_root": str(REPO_ROOT),
        "work_root": str(root),
        "profile": profile.name,
        "tier": profile.tier,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "autocast": bool(USE_AUTOCAST),
        "torch": torch.__version__,
        "mps_available": bool(getattr(torch.backends.mps, "is_available", lambda: False)()),
        "mps_built": bool(getattr(torch.backends.mps, "is_built", lambda: False)()),
        "cuda_available": bool(torch.cuda.is_available()),
        "host_memory_gb": round(host_mem_gb, 2) if host_mem_gb else None,
        "conda_executable": conda_exe,
        "active_conda_env": active_env,
        "expected_conda_env": "aidemo",
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "mps_high_watermark_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
        "mps_low_watermark_ratio": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
    }


def log_root_event(ctx: SweepContext, event: str, **payload: Any) -> None:
    append_jsonl(
        ctx.events_path,
        {
            "ts": now_ts(),
            "event": event,
            **payload,
        },
    )


def log_run_event(path: Path, event: str, **payload: Any) -> None:
    append_jsonl(
        path,
        {
            "ts": now_ts(),
            "event": event,
            **payload,
        },
    )


def build_sweep_context(args: argparse.Namespace, profile: SweepProfile) -> SweepContext:
    root = Path(args.work_root).expanduser().resolve()
    runs_dir = root / "runs"
    cache_dir = root / "cache"
    reports_dir = root / "reports"
    for directory in (root, runs_dir, cache_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env_snapshot = detect_environment_snapshot(profile, root)
    atomic_write_json(root / "environment.json", env_snapshot)

    def _load_tok():
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    tokenizer = with_retries("tokenizer load", _load_tok)
    tok_ok = sanity_check_tokenizer(tokenizer)
    log_path = root / "events.jsonl"
    append_jsonl(
        log_path,
        {
            "ts": now_ts(),
            "event": "tokenizer_check",
            "ok": tok_ok,
        },
    )
    if not tok_ok:
        raise SweepError("passkey tokenizer sanity check failed")

    base_seq_len = min(profile.seq_lens)

    def _load_train():
        if args.dataset == "local_wikitext":
            return load_local_wikitext_tokens(tokenizer, split="train", max_tokens=profile.train_tokens)
        data = load_data(
            tokenizer,
            max_tokens=profile.train_tokens,
            seq_len=base_seq_len,
            dataset=args.dataset,
            cache_dir=str(cache_dir),
        )
        return data.reshape(-1)

    def _load_val():
        if args.dataset == "local_wikitext":
            return load_local_wikitext_tokens(
                tokenizer,
                split="validation",
                max_tokens=profile.val_tokens,
            )
        data = load_val(
            tokenizer,
            max_tokens=profile.val_tokens,
            dataset=args.dataset,
            cache_dir=str(cache_dir),
        )
        if data.dim() > 1:
            data = data.reshape(-1)
        return data

    train_flat = with_retries("train data load", _load_train)
    val_flat = with_retries("val data load", _load_val)
    filler_tokens = val_flat.clone()

    base_cfg = dict(TIER_CONFIGS[profile.tier])
    ctx = SweepContext(
        args=args,
        profile=profile,
        root=root,
        runs_dir=runs_dir,
        cache_dir=cache_dir,
        reports_dir=reports_dir,
        events_path=log_path,
        env_snapshot=env_snapshot,
        tokenizer=tokenizer,
        train_flat=train_flat,
        val_flat=val_flat,
        filler_tokens=filler_tokens,
        base_cfg=base_cfg,
    )
    log_root_event(
        ctx,
        "context_ready",
        train_tokens=int(train_flat.numel()),
        val_tokens=int(val_flat.numel()),
        base_seq_len=base_seq_len,
    )
    return ctx


def build_run_spec(
    profile: SweepProfile,
    seq_len: int,
    num_heads: int,
    tau: float,
    seed: int,
    stage: str,
    tier_cfg: Dict[str, Any],
) -> RunSpec:
    head_dim = int(tier_cfg["hidden_size"]) // int(num_heads)
    theory_tau = round_tau(predicted_tau(head_dim, seq_len))
    dsr_eval_length = min(seq_len * 4, 8192)
    return RunSpec(
        stage=stage,
        run_id=f"{stage}_{config_key(seq_len, num_heads, head_dim)}_tau{tau:.2f}_seed{seed}",
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        tau=round_tau(tau),
        theory_tau=theory_tau,
        seed=seed,
        tier=profile.tier,
        train_tokens=profile.train_tokens,
        eval_lengths=extrapolation_lengths(seq_len),
        passkey_lengths=passkey_lengths(seq_len),
        passkey_trials=(
            profile.passkey_trials_pilot if stage == "pilot" else profile.passkey_trials_confirm
        ),
        eval_chunks=profile.eval_chunks,
        dsr_eval_length=dsr_eval_length,
        dsr_distances=dsr_distances(seq_len, dsr_eval_length),
        dsr_trials=(0 if stage == "pilot" else profile.dsr_trials),
        passkey_mix_ratio=profile.passkey_mix_ratio,
    )


def build_pilot_specs(profile: SweepProfile, base_cfg: Dict[str, Any]) -> List[RunSpec]:
    pilot_seed = profile.seeds[0]
    out: List[RunSpec] = []
    for seq_len in profile.seq_lens:
        cfg = build_cfg(base_cfg, seq_len, profile.head_counts[0])
        hidden = int(cfg["hidden_size"])
        for num_heads in profile.head_counts:
            if hidden % num_heads != 0:
                continue
            head_dim = hidden // num_heads
            theory = predicted_tau(head_dim, seq_len)
            taus = build_tau_grid(theory, profile.tau_multipliers)
            for tau in taus:
                out.append(
                    build_run_spec(
                        profile=profile,
                        seq_len=seq_len,
                        num_heads=num_heads,
                        tau=tau,
                        seed=pilot_seed,
                        stage="pilot",
                        tier_cfg=cfg,
                    )
                )
    return out


def run_dir_for(ctx: SweepContext, spec: RunSpec) -> Path:
    return ctx.runs_dir / spec.run_id


def result_path_for(ctx: SweepContext, spec: RunSpec) -> Path:
    return run_dir_for(ctx, spec) / "result.json"


def status_path_for(ctx: SweepContext, spec: RunSpec) -> Path:
    return run_dir_for(ctx, spec) / "status.json"


def checkpoint_path_for(ctx: SweepContext, spec: RunSpec) -> Path:
    return run_dir_for(ctx, spec) / "checkpoint_last.pt"


def selection_key(spec: RunSpec) -> str:
    return config_key(spec.seq_len, spec.num_heads, spec.head_dim)


def load_completed_results(specs: Iterable[RunSpec], ctx: SweepContext) -> Dict[str, Dict[str, Any]]:
    out = {}
    for spec in specs:
        path = result_path_for(ctx, spec)
        if path.exists():
            out[spec.run_id] = read_json(path, {})
    return out


def build_confirm_selection(
    profile: SweepProfile,
    pilot_specs: List[RunSpec],
    pilot_results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Tuple[RunSpec, Dict[str, Any]]]] = defaultdict(list)
    for spec in pilot_specs:
        result = pilot_results.get(spec.run_id)
        if result:
            grouped[selection_key(spec)].append((spec, result))

    selection: Dict[str, Any] = {
        "generated_at": now_ts(),
        "top_k": profile.top_k,
        "configs": {},
    }

    for key, rows in grouped.items():
        rows_sorted = sorted(
            rows,
            key=lambda item: float(item[1].get("selection_score", -1e9)),
            reverse=True,
        )
        if not rows_sorted:
            continue
        theory_tau = rows_sorted[0][0].theory_tau
        chosen: List[float] = []
        for forced in (theory_tau, 0.0):
            if forced not in chosen:
                chosen.append(forced)
        for spec, _result in rows_sorted:
            if spec.tau not in chosen:
                chosen.append(spec.tau)
            if len(chosen) >= max(profile.top_k + 1, len(set([theory_tau, 0.0]))):
                break
        selection["configs"][key] = {
            "theory_tau": theory_tau,
            "chosen_taus": chosen,
            "pilot_ranking": [
                {
                    "tau": spec.tau,
                    "selection_score": round(float(result.get("selection_score", -1e9)), 6),
                    "ppl_score": result.get("ppl_score"),
                    "passkey_score": result.get("passkey_score"),
                }
                for spec, result in rows_sorted
            ],
        }

    atomic_write_json(output_path, selection)
    return selection


def build_confirm_specs(
    profile: SweepProfile,
    base_cfg: Dict[str, Any],
    selection: Dict[str, Any],
) -> List[RunSpec]:
    out: List[RunSpec] = []
    extra_seeds = profile.seeds[1:]
    if not extra_seeds:
        return out
    for seq_len in profile.seq_lens:
        hidden = int(base_cfg["hidden_size"])
        for num_heads in profile.head_counts:
            if hidden % num_heads != 0:
                continue
            head_dim = hidden // num_heads
            key = config_key(seq_len, num_heads, head_dim)
            config_sel = selection.get("configs", {}).get(key)
            if not config_sel:
                continue
            cfg = build_cfg(base_cfg, seq_len, num_heads)
            for tau in config_sel.get("chosen_taus", []):
                for seed in extra_seeds:
                    out.append(
                        build_run_spec(
                            profile=profile,
                            seq_len=seq_len,
                            num_heads=num_heads,
                            tau=float(tau),
                            seed=seed,
                            stage="confirm",
                            tier_cfg=cfg,
                        )
                    )
    return out


def save_status(path: Path, status: Dict[str, Any]) -> None:
    prev = read_json(path, {})
    merged = dict(prev)
    merged.update(status)
    merged["updated_at"] = now_ts()
    atomic_write_json(path, merged)


def build_probe_candidates(target_effective_batch: int) -> List[int]:
    raw = [
        target_effective_batch,
        64,
        48,
        32,
        24,
        16,
        12,
        8,
        6,
        4,
        2,
        1,
    ]
    out = []
    for value in raw:
        if value >= 1 and value not in out:
            out.append(value)
    return sorted(out, reverse=True)


def probe_micro_batch(
    ctx: SweepContext,
    cfg: Dict[str, Any],
    seq_len: int,
    num_heads: int,
    target_effective_batch: int,
) -> BatchPlan:
    head_dim = cfg["head_dim"]
    key = config_key(seq_len, num_heads, head_dim)
    probe_path = ctx.root / "batch_probes" / f"{key}.json"
    cached = read_json(probe_path, {})
    if cached:
        return BatchPlan(**cached)

    train_sample = chunk_tokens(ctx.train_flat, seq_len, max_tokens=seq_len * 128)
    inv_freq = evq_cosh_inv_freq(head_dim, 0.0)
    candidates = build_probe_candidates(target_effective_batch)
    success_micro = 1

    for micro in candidates:
        if micro > len(train_sample):
            continue
        maybe_clear_device_cache()
        try:
            set_seed(1234)
            model = GPT(cfg, inv_freq).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
            batch = train_sample[:micro].to(DEVICE)
            ctx_autocast = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
            with ctx_autocast:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            success_micro = micro
            del batch, opt, model, logits, loss
            maybe_clear_device_cache()
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[probe] {key} micro_batch={micro} OOM; trying smaller batch", flush=True)
                maybe_clear_device_cache()
                continue
            raise

    micro_batch_size = min(success_micro, target_effective_batch)
    grad_accum = max(1, math.ceil(target_effective_batch / micro_batch_size))
    plan = BatchPlan(
        config_key=key,
        probed_micro_batch=success_micro,
        target_effective_batch=target_effective_batch,
        micro_batch_size=micro_batch_size,
        grad_accum=grad_accum,
        effective_batch_size=micro_batch_size * grad_accum,
        tokens_per_step=micro_batch_size * grad_accum * seq_len,
        probe_candidates=candidates,
    )
    atomic_write_json(probe_path, asdict(plan))
    return plan


def target_effective_batch(base_cfg: Dict[str, Any], seq_len: int) -> int:
    target_tokens = int(base_cfg["batch_size"]) * int(base_cfg["seq_len"])
    return max(1, target_tokens // seq_len)


def load_or_init_model(
    ctx: SweepContext,
    spec: RunSpec,
    cfg: Dict[str, Any],
    checkpoint_path: Path,
) -> Tuple[GPT, Optional[Dict[str, Any]]]:
    inv_freq = evq_cosh_inv_freq(spec.head_dim, spec.tau)
    model = GPT(cfg, inv_freq).to(DEVICE)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state["model"])
        return model, state
    return model, None


def deterministic_batch_indices(
    dataset_size: int,
    micro_batch_size: int,
    seed: int,
    global_micro_step: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 1_000_003 + int(global_micro_step))
    return torch.randint(0, dataset_size, (micro_batch_size,), generator=generator)


def safe_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    batch_plan: BatchPlan,
    spec: RunSpec,
) -> None:
    payload = {
        "saved_at": now_ts(),
        "step": step,
        "spec": asdict(spec),
        "batch_plan": asdict(batch_plan),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train_one_run(
    ctx: SweepContext,
    spec: RunSpec,
    cfg: Dict[str, Any],
    batch_plan: BatchPlan,
    run_dir: Path,
    checkpoint_path: Path,
) -> GPT:
    train_chunks = chunk_tokens(ctx.train_flat, spec.seq_len, spec.train_tokens)
    train_data = MixedDataset(
        lm_data=train_chunks,
        filler_tokens=ctx.filler_tokens,
        tokenizer=ctx.tokenizer,
        passkey_ratio=spec.passkey_mix_ratio,
        seq_len=spec.seq_len,
    )

    total_steps = max(1, math.ceil(spec.train_tokens / batch_plan.tokens_per_step))
    warmup_steps = max(1, int(total_steps * 0.05))
    checkpoint_interval = max(1, total_steps // 4)
    log_interval = max(1, total_steps // 10)

    model, resume_state = load_or_init_model(ctx, spec, cfg, checkpoint_path)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    start_step = 0
    if resume_state:
        start_step = int(resume_state.get("step", 0))
        opt.load_state_dict(resume_state["optimizer"])
        print(f"[resume] {spec.run_id}: step {start_step}/{total_steps}", flush=True)

    t0 = time.time()
    ctx_autocast = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    try:
        for step in range(start_step + 1, total_steps + 1):
            if STOP_REQUESTED:
                safe_checkpoint(checkpoint_path, model, opt, step - 1, batch_plan, spec)
                raise SweepError(f"stop requested while training {spec.run_id}")

            if step <= warmup_steps:
                cur_lr = cfg["lr"] * step / warmup_steps
            else:
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                cur_lr = cfg["lr"] * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for group in opt.param_groups:
                group["lr"] = cur_lr

            opt.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_idx in range(batch_plan.grad_accum):
                global_micro_step = (step - 1) * batch_plan.grad_accum + micro_idx
                indices = deterministic_batch_indices(
                    dataset_size=len(train_data),
                    micro_batch_size=batch_plan.micro_batch_size,
                    seed=spec.seed,
                    global_micro_step=global_micro_step,
                )
                try:
                    batch = get_batch_from_data(train_data, indices).to(DEVICE)
                    with ctx_autocast:
                        logits = model(batch[:, :-1])
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            batch[:, 1:].reshape(-1),
                        )
                        scaled_loss = loss / batch_plan.grad_accum
                    scaled_loss.backward()
                    accum_loss += float(loss.item()) / batch_plan.grad_accum
                    del batch, logits, loss, scaled_loss
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        safe_checkpoint(checkpoint_path, model, opt, step - 1, batch_plan, spec)
                        maybe_clear_device_cache()
                        raise TrainingOOMError(
                            f"OOM at step={step}, micro={micro_idx}, "
                            f"micro_batch={batch_plan.micro_batch_size}"
                        ) from exc
                    raise

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % log_interval == 0 or step in {1, total_steps}:
                elapsed = time.time() - t0
                rate = (step * batch_plan.tokens_per_step) / max(elapsed, 1e-6)
                progress = {
                    "stage": spec.stage,
                    "step": step,
                    "total_steps": total_steps,
                    "loss": round(accum_loss, 6),
                    "lr": cur_lr,
                    "tokens_per_sec": round(rate, 2),
                    "micro_batch_size": batch_plan.micro_batch_size,
                    "grad_accum": batch_plan.grad_accum,
                    "effective_batch_size": batch_plan.effective_batch_size,
                }
                atomic_write_json(run_dir / "progress.json", progress)
                log_run_event(run_dir / "events.jsonl", "train_progress", **progress)

            if step % checkpoint_interval == 0 or step == total_steps:
                safe_checkpoint(checkpoint_path, model, opt, step, batch_plan, spec)

    finally:
        del opt
        maybe_clear_device_cache()

    return model


def safe_eval_ppl(
    model: GPT,
    val_flat: torch.Tensor,
    eval_lengths: Sequence[int],
    eval_chunks: int,
) -> Dict[str, float]:
    model.eval()
    results: Dict[str, float] = {}
    rng = np.random.RandomState(9999)
    ctx_autocast = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    max_len = max(eval_lengths)
    model.extend_rope(max_len + 64)

    for length in eval_lengths:
        losses: List[float] = []
        max_start = int(val_flat.numel()) - int(length)
        if max_start <= 0:
            continue
        offsets = sorted(
            rng.choice(max_start, size=min(eval_chunks, max(1, max_start // max(length, 1))), replace=False)
        )
        for offset in offsets:
            chunk = val_flat[offset : offset + length].unsqueeze(0).to(DEVICE)
            try:
                with torch.no_grad(), ctx_autocast:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        chunk[:, 1:].reshape(-1),
                    )
                losses.append(float(loss.item()))
                del logits, loss
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"[eval] PPL L={length} OOM, skipping this length", flush=True)
                    maybe_clear_device_cache()
                    break
                raise
            finally:
                del chunk
                maybe_clear_device_cache()
        if losses:
            results[str(length)] = round(math.exp(sum(losses) / len(losses)), 4)
    return results


def safe_eval_passkey(
    model: GPT,
    tokenizer,
    filler_tokens: torch.Tensor,
    lengths: Sequence[int],
    num_trials: int,
    seed: int,
) -> Dict[str, Any]:
    trials = int(num_trials)
    eval_lengths = list(lengths)
    for attempt in range(1, 4):
        try:
            return eval_passkey_nll_gap(
                model=model,
                tokenizer=tokenizer,
                filler_tokens=filler_tokens,
                lengths=eval_lengths,
                depths=[0.5],
                num_trials=trials,
                seed=seed,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            maybe_clear_device_cache()
            print(
                f"[eval] passkey OOM on attempt {attempt}; lengths={eval_lengths}, trials={trials}",
                flush=True,
            )
            if len(eval_lengths) > 1:
                eval_lengths = eval_lengths[:-1]
            else:
                trials = max(2, trials // 2)
    return {
        "details": {},
        "summary": {},
        "global": {
            "retrieval_rate": None,
            "ar_exact_match": None,
            "mean_nll_gap": None,
            "error": "passkey_eval_oom",
        },
    }


def safe_eval_dsr(
    model: GPT,
    spec: RunSpec,
    tokenizer,
    filler_tokens: torch.Tensor,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if spec.dsr_trials <= 0 or not spec.dsr_distances:
        return None, None
    inv_freq = evq_cosh_inv_freq(spec.head_dim, spec.tau)
    trials = int(spec.dsr_trials)
    for attempt in range(1, 4):
        try:
            curve = eval_dsr_single_model(
                model=model,
                inv_freq_orig=inv_freq,
                head_dim=spec.head_dim,
                train_len=spec.seq_len,
                base=500000.0,
                filler_tokens=filler_tokens,
                tokenizer=tokenizer,
                eval_length=spec.dsr_eval_length,
                distances=spec.dsr_distances,
                num_trials=trials,
                batch_size=4 if DEVICE == "mps" else 8,
                apply_yarn=False,
                seed=spec.seed,
            )
            return curve, compute_dsr_summary(curve, train_len=spec.seq_len)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            maybe_clear_device_cache()
            trials = max(2, trials // 2)
            print(
                f"[eval] DSR OOM on attempt {attempt}; reducing trials to {trials}",
                flush=True,
            )
    return None, {"error": "dsr_eval_oom"}


def compute_ppl_score(ppl: Dict[str, float], train_len: int) -> Optional[float]:
    weighted_sum = 0.0
    weight_total = 0.0
    for key, value in ppl.items():
        length = int(key)
        if length <= train_len or value is None:
            continue
        ratio = length / float(train_len)
        weight = math.log2(ratio + 1.0)
        weighted_sum += weight * math.log(float(value))
        weight_total += weight
    if weight_total <= 0:
        return None
    return round(-weighted_sum / weight_total, 6)


def compute_passkey_score(passkey_result: Dict[str, Any]) -> Optional[float]:
    global_part = passkey_result.get("global", {})
    retrieval = global_part.get("retrieval_rate")
    ar_exact = global_part.get("ar_exact_match")
    if ar_exact is None:
        ar_exact = global_part.get("ar_exact_match_rate")
    parts = [float(x) for x in (retrieval, ar_exact) if isinstance(x, (int, float))]
    if not parts:
        return None
    return round(sum(parts) / len(parts), 6)


def compute_selection_score(
    ppl_score: Optional[float],
    passkey_score: Optional[float],
    dsr_summary: Optional[Dict[str, Any]],
) -> float:
    score = ppl_score if ppl_score is not None else -1e9
    if passkey_score is not None:
        score += 0.10 * passkey_score
    if dsr_summary and isinstance(dsr_summary.get("auc_extrap"), (int, float)):
        score += 0.10 * float(dsr_summary["auc_extrap"])
    return round(float(score), 6)


def aggregate_metric(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=1)), 6) if len(arr) > 1 else None,
        "n": int(len(arr)),
    }


def execute_run(ctx: SweepContext, spec: RunSpec) -> Dict[str, Any]:
    run_dir = run_dir_for(ctx, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_path_for(ctx, spec)
    result_path = result_path_for(ctx, spec)
    checkpoint_path = checkpoint_path_for(ctx, spec)
    spec_path = run_dir / "spec.json"
    atomic_write_json(spec_path, asdict(spec))

    if result_path.exists():
        save_status(status_path, {"status": "completed", "result_path": str(result_path)})
        return read_json(result_path, {})

    cfg = build_cfg(ctx.base_cfg, spec.seq_len, spec.num_heads)
    eff_batch = target_effective_batch(ctx.base_cfg, spec.seq_len)
    batch_plan = probe_micro_batch(ctx, cfg, spec.seq_len, spec.num_heads, eff_batch)
    batch_plan_path = run_dir / "batch_plan.json"
    atomic_write_json(batch_plan_path, asdict(batch_plan))

    status_snapshot = read_json(status_path, {})
    last_error_snapshot = str(status_snapshot.get("last_error", "")).lower()
    attempts = int(status_snapshot.get("attempts", 0))
    if "stop requested" in last_error_snapshot:
        attempts = 0
        save_status(
            status_path,
            {
                "status": "pending",
                "attempts": 0,
                "last_error": "",
            },
        )
    max_attempts = 3
    last_error = ""

    while attempts < max_attempts:
        attempts += 1
        save_status(
            status_path,
            {
                "status": "running",
                "attempts": attempts,
                "started_at": read_json(status_path, {}).get("started_at", now_ts()),
                "stage": spec.stage,
                "tau": spec.tau,
                "seq_len": spec.seq_len,
                "num_heads": spec.num_heads,
            },
        )
        log_root_event(ctx, "run_start", run_id=spec.run_id, attempt=attempts)
        log_run_event(run_dir / "events.jsonl", "run_start", attempt=attempts)

        try:
            train_t0 = time.time()
            model = train_one_run(ctx, spec, cfg, batch_plan, run_dir, checkpoint_path)
            train_elapsed = time.time() - train_t0

            eval_t0 = time.time()
            ppl = safe_eval_ppl(model, ctx.val_flat, spec.eval_lengths, spec.eval_chunks)
            passkey = safe_eval_passkey(
                model=model,
                tokenizer=ctx.tokenizer,
                filler_tokens=ctx.filler_tokens,
                lengths=spec.passkey_lengths,
                num_trials=spec.passkey_trials,
                seed=spec.seed,
            )
            dsr_curve, dsr_summary = safe_eval_dsr(
                model=model,
                spec=spec,
                tokenizer=ctx.tokenizer,
                filler_tokens=ctx.filler_tokens,
            )
            eval_elapsed = time.time() - eval_t0

            ppl_score = compute_ppl_score(ppl, spec.seq_len)
            passkey_score = compute_passkey_score(passkey)
            selection_score = compute_selection_score(ppl_score, passkey_score, dsr_summary)

            result = {
                "completed_at": now_ts(),
                "run_id": spec.run_id,
                "stage": spec.stage,
                "tier": spec.tier,
                "seed": spec.seed,
                "seq_len": spec.seq_len,
                "num_heads": spec.num_heads,
                "head_dim": spec.head_dim,
                "tau": spec.tau,
                "theory_tau": spec.theory_tau,
                "inv_freq_hash": sha256_tensor(evq_cosh_inv_freq(spec.head_dim, spec.tau)),
                "batch_plan": asdict(batch_plan),
                "train_time_sec": round(train_elapsed, 2),
                "eval_time_sec": round(eval_elapsed, 2),
                "ppl": ppl,
                "passkey": passkey,
                "dsr_curve": dsr_curve,
                "dsr_summary": dsr_summary,
                "ppl_score": ppl_score,
                "passkey_score": passkey_score,
                "selection_score": selection_score,
            }
            atomic_write_json(result_path, result)
            save_status(status_path, {"status": "completed", "result_path": str(result_path)})
            log_root_event(ctx, "run_complete", run_id=spec.run_id, selection_score=selection_score)
            log_run_event(run_dir / "events.jsonl", "run_complete", selection_score=selection_score)
            del model
            maybe_clear_device_cache()
            return result

        except TrainingOOMError as exc:
            last_error = str(exc)
            if batch_plan.micro_batch_size <= 1:
                break
            batch_plan.micro_batch_size = max(1, batch_plan.micro_batch_size // 2)
            batch_plan.grad_accum = max(1, math.ceil(batch_plan.target_effective_batch / batch_plan.micro_batch_size))
            batch_plan.effective_batch_size = batch_plan.micro_batch_size * batch_plan.grad_accum
            batch_plan.tokens_per_step = batch_plan.effective_batch_size * spec.seq_len
            atomic_write_json(batch_plan_path, asdict(batch_plan))
            save_status(
                status_path,
                {
                    "status": "retrying",
                    "last_error": last_error,
                    "micro_batch_size": batch_plan.micro_batch_size,
                    "grad_accum": batch_plan.grad_accum,
                },
            )
            log_run_event(
                run_dir / "events.jsonl",
                "oom_recovery",
                attempt=attempts,
                new_micro_batch_size=batch_plan.micro_batch_size,
                new_grad_accum=batch_plan.grad_accum,
                error=last_error,
            )
            maybe_clear_device_cache()
            continue
        except SweepError:
            raise
        except Exception as exc:  # pragma: no cover - runtime recovery
            last_error = str(exc)
            save_status(status_path, {"status": "retrying", "last_error": last_error})
            log_run_event(run_dir / "events.jsonl", "run_error", attempt=attempts, error=last_error)
            maybe_clear_device_cache()
            time.sleep(2.0 * attempts)

    save_status(status_path, {"status": "failed", "last_error": last_error})
    log_root_event(ctx, "run_failed", run_id=spec.run_id, error=last_error)
    raise SweepError(f"{spec.run_id} failed after {max_attempts} attempts: {last_error}")


def execute_specs(ctx: SweepContext, specs: List[RunSpec]) -> None:
    total = len(specs)
    for index, spec in enumerate(specs, start=1):
        result_path = result_path_for(ctx, spec)
        if result_path.exists():
            print(f"[skip] {index}/{total} {spec.run_id}", flush=True)
            continue
        print(
            f"\n{'=' * 90}\n"
            f"[run] {index}/{total} {spec.run_id}\n"
            f"{'=' * 90}",
            flush=True,
        )
        try:
            execute_run(ctx, spec)
        except SweepError as exc:
            print(f"[warn] {exc}", flush=True)
            if ctx.args.fail_fast:
                raise
        if STOP_REQUESTED:
            raise SweepError("stop requested")


def generate_report(
    ctx: SweepContext,
    pilot_specs: List[RunSpec],
    confirm_specs: List[RunSpec],
    selection: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    all_specs = pilot_specs + confirm_specs
    results = load_completed_results(all_specs, ctx)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for spec in all_specs:
        result = results.get(spec.run_id)
        if result:
            grouped[selection_key(spec)].append(result)

    summary: Dict[str, Any] = {
        "generated_at": now_ts(),
        "profile": asdict(ctx.profile),
        "environment": ctx.env_snapshot,
        "selection": selection,
        "configs": {},
    }

    lines: List[str] = []
    lines.append(f"# Phase16 Formula Optimality Sweep Report")
    lines.append("")
    lines.append(f"- Generated: {summary['generated_at']}")
    lines.append(f"- Profile: `{ctx.profile.name}`")
    lines.append(f"- Device: `{ctx.env_snapshot['device']}`")
    lines.append(f"- Active conda env: `{ctx.env_snapshot.get('active_conda_env') or 'unknown'}`")
    lines.append("")
    lines.append("## Config Summary")
    lines.append("")
    lines.append("| Config | Theory tau | Best tau | Theory rank | Complete seeds | Best mean score | |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    for key, rows in sorted(grouped.items()):
        tau_groups: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            tau_groups[float(row["tau"])].append(row)

        tau_summary: Dict[str, Any] = {}
        ranking: List[Tuple[float, Dict[str, Any]]] = []
        theory_tau = None
        for tau, tau_rows in sorted(tau_groups.items()):
            scores = [float(r["selection_score"]) for r in tau_rows if r.get("selection_score") is not None]
            tau_rows_sorted = sorted(tau_rows, key=lambda r: int(r["seed"]))
            theory_tau = theory_tau if theory_tau is not None else float(tau_rows_sorted[0]["theory_tau"])
            score_stats = aggregate_metric(scores)
            tau_summary[str(tau)] = {
                "seeds": [int(r["seed"]) for r in tau_rows_sorted],
                "selection_score": score_stats,
                "ppl": {},
                "passkey_retrieval_rate": aggregate_metric(
                    [
                        float(r["passkey"]["global"]["retrieval_rate"])
                        for r in tau_rows
                        if isinstance(r.get("passkey", {}).get("global", {}).get("retrieval_rate"), (int, float))
                    ]
                ),
                "passkey_ar_exact": aggregate_metric(
                    [
                        float(
                            r["passkey"]["global"].get(
                                "ar_exact_match",
                                r["passkey"]["global"].get("ar_exact_match_rate"),
                            )
                        )
                        for r in tau_rows
                        if isinstance(
                            r.get("passkey", {}).get("global", {}).get("ar_exact_match"),
                            (int, float),
                        )
                        or isinstance(
                            r.get("passkey", {}).get("global", {}).get("ar_exact_match_rate"),
                            (int, float),
                        )
                    ]
                ),
                "dsr_auc_extrap": aggregate_metric(
                    [
                        float((r.get("dsr_summary") or {})["auc_extrap"])
                        for r in tau_rows
                        if isinstance((r.get("dsr_summary") or {}).get("auc_extrap"), (int, float))
                    ]
                ),
            }
            first_ppl = tau_rows_sorted[0].get("ppl", {})
            for length_key in sorted(first_ppl.keys(), key=int):
                values = [
                    float(r["ppl"][length_key])
                    for r in tau_rows
                    if length_key in r.get("ppl", {})
                ]
                tau_summary[str(tau)]["ppl"][length_key] = aggregate_metric(values)
            ranking.append((tau, tau_summary[str(tau)]))

        ranking.sort(
            key=lambda item: (
                item[1]["selection_score"]["mean"]
                if item[1]["selection_score"]["mean"] is not None
                else -1e9
            ),
            reverse=True,
        )
        theory_rank = None
        best_tau = None
        for rank, (tau, _stats) in enumerate(ranking, start=1):
            if best_tau is None:
                best_tau = tau
            if theory_tau is not None and abs(float(tau) - float(theory_tau)) <= 1e-8:
                theory_rank = rank
        summary["configs"][key] = {
            "theory_tau": theory_tau,
            "best_tau": best_tau,
            "theory_rank": theory_rank,
            "taus": tau_summary,
        }
        best_stats = ranking[0][1]["selection_score"] if ranking else {"mean": None}
        n_complete = max(
            [
                len(stats["seeds"])
                for stats in tau_summary.values()
            ]
            or [0]
        )
        lines.append(
            f"| `{key}` | {theory_tau if theory_tau is not None else 'NA'} | "
            f"{best_tau if best_tau is not None else 'NA'} | "
            f"{theory_rank if theory_rank is not None else 'NA'} | "
            f"{n_complete} | "
            f"{best_stats['mean'] if best_stats['mean'] is not None else 'NA'} | |"
        )

    if not grouped:
        lines.append("")
        lines.append("No completed results yet.")

    report_md = "\n".join(lines) + "\n"
    report_path = ctx.reports_dir / "report.md"
    summary_path = ctx.reports_dir / "summary.json"
    report_path.write_text(report_md, encoding="utf-8")
    atomic_write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 16 formula optimality sweep harness")
    parser.add_argument("--profile", default="local_m4", choices=sorted(PROFILE_PRESETS))
    parser.add_argument("--mode", default="all", choices=["all", "run", "report"])
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "results" / "theory" / "phase16_formula_optimality_sweep"),
        help="Root directory for manifests, checkpoints, logs, and reports",
    )
    parser.add_argument("--dataset", default="fineweb-edu")
    parser.add_argument("--tier", default="")
    parser.add_argument("--seq-lens", default="")
    parser.add_argument("--head-counts", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--tau-multipliers", default="")
    parser.add_argument("--train-tokens", type=int, default=None)
    parser.add_argument("--val-tokens", type=int, default=None)
    parser.add_argument("--passkey-mix-ratio", type=float, default=None)
    parser.add_argument("--eval-chunks", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args)
    root = Path(args.work_root).expanduser().resolve()

    print("#" * 90)
    print("Phase 16 Formula Optimality Sweep")
    print("#" * 90)
    print(f"profile={profile.name} tier={profile.tier} device={DEVICE} dtype={DTYPE}")
    print(f"seq_lens={profile.seq_lens} head_counts={profile.head_counts} seeds={profile.seeds}")
    print(f"train_tokens={profile.train_tokens} val_tokens={profile.val_tokens}")
    print(f"work_root={root}")
    print("#" * 90, flush=True)

    with SweepLock(root / "sweep.lock"):
        ctx = build_sweep_context(args, profile)
        pilot_specs = build_pilot_specs(profile, ctx.base_cfg)
        atomic_write_json(ctx.root / "pilot_plan.json", [asdict(spec) for spec in pilot_specs])

        confirm_specs: List[RunSpec] = []
        selection_path = ctx.reports_dir / "selection.json"
        selection = read_json(selection_path, None)

        if args.mode in {"all", "run"}:
            execute_specs(ctx, pilot_specs)
            pilot_results = load_completed_results(pilot_specs, ctx)
            selection = build_confirm_selection(profile, pilot_specs, pilot_results, selection_path)
            confirm_specs = build_confirm_specs(profile, ctx.base_cfg, selection)
            atomic_write_json(ctx.root / "confirm_plan.json", [asdict(spec) for spec in confirm_specs])
            execute_specs(ctx, confirm_specs)
        elif args.mode == "report":
            if selection:
                confirm_specs = build_confirm_specs(profile, ctx.base_cfg, selection)

        summary = generate_report(ctx, pilot_specs, confirm_specs, selection)
        log_root_event(
            ctx,
            "report_written",
            report_path=str(ctx.reports_dir / "report.md"),
            summary_path=str(ctx.reports_dir / "summary.json"),
            n_configs=len(summary.get("configs", {})),
        )
        print(f"[done] report: {ctx.reports_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
