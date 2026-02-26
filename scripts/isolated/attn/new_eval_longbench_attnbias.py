#!/usr/bin/env python3
"""
Isolated evaluator for attention-integrated runs.

Runs:
- LongBench lb21 (full samples, per-sample traces)
- NIAH
- Passkey teacher-forcing

All with optional runtime attention-logit bias patch applied through model-loader monkey patching.
No existing eval script is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from transformers import AutoConfig

from attn_patch_llama_attention_bias import AttentionBiasConfig, PatchHandle, apply_llama_attention_bias_patch


MODEL_LOCK_NAME = "Meta-Llama-3-8B-Instruct"
MODEL_LOCK_MAX_POS = 8192
MODEL_LOCK_DEFAULT = "meta-llama/Meta-Llama-3-8B-Instruct"
ROPE_THETA_LOCK = 500000.0


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def assert_model_lock(base_model_path: str) -> Dict[str, object]:
    low = base_model_path.lower()
    if MODEL_LOCK_NAME.lower() not in low:
        raise RuntimeError(f"Model lock violation: expected {MODEL_LOCK_NAME}, got {base_model_path}")
    if "3.1" in low or "128k" in low:
        raise RuntimeError("Model lock violation: Llama-3.1/128K model forbidden.")
    cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
    max_pos = int(getattr(cfg, "max_position_embeddings", -1))
    theta = float(getattr(cfg, "rope_theta", ROPE_THETA_LOCK))
    if max_pos != MODEL_LOCK_MAX_POS:
        raise RuntimeError(f"Model lock violation: max_position_embeddings={max_pos}")
    if abs(theta - ROPE_THETA_LOCK) > 1e-6:
        raise RuntimeError(f"Theta lock violation: rope_theta={theta}")
    return {
        "base_model_path": base_model_path,
        "max_position_embeddings": max_pos,
        "rope_theta": theta,
    }


def _run_module_main(module_name: str, argv: list[str]) -> int:
    mod = importlib.import_module(module_name)
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        mod.main()
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    finally:
        sys.argv = old_argv
    return 0


def _patch_loader(module_name: str, cfg: AttentionBiasConfig) -> Tuple[object, object]:
    mod = importlib.import_module(module_name)
    original = mod.load_model_and_tokenizer

    def wrapped_loader(args):
        out = original(args)
        # eval_longbench returns 5-tuple; niah loader also returns 5-tuple.
        model = out[0]
        if cfg.enabled and cfg.mode != "off":
            handle: PatchHandle = apply_llama_attention_bias_patch(model, cfg)
            setattr(model, "_attn_bias_patch_handle", handle)
        return out

    mod.load_model_and_tokenizer = wrapped_loader
    return mod, original


def _restore_loader(mod: object, original: object) -> None:
    setattr(mod, "load_model_and_tokenizer", original)


def _bucket_name(n: int) -> str:
    if n <= 2000:
        return "<=2k"
    if n <= 4000:
        return "2k-4k"
    if n <= 8000:
        return "4k-8k"
    return ">8k"


def build_length_buckets(longbench_json: Path, out_path: Path) -> Dict[str, object]:
    if not longbench_json.exists():
        payload = {"error": f"missing {longbench_json}"}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    raw = json.loads(longbench_json.read_text(encoding="utf-8"))
    rows = raw.get("per_sample_scores_raw", [])
    buckets: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        tok = int(row.get("input_tokens_after_trunc", row.get("input_tokens", 0)) or 0)
        score = float(row.get("score_raw", row.get("score", 0.0)))
        b = _bucket_name(tok)
        if b not in buckets:
            buckets[b] = {"count": 0.0, "score_sum_raw": 0.0}
        buckets[b]["count"] += 1.0
        buckets[b]["score_sum_raw"] += score

    out = {}
    for b, v in buckets.items():
        c = max(1.0, v["count"])
        out[b] = {
            "count": int(v["count"]),
            "mean_raw": v["score_sum_raw"] / c,
            "mean_pct": (v["score_sum_raw"] / c) * 100.0,
        }
    payload = {"length_buckets": out}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_inv_freq_tensor(path: Path) -> "torch.Tensor":
    import torch

    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for k in ("inv_freq", "custom_inv_freq", "tensor", "data"):
            if k in raw:
                raw = raw[k]
                break
    inv = torch.as_tensor(raw, dtype=torch.float32).view(-1)
    if inv.numel() <= 1:
        raise RuntimeError(f"Invalid inv_freq payload in {path}")
    return inv


def build_s2_by_delta_from_inv_freq(inv_freq_1d: "torch.Tensor", max_delta: int) -> "torch.Tensor":
    import torch

    inv = inv_freq_1d.detach().to(dtype=torch.float32, device="cpu").view(-1)
    d = torch.arange(int(max_delta), dtype=torch.float32).view(-1, 1)
    s = torch.cos(d * inv.view(1, -1)).mean(dim=1)
    return (s * s).clamp_min(1e-8)


def resolve_import_roots(this_file: Path) -> Tuple[Path, Path]:
    # Prefer robust discovery: locate scripts/eval_longbench.py from current path.
    for parent in [this_file.parent] + list(this_file.parents):
        scripts_candidate = parent / "scripts"
        if (scripts_candidate / "eval_longbench.py").exists():
            return scripts_candidate, parent
    # Backward-compatible fallback for current repository layout.
    scripts_root = this_file.resolve().parents[2]
    return scripts_root, scripts_root.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate longbench/niah/passkey with optional attention bias patch.")
    ap.add_argument("--base_model_path", type=str, default=MODEL_LOCK_DEFAULT)
    ap.add_argument("--adapter_path", type=str, required=True)
    ap.add_argument("--custom_inv_freq_path", type=str, default="")
    ap.add_argument("--output_root", type=Path, default=Path("artifacts/new_attnbias_v1/eval"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_input_tokens", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument(
        "--max_batch_input_tokens",
        type=int,
        default=0,
        help=(
            "Soft token budget per generation micro-batch forwarded to eval_longbench. "
            "0 means batch_size * max_input_tokens."
        ),
    )
    ap.add_argument("--attn_implementation", type=str, default="auto")
    ap.add_argument("--longbench_local_data_dir", type=str, default="/root/autodl-tmp/dfrope/ms_datasets/LongBench/data")
    ap.add_argument("--attn_bias_mode", type=str, choices=["off", "bias", "bias+gate"], default="off")
    ap.add_argument("--gamma_mode", type=str, choices=["constant", "per-layer", "head-group"], default="constant")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--gamma_by_layer", type=str, default="")
    ap.add_argument("--gamma_head_low", type=float, default=0.0)
    ap.add_argument("--gamma_head_high", type=float, default=0.0)
    ap.add_argument("--gate_tau", type=float, default=0.0)
    ap.add_argument("--gate_tg", type=float, default=1.0)
    ap.add_argument("--s2_power", type=float, default=2.0)
    ap.add_argument("--s2_table_path", type=str, default="")
    ap.add_argument("--require_s2_table", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    if int(args.max_input_tokens) > int(MODEL_LOCK_MAX_POS):
        print(
            f"[WARNING] max_input_tokens={args.max_input_tokens} exceeds model lock {MODEL_LOCK_MAX_POS}; "
            f"clamping to {MODEL_LOCK_MAX_POS}."
        )
        args.max_input_tokens = int(MODEL_LOCK_MAX_POS)

    if args.attn_bias_mode != "off":
        if args.attn_implementation in {"auto", "flash_attention_2", "flash_attention_3"}:
            print(
                "[WARNING] attn_bias_mode enabled; forcing attn_implementation=sdpa "
                "to avoid flash-attention additive-mask incompatibility while keeping throughput."
            )
            args.attn_implementation = "sdpa"
        if int(args.batch_size) > 1:
            print("[WARNING] attn_bias_mode enabled; forcing batch_size=1 for eval stability.")
            args.batch_size = 1

    model_info = assert_model_lock(args.base_model_path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    traces_dir = args.output_root / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    resolved_s2_table_path = str(args.s2_table_path).strip()
    if args.attn_bias_mode != "off":
        if not resolved_s2_table_path:
            if not str(args.custom_inv_freq_path).strip():
                if bool(args.require_s2_table):
                    raise RuntimeError(
                        "attn_bias_mode requires s2 table: provide --s2_table_path or --custom_inv_freq_path."
                    )
            else:
                inv = load_inv_freq_tensor(Path(args.custom_inv_freq_path))
                auto_s2 = build_s2_by_delta_from_inv_freq(inv, max_delta=int(args.max_input_tokens))
                auto_s2_path = traces_dir / "s2_by_delta.pt"
                import torch

                torch.save(
                    {
                        "s2_by_delta": auto_s2,
                        "meta": {
                            "source": "from_custom_inv_freq",
                            "max_delta": int(args.max_input_tokens),
                            "custom_inv_freq_path": str(args.custom_inv_freq_path),
                            "custom_inv_freq_sha256": sha256_file(Path(args.custom_inv_freq_path)),
                        },
                    },
                    auto_s2_path,
                )
                resolved_s2_table_path = auto_s2_path.as_posix()
        elif bool(args.require_s2_table) and (not Path(resolved_s2_table_path).exists()):
            raise FileNotFoundError(f"s2_table_path not found: {resolved_s2_table_path}")

    cfg = AttentionBiasConfig(
        mode=args.attn_bias_mode,
        gamma_mode=args.gamma_mode,
        gamma=float(args.gamma),
        gamma_by_layer=args.gamma_by_layer,
        gamma_head_low=float(args.gamma_head_low),
        gamma_head_high=float(args.gamma_head_high),
        tau=float(args.gate_tau),
        tg=float(args.gate_tg),
        s2_power=float(args.s2_power),
        s2_table_path=resolved_s2_table_path,
        require_s2_table=bool(args.require_s2_table),
        enabled=(args.attn_bias_mode != "off"),
    )

    scripts_root, repo_root = resolve_import_roots(Path(__file__).resolve())
    for p in [scripts_root, repo_root]:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

    lb_mod, lb_orig_loader = _patch_loader("eval_longbench", cfg)
    niah_mod, niah_orig_loader = _patch_loader("eval_niah_recall", cfg)
    try:
        longbench_json = args.output_root / "longbench_lb21.json"
        longbench_manifest = args.output_root / "longbench_manifest.json"
        repro_manifest = args.output_root / "repro_manifest_longbench"
        lb_argv = [
            "eval_longbench.py",
            "--base_model_path",
            args.base_model_path,
            "--adapter_path",
            args.adapter_path,
            "--model_alias",
            "attnbias_eval",
            "--skip_base_unfinetuned",
            "--task_set",
            "lb21",
            "--max_samples_per_task",
            "0",
            "--max_input_tokens",
            str(args.max_input_tokens),
            "--batch_size",
            str(args.batch_size),
            "--max_batch_input_tokens",
            str(int(args.max_batch_input_tokens)),
            "--prompt_source",
            "official",
            "--chat_template",
            "auto",
            "--truncate_mode",
            "middle",
            "--max_new_tokens_policy",
            "official",
            "--score_scale",
            "pct",
            "--strict_parity_check",
            "--seed",
            str(args.seed),
            "--longbench_local_data_dir",
            args.longbench_local_data_dir,
            "--save_per_sample_traces",
            "1",
            "--trace_output_max_chars",
            "1024",
            "--manifest_json",
            str(longbench_manifest),
            "--repro_manifest_dir",
            str(repro_manifest),
            "--attn_implementation",
            args.attn_implementation,
            "--output_json",
            str(longbench_json),
        ]
        if args.custom_inv_freq_path.strip():
            lb_argv.extend(["--custom_inv_freq_path", args.custom_inv_freq_path])
        rc_lb = _run_module_main("eval_longbench", lb_argv)

        niah_dir = args.output_root / "niah"
        niah_argv = [
            "eval_niah_recall.py",
            "--base_model_path",
            args.base_model_path,
            "--adapter_path",
            args.adapter_path,
            "--variant",
            "custom" if args.custom_inv_freq_path.strip() else "base",
            "--output_dir",
            str(niah_dir),
            "--lengths",
            "2048,4096,8192",
            "--depths",
            "10,50,90",
            "--trials_per_cell",
            "1",
            "--seed",
            str(args.seed),
            "--attn_implementation",
            args.attn_implementation,
        ]
        if args.custom_inv_freq_path.strip():
            niah_argv.extend(["--custom_inv_freq_path", args.custom_inv_freq_path])
        rc_niah = _run_module_main("eval_niah_recall", niah_argv)

        # eval_passkey_teacher_forcing imports loader from eval_niah_recall;
        # since eval_niah_recall loader is already patched, passkey path gets the same patched model.
        passkey_dir = args.output_root / "passkey"
        passkey_argv = [
            "eval_passkey_teacher_forcing.py",
            "--base_model_path",
            args.base_model_path,
            "--adapter_path",
            args.adapter_path,
            "--variant",
            "custom" if args.custom_inv_freq_path.strip() else "base",
            "--output_dir",
            str(passkey_dir),
            "--lengths",
            "2048,4096,8192",
            "--depths",
            "10,50,90",
            "--trials_per_cell",
            "20",
            "--seed",
            str(args.seed),
            "--attn_implementation",
            args.attn_implementation,
        ]
        if args.custom_inv_freq_path.strip():
            passkey_argv.extend(["--custom_inv_freq_path", args.custom_inv_freq_path])
        rc_passkey = _run_module_main("eval_passkey_teacher_forcing", passkey_argv)

    finally:
        _restore_loader(lb_mod, lb_orig_loader)
        _restore_loader(niah_mod, niah_orig_loader)

    length_bucket_path = args.output_root / "length_buckets.json"
    bucket_payload = build_length_buckets(args.output_root / "longbench_lb21.json", length_bucket_path)

    agg = {
        "timestamp": now(),
        "model_lock": model_info,
        "adapter_path": args.adapter_path,
        "custom_inv_freq_path": args.custom_inv_freq_path,
        "attn_bias": vars(cfg),
        "s2_table_path_resolved": resolved_s2_table_path,
        "eval_batching": {
            "batch_size": int(args.batch_size),
            "max_batch_input_tokens": int(args.max_batch_input_tokens),
            "max_input_tokens": int(args.max_input_tokens),
        },
        "return_codes": {"longbench": rc_lb, "niah": rc_niah, "passkey": rc_passkey},
        "outputs": {
            "longbench_json": (args.output_root / "longbench_lb21.json").as_posix(),
            "length_buckets_json": length_bucket_path.as_posix(),
            "niah_dir": (args.output_root / "niah").as_posix(),
            "passkey_dir": (args.output_root / "passkey").as_posix(),
            "traces_dir": traces_dir.as_posix(),
        },
        "length_buckets": bucket_payload.get("length_buckets", {}),
        "code_hashes": {
            "new_eval_longbench_attnbias.py": sha256_file(Path(__file__)),
            "attn_patch_llama_attention_bias.py": sha256_file(Path(__file__).resolve().parent / "attn_patch_llama_attention_bias.py"),
        },
    }
    (args.output_root / "results.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] results -> {(args.output_root / 'results.json').as_posix()}")


if __name__ == "__main__":
    main()
