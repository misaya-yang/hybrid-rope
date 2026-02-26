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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate longbench/niah/passkey with optional attention bias patch.")
    ap.add_argument("--base_model_path", type=str, default=MODEL_LOCK_DEFAULT)
    ap.add_argument("--adapter_path", type=str, required=True)
    ap.add_argument("--custom_inv_freq_path", type=str, default="")
    ap.add_argument("--output_root", type=Path, default=Path("artifacts/new_attnbias_v1/eval"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_input_tokens", type=int, default=16384)
    ap.add_argument("--batch_size", type=int, default=2)
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
    args = ap.parse_args()

    model_info = assert_model_lock(args.base_model_path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    traces_dir = args.output_root / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    cfg = AttentionBiasConfig(
        mode=args.attn_bias_mode,
        gamma_mode=args.gamma_mode,
        gamma=float(args.gamma),
        gamma_by_layer=args.gamma_by_layer,
        gamma_head_low=float(args.gamma_head_low),
        gamma_head_high=float(args.gamma_head_high),
        tau=float(args.gate_tau),
        tg=float(args.gate_tg),
        enabled=(args.attn_bias_mode != "off"),
    )

    repo_root = Path(__file__).resolve().parent
    scripts_root = repo_root / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))

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
