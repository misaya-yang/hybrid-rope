#!/usr/bin/env python3
"""External-control evaluation (Qwen2.5 family) on round-12 tasks.

Two input modes:
  legacy (default): rows carry OLMo ids; OLMo decode -> text -> target encode.
  --native-tasks: prompt_ids are native target-tokenizer ids (round-12 EXT
    tasks, built at exact target token counts for matched 2x/4x factors).

Optional YaRN injection (--yarn-factor F): before load, config.rope_scaling
gets rope_type=yarn, factor F, original_max_position_embeddings=
--original-max-pos, beta_fast=32/beta_slow=1, and max_position_embeddings is
raised to original_max_pos*F (keeps the validator identity factor =
post/pre context ratio). This is exactly the shipped Qwen 128K recipe:
native 32K + static YaRN factor 4, NO fine-tuning.

Attention is SDPA with only FLASH_ATTENTION / EFFICIENT_ATTENTION kernels
permitted (the MATH backend is O(n^2) memory and never allowed at 64K+).
Rows that OOM are recorded as skipped_oom (scored zero) and the run continues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import scoring


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@torch.inference_mode()
def generate_one(model, tok, prompt_ids, budget, device, chunk=2048):
    """Greedy decode with CHUNKED prefill.

    A monolithic 128K prefill spikes one layer's FFN intermediates
    (intermediate_size x seq_len, ~10GB at 7B) on top of weights + full KV
    and OOMs a 32GB card; even with chunks, lm_head over a whole chunk is
    chunk x vocab x 2B (~5GB at 16K), and HF materializes a 4D causal mask
    (q_len x kv_len, fp32) for SDPA whenever a chunk attends to cached
    prefix — ~8.6GB at q=16K/kv=128K. So prefill runs through the BASE
    model (hidden states only, KV grows incrementally) with small chunks
    (2048: mask <= ~1GB) and lm_head is applied to the final position
    alone. Semantics are identical to greedy generate(): argmax per step,
    stop at any EOS/special terminator.
    """
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    max_new = int(budget)
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    extra = getattr(tok, "additional_special_tokens_ids", None) or []
    # Qwen2.5 family: im_end/im_sep-style specials commonly terminate answers.
    for tid in extra:
        eos_ids.add(tid)

    base = model.model          # Qwen2Model: hidden states, no lm_head
    lm_head = model.lm_head

    L = input_ids.shape[1]
    past = None
    hidden = None
    for start in range(0, L, chunk):
        out = base(input_ids[:, start:start + chunk], past_key_values=past,
                   use_cache=True)
        past = out.past_key_values
        hidden = out.last_hidden_state

    new_tokens = []
    next_id = int(lm_head(hidden[0, -1, :]).argmax(-1).item())
    for _ in range(max_new):
        new_tokens.append(next_id)
        if next_id in eos_ids:
            break
        step = base(torch.tensor([[next_id]], dtype=torch.long, device=device),
                    past_key_values=past, use_cache=True)
        past = step.past_key_values
        next_id = int(lm_head(step.last_hidden_state[0, -1, :]).argmax(-1).item())

    ended_eos = bool(new_tokens) and new_tokens[-1] in eos_ids
    stop = "eos" if ended_eos else ("budget" if len(new_tokens) >= max_new else "other")
    text = tok.decode(new_tokens[:-1] if ended_eos else new_tokens,
                      skip_special_tokens=False)
    return text, ended_eos, stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--prompt-tokenizer", default=None,
                    help="tokenizer that built prompt_ids (legacy round-trip mode)")
    ap.add_argument("--native-tasks", action="store_true",
                    help="prompt_ids are native target-tokenizer ids")
    ap.add_argument("--chat-template", action="store_true",
                    help="wrap each row as a single user turn via "
                         "apply_chat_template + add_generation_prompt "
                         "(REQUIRED for Instruct models: raw completion mode "
                         "collapses into filler continuation at ALL lengths)")
    ap.add_argument("--families", nargs="*", default=["single_evidence"])
    ap.add_argument("--lengths", type=int, nargs="*", default=None)
    ap.add_argument("--yarn-factor", type=float, default=None)
    ap.add_argument("--original-max-pos", type=int, default=32768)
    ap.add_argument("--context-max", type=int, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-rows-per-cell", type=int, default=10**9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.native_tasks and args.prompt_tokenizer:
        raise SystemExit("--native-tasks and --prompt-tokenizer are mutually exclusive")
    if not args.native_tasks and not args.prompt_tokenizer:
        raise SystemExit("legacy round-trip mode requires --prompt-tokenizer")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("output complete already; preserved, not overwritten")

    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.model)
    # transformers 5.x standardized rope_theta INTO the rope_scaling/rope_parameters
    # dict; replacing the dict wholesale drops it (yarn then gets base=None).
    base_theta = None
    for src in (getattr(cfg, "rope_scaling", None), getattr(cfg, "rope_parameters", None)):
        if isinstance(src, dict) and src.get("rope_theta") is not None:
            base_theta = float(src["rope_theta"])
            break
    if base_theta is None:
        base_theta = float(getattr(cfg, "rope_theta", 10000.0))
    yarn_cfg = None
    if args.yarn_factor is not None:
        yarn_cfg = {
            "rope_type": "yarn",
            "factor": float(args.yarn_factor),
            "original_max_position_embeddings": int(args.original_max_pos),
            "beta_fast": 32,
            "beta_slow": 1,
            "rope_theta": base_theta,
        }
        cfg.rope_scaling = dict(yarn_cfg)
        cfg.max_position_embeddings = int(args.original_max_pos * args.yarn_factor)

    tok = AutoTokenizer.from_pretrained(args.model)
    src_tok = None if args.native_tasks else AutoTokenizer.from_pretrained(args.prompt_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.eval().to(args.device)

    # --- Runtime YaRN audit: the injected config must have taken effect. ---
    rotary = model.model.rotary_emb
    inv_freq = rotary.inv_freq.detach().float().cpu().numpy()
    audit = {
        "rotary_class": type(rotary).__name__,
        "rope_theta": base_theta,
        "inv_freq_head3": inv_freq[:3].tolist(),
        "inv_freq_tail3": inv_freq[-3:].tolist(),
        "attention_factor": getattr(rotary, "attention_factor", None),
        "attention_scaling": getattr(rotary, "attention_scaling", None),
    }
    if args.yarn_factor is not None:
        import math
        import numpy as np
        dim = inv_freq.shape[0] * 2
        base = base_theta
        L0 = float(args.original_max_pos)
        # Expected inv_freq EXACTLY per HF modeling_rope_utils._compute_yarn_parameters
        # (float32 arithmetic, linear ramp in dim-index space, truncated bounds):
        pos = np.float32(base) ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim))
        extrap = (1.0 / pos).astype(np.float32)
        interp = (1.0 / (np.float32(args.yarn_factor) * pos)).astype(np.float32)

        def find_correction_dim(num_rotations):
            return dim * math.log(L0 / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        low_r, high_r = find_correction_dim(32), find_correction_dim(1)
        low = max(math.floor(low_r), 0)
        high = min(math.ceil(high_r), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
        expected = interp * ramp + extrap * (1 - ramp)
        dev = float(np.abs(inv_freq - expected).max() / float(expected.max()))
        audit["yarn_bounds_low_high"] = [int(low), int(high)]
        audit["inv_freq_vs_hf_yarn_formula_max_rel_dev"] = dev
        # Informational: deviation vs the smoothstep-in-wavelength variant
        # (the one Y2 used on OLMo).
        omega64 = base ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
        wl = 2.0 * np.pi / omega64
        u = np.clip((wl - L0 / 32.0) / (L0 - L0 / 32.0), 0.0, 1.0)
        gamma = u * u * (3.0 - 2.0 * u)
        smooth = (omega64 * (1.0 - gamma) + (omega64 / args.yarn_factor) * gamma).astype(np.float32)
        audit["inv_freq_vs_smoothstep_variant_max_rel_dev"] = \
            float(np.abs(inv_freq - smooth).max() / float(expected.max()))
        assert dev < 1e-5, \
            f"rotary inv_freq does not match the HF YaRN blend (rel dev {dev}); " \
            "refusing to evaluate without the intended extrapolation config"

    rows = [json.loads(l) for l in Path(args.tasks).read_text().splitlines()]
    if args.families:
        rows = [r for r in rows if r["family"] in set(args.families)]
    if args.lengths:
        rows = [r for r in rows if int(r["length_cap"]) in set(args.lengths)]
    if args.smoke:
        rows = rows[:2]

    ctx_cap = args.context_max
    if ctx_cap is None and args.yarn_factor is not None:
        ctx_cap = int(args.original_max_pos * args.yarn_factor)
    if ctx_cap is None and args.lengths:
        ctx_cap = int(args.lengths[-1])

    from torch.nn.attention import SDPBackend, sdpa_kernel
    BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]

    examples_path = out / "examples.jsonl"
    t0 = time.time()
    n_ok = n_trunc = n_oom = 0
    with sdpa_kernel(BACKENDS), examples_path.open("w") as fh:
        for idx, row in enumerate(rows):
            if args.native_tasks:
                ids = list(row["prompt_ids"])
            else:
                text_prompt = src_tok.decode(row["prompt_ids"], skip_special_tokens=True)
                enc = tok(text_prompt, add_special_tokens=False, return_tensors="pt")
                ids = enc["input_ids"][0].tolist()
            if args.chat_template:
                text = tok.decode(ids, skip_special_tokens=False)
                ct = tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=True)
                # transformers>=5.15 returns a BatchEncoding (dict-like),
                # not a bare id list; list() on it yields the key strings.
                if hasattr(ct, "input_ids"):
                    ct = ct["input_ids"]
                    if ct and isinstance(ct[0], (list, tuple)):
                        ct = ct[0]
                ids = list(ct)
            # Track A convention: the row fills the context exactly and
            # generation continues ABOVE it (positions length..length+budget);
            # only a hard overflow (round-trip inflation) gets truncated.
            # Chat-template mode is exempt: the template adds a fixed ~11-token
            # envelope that must stay intact (yarn rotary extends past
            # max_pos dynamically for the envelope + budget).
            if not args.chat_template and ctx_cap is not None and len(ids) > ctx_cap:
                ids = ids[:ctx_cap]
                n_trunc += 1
            try:
                text, eos, stop = generate_one(model, tok, ids,
                                               row["generation_budget"], args.device)
            except RuntimeError as e:
                oom = isinstance(e, torch.cuda.OutOfMemoryError) or \
                    "out of memory" in str(e).lower()
                if not oom:
                    raise
                n_oom += 1
                torch.cuda.empty_cache()
                rec = scoring.score_row(row, "", False, "oom_skip")
                rec.update(model=args.model_id, system=args.model_id,
                           input_tokens=len(ids), skipped_oom=True)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_ok += 1
                print(f"[{idx}/{len(rows)}] OOM_SKIP L{row['length_cap']}", flush=True)
                continue
            rec = scoring.score_row(row, text, eos, stop)
            rec.update(model=args.model_id, system=args.model_id,
                       input_tokens=len(ids))
            if not args.native_tasks:
                rec["src_input_tokens"] = len(row["prompt_ids"])
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_ok += 1
            if idx % 8 == 0:
                print(f"[{idx}/{len(rows)}] {rec['family']} L{rec['length_cap']} "
                      f"strict={rec['strict_exact_eos']} f1={rec['qa_f1']}", flush=True)

    from collections import defaultdict
    cells = defaultdict(list)
    for line in examples_path.read_text().splitlines():
        r = json.loads(line)
        cells[(r["family"], r["layout"], r["length_cap"])].append(r)
    summary = {}
    for key, rs in sorted(cells.items()):
        groups = {}
        for r in rs:
            groups.setdefault(r["group_id"], {})[r["world"]] = r["strict_exact_eos"]
        summary[f"{key[0]}:{key[1]}:{key[2]}"] = {
            "rows": len(rs),
            "groups": len(groups),
            "strict_rows": sum(1 for r in rs if r["strict_exact_eos"]),
            "strict_groups": sum(1 for ws in groups.values() if all(ws.values())),
            "ruler_official": sum(r["ruler_official_contains"] for r in rs),
            "qa_em": sum(r["qa_em"] for r in rs),
            "qa_f1_mean": round(sum(r["qa_f1"] for r in rs) / max(1, len(rs)), 4),
            "eos_rate": round(sum(r["ended_with_eos"] for r in rs) / max(1, len(rs)), 3),
            "lenient_contains": sum(r["lenient_contains_any_gold"] for r in rs),
            "oom_skipped": sum(1 for r in rs if r.get("skipped_oom")),
        }
    manifest = {
        "status": "CROSS_FAMILY_EVAL_COMPLETE_V2",
        "model": args.model, "model_id": args.model_id,
        "mode": "native_tasks" if args.native_tasks else
                f"round_trip(OLMo decode -> target encode, src={args.prompt_tokenizer})",
        "chat_template": bool(args.chat_template),
        "yarn": yarn_cfg,
        "config_max_position_embeddings": int(cfg.max_position_embeddings),
        "rotary_audit": audit,
        "attention": "sdpa; kernels forced to FLASH_ATTENTION+EFFICIENT_ATTENTION "
                     "(MATH backend disallowed)",
        "context_max": ctx_cap,
        "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "n_rows": n_ok, "n_prompt_truncated": n_trunc, "n_oom_skipped": n_oom,
        "wall_seconds": round(time.time() - t0, 1),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "summary": summary,
        "tasks_sha256": sha256_bytes(Path(args.tasks).read_bytes()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
