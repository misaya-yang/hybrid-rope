#!/usr/bin/env python3
"""Precompute the frozen native teacher's KL target distributions (one-shot).

native_pool_v3's 512 train rows ALL carry prediction_positions (verified:
512/512, 2-42 positions/row), so track_b_train.py's kl_replay_loss NEVER
enters its random-resample branch on this pool: the teacher target is a pure
function of (frozen teacher, row, positions). Precompute it once:

  for each train row: ids = row[:2048] (V1 truncation); if len >= 8 and any
  position survives (1 <= p < L): one teacher forward, gather logits at the
  surviving positions, softmax in fp32, store bf16.

The trainer (track_b_train_v2.py) loads this cache into RAM (~0.6GB) and
computes KL(teacher || student) without ever loading the teacher — required
for 7B on a 32GB card (student + teacher would not co-reside with 16K
activations). Math is identical to V1 live KL up to bf16 storage rounding
(verified at build: one row recomputed and compared).

Usage:
  python build_kl_cache.py --model <olmo dir> \
      --replay-manifest <native_pool_v3/manifest.json> \
      --out <kl_cache_olmo7b.npz>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--replay-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.model.rotary_emb.attention_scaling = 1.0
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)

    mp = Path(args.replay_manifest)
    m = json.loads(mp.read_text())
    assert m.get("status") == "NATIVE_REPLAY_POOL_V1"
    rows_file = mp.parent / m["rows_path"]
    assert hashlib.sha256(rows_file.read_bytes()).hexdigest() == m["rows_sha256"]

    arrays, index = {}, []
    with torch.inference_mode():
        for i, line in enumerate(rows_file.read_text().splitlines()):
            r = json.loads(line)
            if r.get("split", "train") != "train":
                continue
            ids = r.get("input_ids") or (list(r["prompt_ids"]) + list(r["target_ids"]))
            ids = ids[:args.max_len]
            L = len(ids)
            cand = [p for p in (r.get("prediction_positions") or []) if 1 <= p < L]
            if L < 8 or not cand:
                index.append({"row": i, "n_pos": 0})
                continue
            t = torch.tensor([ids], dtype=torch.long, device=args.device)
            pos = torch.tensor(cand, dtype=torch.long, device=args.device)
            h = model.model(input_ids=t).last_hidden_state
            logits = F.linear(h[0, pos], model.lm_head.weight.to(h.dtype))
            probs = F.softmax(logits.float(), dim=-1).to(torch.bfloat16).cpu().numpy()
            arrays[f"p_{i}"] = probs
            index.append({"row": i, "n_pos": len(cand),
                          "positions": cand})
            if i % 64 == 0:
                print(f"cached row {i} ({len(cand)} pos, {L} tok)", flush=True)

    arrays["index_json"] = np.frombuffer(
        json.dumps(index).encode(), dtype=np.uint8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    # ---- verify: recompute row 0 (or first non-empty) and compare ----------
    first = next(ix for ix in index if ix["n_pos"] > 0)
    line = [l for l in rows_file.read_text().splitlines()][first["row"]]
    r = json.loads(line)
    ids = (r.get("input_ids") or (list(r["prompt_ids"]) + list(r["target_ids"])))[
        :args.max_len]
    pos = torch.tensor(first["positions"], dtype=torch.long, device=args.device)
    with torch.inference_mode():
        h = model.model(input_ids=torch.tensor([ids], device=args.device)
                        ).last_hidden_state
        logits = F.linear(h[0, pos], model.lm_head.weight.to(h.dtype))
        re = F.softmax(logits.float(), dim=-1).to(torch.bfloat16).cpu().numpy()
    dev = float(np.abs(re.astype(np.float32)
                       - arrays[f"p_{first['row']}"].astype(np.float32)).max())
    meta = {"status": "KL_TEACHER_CACHE_V1", "model": args.model,
            "replay_manifest_sha256": m["rows_sha256"],
            "n_rows": len(index),
            "n_rows_with_positions": sum(1 for ix in index if ix["n_pos"] > 0),
            "max_len": args.max_len, "storage": "bf16 probs, softmax fp32",
            "recompute_check_row": first["row"], "recompute_max_abs_dev": dev,
            "sha256": sha}
    assert dev == 0.0, f"cache recompute mismatch {dev}"
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print("KL_CACHE_BUILT", flush=True)


if __name__ == "__main__":
    main()
