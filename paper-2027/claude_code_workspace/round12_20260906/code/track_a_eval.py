#!/usr/bin/env python3
"""Track A: zero-training static position-system evaluation (route A of the plan).

Fixed matrix: model in {OLMo-2-0425-1B-Instruct, OLMo-2-1124-7B-Instruct} x
system in {N native, Z, Y YaRN-s4, M MrRoPE-Pro-s4}. One static table per arm,
same table+gain at every request length; NO native routing; frozen checkpoints,
zero training. Greedy decoding; raw outputs and ALL declared scores saved.

Table install is byte-compatible with frozen engine code_release_008:
  inv_freq := table (float32, K pairs); attention_scaling := rotary_amplitude.
Position x frequency and sin/cos are computed in FP32 inside the model's rotary
forward (transformers forces float32 there).

Usage:
  python track_a_eval.py --model /path/to/olmo --model-id olmo7b \
      --tables /path/round12_tables --arm M \
      --tasks /path/round12_tasks.jsonl \
      --families ruler_single_key ruler_multi_key single_evidence \
      --lengths 4096 8192 16384 \
      --output /path/out_dir [--max-rows-per-cell 64]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import scoring


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_and_install(model_path: str, tables_dir: Path, arm: str,
                     adapter_dir: str | None = None, extra_path: str | None = None,
                     dtype=torch.bfloat16):
    """Load base model, install the frozen static table + gain, and (optionally,
    for trained-product evaluation) wrap with the saved LoRA adapter and load
    the norm/embedding deltas. The table is installed on the BASE model before
    any PEFT wrapping; PEFT never touches the rotary."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = json.loads((tables_dir / "manifest_round12.json").read_text())
    assert manifest["status"] == "ROUND12_STATIC_TABLES_FROZEN_V1", "table manifest not frozen"
    entry = manifest["arms"][arm]
    table = np.load(tables_dir / entry["path"], allow_pickle=False)
    assert table.dtype == np.float32
    assert sha256_bytes(np.ascontiguousarray(table).tobytes()) == entry["float32_sha256"], \
        "table hash mismatch vs frozen manifest"
    gain = float(entry["rotary_amplitude"])

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.eval()

    rotary = model.model.rotary_emb
    assert rotary.inv_freq.shape == table.shape, \
        f"inv_freq shape {rotary.inv_freq.shape} != table {table.shape} (head_dim mismatch?)"
    rotary.inv_freq.copy_(torch.from_numpy(table).to(rotary.inv_freq))
    rotary.original_inv_freq = rotary.inv_freq.detach().clone()
    rotary.attention_scaling = gain

    adapter_sha = None
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        model.eval()
        adapter_sha = sha256_bytes(
            (Path(adapter_dir) / "adapter_model.safetensors").read_bytes()) \
            if (Path(adapter_dir) / "adapter_model.safetensors").exists() else None
        if extra_path:
            from safetensors.torch import load_file
            extra = load_file(extra_path)
            names = dict(model.named_parameters())
            for k, v in extra.items():
                assert k in names, f"extra param {k} not found in wrapped model"
                names[k].data.copy_(v.to(names[k].dtype))
                names[k].requires_grad_(False)

    # Runtime identity check (mirrors engine drift guard).
    rotary2 = model.get_base_model().model.rotary_emb if adapter_dir else model.model.rotary_emb
    actual = rotary2.inv_freq.detach().float().cpu().numpy()
    assert sha256_bytes(np.ascontiguousarray(actual, dtype=np.float32).tobytes()) \
        == entry["float32_sha256"], "runtime rotary tensor drift"
    assert float(rotary2.attention_scaling) == gain, "runtime gain drift"
    return model, tok, {"arm": arm, "table_sha256": entry["float32_sha256"],
                        "gain": gain, "manifest_status": manifest["status"],
                        "adapter_dir": adapter_dir, "adapter_sha256": adapter_sha,
                        "extra_path": extra_path}


@torch.inference_mode()
def generate_one(model, tok, prompt_ids: list[int], budget: int, device: str):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    max_new = int(budget)
    out = model.generate(input_ids, do_sample=False, max_new_tokens=max_new,
                         pad_token_id=tok.eos_token_id, return_dict_in_generate=False)
    new_tokens = out[0, input_ids.shape[1]:].tolist()
    ended_eos = bool(new_tokens and new_tokens[-1] == tok.eos_token_id)
    stop = "eos" if ended_eos else ("budget" if len(new_tokens) >= max_new else "other")
    if new_tokens and new_tokens[-1] == tok.eos_token_id:
        new_tokens = new_tokens[:-1]
    text = tok.decode(new_tokens, skip_special_tokens=False)
    return text, ended_eos, stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", required=True, help="e.g. olmo1b / olmo7b")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--arm", required=True, choices=["N", "Z", "Y", "M", "Y2"])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--lengths", type=int, nargs="*", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-rows-per-cell", type=int, default=10**9)
    ap.add_argument("--adapter", default=None,
                    help="optional PEFT adapter dir for trained-product eval")
    ap.add_argument("--extra", default=None,
                    help="optional extra_norm_embedding.safetensors path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="2 rows only, for runtime checks")
    ap.add_argument("--chat-template", action="store_true",
                    help="audit mode: wrap each row as a single user turn via "
                         "apply_chat_template + add_generation_prompt (Instruct "
                         "models collapse in raw completion mode on Qwen; audit "
                         "whether OLMo Track A raw-mode results are affected)")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("output complete already; preserved, not overwritten")

    tables_dir = Path(args.tables)
    model, tok, identity = load_and_install(args.model, tables_dir, args.arm)
    model.to(args.device)

    rows = [json.loads(l) for l in Path(args.tasks).read_text().splitlines()]
    if args.families:
        rows = [r for r in rows if r["family"] in set(args.families)]
    if args.lengths:
        rows = [r for r in rows if int(r["length_cap"]) in set(args.lengths)]
    if args.smoke:
        rows = rows[:2]

    examples_path = out / "examples.jsonl"
    t0 = time.time()
    n_ok = 0
    with examples_path.open("w") as fh:
        for idx, row in enumerate(rows):
            ids = list(row["prompt_ids"])
            if args.chat_template:
                text_prompt = tok.decode(ids, skip_special_tokens=False)
                ct = tok.apply_chat_template(
                    [{"role": "user", "content": text_prompt}],
                    add_generation_prompt=True, tokenize=True)
                # transformers>=5.15 returns a BatchEncoding, not a bare id
                # list; list() on it yields the key strings.
                if hasattr(ct, "input_ids"):
                    ct = ct["input_ids"]
                    if ct and isinstance(ct[0], (list, tuple)):
                        ct = ct[0]
                ids = list(ct)
            text, eos, stop = generate_one(model, tok, ids,
                                           row["generation_budget"], args.device)
            rec = scoring.score_row(row, text, eos, stop)
            rec.update(model=args.model_id, system=args.arm,
                       input_tokens=len(ids))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_ok += 1
            if idx % 16 == 0:
                print(f"[{idx}/{len(rows)}] {rec['family']} L{rec['length_cap']} "
                      f"strict={rec['strict_exact_eos']} ruler={rec['ruler_official_contains']} "
                      f"f1={rec['qa_f1']}", flush=True)

    # Summary: per family/layout/length cell
    from collections import Counter, defaultdict
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
            "strict_groups": sum(1 for worlds in groups.values() if all(worlds.values())),
            "ruler_official": sum(r["ruler_official_contains"] for r in rs),
            "qa_em": sum(r["qa_em"] for r in rs),
            "qa_f1_mean": round(sum(r["qa_f1"] for r in rs) / max(1, len(rs)), 4),
            "eos_rate": round(sum(r["ended_with_eos"] for r in rs) / max(1, len(rs)), 3),
            "lenient_contains": sum(r["lenient_contains_any_gold"] for r in rs),
        }
    manifest = {
        "status": "TRACKA_EVAL_COMPLETE",
        "model": args.model, "model_id": args.model_id,
        "arm_identity": identity,
        "chat_template": bool(args.chat_template),
        "n_rows": n_ok, "wall_seconds": round(time.time() - t0, 1),
        "decoding": "greedy, max_new_tokens=generation_budget",
        "summary": summary,
        "tasks_sha256": sha256_bytes(Path(args.tasks).read_bytes()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
