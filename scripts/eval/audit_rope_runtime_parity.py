#!/usr/bin/env python3
"""Compare the research RULER path with stock HF SDPA on identical token IDs.

This is an implementation diagnostic, not a new RoPE candidate. All attention
calls are Flash-only; Native frequencies and weights stay fixed throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--tasks", nargs="+", default=["niah_single_1", "niah_multikey_2"])
    parser.add_argument("--limit-per-cell", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
        configure_cuda, configure_ruler_flash_attention, greedy_generate,
    )
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer import (
        _validate_data, official_task_score,
    )
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_instruct_ruler_transfer import (
        TASK_CONFIGS, chat_input_ids,
    )
    from scripts.lib.checkpoint_identity import safetensors_weight_set_sha256

    checkpoint = args.checkpoint.resolve()
    weight_hash = safetensors_weight_set_sha256(checkpoint)
    if weight_hash != args.expected_weight_sha256:
        raise RuntimeError("checkpoint weight identity changed")
    rows, data_receipts = [], []
    for root in args.data_root:
        receipt, selected = _validate_data(
            root=root.resolve(), checkpoint=checkpoint,
            requested_tasks=tuple(args.tasks), requested_lengths=None,
            limit_per_cell=args.limit_per_cell,
        )
        data_receipts.append(receipt)
        rows.extend(selected)
    row_keys = [(r["_task"], r["_nominal_length"], r["_local_index"]) for r in rows]
    if len(set(row_keys)) != len(row_keys):
        raise RuntimeError("overlapping data roots")
    args.output.mkdir(parents=True, exist_ok=True)
    examples_path = args.output / "examples.jsonl"
    if examples_path.exists():
        raise RuntimeError("diagnostic output already exists; use a fresh directory")
    configure_cuda()
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    rotary = model.model.rotary_emb

    def tensor_hash(tensor):
        return hashlib.sha256(tensor.detach().cpu().float().numpy().tobytes()).hexdigest()

    initial_inv = rotary.inv_freq.detach().clone()
    source_file = Path(inspect.getsourcefile(type(model)))
    manifest = {
        "status": "FROZEN_RUNTIME_PARITY_DIAGNOSTIC",
        "script_sha256": sha256_file(Path(__file__)),
        "weights_sha256": weight_hash,
        "config_sha256": sha256_file(checkpoint / "config.json"),
        "transformers": transformers.__version__, "torch": torch.__version__,
        "model_implementation_sha256": sha256_file(source_file),
        "model_class": type(model).__name__,
        "model_config": model.config.to_dict(),
        "generation_config": model.generation_config.to_dict(),
        "native_inv_sha256_float32": tensor_hash(initial_inv),
        "native_inv_dtype": str(initial_inv.dtype),
        "attention_scaling": float(rotary.attention_scaling),
        "mlp_activation": type(model.model.layers[0].mlp.act_fn).__name__,
        "embedding_class": type(model.model.embed_tokens).__name__,
        "data": data_receipts,
        "protocol": "identical unpadded IDs; native; stock SDPA versus custom Flash; matched greedy EOS",
        "flash_only": True, "shutdown": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    torch.cuda.reset_peak_memory_stats()
    results = []

    @torch.inference_mode()
    def logits_pair(ids, forced=None, check_uncached=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            first = out.logits[:, -1, :].float()
            token = first.argmax(-1) if forced is None else forced
            cached_length = int(out.past_key_values.get_seq_length())
            step = model(input_ids=token[:, None], past_key_values=out.past_key_values,
                         use_cache=True, logits_to_keep=1)
            second = step.logits[:, -1, :].float()
            del out, step
            cache_delta = None
            if check_uncached:
                full = model(input_ids=torch.cat((ids, token[:, None]), dim=1),
                             use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                cache_delta = {"max_abs": float((full - second).abs().max()),
                               "same_argmax": bool(torch.equal(full.argmax(-1), second.argmax(-1)))}
            if not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):
                raise RuntimeError("nonfinite diagnostic logits")
            return first, second, token, cached_length, cache_delta

    for row in sorted(rows, key=lambda r: (r["_nominal_length"], r["_task"])):
        started = time.perf_counter()
        chat = chat_input_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": row["input"]}],
            add_generation_prompt=True, return_tensors="pt",
        ))
        prefix = tokenizer(row.get("answer_prefix", ""), add_special_tokens=False,
                           return_tensors="pt").input_ids
        ids = torch.cat((chat, prefix), dim=1).to("cuda")
        budget = int(row["_generation_tokens"])
        if ids.shape[1] + budget > row["_nominal_length"]:
            raise RuntimeError("prompt exceeds registered cell")
        model.config._attn_implementation = "sdpa"
        stock_first, stock_second, forced, cache_length, cache_delta = logits_pair(ids, check_uncached=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            stock_ids = model.generate(
                input_ids=ids, do_sample=False, max_new_tokens=budget,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )[:, ids.shape[1]:]
        configure_ruler_flash_attention(model)
        custom_first, custom_second, _, _, _ = logits_pair(ids, forced=forced)
        custom_ids = greedy_generate(model, ids, max_new_tokens=budget,
                                     eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        if not torch.equal(rotary.inv_freq, initial_inv):
            raise RuntimeError("Native rotary buffer mutated")
        decoded = [tokenizer.decode(value[0].tolist(), skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
                   for value in (stock_ids, custom_ids)]
        references = [str(value) for value in row["outputs"]]
        metric = TASK_CONFIGS[row["_task"]]["official_metric"]
        result = {
            "task": row["_task"], "nominal_length": row["_nominal_length"],
            "local_index": row["_local_index"], "prompt_tokens": ids.shape[1],
            "prompt_ids_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            "references": references, "stock_tokens": stock_ids[0].tolist(),
            "custom_tokens": custom_ids[0].tolist(),
            "stock_prediction": decoded[0], "custom_prediction": decoded[1],
            "stock_score": official_task_score(decoded[0], references, metric),
            "custom_score": official_task_score(decoded[1], references, metric),
            "exact_generated_tokens_equal": bool(torch.equal(stock_ids, custom_ids)),
            "prefill_max_abs": float((stock_first-custom_first).abs().max()),
            "prefill_argmax_equal": bool(torch.equal(stock_first.argmax(-1), custom_first.argmax(-1))),
            "forced_decode_max_abs": float((stock_second-custom_second).abs().max()),
            "forced_decode_argmax_equal": bool(torch.equal(stock_second.argmax(-1), custom_second.argmax(-1))),
            "stock_cache_length": cache_length, "stock_cached_vs_uncached": cache_delta,
            "elapsed_seconds": time.perf_counter() - started,
        }
        results.append(result)
        with examples_path.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        print(json.dumps({k: result[k] for k in ("task", "nominal_length", "stock_score",
              "custom_score", "exact_generated_tokens_equal", "prefill_max_abs", "elapsed_seconds")}), flush=True)
    summary = {
        "status": "RUNTIME_PARITY_DIAGNOSTIC_COMPLETE", "rows": len(results),
        "all_generated_tokens_equal": all(r["exact_generated_tokens_equal"] for r in results),
        "max_prefill_abs": max(r["prefill_max_abs"] for r in results),
        "max_forced_decode_abs": max(r["forced_decode_max_abs"] for r in results),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "native_inv_final_sha256_float32": tensor_hash(rotary.inv_freq),
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
