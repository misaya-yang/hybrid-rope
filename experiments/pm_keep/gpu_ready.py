"""Bounded real-model integration parity and one complete-input readiness run.

The 64-token truncation is exclusively an integration probe, never a task
result. The scientific smoke row is consumed in full by every arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import AdapterConfig, PrefillSession
from .baselines import ea_prefix_scores, keydiff_prefix_scores, source_receipt


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def checkpoint_loading_view(source: Path, destination: Path):
    """Recover only a missing shard index in a new symlink view, never weights.

    An index is a map from tensor names to their actual shard filenames. Tensor
    headers provide that exact map without loading, editing, or guessing weights.
    Full HF checkpoint loading below still rejects missing/unexpected parameters.
    """
    source = source.resolve()
    if (source / "model.safetensors").exists() or (source / "model.safetensors.index.json").exists():
        return source, None
    shards = sorted(source.glob("model-*-of-*.safetensors"))
    if not shards:
        return source, None
    destination.mkdir(parents=True, exist_ok=False)
    weight_map, total_size = {}, 0
    for shard in shards:
        with shard.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            if header_length > 100_000_000:
                raise ValueError("invalid safetensors header length")
            header = json.loads(stream.read(header_length))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in weight_map:
                raise ValueError(f"duplicate tensor in checkpoint shards: {name}")
            weight_map[name] = shard.name
            begin, end = tensor["data_offsets"]
            total_size += end - begin
    for path in source.iterdir():
        if path.is_file():
            (destination / path.name).symlink_to(path)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    write_json(destination / "model.safetensors.index.json", index)
    receipt = {
        "reason": "original model directory has sharded safetensors but no weight index",
        "original_directory": str(source), "loading_view": str(destination.resolve()),
        "original_files_changed": False, "tensor_count": len(weight_map),
        "index_sha256": file_hash(destination / "model.safetensors.index.json"),
        "map_source": "actual safetensors shard headers; no inferred tensor contents",
    }
    return destination.resolve(), receipt


@torch.inference_mode()
def keep_all_parity(model, input_ids, eos):
    ids = torch.as_tensor(input_ids[:64], dtype=torch.long, device="cuda")[None]
    if ids.shape[1] != 64:
        raise ValueError("readiness parity requires at least 64 input tokens")
    complete = model(ids, use_cache=True, logits_to_keep=1)
    native_prefix = model(ids[:, :32], use_cache=True, logits_to_keep=1)
    session = PrefillSession(model, ids[:, :32], AdapterConfig(
        keep_fraction=1.0, samples_per_head=16, horizon=8,
        sink_tokens=4, recent_tokens=16,
    )).prefill()
    prefill_delta = float((native_prefix.logits[:, -1].float() - session.last_logits.float()).abs().max())
    cache_delta = max(float((a.keys.float() - b.keys.float()).abs().max())
                      for a, b in zip(native_prefix.past_key_values.layers, session.cache.layers))
    branch = session.branch("P")  # exercise actual score, selection, and gather at B=T
    native = native_prefix
    incremental_delta = 0.0
    for token in ids[0, 32:]:
        native = model(token.reshape(1, 1), past_key_values=native.past_key_values,
                       use_cache=True, logits_to_keep=1)
        actual = branch.step(int(token))
        incremental_delta = max(incremental_delta,
                                float((actual.float() - native.logits[:, -1].float()).abs().max()))
    full_delta = (complete.logits[:, -1].float() - branch.last_logits.float()).abs()
    # Compare the original complete input with the adapter's same complete
    # prefill. A 32+32 incremental path is a separate backend partition probe,
    # not an appropriate tolerance baseline for the 64-token full call.
    full_session = PrefillSession(model, ids, AdapterConfig(
        keep_fraction=1.0, samples_per_head=16, horizon=8,
        sink_tokens=4, recent_tokens=16,
    )).prefill()
    full_branch = full_session.branch("P")
    matched_full_delta = float((complete.logits[:, -1].float() - full_branch.last_logits.float()).abs().max())
    matched_full_cache_delta = max(float((a.keys.float() - b.keys.float()).abs().max())
                                  for a, b in zip(complete.past_key_values.layers, full_session.cache.layers))
    complete_generated = []
    for index in range(8):
        token = int(complete.logits[0, -1].argmax())
        complete_generated.append(token)
        if token in eos or index == 7:
            break
        complete = model(torch.tensor([[token]], device="cuda"),
                         past_key_values=complete.past_key_values, use_cache=True, logits_to_keep=1)
    generated = branch.generate(8, eos)
    matched_full_generated = full_branch.generate(8, eos)
    # An explicit keep-set must use exactly the same reader for every arm name.
    same_a = session.branch("F")
    same_b = session.branch("same_keep_set_control", same_a.keep_indices)
    same_a.consume(ids[:, 32:])
    same_b.consume(ids[:, 32:])
    same_set_delta = float((same_a.last_logits.float() - same_b.last_logits.float()).abs().max())
    same_set_greedy = same_a.generate(4, eos)["generated_ids"] == same_b.generate(4, eos)["generated_ids"]
    positions = generated["logical_positions"]
    correct_positions = (positions == list(range(32, 32 + len(positions)))
                         and generated["physical_cache_lengths"] == [p + 1 for p in positions])
    # Both matched full and matched incremental computational paths must agree
    # bitwise. Native full-vs-incremental BF16 variation is separately reported;
    # it cannot be attributed to this cache adapter when native Q=1 is identical.
    passed = (prefill_delta == cache_delta == incremental_delta == same_set_delta == 0.0
              and matched_full_delta == matched_full_cache_delta == 0.0
              and same_set_greedy and correct_positions
              and complete_generated == matched_full_generated["generated_ids"]
              and complete_generated == generated["generated_ids"])
    return {
        "status": "PASS" if passed else "BLOCKED_INTEGRATION",
        "scope": "64-token integration-only prefix truncation; not a task score",
        "prefill_tokens": 32, "complete_input_tokens": 64,
        "native_vs_hook_prefill_max_abs_logit": prefill_delta,
        "native_vs_hook_cache_key_max_abs": cache_delta,
        "native_vs_adapter_incremental_max_abs_logit": incremental_delta,
        "single_full_vs_adapter_incremental_max_abs_logit": float(full_delta.max()),
        "single_full_vs_adapter_incremental_mean_abs_logit": float(full_delta.mean()),
        "full_vs_incremental_role": "native backend partition diagnostic; same incremental native path is compared separately",
        "matched_complete_prefill_max_abs_logit": matched_full_delta,
        "matched_complete_prefill_cache_key_max_abs": matched_full_cache_delta,
        "matched_complete_prefill_keep_all_greedy_ids": matched_full_generated["generated_ids"],
        "single_full_greedy_ids": complete_generated,
        "keep_all_greedy_ids": generated["generated_ids"],
        "greedy_identical": complete_generated == generated["generated_ids"],
        "cache_positions_correct": correct_positions,
        "same_keep_set_max_abs_logit": same_set_delta,
        "same_keep_set_greedy_identical": same_set_greedy,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default="F,E,P,C,U,K")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no automatic instance start")
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    row = rows[0]
    if row["prompt_ids"] != row["prefix_ids"] + row["suffix_ids"]:
        raise ValueError("prepared token boundary is inconsistent")
    if row.get("future_question_visible_to_scorer") is not False:
        raise ValueError("readiness requires the explicit unseen-question protocol")
    arms = args.arms.split(",")
    if len(set(arms)) != len(arms) or not set(arms) <= {"F", "E", "P", "C", "U", "K"}:
        raise ValueError("readiness arms must be unique F/E/P/C/U/K")
    gpu = torch.cuda.get_device_properties(0)
    status = {
        "status": "LOADING", "pid": os.getpid(), "model": str(args.model),
        "data": str(args.data), "data_sha256": file_hash(args.data),
        "row_id": row["row_id"], "arms": arms,
        "gpu": gpu.name, "gpu_total_memory_bytes": gpu.total_memory,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "dtype": "bfloat16", "attention_backend": "sdpa_flash_only",
        "generation": "raw argmax; no sampling/repetition/logit processors; identical across arms",
        "scope": "one full-input readiness example, not a method evaluation",
        "source_sha256": {path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")},
        "model_config_sha256": file_hash(args.model / "config.json"),
        "weight_files": {path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                         for path in args.model.glob("*.safetensors")},
    }
    state_path = args.output / "status.json"
    write_json(state_path, status)
    try:
        loading_path, index_receipt = checkpoint_loading_view(args.model, args.output / "model_view")
        status["model_loading_path"] = str(loading_path)
        if index_receipt:
            write_json(args.output / "derived_index_receipt.json", index_receipt)
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        rendered = tok(row["full_prompt"], add_special_tokens=False)["input_ids"]
        if rendered != row["prompt_ids"]:
            raise ValueError("runtime tokenizer does not reproduce the frozen full prompt IDs")
        model, loading = AutoModelForCausalLM.from_pretrained(
            loading_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", output_loading_info=True,
        )
        if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")):
            raise ValueError(f"checkpoint load mismatch: {loading}")
        model = model.cuda().eval()
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        status.update(status="PARITY", eos_token_ids=sorted(eos),
                      native_max_position_embeddings=model.config.max_position_embeddings)
        write_json(state_path, status)
        parity = keep_all_parity(model, row["prompt_ids"], eos)
        write_json(args.output / "parity.json", parity)
        print(json.dumps({"event": "parity", **parity}), flush=True)
        if parity["status"] != "PASS":
            raise RuntimeError("real-model keep-all integration parity failed; inspect parity.json")
        torch.cuda.empty_cache()
        callbacks = {key: callback for key, callback in
                     (("E", ea_prefix_scores), ("K", keydiff_prefix_scores)) if key in arms}
        if callbacks:
            write_json(args.output / "baseline_source.json", source_receipt())
        torch.cuda.reset_peak_memory_stats()
        status.update(status="FULL_INPUT_PREFILL")
        write_json(state_path, status)
        session = PrefillSession(model, row["prefix_ids"]).prefill(callbacks)
        torch.cuda.synchronize()
        prefill_peak = torch.cuda.max_memory_allocated()
        print(json.dumps({"event": "prefix_complete", "prefix_tokens": session.prefix_length,
                          "timings": session.timings}), flush=True)
        with (args.output / "predictions.jsonl").open("x") as stream:
            for arm in arms:
                status.update(status="GENERATING", active_arm=arm)
                write_json(state_path, status)
                torch.cuda.reset_peak_memory_stats()
                begin = time.perf_counter()
                branch = session.branch(arm)
                kept = [indices.cpu() for indices in branch.keep_indices]
                torch.save({"original_prefix_positions_per_layer_kv_head": kept,
                            "prefix_length": session.prefix_length}, args.output / f"keep_{arm}.pt")
                branch.consume(row["suffix_ids"])
                result = branch.generate(row["max_new_tokens"], eos)
                torch.cuda.synchronize()
                raw = result["generated_ids"]
                answer_ids = raw[:-1] if result["ended_eos"] else raw
                text = tok.decode(answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                exact = text == row["expected"]
                record = {
                    "row_id": row["row_id"], "task": row["task"], "arm": arm,
                    "scope": "single complete-input GPU readiness example",
                    "prefix_tokens": session.prefix_length, "input_tokens": len(row["prompt_ids"]),
                    "expected": row["expected"], "output_text": text,
                    "eos_token_ids": sorted(eos), "full_exact": exact,
                    "full_exact_and_eos": exact and result["ended_eos"],
                    "score_contract": row["score_contract"], "prompt_ids_sha256": row["prompt_ids_sha256"],
                    "arm_wall_seconds_after_shared_prefill": time.perf_counter() - begin,
                    "score_seconds": session.timings["score_seconds"].get(arm, 0.0),
                    "shared_prefill_timings": dict(session.timings),
                    "shared_prefill_peak_cuda_bytes": prefill_peak,
                    "branch_peak_cuda_bytes_including_retained_full_prefix": torch.cuda.max_memory_allocated(),
                    **result,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps({"event": "arm_complete", "arm": arm, "output_text": text,
                                  "full_exact_and_eos": record["full_exact_and_eos"],
                                  "seconds": record["arm_wall_seconds_after_shared_prefill"],
                                  "score_seconds": record["score_seconds"]}), flush=True)
                del branch
        write_json(args.output / "session_receipt.json", {
            "timings": session.timings, "memory": session.memory_receipt(),
            "prefill_tokens": session.prefix_length, "compressed_prefix_budget": session.total_budget,
            "peak_memory_caveat": "paired run retains a full prefix for later branches; not deployment peak reduction",
        })
        status.update(status="READY", completed_arms=arms, seconds=time.perf_counter() - start,
                      method_quality_conclusion="none: readiness is one independent input")
    except Exception as error:
        status.update(status="FAILED", error_type=type(error).__name__, error=str(error),
                      seconds=time.perf_counter() - start)
        raise
    finally:
        write_json(state_path, status)


if __name__ == "__main__":
    main()
