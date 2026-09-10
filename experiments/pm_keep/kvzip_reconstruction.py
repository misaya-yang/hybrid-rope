"""KVPress author's KVzip reconstruction score with local fixed-budget gathering.

This is a strong-control/mechanism-repair adaptation, NOT PM positional novelty
or a reproduction of KVzip's cross-layer fake-pruning policy. Only pinned author
prepare/score_kvzip are reused. Known prefix tokens are teacher-forced as context
reconstruction input; no future task question or answer enters the scorer. Task
answers are subsequently freely generated through the existing native reader.

Default is a CPU preparation preview. Root owns GPU scheduling: use --execute.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch

from .adapter import AdapterConfig, PrefillSession, _sync, cache_nbytes
from .baselines import load_author_class, source_receipt
from .run import digest, model_identity, records, score, write_json


LABEL = "KVPress_KVzip_reconstruction_score_fixed_per_head_budget_native_gather_v1"


def template_frame(tokenizer):
    """Same one-user chat frame extraction as the pinned author's __call__."""
    if tokenizer.chat_template is None:
        prefix_text, suffix_text = "", "\n"
    else:
        dummy = "dummy context"
        separator = "\n" + "#" * len(dummy)
        text = tokenizer.apply_chat_template([{"role": "user", "content": dummy + separator}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False)
        prefix_part, suffix_text = text.split(separator)
        prefix_text = prefix_part.split(dummy)[0]
    return (tokenizer.encode(prefix_text, return_tensors="pt", add_special_tokens=False),
            tokenizer.encode(suffix_text, return_tensors="pt", add_special_tokens=False))


def prepare_author(press, model, tokenizer, prefix_ids, chunk_size):
    prefix = torch.as_tensor(prefix_ids, dtype=torch.long).cpu().reshape(1, -1)
    header, suffix = template_frame(tokenizer)
    header_length, length = header.shape[-1], prefix.shape[-1]
    if header_length >= length or not torch.equal(prefix[:, :header_length], header):
        raise ValueError("supplied prefix does not match the author's one-user tokenizer frame")
    if chunk_size < 1:
        raise ValueError("reconstruction chunk size must be positive")
    press.context_length = length
    press.prefix_length = header_length  # author's wrapper length, not our complete KV prefix T
    press._context_ids = prefix
    press._suffix_ids = suffix
    pairs = press.prepare(model, tokenizer, chunk_size=chunk_size)
    limit = model.config.max_position_embeddings
    if any(length + repeat.shape[-1] > limit for _, repeat in pairs):
        raise ValueError("prefix plus reconstruction chunk exceeds native window; no truncation or RoPE change")
    if sum(piece.shape[-1] for piece, _ in pairs) != length - header_length:
        raise RuntimeError("author preparation did not cover the complete known context")
    return pairs


@torch.inference_mode()
def reconstruction_session(model, tokenizer, prefix_ids, *, keep_fraction=.25,
                           chunk_size=2048, sink_tokens=4, recent_tokens=256, source_root=None):
    """Prefix-only callable; returns a normal PrefillSession with scores['R']."""
    cls = load_author_class("KVzipPress", source_root)
    press = cls(compression_ratio=0.0, layerwise=False, n_sink=sink_tokens,
                kvzip_plus_normalization=False)
    # We need native validation/cache branching, not PM query scoring. One
    # unused prefix-Q sample satisfies that existing lightweight interface.
    config = AdapterConfig(samples_per_head=1, horizon=1, sink_tokens=sink_tokens,
                           recent_tokens=recent_tokens, keep_fraction=keep_fraction)
    session = PrefillSession(model, prefix_ids, config)
    if session.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(session.device)
    _sync(session.device)
    started = time.perf_counter()
    pairs = prepare_author(press, model, tokenizer, session.prefix_ids, chunk_size)
    _sync(session.device)
    prepare_seconds = time.perf_counter() - started
    session.prefill()
    temporary = session.branch("F")
    prefix_length = session.prefix_length
    chunks, handles = [], []
    score_calls = []

    def hook(module, args, kwargs, output):
        cache = kwargs["past_key_values"]
        if cache is not temporary.cache:
            raise RuntimeError("author scorer was called outside the isolated reconstruction branch")
        layer = cache.layers[module.layer_idx]
        # Use author score math unchanged. The returned cropped views are not
        # installed mid-forward; the complete temporary cache is cropped below.
        press.score_kvzip(module, kwargs["hidden_states"], layer.keys, layer.values,
                          output[1], kwargs)
        score_calls.append((int(module.layer_idx), press.start_idx, press.end_idx))

    for layer in model.model.layers:
        handles.append(layer.self_attn.register_forward_hook(hook, with_kwargs=True))
    _sync(session.device)
    reconstruction_started = time.perf_counter()
    try:
        press.start_idx = press.prefix_length
        for index, (known_chunk, repeat_ids) in enumerate(pairs):
            press.end_idx = press.start_idx + known_chunk.shape[-1]
            if any(layer.get_seq_length() != prefix_length for layer in temporary.cache.layers):
                raise RuntimeError("temporary cache contains prior reconstruction tokens")
            repeat = repeat_ids.to(session.device)
            positions = torch.arange(prefix_length, prefix_length + repeat.shape[-1], device=session.device)
            output = model(repeat, past_key_values=temporary.cache,
                           position_ids=positions[None], cache_position=positions,
                           use_cache=True, logits_to_keep=1)
            if not torch.isfinite(output.logits).all():
                raise FloatingPointError("nonfinite reconstruction logits")
            before_crop = [layer.get_seq_length() for layer in temporary.cache.layers]
            if any(n != prefix_length + repeat.shape[-1] for n in before_crop):
                raise RuntimeError("reconstruction cache does not use native absolute positions")
            temporary.cache.crop(prefix_length)
            chunks.append({"chunk_index": index, "original_context_start": press.start_idx,
                           "original_context_end_exclusive": press.end_idx,
                           "source_copy_tokens": int(known_chunk.shape[-1]),
                           "reconstruction_input_tokens": int(repeat.shape[-1]),
                           "logical_query_start": prefix_length,
                           "logical_query_end_exclusive": prefix_length + int(repeat.shape[-1]),
                           "temporary_cache_lengths_before_crop": before_crop,
                           "temporary_cache_lengths_after_crop": [l.get_seq_length() for l in temporary.cache.layers],
                           "repeat_ids_sha256": digest(repeat_ids.flatten().tolist())})
            press.start_idx = press.end_idx
        _sync(session.device)
    finally:
        for handle in handles:
            handle.remove()
        temporary.cache.crop(prefix_length)
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    expected_calls = len(pairs) * len(model.model.layers)
    if len(score_calls) != expected_calls or not torch.isfinite(press.score_val).all():
        raise RuntimeError("author reconstruction score is incomplete or nonfinite")
    if any(layer.get_seq_length() != prefix_length for layer in session.cache.layers):
        raise RuntimeError("the original prefix cache was modified")
    session.scores["R"] = [press.score_val[i, 0].detach().float().clone() for i in range(len(model.model.layers))]
    session.timings["score_seconds"]["R"] = prepare_seconds + reconstruction_seconds + temporary.timings["gather_seconds"]
    receipt = {"label": LABEL, "author_score": "unchanged prepare and score_kvzip from pinned KVPress source",
               "allocation": "existing select_fixed_budget per native KV head/layer, sink/recent inside budget; original-position physical gather",
               "not_used": ["KVzipPress.__call__", "compress_post", "masked_key_indices fake pruning", "future task question or answer"],
               "reconstruction_input": "fixed author Repeat instructions, tokenizer chat suffix, and copies of already-visible prefix tokens; teacher-forced known context",
               "prefix_length": prefix_length, "author_template_header_tokens": press.prefix_length,
               "chunks": chunks, "author_score_calls": len(score_calls),
               "reconstruction_input_tokens_total": sum(c["reconstruction_input_tokens"] for c in chunks),
               "source_copy_tokens_total": sum(c["source_copy_tokens"] for c in chunks),
               "prepare_seconds": prepare_seconds,
               "initial_prefill_seconds": session.timings["prefill_total_seconds"],
               "temporary_full_gather_seconds": temporary.timings["gather_seconds"],
               "reconstruction_scoring_seconds": reconstruction_seconds,
               "prefix_and_reconstruction_total_seconds": time.perf_counter() - started,
               "full_prefix_kv_bytes": cache_nbytes(session.cache),
               "score_bytes": sum(s.numel() * s.element_size() for s in session.scores["R"]),
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(session.device) if session.device.type == "cuda" else None,
               "peak_scope": "shared original Full prefix + isolated temporary Full reconstruction cache + scoring workspace",
               "reader_unchanged": True, "original_prefix_cache_length_after_scoring": session.cache.get_seq_length(),
               "no_future_labels_accepted_by_scorer": True}
    del temporary, press, pairs
    return session, receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--row-ids", nargs="+")
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--per-task", type=int)
    p.add_argument("--keep-fraction", type=float, default=.25)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--broad-scoring", action="store_true")
    p.add_argument("--root", default="/root/autodl-tmp/position_overnight_20260909")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    source = source_receipt()
    wanted = set(a.row_ids or [])
    rows = [r for r in records(a.data) if r["split"] == a.split and (not wanted or r["row_id"] in wanted)
            and (not a.tasks or r["task"] in a.tasks)]
    if wanted and {r["row_id"] for r in rows} != wanted:
        raise ValueError("selected row IDs missing or filtered out")
    if a.per_task:
        grouped = {}
        for row in rows:
            grouped.setdefault(row["task"], []).append(row)
        rows = [r for group in grouped.values() for r in group[:a.per_task]]
    if not rows or any(r["prefix_ids"] + r["suffix_ids"] != r["prompt_ids"] for r in rows):
        raise ValueError("nonempty rows with exact prefix/suffix boundaries required")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    cfg = AutoConfig.from_pretrained(a.model, local_files_only=True)
    proxy_model = SimpleNamespace(config=cfg, dtype=torch.float32, device=torch.device("cpu"))
    cls = load_author_class("KVzipPress")
    preview = []
    for row in rows:
        press = cls(compression_ratio=0.0, layerwise=False, n_sink=4, kvzip_plus_normalization=False)
        pairs = prepare_author(press, proxy_model, tokenizer, row["prefix_ids"], a.chunk_size)
        preview.append({"row_id": row["row_id"], "prefix_length": len(row["prefix_ids"]),
                        "reconstruction_chunks": len(pairs),
                        "reconstruction_input_tokens": sum(p.shape[-1] for _, p in pairs),
                        "max_logical_position_exclusive": len(row["prefix_ids"]) + max(p.shape[-1] for _, p in pairs)})
    contract = {"label": LABEL, "model": model_identity(a.model), "author_source": source,
                "source_sha256": digest(Path(__file__).read_text()), "keep_fraction": a.keep_fraction,
                "chunk_size": a.chunk_size, "dtype": a.dtype, "preview": preview,
                "input_sha256": {r["row_id"]: digest(r["prompt_ids"]) for r in rows},
                "dry_run": not a.execute}
    print(json.dumps(contract, ensure_ascii=False), flush=True)
    if not a.execute:
        return
    if (Path(a.root) / "STOP").exists():
        raise SystemExit("STOP exists; root owns execution")
    out = Path(a.output)
    if (out / "status.json").exists():
        raise FileExistsError("preserve existing evidence and choose a new output directory")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent fallback")
    torch.set_num_threads(4)
    write_json(out / "contract.json", contract)
    write_json(out / "status.json", {"status": "LOADING"})
    scorer = score
    if a.broad_scoring:
        from experiments.broad_position_eval.scoring import score as scorer
    started = time.perf_counter()
    completed = []
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
                torch_dtype=getattr(torch, a.dtype), attn_implementation="sdpa").to(a.device).eval()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        with (out / "per_example.jsonl").open("w") as stream:
            for row in rows:
                if (Path(a.root) / "STOP").exists():
                    break
                write_json(out / "status.json", {"status": "RUNNING", "row_id": row["row_id"], "completed_rows": completed})
                row_started = time.perf_counter()
                session, receipt = reconstruction_session(model, tokenizer, row["prefix_ids"],
                    keep_fraction=a.keep_fraction, chunk_size=a.chunk_size)
                indices = session.keep_indices("R")
                trace_path = out / "keep_traces" / (row["row_id"] + ".pt")
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"keep_indices": {"R": [x.detach().cpu() for x in indices]},
                            "prefix_ids_sha256": digest(row["prefix_ids"])}, trace_path)
                branch = session.branch("R", indices)
                branch.consume(row["suffix_ids"])
                generated = branch.generate(row["max_new_tokens"], eos)
                result = {"row_id": row["row_id"], "task": row["task"], "arm": "KVzip_fixed_budget",
                          "label": LABEL, "reconstruction": receipt,
                          "keep_indices_sha256": digest([x.cpu().tolist() for x in indices]),
                          "keep_trace": str(trace_path),
                          "kept_per_head": session.total_budget,
                          "generated_token_ids": generated["generated_ids"],
                          "generation": generated,
                          **scorer(row, generated["generated_ids"], tokenizer, eos),
                          "total_seconds": time.perf_counter() - row_started}
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                completed.append(row["row_id"])
                del session, branch, indices
        write_json(out / "status.json", {"status": "COMPLETE" if len(completed) == len(rows) else "STOPPED",
                   "completed_rows": completed, "elapsed_seconds": time.perf_counter() - started})
    except BaseException as error:
        write_json(out / "status.json", {"status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
