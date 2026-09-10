"""Run complete-output NOSA comparisons and persist each paired observation.

All runs identify the source-derived attention backend. GPU qualification is
recorded separately from scientific outcomes. No prompt truncation, answer
injection, constrained decoding, silent model fallback, or network downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import torch

from .runtime import AttentionSettings, BACKEND_LABEL, NosaReferenceForCausalLM, native_select
from .selector_controls import MODES, BlockSummarySelector


STOP_REQUESTED = False


def _stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("runtime.py"),
                      Path(__file__).with_name("selector_controls.py"))}


def score_output(row, generated, tokenizer, eos_ids):
    ended = bool(generated and generated[-1] in eos_ids)
    body = generated[:-1] if ended else generated
    text = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    raw_text = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    score = {"ended_with_eos": ended, "output_text": text, "output_text_with_special_tokens": raw_text}
    contract = row["score_contract"]
    if contract == "literal_full_string_plus_terminal_eos_v1":
        score.update(exact_string=text == row["expected"], exact_plus_eos=ended and text == row["expected"])
    elif contract == "ruler_official_string_match_all_v1":
        refs = row["references"]
        if not refs:
            raise ValueError("official RULER row has no references")
        # Upstream string_match_all averages reference-wise case-insensitive
        # substring recall. It is deliberately NOT labelled exact generation.
        score["official_recall"] = sum(ref.lower() in text.lower() for ref in refs) / len(refs)
    else:
        raise ValueError(f"unknown scoring contract {contract}")
    return score


@torch.inference_mode()
def generate(model, ids, max_new_tokens, eos_ids, chunk_size, device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = model.prefill(ids, chunk_size=chunk_size)
    if device == "cuda":
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    if not bool(torch.isfinite(output.logits).all()):
        raise FloatingPointError("nonfinite prefill logits")
    cache_bytes = sum(t.numel() * t.element_size() for layer in output.past_key_values for t in (layer.k, layer.v, layer.cis))
    generated = []
    for step in range(max_new_tokens):
        token = int(output.logits[0, -1].argmax())
        generated.append(token)
        if token in eos_ids or step + 1 == max_new_tokens:
            break
        output = model(torch.tensor([[token]], device=ids.device), past_key_values=output.past_key_values,
                       num_logits_to_keep=1)
        if not bool(torch.isfinite(output.logits).all()):
            raise FloatingPointError("nonfinite decode logits")
    if device == "cuda":
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started
    return generated, {"prefill_seconds": prefill_seconds, "total_seconds": total_seconds,
                       "decode_seconds": total_seconds - prefill_seconds,
                       "prefill_tokens_per_second": ids.numel() / max(prefill_seconds, 1e-9),
                       "persistent_raw_kv_cis_bytes_at_prompt": cache_bytes,
                       "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                       "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None}


def select_rows(rows, args):
    rows = [r for r in rows if r["split"] == args.split and (not args.lengths or r["length_cap"] in args.lengths)
            and (not args.tasks or r["task"] in args.tasks)]
    if args.row_ids:
        wanted = set(json.loads(Path(args.row_ids).read_text()))
        rows = [r for r in rows if r["row_id"] in wanted]
        missing = wanted - {r["row_id"] for r in rows}
        if missing:
            raise ValueError(f"requested row IDs are missing or excluded: {sorted(missing)[:3]}")
    elif args.per_cell:
        # Deterministic prefix in each task/length cell. CF rows are only
        # limited through family groups to avoid separating the four worlds.
        groups = {}
        for row in rows:
            groups.setdefault((row["task"], row["length_cap"]), []).append(row)
        picked = []
        for group in groups.values():
            if group[0]["score_contract"].startswith("literal_"):
                families = sorted({r["family_id"] for r in group})[:args.per_cell]
                picked.extend(r for r in group if r["family_id"] in families)
            else:
                picked.extend(group[:args.per_cell])
        rows = picked
    # Interleave tasks/lengths by within-cell ordinal, not by observed outcome.
    cells = {}
    for row in rows:
        cells.setdefault((row["task"], row["length_cap"]), []).append(row)
    ordered = []
    for idx in range(max((len(v) for v in cells.values()), default=0)):
        for key in sorted(cells):
            if idx < len(cells[key]):
                ordered.append(cells[key][idx])
    return ordered


def atomic_units(rows):
    """Keep complete counterfactual material families across tasks/lengths.

    Public benchmark rows are independent units. A hard external interruption
    can still leave an incomplete unit, which analysis must mark as partial.
    """
    groups = {}
    for row in rows:
        if row["score_contract"].startswith("literal_") and row.get("material_cluster_id"):
            key = ("counterfactual", row["material_cluster_id"])
        else:
            key = ("row", row["row_id"])
        groups.setdefault(key, []).append(row)
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/autodl-tmp/NOSA-1B")
    parser.add_argument("--data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-cache", help="Reuse completed fixed-control generations across candidate versions")
    parser.add_argument("--reuse-from", nargs="+", default=[], help="Import matching completed candidate rows from these runs")
    parser.add_argument("--selectors", nargs="+", choices=("native", "dense", *MODES), default=["native"])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--per-cell", type=int)
    parser.add_argument("--row-ids")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--select-blocks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--attention-query-chunk-size", type=int, default=16)
    parser.add_argument("--baseline-attention-query-chunk-size", type=int,
                        help="Reuse control cache from a query chunk with verified execution parity")
    parser.add_argument("--budget-seconds", type=float, default=None,
                        help="Optional only; default has no time cutoff. User stops the server.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke and not args.data:
        parser.error("--data is required for scientific evaluation")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    eos = json.loads((Path(args.model) / "config.json").read_text()).get("eos_token_id", tokenizer.eos_token_id)
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    if args.smoke:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": "What is 1 + 1? Reply with the answer only."}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
        rows = [{"row_id": "runtime_smoke", "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False),
                 "expected": "2", "references": ["2"], "score_contract": "literal_full_string_plus_terminal_eos_v1",
                 "max_new_tokens": 16, "task": "runtime_smoke", "family_id": "runtime_smoke", "split": "dev", "length_cap": 256}]
    else:
        rows = select_rows([json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()], args)
    if not rows:
        raise ValueError("empty evaluation selection")
    contract = {"backend": BACKEND_LABEL, "model": str(Path(args.model).resolve()), "selectors": args.selectors,
                "device": args.device, "dtype": args.dtype, "split": args.split, "topk": args.topk,
                "select_blocks": args.select_blocks, "chunk_size": args.chunk_size,
                "attention_query_chunk_size": args.attention_query_chunk_size,
                "baseline_attention_query_chunk_size": args.baseline_attention_query_chunk_size,
                "source_hashes": source_hashes(), "row_ids": [r["row_id"] for r in rows],
                "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest() if args.data else None,
                "eos_ids": sorted(eos_ids), "generation": "greedy, no constraints; never truncate input",
                "score_scope": "official subset recall and separate literal exact+EOS; no full-benchmark claim"}
    model_dir = Path(args.model)
    model_id = {"config": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
                "tokenizer": hashlib.sha256((model_dir / "tokenizer.json").read_bytes()).hexdigest(),
                "weights": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(model_dir.iterdir())
                            if p.suffix in (".bin", ".safetensors")}}
    fixed_controls = {"native", "dense", "weighted_mean", "cobs_rank1", "cobs_rank2", "split2", "split4", "quest"}
    baseline_dir = Path(args.baseline_cache) if args.baseline_cache else None
    if baseline_dir:
        baseline_dir.mkdir(parents=True, exist_ok=True)

    def control_path(row, name):
        if baseline_dir is None or name not in fixed_controls:
            return None
        payload = {"version": "nosa_fixed_controls_v1", "model": model_id, "ids": row["prompt_ids"],
                   "scoring": [row["score_contract"], row.get("expected"), row.get("references")],
                   "max_new_tokens": row["max_new_tokens"], "selector": name, "topk": args.topk,
                   "select_blocks": args.select_blocks, "dtype": args.dtype,
                   "backend": BACKEND_LABEL, "chunk_size": args.chunk_size,
                   "attention_query_chunk_size": args.baseline_attention_query_chunk_size or args.attention_query_chunk_size}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return baseline_dir / (key + ".json")
    contract_path = output_dir / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("run contract changed: use a new output directory rather than mix results")
    write_json(contract_path, contract)
    snapshot_dir = output_dir / "code_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    for filename in contract["source_hashes"]:
        (snapshot_dir / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    existing_path = output_dir / "generations.jsonl"
    if args.reuse_from:
        from experiments.position_overnight.reuse import append_missing, candidate_rows
        for source in args.reuse_from:
            count = append_missing(existing_path, candidate_rows(source, contract, set(contract["row_ids"]),
                                    args.selectors, kind="pc2"), "selector")
            print(json.dumps({"reused_candidate_rows": count, "source": source}), flush=True)
    existing = [json.loads(line) for line in existing_path.read_text().splitlines()] if existing_path.exists() else []
    done = {(row["row_id"], row["selector"]) for row in existing}
    started = time.time()
    status = {"status": "LOADING", "pid": os.getpid(), "started_at": started,
              "total_pairs": len(rows) * len(args.selectors), "completed_pairs": len(done), "backend": BACKEND_LABEL}
    write_json(output_dir / "status.json", status)
    try:
        model = NosaReferenceForCausalLM.from_pretrained(args.model, device=args.device,
                                                        dtype=getattr(torch, args.dtype))
    except BaseException as error:
        status.update(status="FAILED", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.time() - started)
        write_json(output_dir / "status.json", status)
        raise
    load_report = {**model.load_report, "parameters": sum(p.numel() for p in model.parameters()),
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                   "load_seconds": time.time() - started, "scientific_quality_verified": False}
    write_json(output_dir / "load_report.json", load_report)
    print(json.dumps({"loaded": load_report}), flush=True)
    try:
        with existing_path.open("a") as stream:
            row_index = 0
            for unit in atomic_units(rows):
                if STOP_REQUESTED or (args.budget_seconds is not None and time.time() - started >= args.budget_seconds):
                    break
                for row in unit:
                    ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=args.device)
                    if len(row["prompt_ids"]) + row["max_new_tokens"] > row["length_cap"]:
                        raise ValueError(f"input plus generation cap exceeds declared length: {row['row_id']}")
                    # Complete this material family before any optional cutoff.
                    # Default operation has no time cap or automatic shutdown.
                    offset = row_index % len(args.selectors)
                    order = args.selectors[offset:] + args.selectors[:offset]
                    row_index += 1
                    for name in order:
                        if (row["row_id"], name) in done:
                            continue
                        cache_path = control_path(row, name)
                        if cache_path is not None and cache_path.exists():
                            result = {**json.loads(cache_path.read_text()), "reused_baseline": True}
                            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                            stream.flush()
                            done.add((row["row_id"], name))
                            status["completed_pairs"] = len(done)
                            write_json(output_dir / "status.json", status)
                            print(json.dumps({"row_id": row["row_id"], "selector": name, "reused_baseline": True}), flush=True)
                            continue
                        selector = native_select if name in ("native", "dense") else BlockSummarySelector(name)
                        model.selector = selector
                        model.settings = AttentionSettings(topk=args.topk, select_blocks=args.select_blocks,
                                                           dense=name == "dense", attention_query_chunk_size=args.attention_query_chunk_size)
                        status.update(status="RUNNING", row_id=row["row_id"], selector=name)
                        write_json(output_dir / "status.json", status)
                        generated, timing = generate(model, ids, row["max_new_tokens"], eos_ids, args.chunk_size, args.device)
                        result = {key: row.get(key) for key in ("row_id", "task", "suite", "family_id", "material_cluster_id",
                                                               "split", "length_cap", "score_contract", "expected", "references",
                                                               "content_swap", "query_swap", "query_ordinal", "seed")}
                        result.update(selector=name, input_tokens=ids.numel(), generated_token_ids=generated,
                                      generated_tokens=len(generated), topk=None if name == "dense" else args.topk,
                                      reader_support="all_causal_KV" if name == "dense" else "selected_blocks",
                                      backend=BACKEND_LABEL,
                                      chunk_size=args.chunk_size, attention_query_chunk_size=args.attention_query_chunk_size,
                                      selector_metrics=dict(selector.metrics) if isinstance(selector, BlockSummarySelector) else {},
                                      **score_output(row, generated, tokenizer, eos_ids), **timing)
                        result["reused_baseline"] = False
                        if cache_path is not None:
                            write_json(cache_path, result)
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stream.flush()
                        done.add((row["row_id"], name))
                        status["completed_pairs"] = len(done)
                        print(json.dumps({k: result[k] for k in ("row_id", "selector", "input_tokens", "generated_tokens", "prefill_seconds", "output_text")}, ensure_ascii=False), flush=True)
                        write_json(output_dir / "status.json", status)
        status.update(status="COMPLETE" if len(done) == status["total_pairs"] else "PARTIAL_BUDGET",
                      elapsed_seconds=time.time() - started)
        write_json(output_dir / "status.json", status)
    except BaseException as error:
        status.update(status="FAILED", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.time() - started)
        write_json(output_dir / "status.json", status)
        raise


if __name__ == "__main__":
    main()
