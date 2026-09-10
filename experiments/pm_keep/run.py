"""Full generation for PM-Keep, with reusable fixed baseline results.

No time cutoff or server shutdown. The user/Sol owns stopping and method
iteration. A result cache avoids repeating F/E/KeyDiff when P/C/U change.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import string
import time

import torch

from .adapter import AdapterConfig, PrefillSession
from .baselines import ea_official_keep_indices, ea_prefix_scores, keydiff_prefix_scores, source_receipt, strong_baseline_status


STOP = False
BASELINE_VERSION = "pm_native_qwen2_logical_positions_v1"


def stop(signum, frame):
    global STOP
    STOP = True


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def records(path, recover_tail=False):
    """Recover a interrupted final JSONL write; internal corruption is fatal."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    parsed, good = [], 0
    for index, line in enumerate(lines):
        try:
            parsed.append(json.loads(line))
            good += len(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not recover_tail or index != len(lines) - 1:
                raise
            path.with_suffix(".interrupted_tail").write_bytes(raw[good:])
            path.write_bytes(raw[:good])
    return parsed


def model_identity(path):
    path = Path(path)
    return {"path": str(path.resolve()),
            "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest(),
            "weights": {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                        for p in sorted(path.iterdir()) if p.suffix in (".safetensors", ".bin")}}


def baseline_key(model_id, row, arm, config, dtype):
    allocation = {} if arm == "F" else {"keep_fraction": config.keep_fraction, "sink": config.sink_tokens,
                                        "recent": 0 if arm == "E_author_policy" else config.recent_tokens}
    # PM sampling/horizon/query policy never change the fixed EA or Full KV.
    # If the reader/baseline semantics change, update BASELINE_VERSION.
    return digest({"version": BASELINE_VERSION, "model": model_id, "prompt_ids": row["prompt_ids"],
                   "prefix_length": row["prefix_length"], "answer_limit": row["max_new_tokens"],
                   "scoring": [row["score_contract"], row.get("expected"), row.get("references")],
                   "arm": arm, "allocation": allocation, "dtype": dtype,
                   "decode": "raw_greedy_argmax", "official_EA_horizon": 512})


def normalize_answer(value):
    value = "".join(c for c in value.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def score(row, generated, tokenizer, eos_ids):
    ended = bool(generated and generated[-1] in eos_ids)
    body = generated[:-1] if ended else generated
    text = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    result = {"output_text": text, "ended_with_eos": ended,
              "output_text_with_special_tokens": tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)}
    if row["score_contract"] == "literal_full_string_plus_terminal_eos_v1":
        result.update(exact_string=text == row["expected"], exact_plus_eos=ended and text == row["expected"])
    elif row["score_contract"] == "longbench_qa_f1_context_first_v1":
        pred = normalize_answer(text).split()
        scores = []
        for answer in row["references"]:
            ref = normalize_answer(answer).split()
            common = sum((Counter(pred) & Counter(ref)).values())
            scores.append(2 * common / (len(pred) + len(ref)) if common else 0.0)
        result["qa_f1"] = max(scores, default=0.0)
        result["qa_normalized_em"] = float(any(normalize_answer(text) == normalize_answer(a) for a in row["references"]))
    else:
        raise ValueError(f"unsupported score contract {row['score_contract']}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/root/autodl-tmp/rope_qwen_baseline_20260907/model")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline-cache", required=True)
    p.add_argument("--reuse-from", nargs="+", default=[], help="Import matching completed candidate rows from these runs")
    p.add_argument("--arms", nargs="+", choices=("F", "E", "E_author_policy", "C", "P", "U", "K"), default=["F", "E", "C", "P", "U", "K"])
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--per-task", type=int)
    p.add_argument("--horizon", type=int, default=512)
    p.add_argument("--query-policy", choices=("uniform_prefix", "recent_prefix"), default="uniform_prefix")
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--keep-fraction", type=float, default=.25)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = p.parse_args()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing a silent CPU fallback")
    rows = [r for r in records(args.data) if r["split"] == args.split and (not args.tasks or r["task"] in args.tasks)]
    cells = {}
    for row in rows:
        cells.setdefault(row["task"], []).append(row)
    if args.per_task:
        cells = {k: v[:args.per_task] for k, v in cells.items()}
    rows = [cells[k][i] for i in range(max((len(v) for v in cells.values()), default=0))
            for k in sorted(cells) if i < len(cells[k])]
    if not rows:
        raise ValueError("empty task selection")
    out, baseline_dir = Path(args.output), Path(args.baseline_cache)
    out.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    cfg = AdapterConfig(samples_per_head=args.samples, horizon=args.horizon, seed=args.seed,
                        query_policy=args.query_policy, keep_fraction=args.keep_fraction)
    identity = model_identity(args.model)
    sources = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")
               if not f.name.startswith("test_")}
    contract = {"model": identity, "config": asdict(cfg), "arms": args.arms, "dtype": args.dtype,
                "backend": "native_HF_Qwen2_SDPA_original_position_cache_v1", "decode": "raw_greedy_argmax",
                "rows": [r["row_id"] for r in rows], "sources": sources, "baseline_version": BASELINE_VERSION,
                "task_protocol": "context first, question unseen by compression; not original LongBench prompt order",
                "time_limit": None, "automatic_shutdown": False, "strong_baseline_status": strong_baseline_status()}
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("output directory belongs to a different candidate version; use a new output directory")
    write_json(manifest, contract)
    snapshot = out / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in sources:
        (snapshot / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    old = records(out / "per_example.jsonl", recover_tail=True)
    if args.reuse_from:
        from experiments.position_overnight.reuse import append_missing, candidate_rows
        keys = {(r["row_id"], arm): baseline_key(identity, r, arm, cfg, args.dtype)
                for r in rows for arm in args.arms}
        for source in args.reuse_from:
            count = append_missing(out / "per_example.jsonl", candidate_rows(source, contract, set(contract["rows"]),
                                   args.arms, kind="pm", row_keys=keys), "arm")
            print(json.dumps({"reused_candidate_rows": count, "source": source}), flush=True)
        old = records(out / "per_example.jsonl")
    done = {(r["row_id"], r["arm"]) for r in old}
    needed = []
    for row in rows:
        for arm in args.arms:
            if (row["row_id"], arm) not in done:
                needed.append((row, arm))
    status = {"status": "LOADING", "pid": os.getpid(), "completed": len(done), "total": len(rows) * len(args.arms),
              "started_at": time.time(), "time_limit": None}
    write_json(out / "status.json", status)
    if not needed:
        status["status"] = "COMPLETE"
        write_json(out / "status.json", status)
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=getattr(torch, args.dtype),
                                               attn_implementation="sdpa").to(args.device).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    try:
        with (out / "per_example.jsonl").open("a") as stream:
            for ri, row in enumerate(rows):
                if STOP or (out / "STOP").exists():
                    break
                if row["prefix_ids"] + row["suffix_ids"] != row["prompt_ids"] or len(row["prefix_ids"]) != row["prefix_length"]:
                    raise ValueError("prefix/suffix is not the original complete tokenization")
                pending = [arm for arm in args.arms if (row["row_id"], arm) not in done]
                if not pending:
                    continue
                cached, active = {}, []
                for arm in pending:
                    key = baseline_key(identity, row, arm, cfg, args.dtype)
                    cache_path = baseline_dir / (key + ".json")
                    if arm in ("F", "E", "E_author_policy", "K") and cache_path.exists():
                        cached[arm] = json.loads(cache_path.read_text())
                    else:
                        active.append(arm)
                session = None
                if active:
                    status.update(status="RUNNING", row_id=row["row_id"], arms=active)
                    write_json(out / "status.json", status)
                    callbacks = {}
                    if any(a in active for a in ("E", "E_author_policy")):
                        callbacks["E"] = ea_prefix_scores
                    if "K" in active:
                        callbacks["K"] = keydiff_prefix_scores
                    if callbacks:
                        write_json(out / "baseline_sources.json", source_receipt())
                    if args.device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    session = PrefillSession(model, row["prefix_ids"], cfg).prefill(external_scorers=callbacks)
                offset = ri % len(pending)
                for arm in pending[offset:] + pending[:offset]:
                    if arm in cached:
                        result = {**cached[arm], "reused_baseline": True, "baseline_cache_key": baseline_key(identity, row, arm, cfg, args.dtype)}
                    else:
                        if arm == "E_author_policy":
                            indices = [ea_official_keep_indices(s, session.total_budget) for s in session.score("E")]
                            branch = session.branch("E", indices=indices)
                        else:
                            branch = session.branch(arm)
                        branch.consume(row["suffix_ids"])
                        generated = branch.generate(row["max_new_tokens"], eos_ids)
                        keep_hash = digest([idx.cpu().tolist() for idx in branch.keep_indices])
                        timing = dict(generated["timings"])
                        scoring_arm = "E" if arm == "E_author_policy" else arm
                        timing["scoring_seconds"] = session.timings["score_seconds"].get(scoring_arm, 0.0)
                        timing["query_capture_seconds"] = session.timings["query_capture_seconds"] if arm in ("P", "C", "U") else 0.0
                        timing["native_prefill_seconds"] = session.timings["native_prefill_residual_seconds"]
                        timing["estimated_standalone_total_seconds"] = sum(timing.values())
                        result = {k: row.get(k) for k in ("row_id", "task", "split", "doc_id", "context_sha256", "expected", "references", "score_contract", "prefix_length", "suffix_tokens", "input_tokens")}
                        result.update(arm=arm, config=asdict(cfg), generated_token_ids=generated["generated_ids"],
                                      generated_tokens=len(generated["generated_ids"]), timings=timing,
                                      prefix_kv_bytes=generated["initial_prefix_kv_bytes"], retained_indices_bytes=generated["retained_indices_bytes"],
                                      final_kv_bytes=generated["final_kv_bytes"], keep_indices_sha256=keep_hash,
                                      memory_accounting=session.memory_receipt(),
                                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                                      peak_scope="shared prefix plus paired branches/scoring; not an isolated deployment peak",
                                      baseline_cache_key=baseline_key(identity, row, arm, cfg, args.dtype), reused_baseline=False,
                                      **score(row, generated["generated_ids"], tokenizer, eos_ids))
                        if arm in ("F", "E", "E_author_policy", "K"):
                            write_json(baseline_dir / (result["baseline_cache_key"] + ".json"), result)
                        del branch
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    done.add((row["row_id"], arm))
                    status["completed"] = len(done)
                    write_json(out / "status.json", status)
                    print(json.dumps({"row_id": row["row_id"], "arm": arm, "reused": result["reused_baseline"],
                                      "output": result["output_text"], "score": result.get("exact_plus_eos", result.get("qa_f1"))}, ensure_ascii=False), flush=True)
                del session
        status.update(status="COMPLETE" if len(done) == status["total"] else "STOPPED_BY_USER",
                      elapsed_seconds=time.time() - status["started_at"])
    except BaseException as error:
        status.update(status="FAILED", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.time() - status["started_at"])
        raise
    finally:
        write_json(out / "status.json", status)


if __name__ == "__main__":
    main()
