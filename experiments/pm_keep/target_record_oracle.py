"""Privileged DEV target-record retention intervention; not a deployable method.

Only the fixed first two Full-correct prose256 DEV inputs (002/003) are used.
This headroom selection cannot estimate an overall treatment effect. Future query
key/ordinal locates the source Record, including its known literal value for span
identification; no answer tokens are injected into model input. All keep sets are
frozen before actual question ingestion. Default CPU dryrun; root supplies the
single external GPU lock and --execute. No internal competing lock is acquired.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments.broad_position_eval.scoring import VERSION, score
from experiments.position_overnight.reuse import candidate_rows
from .adapter import cache_nbytes
from .balanced_queries import BalancedConfig, BalancedValueSession
from .causal_probe import paired_swap_sets
from .ops import _seed
from .retention_evidence import record_spans
from .run import BASELINE_VERSION, digest, model_identity, records, write_json


ROW_IDS = ("broad_retrieval_dev_002_prose_256", "broad_retrieval_dev_003_prose_256")


@dataclass(frozen=True)
class OracleConfig(BalancedConfig):
    broad_scoring_version: str = VERSION  # reporting only; same Balanced computation


def target_spans(row, tokenizer):
    if row["split"] != "dev" or row["row_id"] not in ROW_IDS:
        raise ValueError("only the two preselected Full-correct DEV rows are authorized")
    if row["prefix_ids"] + row["suffix_ids"] != row["prompt_ids"]:
        raise ValueError("frozen prefix/suffix boundary mismatch")
    encoded = tokenizer(row["prefix_text"], add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != row["prefix_ids"]:
        raise ValueError("retokenization differs from original prefix IDs")
    spans = [s for s in record_spans(row, encoded["offset_mapping"]) if s["target"]]
    target = sorted({i for s in spans for i in s["record"]})
    if not target:
        raise ValueError("no complete target Record span")
    return spans, target


@torch.inference_mode()
def force_record_sets(control, scores, target_ids, prefix_length, *, sink_tokens=4,
                      recent_tokens=256, seed=20260909, layer_idx=0):
    """Fixed B; remove lowest P-score unprotected non-target slots first."""
    if control.ndim != 2 or scores.shape != (control.shape[0], prefix_length):
        raise ValueError("expected P set [Hkv,B] and scores [Hkv,T]")
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite P scores")
    target = set(int(i) for i in target_ids)
    if not target or min(target) < 0 or max(target) >= prefix_length:
        raise ValueError("target IDs must be original prefix positions")
    protected = set(range(min(sink_tokens, prefix_length))) | set(range(max(0, prefix_length - recent_tokens), prefix_length))
    budget = control.shape[1]
    if len(protected | target) > budget:
        raise ValueError("entire target plus protected union cannot fit the fixed budget")
    proposal, intended = [], []
    score_rows = scores.detach().cpu().tolist()  # avoid a GPU sync for every sortable slot
    for head, original in enumerate(control.cpu().tolist()):
        current = set(original)
        if len(current) != budget or original != sorted(original) or not protected <= current:
            raise ValueError("P must be sorted/unique and retain the common protected slots")
        if any(i < 0 or i >= prefix_length for i in current):
            raise ValueError("P set outside prefix")
        missing = sorted(target - current)
        candidates = sorted(current - protected - target,
                            key=lambda i: (score_rows[head][i], i))
        if len(candidates) < len(missing):
            raise ValueError("insufficient removable slots at unchanged budget")
        removed = candidates[:len(missing)]
        proposal.append(sorted((current - set(removed)) | target))
        intended.append({"head": head, "target_missing_before": missing,
                         "removed_by_lowest_P_score": removed})
    proposal = torch.tensor(proposal, device=control.device, dtype=control.dtype)
    exact, sham, swaps = paired_swap_sets(control, proposal, prefix_length,
        seed=_seed(seed, layer_idx, 0, "target_record_oracle_sham"), max_swaps=0)
    if not torch.equal(exact, proposal):
        raise RuntimeError("paired_swap_sets truncated target proposal; do not run partial oracle")
    for head, receipt in enumerate(swaps):
        if receipt["swaps"] != len(intended[head]["target_missing_before"]):
            raise RuntimeError("sham swap count differs from full target repair")
        if set(receipt["removed"]) != set(intended[head]["removed_by_lowest_P_score"]):
            raise RuntimeError("sham removals differ from intended lowest-score removals")
        if not (protected | target) <= set(exact[head].tolist()) or not protected <= set(sham[head].tolist()):
            raise RuntimeError("protected or target preservation failed")
        receipt.update(intended[head], slots_per_head=budget,
                       target_tokens_total=len(target), target_fully_present_after=True)
    return exact, sham, swaps


def frozen_variants(session, target_ids):
    original = [s.clone() for s in session.keep_indices("P")]
    scores = session.score("P")
    oracle, sham, receipts = [], [], []
    for layer, (p, s) in enumerate(zip(original, scores)):
        fixed, random_control, detail = force_record_sets(p, s, target_ids, session.prefix_length,
            sink_tokens=session.config.sink_tokens, recent_tokens=session.config.recent_tokens,
            seed=session.config.seed, layer_idx=layer)
        oracle.append(fixed)
        sham.append(random_control)
        receipts.append({"layer_idx": layer, "heads": detail})
    return {"P": original, "oracle_target_record": oracle, "sham_same_removals": sham}, receipts


def p_input_key(identity, row, cfg, dtype, version):
    # Same fields as existing run.baseline_key, with its stored broad metric
    # version explicit rather than mutating the original run module's globals.
    return digest({"version": version, "model": identity, "prompt_ids": row["prompt_ids"],
        "prefix_length": row["prefix_length"], "answer_limit": row["max_new_tokens"],
        "scoring": [row["score_contract"], row.get("expected"), row.get("references")],
        "arm": "P", "allocation": {"keep_fraction": cfg.keep_fraction,
            "sink": cfg.sink_tokens, "recent": cfg.recent_tokens}, "dtype": dtype,
        "decode": "raw_greedy_argmax", "official_EA_horizon": 512})


def reusable_p(source, identity, row, cfg, dtype, keep_sha):
    if source is None:
        return None
    version = BASELINE_VERSION + "_" + VERSION
    target = {"model": identity, "backend": "native_HF_Qwen2_SDPA_original_position_cache_v1",
        "dtype": dtype, "config": asdict(cfg), "decode": "raw_greedy_argmax", "baseline_version": version,
        "task_protocol": "context first, question unseen by compression; not original LongBench prompt order",
        "sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("adapter.py", "ops.py")}}
    rows = candidate_rows(source, target, {row["row_id"]}, ["P"], kind="pm",
        row_keys={(row["row_id"], "P"): p_input_key(identity, row, cfg, dtype, version)})
    if not rows:
        raise ValueError("requested old P result is absent; do not silently rerun it")
    saved = rows[0]
    if saved.get("keep_indices_sha256") != keep_sha:
        raise ValueError("current P set differs from old control; reuse is invalid")
    return {**saved, "reused_P_control": True,
            "reuse_scope": "same model/input/scoring/config/native reader and exact current P keep hash; no new P generation"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reuse-p", help="matching original balanced run directory")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--root", default="/root/autodl-tmp/position_overnight_20260909")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    available = {r["row_id"]: r for r in records(a.data)}
    rows = [available[rid] for rid in ROW_IDS]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    descriptions = {}
    for row in rows:
        spans, ids = target_spans(row, tokenizer)
        descriptions[row["row_id"]] = {"source_spans": spans, "target_record_token_ids": ids,
            "prefix_ids_sha256": digest(row["prefix_ids"]), "prompt_ids_sha256": digest(row["prompt_ids"])}
    cfg = OracleConfig(samples_per_head=256, horizon=128, seed=20260909, keep_fraction=.25)
    identity = model_identity(a.model)
    contract = {"probe": "privileged_whole_target_record_same_budget_all_layers_v1",
        "row_ids": list(ROW_IDS), "model": identity, "dtype": a.dtype, "config": asdict(cfg),
        "source_sha256": digest(Path(__file__).read_text()), "targets": descriptions,
        "row_selection": "fixed first two DEV prose256 rows where prior Full generation was correct; selected headroom only, not an overall effect estimate",
        "privilege": "future query key/ordinal identifies original source record; literal value only locates that existing sentence, never injected as answer tokens",
        "arms": ["P", "oracle_target_record", "sham_same_removals"],
        "scheduler": "root must hold its single external GPU lock; this module does not acquire a second lock",
        "dry_run": not a.execute}
    print(json.dumps(contract, ensure_ascii=False), flush=True)
    if not a.execute:
        return
    if (Path(a.root) / "STOP").exists():
        raise SystemExit("STOP exists; root owns execution")
    out = Path(a.output)
    if (out / "status.json").exists():
        raise FileExistsError("preserve previous evidence; choose a new output directory")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(4)
    write_json(out / "contract.json", contract)
    write_json(out / "status.json", {"status": "LOADING"})
    started, completed = time.perf_counter(), []
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
            torch_dtype=getattr(torch, a.dtype), attn_implementation="sdpa").to(a.device).eval()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        with (out / "per_example.jsonl").open("w") as stream:
            for row in rows:
                if (Path(a.root) / "STOP").exists():
                    break
                session = BalancedValueSession(model, row["prefix_ids"], cfg).prefill()
                variants, swaps = frozen_variants(session, descriptions[row["row_id"]]["target_record_token_ids"])
                hashes = {arm: digest([x.cpu().tolist() for x in sets]) for arm, sets in variants.items()}
                old_p = reusable_p(a.reuse_p, identity, row, cfg, a.dtype, hashes["P"])
                evidence = {"row_id": row["row_id"], "swaps_per_layer_head": swaps,
                    "keep_hashes": hashes, "all_sets_frozen_before_question_ingestion": True,
                    "prefix_length": session.prefix_length, "budget_per_head": session.total_budget,
                    "original_prefix_kv_bytes": cache_nbytes(session.cache),
                    "original_prefix_ids_sha256": digest(row["prefix_ids"]),
                    "privileged_target_lookup": descriptions[row["row_id"]]}
                write_json(out / (row["row_id"] + ".intervention.json"), evidence)
                torch.save({arm: [x.cpu() for x in sets] for arm, sets in variants.items()},
                           out / (row["row_id"] + ".keep_sets.pt"))
                for arm, indices in variants.items():
                    if arm == "P" and old_p is not None:
                        result = old_p
                    else:
                        branch = session.branch("P", indices=indices)
                        branch.consume(row["suffix_ids"])
                        generated = branch.generate(row["max_new_tokens"], eos)
                        result = {"row_id": row["row_id"], "task": row["task"], "arm": arm,
                            "config": asdict(cfg), "generated_token_ids": generated["generated_ids"],
                            "keep_indices_sha256": hashes[arm], "reused_P_control": False,
                            "generation": generated, "prefix_kv_bytes": generated["initial_prefix_kv_bytes"],
                            "final_kv_bytes": generated["final_kv_bytes"],
                            **score(row, generated["generated_ids"], tokenizer, eos)}
                        del branch
                    if any(layer.get_seq_length() != session.prefix_length for layer in session.cache.layers):
                        raise RuntimeError("question branch polluted the original prefix")
                    result.update(diagnostic_only=True, privileged_target_record_intervention=True,
                                  headroom_selected_DEV=True)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                completed.append(row["row_id"])
                write_json(out / "status.json", {"status": "RUNNING", "completed_rows": completed})
                del session, variants
        write_json(out / "status.json", {"status": "COMPLETE" if len(completed) == len(rows) else "STOPPED",
                   "completed_rows": completed, "elapsed_seconds": time.perf_counter() - started})
    except BaseException as error:
        write_json(out / "status.json", {"status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
