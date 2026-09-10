"""Privileged two-input question-KV/source cross; no Full final logits reused.

Only previously observed question tokens (excluding the final prompt token)
are imported. Prefix support is fixed to saved P/oracle indices. This is a
state/content-transfer diagnostic, not a same-resource deployment method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .adapter import _sync, cache_nbytes
from .balanced_queries import BalancedValueSession
from .run import digest, model_identity, records, write_json
from .target_record_oracle import OracleConfig, ROW_IDS, score


def tensor_hash(tensors):
    h = hashlib.sha256()
    for x in tensors:
        h.update(str((tuple(x.shape), str(x.dtype))).encode())
        h.update(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def append_question_state(branch, donor, *, donor_prefix_slots, question_tokens):
    """Import only donor's question K/V; deliberately invalidate inherited logits."""
    t, n = branch.prefix_length, question_tokens
    b = branch.cache.get_seq_length()
    if n < 0 or branch.logical_position != t or donor.logical_position != t + n:
        raise ValueError("unexpected logical position at splice")
    if donor.cache.get_seq_length() != donor_prefix_slots + n:
        raise ValueError("donor must stop before the final prompt token")
    prefix_hashes, question_hashes = [], []
    _sync(branch.device)
    started = time.perf_counter()
    for i, (dst, src) in enumerate(zip(branch.cache.layers, donor.cache.layers)):
        if dst.get_seq_length() != b or src.get_seq_length() != donor_prefix_slots + n:
            raise ValueError("layer lengths disagree")
        old_k, old_v = dst.keys, dst.values
        prefix_hashes.append(tensor_hash((old_k, old_v)))
        k = src.keys[:, :, donor_prefix_slots:donor_prefix_slots + n].clone()
        v = src.values[:, :, donor_prefix_slots:donor_prefix_slots + n].clone()
        question_hashes.append(tensor_hash((k, v)))
        if n:
            branch.cache.update(k, v, i)
        after = branch.cache.layers[i]
        if not torch.equal(after.keys[:, :, :b], old_k) or not torch.equal(after.values[:, :, :b], old_v):
            raise RuntimeError("splice changed selected prefix")
        if tensor_hash((after.keys[:, :, b:], after.values[:, :, b:])) != question_hashes[-1]:
            raise RuntimeError("imported question tensors differ from donor")
    branch.logical_position = t + n
    branch.last_logits = None  # consume final token locally before generation
    _sync(branch.device)
    return {"prefix_tensor_hashes": prefix_hashes, "question_tensor_hashes": question_hashes,
            "prefix_slots": b, "question_slots": n, "donor_prefix_slots": donor_prefix_slots,
            "physical_length_before_final": b + n, "logical_position_before_final": t + n,
            "imported_question_logical_positions": [t, t + n],
            "full_prefix_imported": False, "donor_logits_imported": False,
            "splice_and_verification_seconds": time.perf_counter() - started}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--oracle-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    root, old, out = Path(a.root), Path(a.oracle_run), Path(a.output)
    original = json.loads((old / "contract.json").read_text())
    identity = model_identity(a.model)
    for key in ("config_sha256", "tokenizer_sha256", "weights"):
        if identity[key] != original["model"][key]:
            raise ValueError("model identity mismatch: " + key)
    available = {r["row_id"]: r for r in records(a.data)}
    rows = [available[rid] for rid in ROW_IDS]
    old_rows = {(r["row_id"], r["arm"]): r for r in records(old / "per_example.jsonl")}
    cfg = OracleConfig(**original["config"])
    supports, input_receipts = {}, {}
    for row in rows:
        rid = row["row_id"]
        if row["split"] != "dev" or len(row["suffix_ids"]) < 1:
            raise ValueError("fixed DEV inputs with nonempty question required")
        if row["prefix_ids"] + row["suffix_ids"] != row["prompt_ids"]:
            raise ValueError("prefix/suffix boundary mismatch")
        target = original["targets"][rid]
        if digest(row["prefix_ids"]) != target["prefix_ids_sha256"] or digest(row["prompt_ids"]) != target["prompt_ids_sha256"]:
            raise ValueError("input differs from original intervention")
        saved = torch.load(old / (rid + ".keep_sets.pt"), map_location="cpu", weights_only=True)
        receipt = json.loads((old / (rid + ".intervention.json")).read_text())
        supports[rid] = {arm: saved[arm] for arm in ("P", "oracle_target_record")}
        for arm, indices in supports[rid].items():
            sha = digest([x.tolist() for x in indices])
            if sha != receipt["keep_hashes"][arm] or sha != old_rows[rid, arm]["keep_indices_sha256"]:
                raise ValueError("keep identity mismatch")
            if any(x.shape[1] != receipt["budget_per_head"] for x in indices):
                raise ValueError("budget mismatch")
        input_receipts[rid] = {"prefix_ids_sha256": digest(row["prefix_ids"]),
            "prompt_ids_sha256": digest(row["prompt_ids"]), "keep_hashes": receipt["keep_hashes"],
            "prefix_length": len(row["prefix_ids"]), "suffix_length": len(row["suffix_ids"])}
    contract = {"probe": "question_state_cross_before_final_token_v1", "row_ids": list(ROW_IDS),
        "model": identity, "inputs": input_receipts, "dtype": original["dtype"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "old_run": str(old), "new_generations": 4,
        "scope": "Full question-KV content/state transfer with fixed prefix supports; privileged selected DEV diagnostic"}
    if not a.execute:
        print(json.dumps(contract)); return
    if (root / "STOP").exists() or out.exists():
        raise RuntimeError("STOP or existing output; no launch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent fallback")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    torch.set_num_threads(4)
    write_json(out / "contract.json", contract)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "status.json", {"status": "LOADING"})
    started, completed = time.perf_counter(), []
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
            torch_dtype=getattr(torch, original["dtype"]), attn_implementation="sdpa").cuda().eval()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        with (out / "per_example.jsonl").open("w") as stream:
            for row in rows:
                if (root / "STOP").exists():
                    break
                rid, t, n = row["row_id"], len(row["prefix_ids"]), len(row["suffix_ids"]) - 1
                write_json(out / "status.json", {"status": "RUNNING", "row_id": rid, "completed_rows": completed})
                _sync(torch.device("cuda")); prefill_start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                session = BalancedValueSession(model, row["prefix_ids"], cfg).prefill()
                _sync(session.device); prefill_seconds = time.perf_counter() - prefill_start
                donor = session.branch("F").consume(row["suffix_ids"][:-1])
                donor_cost = dict(donor.timings)
                for arm, indices in supports[rid].items():
                    old_result = {**old_rows[rid, arm], "reused_cell": True,
                        "prefix_support": arm, "question_state_source": "own_compressed_branch"}
                    stream.write(json.dumps(old_result) + "\n"); stream.flush()
                    branch_started = time.perf_counter()
                    branch = session.branch(arm, indices)
                    receipt = append_question_state(branch, donor, donor_prefix_slots=t, question_tokens=n)
                    if branch.last_logits is not None:
                        raise RuntimeError("inherited logits must be invalidated")
                    branch.consume(row["suffix_ids"][-1:])
                    if branch.cache.get_seq_length() != branch.keep_indices[0].shape[1] + n + 1:
                        raise RuntimeError("answer-entry physical length mismatch")
                    receipt.update(answer_entry_physical_length=branch.cache.get_seq_length(),
                        answer_entry_logical_next=branch.logical_position,
                        answer_entry_kv_bytes=cache_nbytes(branch.cache),
                        full_prefix_prefill_seconds=prefill_seconds, full_question_source_cost=donor_cost)
                    generated = branch.generate(row["max_new_tokens"], eos)
                    receipt.update(new_branch_total_seconds=time.perf_counter() - branch_started,
                        full_prefix_kv_bytes=cache_nbytes(session.cache),
                        full_donor_kv_bytes=cache_nbytes(donor.cache),
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_scope="shared Full prefix, Full question donor and compressed branch")
                    result = {k: row.get(k) for k in ("row_id", "task", "split", "family_id", "material_cluster_id", "expected", "score_contract")}
                    result.update(arm=arm + "_full_question_state", prefix_support=arm,
                        question_state_source="Full_suffix_excluding_final_token", reused_cell=False,
                        keep_indices_sha256=digest([x.tolist() for x in indices]),
                        generated_token_ids=generated["generated_ids"], first_prediction_token=generated["generated_ids"][0],
                        generation=generated, splice=receipt, diagnostic_only=True,
                        **score(row, generated["generated_ids"], tokenizer, eos))
                    stream.write(json.dumps(result) + "\n"); stream.flush()
                    del branch
                if session.cache.get_seq_length() != t or donor.logical_position != t + n:
                    raise RuntimeError("prefix or donor was changed")
                completed.append(rid)
                del donor, session
        write_json(out / "status.json", {"status": "COMPLETE" if len(completed) == 2 else "STOPPED",
            "completed_rows": completed, "elapsed_seconds": time.perf_counter() - started})
    except BaseException as exc:
        write_json(out / "status.json", {"status": "FAILED", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
