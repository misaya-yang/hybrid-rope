"""DEV diagnostic: prefix-only keep sets evaluated on one Full free-generation path.

Future Q never enters a selector. We freeze BalancedValueSession P/C/U/K first,
record native post-RoPE Q at the last question token and first ingested answer
(default), then reuse the final Full cache with absolute-position visibility.
Only Full generates; masked-read errors are local common-trajectory diagnostics,
not outcomes of compressed generation or deployed future-query access.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from .balanced_queries import BalancedConfig, BalancedValueSession
from .baselines import keydiff_prefix_scores, source_receipt
from .run import digest, model_identity, records, score, write_json


class NativeQueryTrace(AbstractContextManager):
    """Capture only small Q tensors, never per-step K/V snapshots."""

    def __init__(self, model, positions):
        self.model = model
        self.positions = set(int(p) for p in positions)
        self.queries = {i: {} for i in range(len(model.model.layers))}
        self.handles = []
        self.active = {}

    def __enter__(self):
        def before(index):
            def hook(module, args, kwargs):
                self.active.pop(index, None)
                if len(self.queries[index]) == len(self.positions):
                    return
                pos = kwargs["cache_position"]
                if pos.numel() != 1:
                    raise ValueError("trace supports native one-token continuation only")
                absolute = int(pos.item())
                if absolute in self.positions:
                    self.active[index] = (absolute, kwargs["position_embeddings"])
            return hook

        def after_projection(index):
            def hook(module, args, output):
                current = self.active.pop(index, None)
                if current is None:
                    return
                absolute, (cos, sin) = current
                heads = self.model.config.num_attention_heads
                q = output.view(1, 1, heads, -1).transpose(1, 2)
                rotated, _ = apply_rotary_pos_emb(q, q, cos, sin)
                self.queries[index][absolute] = rotated[0, :, 0].detach().clone()
            return hook

        for index, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(before(index), with_kwargs=True))
            self.handles.append(layer.self_attn.q_proj.register_forward_hook(after_projection(index)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.active.clear()
        return False


def _keep_mask(indices, heads, prefix_length, visible_length, device):
    ids = torch.as_tensor(indices, device=device, dtype=torch.long)
    if ids.ndim != 2 or ids.shape[0] != heads or not ids.shape[1]:
        raise ValueError("keep set must be nonempty [Hkv,B]")
    if bool(((ids < 0) | (ids >= prefix_length)).any()):
        raise ValueError("keep set can contain original prefix positions only")
    if ids.shape[1] > 1 and not bool((ids[:, 1:] > ids[:, :-1]).all()):
        raise ValueError("keep set must be sorted and unique")
    allowed = torch.zeros((heads, visible_length), dtype=torch.bool, device=device)
    allowed.scatter_(1, ids, True)
    allowed[:, prefix_length:] = True  # all already-visible question/generated K/V
    return allowed


@torch.inference_mode()
def evaluate_native_query(q_post, final_keys, final_values, query_position, prefix_length,
                          keep_sets, *, attention_scale):
    """FP32 grouped QK evaluation; returns JSON-like metrics and Full output.

    Native inputs: Q[Hq,D], final K/V[Hkv,Tfinal,D]. Full keeps absolute slots
    0..Tfinal-1, so slicing 0..query_position is an explicit causal mask. New
    question/generated keys enter BOTH normalizers; later keys never enter.
    """
    if q_post.ndim != 2 or final_keys.ndim != 3 or final_values.shape != final_keys.shape:
        raise ValueError("expected Q[Hq,D], K/V[Hkv,Tfinal,D]")
    hkv, total, dim = final_keys.shape
    hq = q_post.shape[0]
    if hq % hkv or q_post.shape[-1] != dim:
        raise ValueError("native contiguous GQA groups required")
    visible = int(query_position) + 1
    if not 0 < prefix_length <= query_position < total:
        raise ValueError("query must follow the prefix and exist in the final Full cache")
    if q_post.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("diagnostic FP32 matmul requires TF32 disabled")
    with torch.autocast(device_type=q_post.device.type, enabled=False):
        q = q_post.float().reshape(hkv, hq // hkv, dim)
        k = final_keys[:, :visible].float()
        v = final_values[:, :visible].float()
        logits = torch.einsum("hgd,htd->hgt", q, k) * attention_scale
        p = logits.softmax(-1, dtype=torch.float32)
        full = torch.einsum("hgt,htd->hgd", p, v)
        prefix_mass = p[..., :prefix_length].sum(-1)
        output = {"query_position": int(query_position), "visible_key_count": visible,
                  "new_visible_key_count": visible - prefix_length,
                  "future_key_count_masked": total - visible,
                  "full_prefix_mass_per_query_head": prefix_mass.flatten().tolist(), "arms": {}}
        for arm, indices in keep_sets.items():
            allowed = _keep_mask(indices, hkv, prefix_length, visible, q.device)
            allowed_prefix = allowed[:, None, :prefix_length]
            kept_prefix_mass = (p[..., :prefix_length] * allowed_prefix).sum(-1)
            retained = (p * allowed[:, None]).sum(-1)
            masked_p = logits.masked_fill(~allowed[:, None], -torch.inf).softmax(-1, dtype=torch.float32)
            compact = torch.einsum("hgt,htd->hgd", masked_p, v)
            error = (compact - full).norm(dim=-1)
            conditional = kept_prefix_mass / prefix_mass.clamp_min(torch.finfo(torch.float32).tiny)
            output["arms"][arm] = {
                "kept_prefix_mass_per_query_head": kept_prefix_mass.flatten().tolist(),
                "conditional_prefix_retention_per_query_head": conditional.flatten().tolist(),
                "retained_total_mass_per_query_head": retained.flatten().tolist(),
                "attention_output_l2_error_per_query_head": error.flatten().tolist(),
                "attention_output_relative_l2_error_per_query_head":
                    (error / full.norm(dim=-1).clamp_min(1e-12)).flatten().tolist(),
                "mean_kept_prefix_mass": float(kept_prefix_mass.mean()),
                "mean_conditional_prefix_retention": float(conditional.mean()),
                "mean_retained_total_mass": float(retained.mean()),
                "mean_output_l2_error": float(error.mean()),
            }
    return output, full.reshape(hq, dim)


def freeze_prefix_choices(session, arms=("P", "C", "U", "K")):
    """Called before a question exists in any cache; no future-Q argument."""
    keep = {arm: [x.detach().clone() for x in session.keep_indices(arm)] for arm in arms}
    keep["F"] = session.keep_indices("F")
    proxy = {}
    for objective in arms:
        scores = session.score(objective)
        proxy[objective] = {}
        for arm, chosen in keep.items():
            values = [s.gather(1, idx).sum(-1) for s, idx in zip(scores, chosen)]
            proxy[objective][arm] = {
                "per_layer_kv_head": [v.cpu().tolist() for v in values],
                "mean": float(torch.cat(values).mean()),
            }
    overlaps = {}
    for other in ("C", "U", "K"):
        if "P" not in keep or other not in keep:
            continue
        layers = []
        for p_layer, other_layer in zip(keep["P"], keep[other]):
            heads = []
            for p, c in zip(p_layer.cpu().tolist(), other_layer.cpu().tolist()):
                left, right = set(p), set(c)
                heads.append({"intersection": len(left & right), "union": len(left | right),
                              "p_only": len(left - right), "other_only": len(right - left),
                              "jaccard": len(left & right) / len(left | right),
                              "overlap_fraction_of_P": len(left & right) / len(left)})
            layers.append(heads)
        overlaps[f"P_vs_{other}"] = layers
    report = {"proxy_objective_scores_on_each_keep_set": proxy,
              "P_objective_P_minus_C": proxy["P"]["P"]["mean"] - proxy["P"]["C"]["mean"],
              "proxy_scope": "original sampled prefix objective, raw-value-norm weighted for P/C/U; self-scoring is not independent validation",
              "keep_sha256": {a: digest([x.cpu().tolist() for x in sets]) for a, sets in keep.items()},
              "set_overlaps": overlaps,
              "prefix_length": session.prefix_length,
              "selected_before_question": True}
    return keep, report


@torch.inference_mode()
def full_free_generation_trace(session, suffix_ids, max_new_tokens, eos_ids, *, answer_query_count=1):
    suffix = torch.as_tensor(suffix_ids).flatten().tolist()
    if not suffix or answer_query_count < 0:
        raise ValueError("nonempty question suffix and nonnegative query count required")
    full = session.branch("F")
    # No hooks or KV copies for earlier question tokens.
    full.consume(suffix[:-1])
    last_question = session.prefix_length + len(suffix) - 1
    wanted = [last_question, *range(last_question + 1, last_question + 1 + answer_query_count)]
    with NativeQueryTrace(session.model, wanted) as trace:
        full.consume(suffix[-1:])
        generated = full.generate(max_new_tokens, eos_ids)
    positions = set.intersection(*(set(v) for v in trace.queries.values()))
    if any(set(v) != positions for v in trace.queries.values()):
        raise RuntimeError("not all layers captured the same actual Q positions")
    if last_question not in positions:
        raise RuntimeError("last question query was not captured")
    expected_length = full.logical_position
    if any(layer.get_seq_length() != expected_length for layer in full.cache.layers):
        raise RuntimeError("Full cache is not contiguous in original absolute positions")
    report = {"requested_positions": wanted, "captured_positions": sorted(positions),
              "uncaptured_positions": sorted(set(wanted) - positions),
              "last_question_position": last_question,
              "answer_query_semantics": "Q of an ingested freely generated answer token predicts its next token; first sampled answer uses last-question Q",
              "missing_query_reason": "EOS or generation cap may stop before a sampled answer token is ingested; no extra teacher-forced forward is added",
              "free_generation": generated,
              "final_full_cache_length": expected_length,
              "kv_snapshot_policy": "one prefix cache and one independent final Full branch; Q-only trace, no per-timestep full-KV copies",
              "captured_q_bytes": sum(q.numel() * q.element_size() for layer in trace.queries.values() for q in layer.values())}
    return full, trace.queries, report


def evaluate_trace(session, full, queries, keep):
    per_query = []
    for index, layer in enumerate(full.cache.layers):
        for pos, q in sorted(queries[index].items()):
            result, _ = evaluate_native_query(q, layer.keys[0], layer.values[0], pos,
                    session.prefix_length, {arm: sets[index] for arm, sets in keep.items()},
                    attention_scale=session.model.model.layers[index].self_attn.scaling)
            result["layer_idx"] = index
            result["P_minus_C_kept_prefix_mass"] = (result["arms"]["P"]["mean_kept_prefix_mass"]
                                                    - result["arms"]["C"]["mean_kept_prefix_mass"])
            result["P_minus_C_output_l2_error"] = (result["arms"]["P"]["mean_output_l2_error"]
                                                   - result["arms"]["C"]["mean_output_l2_error"])
            per_query.append(result)
    summaries = {}
    for pos in sorted({r["query_position"] for r in per_query}):
        rows = [r for r in per_query if r["query_position"] == pos]
        summaries[str(pos)] = {
            "P_minus_C_kept_prefix_mass": sum(r["P_minus_C_kept_prefix_mass"] for r in rows) / len(rows),
            "P_minus_C_output_l2_error": sum(r["P_minus_C_output_l2_error"] for r in rows) / len(rows),
            "mean_full_prefix_mass": sum(sum(r["full_prefix_mass_per_query_head"]) / len(r["full_prefix_mass_per_query_head"]) for r in rows) / len(rows),
            "F_max_output_l2_error": max(max(r["arms"]["F"]["attention_output_l2_error_per_query_head"]) for r in rows),
        }
    return {"per_layer_query": per_query, "per_position_summary": summaries,
            "averaging": "query heads equally weighted inside layer, layers equally weighted; per-head/layer data retained",
            "scope": "common Full trajectory masked-read counterfactual; all visible new K/V included, future K/V excluded; not compressed-trajectory answers"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--row-ids", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--root", default="/root/autodl-tmp/position_overnight_20260909")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--keep-fraction", type=float, default=.25)
    p.add_argument("--answer-query-count", type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent backend fallback")
    wanted = set(a.row_ids)
    rows = [r for r in records(a.data) if r["row_id"] in wanted]
    if {r["row_id"] for r in rows} != wanted:
        raise ValueError("requested rows absent")
    if any(r["split"] != "dev" or r["prefix_ids"] + r["suffix_ids"] != r["prompt_ids"] for r in rows):
        raise ValueError("DEV-only trace with exact prefix/suffix boundary required")
    out = Path(a.output)
    if (out / "status.json").exists():
        raise FileExistsError("preserve the existing probe; choose a new output directory")
    cfg = BalancedConfig(horizon=a.horizon, samples_per_head=a.samples, keep_fraction=a.keep_fraction)
    contract = {"probe": "balanced_prefix_choices_actual_full_future_Q_v1", "config": asdict(cfg),
                "model": model_identity(a.model), "row_ids": a.row_ids,
                "dtype": a.dtype, "device": a.device,
                "inputs": {r["row_id"]: {"prompt_sha256": digest(r["prompt_ids"]),
                    "score_contract": r["score_contract"], "max_new_tokens": r["max_new_tokens"]}
                    for r in rows},
                "answer_query_count": a.answer_query_count,
                "future_queries_used_only_after_choices_frozen": True,
                "source_sha256": digest(Path(__file__).read_text()),
                "dependencies": {name: digest(Path(__file__).with_name(name).read_text()) for name in
                    ("adapter.py", "ops.py", "balanced_queries.py", "run_followup.py", "baselines.py")},
                "keydiff": source_receipt()}
    write_json(out / "contract.json", contract)
    if (Path(a.root) / "STOP").exists():
        raise SystemExit("STOP exists; root owns scheduling")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    start = time.perf_counter()
    write_json(out / "status.json", {"status": "LOADING"})
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
                torch_dtype=getattr(torch, a.dtype), attn_implementation="sdpa").to(a.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        completed = []
        for row in rows:
            if (Path(a.root) / "STOP").exists():
                break
            write_json(out / "status.json", {"status": "RUNNING", "row_id": row["row_id"], "completed_rows": completed})
            session = BalancedValueSession(model, row["prefix_ids"], cfg).prefill({"K": keydiff_prefix_scores})
            keep, proxy = freeze_prefix_choices(session)
            full, queries, trajectory = full_free_generation_trace(session, row["suffix_ids"],
                    row["max_new_tokens"], eos, answer_query_count=a.answer_query_count)
            result = {"row_id": row["row_id"], "task": row["task"], "prefix_choices": proxy,
                      "trajectory": trajectory, "diagnostic": evaluate_trace(session, full, queries, keep),
                      "full_answer": score(row, trajectory["free_generation"]["generated_ids"], tokenizer, eos)}
            write_json(out / (row["row_id"] + ".json"), result)
            torch.save({"queries": {i: {pos: q.cpu() for pos, q in qs.items()} for i, qs in queries.items()},
                        "keep_indices": {arm: [x.cpu() for x in sets] for arm, sets in keep.items()}},
                       out / (row["row_id"] + ".trace.pt"))
            completed.append(row["row_id"])
            del full, session, queries, keep
        write_json(out / "status.json", {"status": "COMPLETE" if len(completed) == len(rows) else "STOPPED",
                   "completed_rows": completed, "elapsed_seconds": time.perf_counter() - start})
    except BaseException as error:
        write_json(out / "status.json", {"status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
