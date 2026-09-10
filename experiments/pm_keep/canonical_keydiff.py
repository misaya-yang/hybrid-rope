"""KeyDiff on first native pre-RoPE K projection; fixed-budget local adaptation.

K_pre uses the unchanged pinned author's negative cosine-to-mean score on the
first k_proj output. K_post is scored on the same prefix as an implementation
control. Original post-RoPE K/V, native positions, GQA and reader are unchanged.
No inverse BF16 rotation or second K projection is used. This is neither the
failed post-K-based Q sampler nor an established/new positional method.
Default is CPU dryrun; --execute requires root's external single GPU lock.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from experiments.broad_position_eval.scoring import VERSION, score
from .adapter import AdapterConfig, PrefillSession, _sync
from .baselines import load_author_class, source_receipt
from .run import BASELINE_VERSION, digest, model_identity, records, write_json


ROW_IDS = tuple(f"broad_retrieval_dev_{i:03d}_{cell}" for i in range(8)
                for cell in ("repeat_16", "prose_16", "prose_256"))
LABEL = "author_KeyDiff_pre_native_RoPE_keys_fixed_budget_v1"


class CanonicalKeyDiffSession(PrefillSession):
    @torch.inference_mode()
    def prefill(self, external_scorers=None):
        callbacks = dict(external_scorers or {})
        if {"K_pre", "K_post"} & callbacks.keys():
            raise ValueError("canonical KeyDiff arms cannot be overridden")
        press = load_author_class("KeyDiffPress")(compression_ratio=0.0)
        raw_keys, handles, captures = {}, [], []
        live_max = 0
        def capture(index):
            def hook(module, args, output):
                nonlocal live_max
                if index not in raw_keys:
                    raw_keys[index] = output.detach()
                    captures.append(index)
                    live_max = max(live_max, sum(x.numel() * x.element_size() for x in raw_keys.values()))
            return hook
        for index, layer in enumerate(self.model.model.layers):
            handles.append(layer.self_attn.k_proj.register_forward_hook(capture(index)))
        def post(data):
            return press.score(data.attention_module, data.hidden_states, data.keys,
                               data.values, None, {})[0]
        def pre(data):
            raw = raw_keys.pop(data.layer_idx)
            keys = raw.view(1, self.prefix_length, self.model.config.num_key_value_heads, -1).transpose(1, 2)
            return press.score(data.attention_module, data.hidden_states, keys,
                               data.values, None, {})[0]
        # Consume first native K only after any external scorer re-projections.
        callbacks.update(K_post=post, K_pre=pre)
        try:
            super().prefill(callbacks)
        finally:
            for handle in handles:
                handle.remove()
            raw_keys.clear()
        if captures != list(range(len(self.model.model.layers))):
            raise RuntimeError("native first K projection was not captured once per layer")
        self.projection_receipt = {"source": "first native k_proj output before apply_rotary_pos_emb",
            "captured_layers": captures, "max_live_raw_K_bytes": live_max,
            "raw_K_released_after_each_layer_score": True,
            "inverse_rotation_used": False, "K_reprojection_used_by_candidate": False,
            "original_post_RoPE_cache_and_reader_unchanged": True}
        return self


def baseline_fingerprint(version, identity, row, arm, config, dtype):
    return digest({"version": version, "model": identity, "prompt_ids": row["prompt_ids"],
        "prefix_length": row["prefix_length"], "answer_limit": row["max_new_tokens"],
        "scoring": [row["score_contract"], row.get("expected"), row.get("references")],
        "arm": arm, "allocation": {} if arm == "F" else {"keep_fraction": config.keep_fraction,
            "sink": config.sink_tokens, "recent": config.recent_tokens},
        "dtype": dtype, "decode": "raw_greedy_argmax", "official_EA_horizon": 512})


def cached_references(source, identity, row, config, dtype, post_keep_sha):
    if source is None:
        return []
    folder = Path(source)
    manifest = json.loads((folder / "manifest.json").read_text())
    version = BASELINE_VERSION + "_" + VERSION
    if manifest["model"] != identity or manifest["dtype"] != dtype or manifest["baseline_version"] != version:
        raise ValueError("fixed reference model/dtype/reader-scoring contract differs")
    matches = [r for r in records(folder / "per_example.jsonl") if r["row_id"] == row["row_id"] and r["arm"] in ("F", "K")]
    if {r["arm"] for r in matches} != {"F", "K"}:
        raise ValueError("requested fixed F/K references absent; do not silently regenerate")
    output = []
    for old in matches:
        expected = baseline_fingerprint(version, identity, row, old["arm"], config, dtype)
        if old.get("baseline_cache_key") != expected:
            raise ValueError("fixed reference input/score/allocation fingerprint differs")
        if old["arm"] == "K" and old.get("keep_indices_sha256") != post_keep_sha:
            raise ValueError("same-prefix K_post set does not match existing original K")
        output.append({**old, "reused_fixed_reference": True, "reused_from_run": str(folder.resolve())})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reuse-from", help="completed three-cell run containing original K and F")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--root", default="/root/autodl-tmp/position_overnight_20260909")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    available = {r["row_id"]: r for r in records(a.data)}
    rows = [available[rid] for rid in ROW_IDS]
    if any(r["split"] != "dev" or r["prefix_ids"] + r["suffix_ids"] != r["prompt_ids"] for r in rows):
        raise ValueError("only the fixed 24 DEV rows with exact prefix boundary are supported")
    config = AdapterConfig(samples_per_head=1, horizon=1, keep_fraction=.25, sink_tokens=4, recent_tokens=256)
    identity, author = model_identity(a.model), source_receipt()
    contract = {"label": LABEL, "model": identity, "author_source": author, "config": asdict(config),
        "row_ids": list(ROW_IDS), "dtype": a.dtype, "source_sha256": digest(Path(__file__).read_text()),
        "input_sha256": {r["row_id"]: digest(r["prompt_ids"]) for r in rows},
        "scientific_change": "only KeyDiff score input is native pre-RoPE K instead of post-RoPE K",
        "new_generation_arms": ["K_pre"], "same_prefix_score_control": "K_post",
        "fixed_references": "reuse F/K outputs; never cache K_pre under K",
        "dry_run": not a.execute, "scheduler": "root's external single GPU lock"}
    print(json.dumps(contract, ensure_ascii=False), flush=True)
    if not a.execute:
        return
    if (Path(a.root) / "STOP").exists():
        raise SystemExit("STOP exists")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    out = Path(a.output)
    if (out / "status.json").exists():
        raise FileExistsError("preserve original results; choose a new output directory")
    torch.set_num_threads(4)
    write_json(out / "contract.json", contract)
    write_json(out / "status.json", {"status": "LOADING"})
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started, completed = time.perf_counter(), []
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
                torch_dtype=getattr(torch, a.dtype), attn_implementation="sdpa").to(a.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        with (out / "per_example.jsonl").open("w") as stream:
            for row in rows:
                if (Path(a.root) / "STOP").exists():
                    break
                if a.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                _sync(next(model.parameters()).device)
                row_started = time.perf_counter()
                session = CanonicalKeyDiffSession(model, row["prefix_ids"], config).prefill()
                selection_started = time.perf_counter()
                sets = {arm: session.keep_indices(arm) for arm in ("K_pre", "K_post")}
                _sync(session.device)
                selection_seconds = time.perf_counter() - selection_started
                hashes = {arm: digest([x.cpu().tolist() for x in indices]) for arm, indices in sets.items()}
                fixed = cached_references(a.reuse_from, identity, row, config, a.dtype, hashes["K_post"])
                torch.save({arm: [x.cpu() for x in indices] for arm, indices in sets.items()},
                           out / (row["row_id"] + ".keep_sets.pt"))
                write_json(out / (row["row_id"] + ".selection.json"), {
                    "row_id": row["row_id"], "prefix_ids_sha256": digest(row["prefix_ids"]),
                    "native_projection": session.projection_receipt, "keep_hashes": hashes,
                    "budget_per_head": session.total_budget, "all_sets_frozen_before_question": True,
                    "timings": session.timings})
                branch = session.branch("K_pre", sets["K_pre"]).consume(row["suffix_ids"])
                generated = branch.generate(row["max_new_tokens"], eos)
                timing = dict(generated["timings"])
                timing.update(native_prefill_residual_seconds=session.timings["native_prefill_residual_seconds"],
                    existing_adapter_query_capture_seconds=session.timings["query_capture_seconds"],
                    K_pre_scoring_seconds=session.timings["score_seconds"]["K_pre"],
                    K_post_implementation_check_seconds=session.timings["score_seconds"]["K_post"],
                    pre_post_fixed_budget_selection_seconds=selection_seconds)
                result = {"row_id": row["row_id"], "task": row["task"], "arm": "K_pre", "label": LABEL,
                    "generated_token_ids": generated["generated_ids"], "generation": generated,
                    "keep_indices_sha256": hashes["K_pre"], "timings": timing,
                    "measured_row_end_to_end_seconds": time.perf_counter() - row_started,
                    "accounted_compute_components_seconds": sum(timing.values()),
                    "estimated_K_pre_without_post_score_seconds": sum(v for k, v in timing.items() if k != "K_post_implementation_check_seconds"),
                    "cost_estimate_scope": "estimate still includes shared pre/post sorting; measured row time also includes saved keep evidence and reference I/O",
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if a.device == "cuda" else None,
                    "peak_scope": "shared prefix plus pre/post score verification and one native gathered branch",
                    **score(row, generated["generated_ids"], tokenizer, eos)}
                for entry in [*fixed, result]:
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stream.flush()
                completed.append(row["row_id"])
                write_json(out / "status.json", {"status": "RUNNING", "completed_rows": completed})
                del session, branch, sets
        write_json(out / "status.json", {"status": "COMPLETE" if len(completed) == len(rows) else "STOPPED",
                   "completed_rows": completed, "elapsed_seconds": time.perf_counter() - started})
    except BaseException as error:
        write_json(out / "status.json", {"status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
