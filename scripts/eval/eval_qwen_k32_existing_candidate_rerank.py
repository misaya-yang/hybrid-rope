#!/usr/bin/env python3
"""Rerank three frozen index candidates by their full-sequence far-source contribution."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.data.prepare_qwen_k32_far_evidence_qa import ids_sha256, sha256_file
from scripts.eval import eval_qwen_k32_far_evidence_qa as qa_eval
from scripts.eval import eval_qwen_k32_natural_nll as nll_eval
from scripts.eval.eval_qwen_k32_source_contrast_decode import (
    BOOTSTRAP_SAMPLES, EXPECTED_BRIDGE_ROWS, EXPECTED_FACTORIAL_QA,
    EXPECTED_FACTORIAL_RESULT, stratified_draws,
)
from scripts.eval.longbench_metrics import TASK_METRIC_MAP, post_process_prediction, score_prediction
from scripts.lib.checkpoint_identity import safetensors_weight_set_sha256

STATUS = "QWEN_K32_EXISTING_CANDIDATE_RERANK_COMPLETE"
EXPECTED_SOURCE_RESULT = "f257df787c1d0d638380fccff438da5bf63dba76c980c50d3ea06f25d6381795"
EXPECTED_SOURCE_EXAMPLES = "6cdbbf8e70fd1690977cc526c1ea2fe8b050e75f58631ebf37b8e17698b52686"
BOOTSTRAP_SEED = 202609043
CANDIDATE_ORDER = ("index_unit_greedy", "index_gain_greedy", "index_unit_source_contrast")


def summarize(records: list[dict], baselines: dict[tuple[str, str], dict]) -> dict:
    if len(records) != 30 or len({(row["task"], row["row_sha256"]) for row in records}) != 30:
        raise ValueError("rerank result grid is incomplete")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = {}
    deltas = {}
    for baseline in ("index_unit", "native_unit"):
        deltas[baseline] = {}
        for task in qa_eval.TASKS:
            task_rows = sorted((row for row in records if row["task"] == task),
                               key=lambda row: row["row_sha256"])
            deltas[baseline][task] = np.array([
                row["score"] - baselines[baseline, row["row_sha256"]]["score"]
                for row in task_rows])
        draws = stratified_draws(deltas[baseline], rng)
        metrics["rerank_minus_" + baseline] = {
            "mean_macro": float(np.mean([values.mean() for values in deltas[baseline].values()])),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "by_task": {task: float(values.mean()) for task, values in deltas[baseline].items()},
        }
    versus_index, versus_native = metrics["rerank_minus_index_unit"], metrics["rerank_minus_native_unit"]
    passed = (versus_index["ci95"][0] > 0 and versus_native["ci95"][0] > 0
              and sum(value > 0 for value in versus_native["by_task"].values()) >= 2)
    return {
        "metrics": metrics,
        "rerank_macro": float(np.mean([row["score"] for row in records])),
        "oracle_candidate_macro": float(np.mean([row["oracle_f1_at_candidates"] for row in records])),
        "classification": "PASS" if passed else "NOT_PASS",
        "decision_rule": (
            "PASS iff rerank-minus-index/unit and rerank-minus-Native/unit CI lower >0, "
            "and rerank-minus-Native is positive on at least 2/3 tasks."
        ),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_SAMPLES,
                      "unit": "rows resampled within each fixed task jointly across candidates/baselines"},
    }


def score_candidate(model, prompt_pair, candidate):
    import torch

    far, absent = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=prompt_pair, use_cache=True, return_dict=True, logits_to_keep=1)
        past = outputs.past_key_values
        for index, token in enumerate(candidate):
            log_probs = torch.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
            far.append(float(log_probs[0, token])); absent.append(float(log_probs[1, token]))
            if index + 1 < len(candidate):
                step = torch.tensor([[token], [token]], dtype=torch.long, device=prompt_pair.device)
                outputs = model(input_ids=step, past_key_values=past, use_cache=True,
                                return_dict=True, logits_to_keep=1)
                past = outputs.past_key_values
    far_mean, absent_mean = float(np.mean(far)), float(np.mean(absent))
    return {"mean_logp_far": far_mean, "mean_logp_absent": absent_mean,
            "rerank_score": 2.0 * far_mean - absent_mean}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qa-data-root", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--factorial-root", type=Path, required=True)
    parser.add_argument("--source-contrast-root", type=Path, required=True)
    parser.add_argument("--index-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("candidate-rerank output must be fresh")
    checks = {
        args.bridge_root / "bridge_rows.jsonl": EXPECTED_BRIDGE_ROWS,
        args.factorial_root / "results.json": EXPECTED_FACTORIAL_RESULT,
        args.factorial_root / "qa_examples.jsonl": EXPECTED_FACTORIAL_QA,
        args.source_contrast_root / "results.json": EXPECTED_SOURCE_RESULT,
        args.source_contrast_root / "examples.jsonl": EXPECTED_SOURCE_EXAMPLES,
    }
    for path, digest in checks.items():
        if sha256_file(path) != digest:
            raise ValueError(f"parent evidence identity drift: {path}")

    qa_manifest, qa_rows = qa_eval.load_data(args.qa_data_root, args.checkpoint)
    qa_by_key = {(row["task"], row["row_sha256"]): row for row in qa_rows}
    bridge_rows = [json.loads(line) for line in (args.bridge_root / "bridge_rows.jsonl").read_text().splitlines()]
    prompts = {}
    for row in bridge_rows:
        if row["condition"] in {"far", "ablated"}:
            prompts.setdefault((row["task"], row["row_sha256"]), {})[row["condition"]] = row["input_ids"]
    if set(prompts) != set(qa_by_key) or any(set(pair) != {"far", "ablated"} for pair in prompts.values()):
        raise ValueError("far/ablated prompt grid is incomplete")

    baselines, candidates = {}, {}
    arm_names = {
        "native_table_unit_gain": "native_unit",
        "index_table_unit_gain": "index_unit",
    }
    for line in (args.factorial_root / "qa_examples.jsonl").read_text().splitlines():
        row = json.loads(line); key = (row["task"], row["row_sha256"])
        if row["arm"] in arm_names:
            baselines[arm_names[row["arm"]], row["row_sha256"]] = row
        if row["arm"] == "index_table_unit_gain":
            candidates.setdefault(key, {})["index_unit_greedy"] = row["generated_token_ids"]
        elif row["arm"] == "index_table_index_gain":
            candidates.setdefault(key, {})["index_gain_greedy"] = row["generated_token_ids"]
    for line in (args.source_contrast_root / "examples.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["profile"] == "index_unit":
            candidates.setdefault((row["task"], row["row_sha256"]), {})[
                "index_unit_source_contrast"] = row["generated_token_ids"]
    if (len(baselines) != 60 or set(candidates) != set(qa_by_key)
            or any(tuple(pool) != CANDIDATE_ORDER for pool in candidates.values())):
        raise ValueError("frozen candidate/baseline grid is incomplete")

    weights = safetensors_weight_set_sha256(args.checkpoint)
    if weights != nll_eval.EXPECTED_WEIGHT_SHA256:
        raise ValueError("checkpoint identity drift")
    index = nll_eval.load_table(
        args.index_table, nll_eval.EXPECTED_INDEX_FILE_SHA256, nll_eval.EXPECTED_INDEX_SHA256)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
        configure_cuda, configure_ruler_flash_attention,
    )
    configure_cuda()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval().to("cuda")
    model.requires_grad_(False); model.config.use_cache = True
    configure_ruler_flash_attention(model)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    rotary = model.model.rotary_emb
    if nll_eval.tensor_hash(rotary.inv_freq.detach().cpu().float().numpy()) != nll_eval.EXPECTED_NATIVE_SHA256:
        raise RuntimeError("runtime Native table drift")
    active = torch.from_numpy(index).to(rotary.inv_freq.device, rotary.inv_freq.dtype)
    with torch.no_grad():
        rotary.inv_freq.copy_(active)
        if hasattr(rotary, "original_inv_freq"):
            rotary.original_inv_freq = rotary.inv_freq.detach().clone()
        rotary.attention_scaling = 1.0

    args.output.mkdir(parents=True)
    manifest = {
        "status": "QWEN_K32_EXISTING_CANDIDATE_RERANK_FROZEN",
        "checkpoint_weight_sha256": weights,
        "qa_data_manifest_sha256": qa_manifest["manifest_sha256"],
        "qa_data_rows_sha256": qa_manifest["file"]["sha256"],
        "parent_hashes": {str(path): digest for path, digest in checks.items()},
        "candidate_order": list(CANDIDATE_ORDER),
        "profile": {"name": "index_unit", "tensor_sha256": nll_eval.EXPECTED_INDEX_SHA256,
                    "attention_scaling": 1.0},
        "rerank": "2*mean_logp_far-mean_logp_absent, original tokens including EOS",
        "script_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(Path(inspect.getsourcefile(type(model)))),
        "attention_source_sha256": sha256_file(Path(inspect.getsourcefile(configure_ruler_flash_attention))),
        "model_updates": 0, "profile_selection": False, "use_cache": True,
        "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    raw_path = args.output / "examples.jsonl"; records = []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with raw_path.open("x") as handle:
        for key in sorted(candidates):
            task, row_sha = key; row, pair = qa_by_key[key], prompts[key]
            prompt_pair = torch.tensor([pair["far"], pair["ablated"]], dtype=torch.long, device="cuda")
            scored = []
            for source in CANDIDATE_ORDER:
                token_ids = candidates[key][source]
                scores = score_candidate(model, prompt_pair, token_ids)
                prediction = tokenizer.decode(token_ids, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
                processed = post_process_prediction(task, prediction)
                scored.append({"source": source, "token_ids": token_ids, "prediction": prediction,
                               "f1": score_prediction(task, TASK_METRIC_MAP[task], processed,
                                                      row["references"], row.get("all_classes", [])),
                               **scores})
            chosen = max(scored, key=lambda candidate: candidate["rerank_score"])
            record = {
                "task": task, "row_sha256": row_sha, "score": chosen["f1"],
                "prediction": chosen["prediction"], "selected_source": chosen["source"],
                "generated_token_ids": chosen["token_ids"], "generated_tokens": len(chosen["token_ids"]),
                "candidates": scored, "oracle_f1_at_candidates": max(candidate["f1"] for candidate in scored),
                "far_prompt_sha256": ids_sha256(pair["far"]),
                "absent_prompt_sha256": ids_sha256(pair["ablated"]),
                "table_sha256_float32": nll_eval.EXPECTED_INDEX_SHA256, "attention_scaling": 1.0,
            }
            records.append(record); handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
            if len(records) == 1 or len(records) % 5 == 0:
                print(f"candidate-rerank {len(records)}/30", flush=True)
            del prompt_pair
        os.fsync(handle.fileno())
    report = summarize(records, baselines)
    report.update(status=STATUS, elapsed_seconds=time.perf_counter() - started,
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                  examples_sha256=sha256_file(raw_path), run_manifest_sha256=sha256_file(manifest_path),
                  evidence_limit="Post-result fixed rerank of three frozen candidates on the same 30-row panel.")
    (args.output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
