#!/usr/bin/env python3
"""Test whether source-contrast decoding recovers the frozen index table's far-source signal."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.data.prepare_qwen_k32_far_evidence_qa import ids_sha256, sha256_file
from scripts.eval import eval_qwen_k32_far_evidence_qa as qa_eval
from scripts.eval import eval_qwen_k32_natural_nll as nll_eval
from scripts.eval.longbench_metrics import TASK_METRIC_MAP, post_process_prediction, score_prediction
from scripts.lib.checkpoint_identity import safetensors_weight_set_sha256

STATUS = "QWEN_K32_SOURCE_CONTRAST_DECODE_COMPLETE"
PROFILES = ("native_unit", "index_unit")
ARM_MAP = {
    "native_unit": "native_table_unit_gain",
    "index_unit": "index_table_unit_gain",
}
EXPECTED_BRIDGE_ROWS = "b330d1d774a239cc02477784700944daf932f081d219f128eb974304a687d75b"
EXPECTED_BRIDGE_RESULT = "b83ecd034628c625d1a1b06ed831e42cff1f142bc5d2b3051ceadb061e7a10fc"
EXPECTED_FACTORIAL_RESULT = "5d6f2f2e7dc4cc8961e1d42d87931e03d3cf540a3af4a7b7d9c776d66b5a3d9b"
EXPECTED_FACTORIAL_QA = "11073e6f1948c2ea175e590b5118b738d1b75494eaa103f8e933b2988d20197d"
BOOTSTRAP_SEED = 202609041
BOOTSTRAP_SAMPLES = 10_000


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def stratified_draws(values: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    task_draws = []
    for task in qa_eval.TASKS:
        cell = values[task]
        indices = rng.integers(0, len(cell), size=(BOOTSTRAP_SAMPLES, len(cell)))
        task_draws.append(cell[indices].mean(axis=1))
    return np.stack(task_draws, axis=1).mean(axis=1)


def summarize(contrast: list[dict], greedy: list[dict]) -> dict:
    contrast_cells = {(row["profile"], row["task"], row["row_sha256"]): row["score"]
                      for row in contrast}
    greedy_cells = {(row["profile"], row["task"], row["row_sha256"]): row["score"]
                    for row in greedy}
    if set(contrast_cells) != set(greedy_cells) or len(contrast_cells) != 60:
        raise ValueError("contrast/greedy comparison grid is incomplete")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    raw = {}
    means = {}
    for profile in PROFILES:
        raw[profile] = {}
        means[profile] = {}
        for task in qa_eval.TASKS:
            keys = sorted(key for key in contrast_cells if key[:2] == (profile, task))
            contrast_values = np.array([contrast_cells[key] for key in keys])
            greedy_values = np.array([greedy_cells[key] for key in keys])
            raw[profile][task] = contrast_values - greedy_values
            means[profile][task] = {
                "contrast": float(contrast_values.mean()),
                "greedy": float(greedy_values.mean()),
                "improvement": float(raw[profile][task].mean()),
            }
    metrics = {}
    for profile in PROFILES:
        draws = stratified_draws(raw[profile], rng)
        metrics[profile + "_contrast_minus_greedy"] = {
            "mean_macro": float(np.mean([values.mean() for values in raw[profile].values()])),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "by_task": {task: float(values.mean()) for task, values in raw[profile].items()},
        }
    contrast_delta = {}
    for task in qa_eval.TASKS:
        keys = sorted(key for key in contrast_cells if key[:2] == ("index_unit", task))
        contrast_delta[task] = np.array([
            contrast_cells["index_unit", task, key[2]]
            - contrast_cells["native_unit", task, key[2]] for key in keys])
    draws = stratified_draws(contrast_delta, rng)
    metrics["index_minus_native_contrast"] = {
        "mean_macro": float(np.mean([values.mean() for values in contrast_delta.values()])),
        "ci95": np.quantile(draws, [.025, .975]).tolist(),
        "by_task": {task: float(values.mean()) for task, values in contrast_delta.items()},
    }
    index_recovery = metrics["index_unit_contrast_minus_greedy"]
    index_vs_native = metrics["index_minus_native_contrast"]
    positive_tasks = sum(value > 0 for value in index_vs_native["by_task"].values())
    passed = (index_recovery["ci95"][0] > 0 and positive_tasks >= 2
              and index_vs_native["ci95"][0] > 0)
    return {
        "status": "QWEN_K32_SOURCE_CONTRAST_DECODE_SUMMARIZED",
        "means": means,
        "metrics": metrics,
        "classification": "PASS" if passed else "NOT_PASS",
        "decision_rule": (
            "PASS iff index contrast-minus-greedy CI lower > 0, index contrast beats Native "
            "contrast with CI lower > 0, and index-minus-Native is positive on at least 2/3 tasks."
        ),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_SAMPLES,
                      "unit": "rows resampled within each task jointly across profiles/decoders"},
    }


def source_contrast_generate(model, pair_ids, *, max_new_tokens: int, eos_token_id: int | None):
    import torch

    generated = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=pair_ids, use_cache=True, return_dict=True, logits_to_keep=1)
        past = outputs.past_key_values
        for _ in range(max_new_tokens):
            log_probs = torch.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
            next_token = (2.0 * log_probs[0] - log_probs[1]).argmax(dim=-1)
            generated.append(next_token)
            if eos_token_id is not None and int(next_token) == eos_token_id:
                break
            step = next_token.view(1, 1).expand(2, 1)
            outputs = model(input_ids=step, past_key_values=past, use_cache=True,
                            return_dict=True, logits_to_keep=1)
            past = outputs.past_key_values
    return torch.stack(generated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qa-data-root", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--factorial-root", type=Path, required=True)
    parser.add_argument("--index-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("source-contrast output must be fresh")
    bridge_rows_path = args.bridge_root / "bridge_rows.jsonl"
    bridge_result_path = args.bridge_root / "results.json"
    factorial_result_path = args.factorial_root / "results.json"
    factorial_qa_path = args.factorial_root / "qa_examples.jsonl"
    expected = {
        bridge_rows_path: EXPECTED_BRIDGE_ROWS,
        bridge_result_path: EXPECTED_BRIDGE_RESULT,
        factorial_result_path: EXPECTED_FACTORIAL_RESULT,
        factorial_qa_path: EXPECTED_FACTORIAL_QA,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"parent evidence identity drift: {path}")

    qa_manifest, qa_rows = qa_eval.load_data(args.qa_data_root, args.checkpoint)
    qa_by_key = {(row["task"], row["row_sha256"]): row for row in qa_rows}
    bridge_rows = [json.loads(line) for line in bridge_rows_path.read_text().splitlines() if line]
    prompts = {}
    for row in bridge_rows:
        if row["condition"] not in {"far", "ablated"}:
            continue
        key = (row["task"], row["row_sha256"])
        prompts.setdefault(key, {})[row["condition"]] = row["input_ids"]
    if set(prompts) != set(qa_by_key) or any(set(pair) != {"far", "ablated"} for pair in prompts.values()):
        raise ValueError("bridge far/ablated prompt grid is incomplete")
    for key, pair in prompts.items():
        if pair["far"] != qa_by_key[key]["input_ids"] or len(pair["far"]) != len(pair["ablated"]):
            raise ValueError("bridge prompt pairing drift")

    greedy_rows = []
    for line in factorial_qa_path.read_text().splitlines():
        row = json.loads(line)
        for profile, arm in ARM_MAP.items():
            if row["arm"] == arm:
                greedy_rows.append({"profile": profile, "task": row["task"],
                                    "row_sha256": row["row_sha256"], "score": row["score"]})

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
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False)
    rotary = model.model.rotary_emb
    native = rotary.inv_freq.detach().cpu().float().numpy().copy()
    if nll_eval.tensor_hash(native) != nll_eval.EXPECTED_NATIVE_SHA256:
        raise RuntimeError("runtime Native table drift")
    profiles = (
        {"name": "native_unit", "values": native, "tensor_sha256": nll_eval.EXPECTED_NATIVE_SHA256},
        {"name": "index_unit", "values": index, "tensor_sha256": nll_eval.EXPECTED_INDEX_SHA256},
    )
    for profile in profiles:
        profile["active"] = torch.from_numpy(profile["values"]).to(
            rotary.inv_freq.device, rotary.inv_freq.dtype)

    args.output.mkdir(parents=True)
    manifest = {
        "status": "QWEN_K32_SOURCE_CONTRAST_DECODE_FROZEN",
        "checkpoint_weight_sha256": weights,
        "qa_data_manifest_sha256": qa_manifest["manifest_sha256"],
        "qa_data_rows_sha256": qa_manifest["file"]["sha256"],
        "bridge_rows_sha256": EXPECTED_BRIDGE_ROWS,
        "bridge_result_sha256": EXPECTED_BRIDGE_RESULT,
        "factorial_result_sha256": EXPECTED_FACTORIAL_RESULT,
        "factorial_qa_sha256": EXPECTED_FACTORIAL_QA,
        "profiles": [{"name": profile["name"], "tensor_sha256": profile["tensor_sha256"],
                      "attention_scaling": 1.0} for profile in profiles],
        "decoder": "argmax(2*log_p_far-log_p_source_absent)",
        "contrast_coefficient": 1.0,
        "script_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(Path(inspect.getsourcefile(type(model)))),
        "attention_source_sha256": sha256_file(Path(inspect.getsourcefile(configure_ruler_flash_attention))),
        "model_updates": 0, "profile_selection": False, "use_cache": True,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    raw_path = args.output / "examples.jsonl"
    records = []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with raw_path.open("x") as handle:
        for profile in profiles:
            with torch.no_grad():
                rotary.inv_freq.copy_(profile["active"])
                if hasattr(rotary, "original_inv_freq"):
                    rotary.original_inv_freq = rotary.inv_freq.detach().clone()
                rotary.attention_scaling = 1.0
            for key in sorted(prompts):
                task, row_sha = key
                row, pair = qa_by_key[key], prompts[key]
                pair_ids = torch.tensor([pair["far"], pair["ablated"]], dtype=torch.long, device="cuda")
                batch_started = time.perf_counter()
                generated = source_contrast_generate(
                    model, pair_ids, max_new_tokens=int(row["generation_tokens"]),
                    eos_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                token_ids = [int(value) for value in generated.cpu().tolist()]
                prediction = tokenizer.decode(
                    token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                processed = post_process_prediction(task, prediction)
                record = {
                    "profile": profile["name"], "task": task, "row_sha256": row_sha,
                    "score": score_prediction(task, TASK_METRIC_MAP[task], processed,
                                              row["references"], row.get("all_classes", [])),
                    "prediction": prediction, "generated_token_ids": token_ids,
                    "generated_tokens": len(token_ids), "generation_tokens_budget": row["generation_tokens"],
                    "far_prompt_sha256": ids_sha256(pair["far"]),
                    "absent_prompt_sha256": ids_sha256(pair["ablated"]),
                    "table_sha256_float32": profile["tensor_sha256"], "attention_scaling": 1.0,
                    "batch_seconds": time.perf_counter() - batch_started,
                }
                records.append(record); handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
                if len(records) == 1 or len(records) % 10 == 0:
                    print(f"source-contrast {len(records)}/60", flush=True)
                del pair_ids, generated
        os.fsync(handle.fileno())
    report = summarize(records, greedy_rows)
    report.update(status=STATUS, elapsed_seconds=time.perf_counter() - started,
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                  examples_sha256=sha256_file(raw_path), run_manifest_sha256=sha256_file(manifest_path),
                  evidence_limit="Post-result fixed decoder on the same 30-row derived QA panel; not independent confirmation.")
    (args.output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
