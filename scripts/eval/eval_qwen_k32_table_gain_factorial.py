#!/usr/bin/env python3
"""Complete the frozen Qwen K32 table-by-gain 2x2 on NLL and repaired far-QA."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from scripts.data.prepare_qwen_k32_natural_nll import sha256_file
from scripts.eval import eval_qwen_k32_far_evidence_qa as qa_eval
from scripts.eval import eval_qwen_k32_natural_nll as nll_eval
from scripts.eval.longbench_metrics import TASK_METRIC_MAP, post_process_prediction, score_prediction
from scripts.lib.checkpoint_identity import safetensors_weight_set_sha256

STATUS = "QWEN_K32_TABLE_GAIN_FACTORIAL_COMPLETE"
PROFILE_ORDER = (
    "native_table_unit_gain",
    "native_table_index_gain",
    "index_table_unit_gain",
    "index_table_index_gain",
)
EXPECTED_NLL_RECEIPT = "ba489f47070d2dd9058afe50fa7c9db1229f50eb2bc364445dfdb5c7815712a9"
EXPECTED_QA_RECEIPT = "6109434ea596b42706e5ce295a1348052ec4d01b22c0849529bf23f2dbef793c"
BOOTSTRAP_SEED = 202609032
BOOTSTRAP_SAMPLES = 10_000


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_ci(values: np.ndarray, *, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()


def summarize(nll_rows: list[dict], qa_rows: list[dict], prior_nll: list[dict],
              prior_qa: list[dict]) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    nll = {(row["arm"], row["sample_id"], row["length"]): float(row["nll"])
           for row in prior_nll + nll_rows}
    qa = {(row["arm"], row["task"], row["row_sha256"]): float(row["score"])
          for row in prior_qa + qa_rows}
    table_gain = {
        "native_unit": "native_table_unit_gain",
        "native_index_gain": "native_table_index_gain",
        "index_unit": "index_table_unit_gain",
        "index_index_gain": "index_table_index_gain",
    }
    sample_ids = sorted({key[1] for key in nll if key[0] == "native_table_unit_gain"})
    tasks = tuple(qa_eval.TASKS)
    qa_ids = {task: sorted({key[2] for key in qa
                            if key[0] == "native_table_unit_gain" and key[1] == task})
              for task in tasks}
    if len(sample_ids) != 32 or any(len(ids) != 10 for ids in qa_ids.values()):
        raise ValueError("factorial baseline grid is incomplete")

    cells = {}
    for label, arm in table_gain.items():
        cells[label] = {
            "nll_32k": float(np.mean([nll[arm, sid, 32768] for sid in sample_ids])),
            "nll_64k": float(np.mean([nll[arm, sid, 65536] for sid in sample_ids])),
            "qa_by_task": {task: float(np.mean([qa[arm, task, row_id] for row_id in qa_ids[task]]))
                           for task in tasks},
        }
        cells[label]["qa_macro"] = float(np.mean(list(cells[label]["qa_by_task"].values())))

    def nll_contrast(left: str, right: str, length: int) -> dict:
        values = np.array([nll[table_gain[left], sid, length] - nll[table_gain[right], sid, length]
                           for sid in sample_ids])
        return {"mean": float(values.mean()), "ci95": paired_ci(values, rng=rng)}

    def qa_contrast(left: str, right: str) -> dict:
        by_task = {task: np.array([qa[table_gain[left], task, row_id]
                                   - qa[table_gain[right], task, row_id]
                                   for row_id in qa_ids[task]]) for task in tasks}
        draws = []
        for task in tasks:
            indices = rng.integers(0, len(by_task[task]),
                                   size=(BOOTSTRAP_SAMPLES, len(by_task[task])))
            draws.append(by_task[task][indices].mean(axis=1))
        macro_draws = np.stack(draws, axis=1).mean(axis=1)
        return {"mean_macro": float(np.mean([values.mean() for values in by_task.values()])),
                "ci95": np.quantile(macro_draws, [.025, .975]).tolist(),
                "by_task": {task: float(values.mean()) for task, values in by_task.items()}}

    index_unit_retention = math.exp(cells["native_unit"]["nll_32k"]
                                    - cells["index_unit"]["nll_32k"])
    frequency_only_nll = nll_contrast("index_unit", "native_unit", 65536)
    frequency_only_qa = qa_contrast("index_unit", "native_unit")
    gain_only_nll = nll_contrast("native_index_gain", "native_unit", 65536)
    gain_only_qa = qa_contrast("native_index_gain", "native_unit")
    nll_interaction_values = np.array([
        (nll["index_table_index_gain", sid, 65536] - nll["index_table_unit_gain", sid, 65536])
        - (nll["native_table_index_gain", sid, 65536] - nll["native_table_unit_gain", sid, 65536])
        for sid in sample_ids])
    qa_interaction = {}
    for task in tasks:
        values = np.array([
            (qa["index_table_index_gain", task, row_id] - qa["index_table_unit_gain", task, row_id])
            - (qa["native_table_index_gain", task, row_id] - qa["native_table_unit_gain", task, row_id])
            for row_id in qa_ids[task]])
        qa_interaction[task] = values
    qa_interaction_mean = float(np.mean([values.mean() for values in qa_interaction.values()]))
    interaction_draws = []
    for task, values in qa_interaction.items():
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
        interaction_draws.append(values[indices].mean(axis=1))

    candidate_pass = (
        index_unit_retention >= .875
        and frequency_only_nll["ci95"][1] < 0
        and frequency_only_qa["ci95"][0] > 0
        and sum(value > 0 for value in frequency_only_qa["by_task"].values()) >= 2
    )
    return {
        "status": "QWEN_K32_TABLE_GAIN_FACTORIAL_SUMMARIZED",
        "cells": cells,
        "frequency_only_index": {
            "ppl_retention_32k": index_unit_retention,
            "nll_64k_minus_native": frequency_only_nll,
            "qa_minus_native": frequency_only_qa,
            "joint_operating_point_point_gate": candidate_pass,
        },
        "gain_only_native": {"nll_64k_minus_native": gain_only_nll,
                             "qa_minus_native": gain_only_qa},
        "interaction": {
            "nll_64k": {"mean": float(nll_interaction_values.mean()),
                         "ci95": paired_ci(nll_interaction_values, rng=rng)},
            "qa_macro": {"mean": qa_interaction_mean,
                         "ci95": np.quantile(np.stack(interaction_draws, axis=1).mean(axis=1),
                                             [.025, .975]).tolist()},
        },
        "decision_rule": (
            "The frequency-only point is usable only if 32K retention >=.875, its 64K NLL-vs-Native "
            "CI upper bound <0, its far-QA CI lower bound >0, and at least two task deltas are positive. "
            "No parameter is tuned."
        ),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_SAMPLES},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nll-data-root", type=Path, required=True)
    parser.add_argument("--qa-data-root", type=Path, required=True)
    parser.add_argument("--index-table", type=Path, required=True)
    parser.add_argument("--nll-receipt", type=Path, required=True)
    parser.add_argument("--qa-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("factorial output must be fresh")
    if (sha256_file(args.nll_receipt) != EXPECTED_NLL_RECEIPT
            or sha256_file(args.qa_receipt) != EXPECTED_QA_RECEIPT):
        raise ValueError("prior outcome identity drift")
    nll_data, nll_rows = nll_eval.load_data(args.nll_data_root, args.checkpoint)
    qa_data, qa_rows = qa_eval.load_data(args.qa_data_root, args.checkpoint)
    index = nll_eval.load_table(
        args.index_table, nll_eval.EXPECTED_INDEX_FILE_SHA256, nll_eval.EXPECTED_INDEX_SHA256)
    weights = safetensors_weight_set_sha256(args.checkpoint)
    if weights != nll_eval.EXPECTED_WEIGHT_SHA256:
        raise ValueError("checkpoint identity drift")

    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
        configure_cuda, configure_ruler_flash_attention, greedy_generate)

    configure_cuda()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval().to("cuda")
    model.requires_grad_(False); configure_ruler_flash_attention(model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False)
    rotary = model.model.rotary_emb
    native = rotary.inv_freq.detach().cpu().float().numpy().copy()
    if nll_eval.tensor_hash(native) != nll_eval.EXPECTED_NATIVE_SHA256:
        raise RuntimeError("runtime Native table drift")
    profiles = (
        {"name": PROFILE_ORDER[0], "values": native, "gain": 1.0,
         "tensor_sha256": nll_eval.EXPECTED_NATIVE_SHA256},
        {"name": PROFILE_ORDER[1], "values": native, "gain": nll_eval.INDEX_GAIN,
         "tensor_sha256": nll_eval.EXPECTED_NATIVE_SHA256},
        {"name": PROFILE_ORDER[2], "values": index, "gain": 1.0,
         "tensor_sha256": nll_eval.EXPECTED_INDEX_SHA256},
        {"name": PROFILE_ORDER[3], "values": index, "gain": nll_eval.INDEX_GAIN,
         "tensor_sha256": nll_eval.EXPECTED_INDEX_SHA256},
    )
    for profile in profiles:
        profile["active"] = torch.from_numpy(profile["values"]).to(
            rotary.inv_freq.device, rotary.inv_freq.dtype)

    args.output.mkdir(parents=True)
    manifest = {
        "status": "QWEN_K32_TABLE_GAIN_FACTORIAL_FROZEN",
        "checkpoint_weight_sha256": weights,
        "nll_data_manifest_sha256": nll_data["manifest_sha256"],
        "nll_data_rows_sha256": nll_data["file"]["sha256"],
        "qa_data_manifest_sha256": qa_data["manifest_sha256"],
        "qa_data_rows_sha256": qa_data["file"]["sha256"],
        "nll_receipt_sha256": EXPECTED_NLL_RECEIPT,
        "qa_receipt_sha256": EXPECTED_QA_RECEIPT,
        "profiles": [{key: value for key, value in profile.items() if key not in {"values", "active"}}
                     for profile in profiles],
        "script_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(Path(inspect.getsourcefile(type(model)))),
        "attention_source_sha256": sha256_file(Path(inspect.getsourcefile(configure_ruler_flash_attention))),
        "scorer_source_sha256": sha256_file(Path(__file__).resolve().with_name("longbench_metrics.py")),
        "model_updates": 0, "profile_selection": False, "use_cache_nll": False,
        "use_cache_qa": True, "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    nll_path, qa_path = args.output / "nll_examples.jsonl", args.output / "qa_examples.jsonl"
    new_nll, new_qa, total, done = [], [], len(profiles) * (len(nll_rows) + len(qa_rows)), 0
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with nll_path.open("x") as nll_handle, qa_path.open("x") as qa_handle:
        for profile in profiles:
            with torch.no_grad():
                rotary.inv_freq.copy_(profile["active"])
                if hasattr(rotary, "original_inv_freq"):
                    rotary.original_inv_freq = rotary.inv_freq.detach().clone()
                rotary.attention_scaling = profile["gain"]
            for row in nll_rows:
                ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = model(input_ids=ids, use_cache=False, logits_to_keep=257).logits
                    logits, targets = nll_eval.aligned_suffix(raw, ids)
                    loss = torch.cat([F.cross_entropy(
                        logits[:, left:left + 64].float().transpose(1, 2),
                        targets[:, left:left + 64], reduction="none")
                        for left in range(0, nll_eval.TARGET_TOKENS, 64)], dim=1).mean()
                torch.cuda.synchronize()
                if (not bool(torch.isfinite(loss)) or not torch.equal(rotary.inv_freq, profile["active"])
                        or float(rotary.attention_scaling) != profile["gain"]):
                    raise RuntimeError("factorial NLL profile or finite-loss invariant failed")
                record = {key: value for key, value in row.items() if key != "input_ids"}
                record.update(arm=profile["name"], nll=float(loss),
                              table_sha256_float32=profile["tensor_sha256"],
                              attention_scaling=profile["gain"])
                new_nll.append(record); nll_handle.write(json.dumps(record) + "\n"); nll_handle.flush()
                done += 1
                if done == 1 or done % 16 == 0: print(f"factorial {done}/{total}", flush=True)
                del ids, raw, logits, targets, loss
            for row in qa_rows:
                ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
                generated = greedy_generate(model, ids, max_new_tokens=row["generation_tokens"],
                                            eos_token_id=tokenizer.eos_token_id)[0]
                torch.cuda.synchronize()
                if (not torch.equal(rotary.inv_freq, profile["active"])
                        or float(rotary.attention_scaling) != profile["gain"]):
                    raise RuntimeError("factorial QA profile mutated during generation")
                token_ids = [int(value) for value in generated.detach().cpu().tolist()]
                prediction = tokenizer.decode(token_ids, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
                score = score_prediction(row["task"], TASK_METRIC_MAP[row["task"]],
                                         post_process_prediction(row["task"], prediction),
                                         row["references"], row.get("all_classes", []))
                record = {"arm": profile["name"], "task": row["task"],
                          "row_sha256": row["row_sha256"], "score": score,
                          "prediction": prediction, "generated_token_ids": token_ids,
                          "generated_tokens": len(token_ids),
                          "table_sha256_float32": profile["tensor_sha256"],
                          "attention_scaling": profile["gain"]}
                new_qa.append(record); qa_handle.write(json.dumps(record) + "\n"); qa_handle.flush()
                done += 1
                if done % 16 == 0 or done == total: print(f"factorial {done}/{total}", flush=True)
                del ids, generated
        os.fsync(nll_handle.fileno()); os.fsync(qa_handle.fileno())

    report = summarize(new_nll, new_qa, [], [])
    report.update(status=STATUS, elapsed_seconds=time.perf_counter() - started,
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                  nll_examples_sha256=sha256_file(nll_path), qa_examples_sha256=sha256_file(qa_path),
                  run_manifest_sha256=sha256_file(manifest_path),
                  evidence_limit="Single-checkpoint post-result causal ablation; no parameter sweep or training.")
    (args.output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
