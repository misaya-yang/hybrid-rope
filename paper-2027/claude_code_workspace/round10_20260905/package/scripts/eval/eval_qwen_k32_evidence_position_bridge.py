#!/usr/bin/env python3
"""Diagnose far-evidence use with Native/index gain-one tables and exact prompt controls."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.data.prepare_qwen_k32_far_evidence_qa import (
    EVIDENCE_MARKER, FILLER_MARKER, answer_leaks, ids_sha256, load_sources,
    sanitize_filler, sha256_file, token_ids,
)
from scripts.eval import eval_qwen_k32_far_evidence_qa as qa_eval
from scripts.eval import eval_qwen_k32_natural_nll as nll_eval
from scripts.lib.checkpoint_identity import safetensors_weight_set_sha256

STATUS = "QWEN_K32_EVIDENCE_POSITION_BRIDGE_COMPLETE"
PROFILES = ("native_unit", "index_unit")
CONDITIONS = ("far", "near", "ablated")
EXPECTED_QA_RECEIPT = "6109434ea596b42706e5ce295a1348052ec4d01b22c0849529bf23f2dbef793c"
EXPECTED_FACTORIAL_RESULT = "5d6f2f2e7dc4cc8961e1d42d87931e03d3cf540a3af4a7b7d9c776d66b5a3d9b"
BOOTSTRAP_SEED = 202609033
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


def summarize(records: list[dict]) -> dict:
    cells = {(row["profile"], row["task"], row["row_sha256"], row["condition"]): row["answer_nll"]
             for row in records}
    row_ids = {task: sorted({row["row_sha256"] for row in records if row["task"] == task})
               for task in qa_eval.TASKS}
    expected = {(profile, task, row_id, condition) for profile in PROFILES
                for task in qa_eval.TASKS for row_id in row_ids[task] for condition in CONDITIONS}
    if set(cells) != expected or any(len(ids) != 10 for ids in row_ids.values()):
        raise ValueError("evidence bridge grid is incomplete")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = {}
    raw = {}
    for profile in PROFILES:
        raw[profile] = {"source_use_far": {}, "source_use_near": {}, "position_penalty": {}}
        for task in qa_eval.TASKS:
            raw[profile]["source_use_far"][task] = np.array([
                cells[profile, task, row_id, "ablated"] - cells[profile, task, row_id, "far"]
                for row_id in row_ids[task]])
            raw[profile]["source_use_near"][task] = np.array([
                cells[profile, task, row_id, "ablated"] - cells[profile, task, row_id, "near"]
                for row_id in row_ids[task]])
            raw[profile]["position_penalty"][task] = np.array([
                cells[profile, task, row_id, "far"] - cells[profile, task, row_id, "near"]
                for row_id in row_ids[task]])
        metrics[profile] = {}
        for name, by_task in raw[profile].items():
            draws = stratified_draws(by_task, rng)
            metrics[profile][name] = {
                "mean_macro": float(np.mean([values.mean() for values in by_task.values()])),
                "ci95": np.quantile(draws, [.025, .975]).tolist(),
                "by_task": {task: float(values.mean()) for task, values in by_task.items()},
            }
    contrasts = {}
    for name in ("source_use_far", "source_use_near", "position_penalty"):
        delta = {task: raw["index_unit"][name][task] - raw["native_unit"][name][task]
                 for task in qa_eval.TASKS}
        draws = stratified_draws(delta, rng)
        contrasts["index_minus_native_" + name] = {
            "mean_macro": float(np.mean([values.mean() for values in delta.values()])),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "by_task": {task: float(values.mean()) for task, values in delta.items()},
        }
    near_resolves = metrics["native_unit"]["source_use_near"]["ci95"][0] > 0
    if not near_resolves:
        diagnosis = "UNRESOLVED_NEAR_POSITIVE_CONTROL_FAILED"
    elif contrasts["index_minus_native_position_penalty"]["ci95"][0] > 0:
        diagnosis = "INDEX_SPECIFIC_DISTANCE_PENALTY"
    elif metrics["index_unit"]["source_use_far"]["ci95"][0] > 0:
        diagnosis = "INDEX_USES_FAR_SOURCE_GREEDY_READOUT_REMAINS_LIMITED"
    else:
        diagnosis = "FAR_SOURCE_USE_NOT_ESTABLISHED"
    return {
        "status": "QWEN_K32_EVIDENCE_POSITION_BRIDGE_SUMMARIZED",
        "metrics": metrics,
        "contrasts": contrasts,
        "diagnosis": diagnosis,
        "decision_rule": (
            "Near is a resolving positive control iff Native ablated-minus-near answer-NLL CI lower >0; "
            "index-specific distance penalty requires the index-minus-Native far-minus-near CI lower >0."
        ),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_SAMPLES,
                      "unit": "rows resampled within each fixed task jointly across profiles/conditions"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qa-data-root", type=Path, required=True)
    parser.add_argument("--filler-root", type=Path, required=True)
    parser.add_argument("--longbench-main-zip", type=Path, required=True)
    parser.add_argument("--longbench-data-zip", type=Path, required=True)
    parser.add_argument("--index-table", type=Path, required=True)
    parser.add_argument("--qa-receipt", type=Path, required=True)
    parser.add_argument("--factorial-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("evidence bridge output must be fresh")
    if (sha256_file(args.qa_receipt) != EXPECTED_QA_RECEIPT
            or sha256_file(args.factorial_result) != EXPECTED_FACTORIAL_RESULT):
        raise ValueError("evidence bridge parent result identity drift")
    qa_manifest, qa_rows = qa_eval.load_data(args.qa_data_root, args.checkpoint)
    _, filler_rows = nll_eval.load_data(args.filler_root, args.checkpoint)
    fillers = {row["sample_id"]: row for row in filler_rows if row["length"] == 65536}
    prompts, _, sources = load_sources(args.longbench_main_zip, args.longbench_data_zip)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True,
                                               trust_remote_code=False)
    clean_fillers = {sample_id: sanitize_filler(tokenizer, filler["input_ids"])[0]
                     for sample_id, filler in fillers.items()}
    bridge_rows = []
    for row in qa_rows:
        source = sources[row["task"]][row["source_index"]]
        marker = "__EVIDENCE_BRIDGE_CONTEXT_START__"
        marked = (marker + source["context"] + "\n\n" + EVIDENCE_MARKER
                  + "\n\nThe following text is unrelated distractor material. Ignore it when answering.\n"
                  + FILLER_MARKER)
        content = prompts[row["task"]].format(context=marked, input=source["input"])
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        pre_text, remainder = rendered.split(marker)
        context_text, remainder = remainder.split(EVIDENCE_MARKER)
        label_text, query_text = remainder.split(FILLER_MARKER)
        pre, context, label, query = [token_ids(tokenizer, value)
                                      for value in (pre_text, context_text, label_text, query_text)]
        filler_start = row["query_suffix_start_token"] - row["filler_tokens"]
        existing_filler = row["input_ids"][filler_start:row["query_suffix_start_token"]]
        frozen_evidence = row["input_ids"][:row["evidence_end_token_exclusive"]]
        if abs((len(pre) + len(context)) - len(frozen_evidence)) > 2 or len(pre) >= len(frozen_evidence):
            raise ValueError("template-derived context boundary is incompatible with the frozen prompt")
        pre = frozen_evidence[:len(pre)]
        context = frozen_evidence[len(pre):]
        label = row["input_ids"][row["evidence_end_token_exclusive"]:filler_start]
        query = row["input_ids"][row["query_suffix_start_token"]:]
        clean = clean_fillers[row["filler_sample_id"]]
        if clean[:len(existing_filler)] != existing_filler:
            raise ValueError("frozen far filler identity drift")
        replacement = None
        for sample_id in sorted(clean_fillers):
            candidate_filler = clean_fillers[sample_id]
            upper = len(candidate_filler) - len(context)
            if upper < 0:
                continue
            starts = sorted(set(int(value) for value in np.linspace(0, upper, 17)))
            replacement = next(
                (candidate_filler[start:start + len(context)] for start in starts
                 if not answer_leaks(
                     tokenizer, candidate_filler[start:start + len(context)], row["references"])),
                None,
            )
            if replacement is not None:
                break
        if replacement is None:
            raise ValueError("all deterministic ablation windows leak the frozen answer")
        prompts_by_condition = {
            "far": row["input_ids"],
            "near": pre + label + existing_filler + context + query,
            "ablated": pre + replacement + label + existing_filler + query,
        }
        answer = token_ids(tokenizer, row["references"][0])
        if not answer or any(len(prompt) + len(answer) > 65536 for prompt in prompts_by_condition.values()):
            raise ValueError("canonical answer does not fit the frozen physical budget")
        if any(len(prompt) != len(row["input_ids"]) for prompt in prompts_by_condition.values()):
            raise ValueError("bridge condition changed prompt length")
        for condition, prompt in prompts_by_condition.items():
            bridge_rows.append({
                "task": row["task"], "row_sha256": row["row_sha256"],
                "condition": condition, "input_ids": prompt, "answer_ids": answer,
                "prompt_ids_sha256": ids_sha256(prompt), "answer_ids_sha256": ids_sha256(answer),
                "reference": row["references"][0], "query_start": len(prompt) - len(query),
                "context_tokens": len(context),
            })

    args.output.mkdir(parents=True)
    data_path = args.output / "bridge_rows.jsonl"
    with data_path.open("x") as handle:
        for row in bridge_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    index = nll_eval.load_table(
        args.index_table, nll_eval.EXPECTED_INDEX_FILE_SHA256, nll_eval.EXPECTED_INDEX_SHA256)
    weights = safetensors_weight_set_sha256(args.checkpoint)
    if weights != nll_eval.EXPECTED_WEIGHT_SHA256:
        raise ValueError("checkpoint identity drift")

    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM
    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
        configure_cuda, configure_ruler_flash_attention)

    configure_cuda()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval().to("cuda")
    model.requires_grad_(False); configure_ruler_flash_attention(model)
    rotary = model.model.rotary_emb
    native = rotary.inv_freq.detach().cpu().float().numpy().copy()
    profiles = (
        {"name": "native_unit", "values": native,
         "tensor_sha256": nll_eval.EXPECTED_NATIVE_SHA256},
        {"name": "index_unit", "values": index,
         "tensor_sha256": nll_eval.EXPECTED_INDEX_SHA256},
    )
    for profile in profiles:
        profile["active"] = torch.from_numpy(profile["values"]).to(
            rotary.inv_freq.device, rotary.inv_freq.dtype)
    manifest = {
        "status": "QWEN_K32_EVIDENCE_POSITION_BRIDGE_FROZEN",
        "checkpoint_weight_sha256": weights,
        "qa_data_manifest_sha256": qa_manifest["manifest_sha256"],
        "qa_data_rows_sha256": qa_manifest["file"]["sha256"],
        "qa_receipt_sha256": EXPECTED_QA_RECEIPT,
        "factorial_result_sha256": EXPECTED_FACTORIAL_RESULT,
        "bridge_rows_sha256": sha256_file(data_path),
        "profiles": [{key: value for key, value in profile.items() if key not in {"values", "active"}}
                     for profile in profiles],
        "conditions": list(CONDITIONS), "canonical_reference_rule": "first released answer",
        "script_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(Path(inspect.getsourcefile(type(model)))),
        "attention_source_sha256": sha256_file(Path(inspect.getsourcefile(configure_ruler_flash_attention))),
        "model_updates": 0, "profile_selection": False, "use_cache": False,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    raw_path = args.output / "examples.jsonl"
    records, total, done = [], len(profiles) * len(bridge_rows), 0
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with raw_path.open("x") as handle:
        for profile in profiles:
            with torch.no_grad():
                rotary.inv_freq.copy_(profile["active"])
                if hasattr(rotary, "original_inv_freq"):
                    rotary.original_inv_freq = rotary.inv_freq.detach().clone()
                rotary.attention_scaling = 1.0
            for row in bridge_rows:
                sequence = row["input_ids"] + row["answer_ids"]
                ids = torch.tensor([sequence], dtype=torch.long, device="cuda")
                answer_tokens = len(row["answer_ids"])
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, use_cache=False,
                                   logits_to_keep=answer_tokens + 1).logits[:, :-1].float()
                    labels = ids[:, -answer_tokens:]
                    loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="mean")
                torch.cuda.synchronize()
                if not bool(torch.isfinite(loss)) or not torch.equal(rotary.inv_freq, profile["active"]):
                    raise RuntimeError("bridge finite/profile invariant failed")
                record = {key: value for key, value in row.items()
                          if key not in {"input_ids", "answer_ids"}}
                record.update(profile=profile["name"], answer_nll=float(loss),
                              table_sha256_float32=profile["tensor_sha256"], attention_scaling=1.0)
                records.append(record); handle.write(json.dumps(record) + "\n"); handle.flush()
                done += 1
                if done == 1 or done % 15 == 0 or done == total:
                    print(f"evidence-bridge {done}/{total}", flush=True)
                del ids, logits, labels, loss
        os.fsync(handle.fileno())
    report = summarize(records)
    report.update(status=STATUS, elapsed_seconds=time.perf_counter() - started,
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                  bridge_rows_sha256=sha256_file(data_path), examples_sha256=sha256_file(raw_path),
                  run_manifest_sha256=sha256_file(manifest_path),
                  evidence_limit="Post-result single-checkpoint mechanism bridge; correct-answer NLL only.")
    (args.output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
