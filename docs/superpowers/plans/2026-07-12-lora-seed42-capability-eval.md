# LLaMA-3-8B Seed-42 Capability Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare long- and short-context evaluation examples on CPU and score the identical examples with Geo base, Geo+LoRA seed 42, and EVQ+LoRA seed 42.

**Architecture:** A CPU-only builder writes compact JSONL records containing tokenized prompts, answers, task metadata, and source identities. A single GPU evaluator reuses the strict model/adapter checks from the existing temporal three-arm evaluator, loads the backbone once, attaches both adapters, and switches the adapter and frequency schedule for each arm. Teacher-forced answer NLL and autoregressive task metrics use separate execution paths because cached decoding is incompatible with the packed-free training attention patch.

**Tech Stack:** Python 3.12, PyTorch 2.8, Transformers, PEFT, Datasets, shell launchers, `unittest`/`pytest`.

## Global Constraints

- The only model arms are `geo_base`, `geo_lora_s42`, and `evq_lora_s42`.
- Both adapters must validate as seed 42 and must share the LongAlpaca training-manifest SHA and model-manifest SHA.
- `geo_base` and `geo_lora_s42` use native geometric frequencies; `evq_lora_s42` uses EVQ-Cosh with `tau=1.414`.
- Checkpoint 200 is out of scope; use only completed step-300 adapters.
- Data preparation must run without CUDA and must finish before a GPU evaluation starts.
- Never silently substitute synthetic data when a requested public dataset is unavailable.
- Long-document truncation may shorten only the document body; the question, choices, answer cue, and gold answer must remain intact.
- Do not modify paper metrics, paper text, or existing tracked experimental results.
- Preserve unrelated work in other branches and worktrees.

---

### Task 1: CPU capability-data builder

**Files:**
- Create: `experiments/lora_evq_v2/prepare_seed42_capability_data.py`
- Create: `tests/test_seed42_capability_data.py`

**Interfaces:**
- Consumes: LLaMA tokenizer path; official RULER JSONL; official NoLiMa files; LongBench v1 JSONL; pinned Hugging Face MCQA datasets.
- Produces: `prepare_suite(args: argparse.Namespace) -> dict[str, Any]`, `truncate_document_only(...) -> list[int]`, `build_passkey_examples(...) -> list[dict[str, Any]]`, and an output `manifest.json` plus per-suite JSONL files.

- [ ] **Step 1: Write failing unit tests for length safety and source failures**

```python
def test_truncate_document_only_preserves_suffix():
    ids = truncate_document_only(
        prefix_ids=[1, 2], document_ids=list(range(100)), suffix_ids=[7, 8, 9], max_prompt_tokens=12
    )
    assert ids[:2] == [1, 2]
    assert ids[-3:] == [7, 8, 9]
    assert len(ids) == 12

def test_missing_requested_source_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "missing.jsonl")

def test_passkey_grid_has_all_lengths_depths_and_unique_ids(fake_tokenizer):
    rows = build_passkey_examples(fake_tokenizer, lengths=(8192, 16384, 32768), depths=(10, 25, 50, 75, 90), trials=2, seed=42)
    assert len(rows) == 30
    assert len({row["example_id"] for row in rows}) == 30
```

- [ ] **Step 2: Run the tests and confirm they fail before implementation**

Run: `python -m pytest tests/test_seed42_capability_data.py -q`

Expected: import failure for the new builder module.

- [ ] **Step 3: Implement the common record and document-only truncation**

Every record must contain the following fields and a SHA256 over the integer prompt IDs:

```python
{
    "schema": "evq_cosh.seed42_capability_example.v1",
    "example_id": str,
    "suite": str,
    "task": str,
    "target_length": int,
    "prompt_ids": list[int],
    "answers": list[str],
    "metric": "exact_match" | "qa_f1" | "mcqa",
    "depth_percent": float | None,
    "choices": list[str] | None,
    "answer_index": int | None,
    "source": dict[str, Any],
    "prompt_sha256": str,
}
```

`truncate_document_only` must raise when prefix plus suffix already exceed the target instead of truncating the suffix.

- [ ] **Step 4: Implement deterministic passkey and official-source importers**

Use seed 42 and prepare 20 passkey examples for every combination of lengths `(8192, 16384, 32768)` and depths `(10, 25, 50, 75, 90)`. Import RULER outputs without changing their prompts or references. Build NoLiMa-Hard examples from the official needle set and book haystacks at 16K/32K. Select LongBench NarrativeQA and Qasper only when the complete prompt fits within 32K, using document-only truncation only for explicitly requested fixed-length diagnostics. Load MCQA sources at these exact revisions:

```python
MCQA_REVISIONS = {
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "ybisk/piqa": "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
}
```

If any requested source fails to load, terminate with a non-zero exit rather than producing a partial suite under the expected manifest name.

- [ ] **Step 5: Implement manifest validation and atomic output**

Write each JSONL to `*.incomplete`, validate every record, rename it atomically, then write `manifest.json` last. The manifest records tokenizer identity, source revisions, file SHA256/size/row count, task counts, and prompt-length minima/maxima. Reject duplicate example IDs or prompt hashes within the same task/length cell.

- [ ] **Step 6: Run focused and existing CPU tests**

Run:

```bash
python -m pytest tests/test_seed42_capability_data.py -q
python -m pytest tests/test_temporal_three_arm_eval.py tests/test_legacy_lora_multiseed.py -q
python -m py_compile experiments/lora_evq_v2/prepare_seed42_capability_data.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add experiments/lora_evq_v2/prepare_seed42_capability_data.py tests/test_seed42_capability_data.py
git commit -m "feat: prepare seed42 capability data"
```

---

### Task 2: Strict three-arm capability evaluator

**Files:**
- Create: `experiments/lora_evq_v2/eval_seed42_capability_three_arm.py`
- Create: `tests/test_seed42_capability_eval.py`

**Interfaces:**
- Consumes: Task 1 `manifest.json` and JSONL records; the existing model and LongAlpaca manifests; Geo/EVQ seed-42 adapter directories.
- Produces: `load_three_arm_runtime(...)`, `score_gold_answer(...)`, `score_mcqa(...)`, `summarize_results(...)`, and atomic raw/summary JSON outputs.

- [ ] **Step 1: Write failing tests for the exact three-arm contract**

```python
def test_capability_arm_contract_is_exact_seed42():
    assert capability_arm_contract() == {
        "geo_base": {"frequency": "native_geo", "adapter": None},
        "geo_lora_s42": {"frequency": "native_geo", "adapter": "geo_longalpaca_s42"},
        "evq_lora_s42": {"frequency": "evq_cosh", "tau": 1.414, "adapter": "evq_longalpaca_tau1414_s42"},
    }

def test_summary_keeps_tasks_and_lengths_separate():
    summary = summarize_results(fake_rows())
    assert set(summary["geo_base"]["passkey"]) == {"8K", "16K", "32K"}
```

Also test token-weighted answer NLL, normalized exact match, QA F1 with token multiplicity, MCQA likelihood margins, and duplicate/missing-arm rejection.

- [ ] **Step 2: Run the tests and confirm they fail before implementation**

Run: `python -m pytest tests/test_seed42_capability_eval.py -q`

Expected: import failure for the new evaluator module.

- [ ] **Step 3: Implement strict artifact and model loading**

Copy the validated sequence from `eval_temporal_holdout_three_arm.py`: validate both adapters against seed 42 and the LongAlpaca/model manifest hashes; reconstruct and compare canonical native-Geo and EVQ frequencies; load one BF16 base; attach Geo as adapter `geo` and EVQ as adapter `evq`; run disabled/active adapter canaries.

The fixed switches are:

```python
def activate_arm(model, arm, geo_frequency, evq_frequency):
    if arm == "geo_base":
        model.set_adapter("geo")
        inject_inv_freq(model, geo_frequency)
        return model.disable_adapter()
    if arm == "geo_lora_s42":
        model.set_adapter("geo")
        inject_inv_freq(model, geo_frequency)
        return contextlib.nullcontext()
    if arm == "evq_lora_s42":
        model.set_adapter("evq")
        inject_inv_freq(model, evq_frequency)
        return contextlib.nullcontext()
    raise ValueError(arm)
```

- [ ] **Step 4: Implement memory-bounded answer scoring**

Run the causal backbone once for `prompt_ids + answer_ids`, select only the hidden states predicting answer tokens, and apply `lm_head` only to those positions. Do not materialize `[sequence_length, vocabulary_size]` logits. Accumulate raw NLL sum and answer-token count before computing mean NLL.

For MCQA, score each option using the same prompt, normalize by option-token count, choose the highest mean log probability, and record the correct-minus-best-wrong margin.

- [ ] **Step 5: Implement generation with the normal cache-capable attention path**

Do not call `configure_packed_free_causal_sdpa` for generation. Use Transformers SDPA/Flash with `use_cache=True`, greedy decoding, task-specific small `max_new_tokens`, and normalized official-style metrics. Run answer NLL for all prepared records; make AR generation selectable as `--generation full|pilot|off`, where pilot deterministically takes the first five examples of each task/length/depth cell.

- [ ] **Step 6: Implement atomic per-example and grouped outputs**

The raw output must include arm, adapter SHA, prompt SHA, prediction, references, NLL sum, answer-token count, task metric, runtime, and peak CUDA memory. The summary must retain each task, length, and depth cell, with 95% Wilson intervals for binary accuracy and no implicit overall headline score.

- [ ] **Step 7: Run focused and regression tests**

Run:

```bash
python -m pytest tests/test_seed42_capability_eval.py tests/test_seed42_capability_data.py -q
python -m pytest tests/test_temporal_three_arm_eval.py tests/test_legacy_lora_multiseed.py -q
python -m py_compile experiments/lora_evq_v2/eval_seed42_capability_three_arm.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add experiments/lora_evq_v2/eval_seed42_capability_three_arm.py tests/test_seed42_capability_eval.py
git commit -m "feat: evaluate seed42 capability arms"
```

---

### Task 3: CPU preparation and GPU launchers

**Files:**
- Create: `scripts/2026-07/08_prepare_lora_seed42_capability_data.sh`
- Create: `scripts/2026-07/09_lora_seed42_capability_eval.sh`
- Modify: `tests/test_seed42_capability_eval.py`

**Interfaces:**
- Consumes: Tasks 1 and 2 CLIs plus explicit environment variables for all server paths.
- Produces: a repeatable CPU preparation command and a fail-closed single-GPU three-arm evaluation command.

- [ ] **Step 1: Add launcher contract tests**

Assert that the CPU launcher contains no CUDA requirement, pins official sources, and calls the builder only after all downloads succeed. Assert that the GPU launcher requires seed-42 Geo/EVQ adapter files, model/training/data manifests, a GPU lock, a non-existing output path, and passes only seed 42.

- [ ] **Step 2: Implement the CPU launcher**

Pin the external sources to:

```text
NVIDIA/RULER main: 38da79d79519ef87aa46ae804f838e1eab7f86d7
adobe-research/NoLiMa: cb14780b249fecf2851127b2101a062c1b2c6430
amodaresi/NoLiMa data: 378115b1f136b6ba78f90f78682bc55f70ec3ddd
THUDM/LongBench: 5e628be450b7e67fb7ae6e201bd6d8f7056f7672
```

Use resumable downloads, verify Git commit IDs, generate RULER 8K/16K/32K with the LLaMA tokenizer, download NoLiMa-Hard plus its book haystacks, reuse the existing LongBench v1 extraction, and invoke the Task 1 builder. The final command must run `--validate_only` before reporting success.

- [ ] **Step 3: Implement the GPU launcher**

Require these environment variables: Python, model, model manifest, LongAlpaca manifest, Geo adapter, EVQ adapter, capability-data root, raw output, and summary output. Acquire the global GPU lock before model loading. Run a one-example-per-suite pilot preflight first; only run the requested `pilot` or `full` mode if that preflight succeeds.

- [ ] **Step 4: Run shell, Python, and focused tests**

Run:

```bash
bash -n scripts/2026-07/08_prepare_lora_seed42_capability_data.sh
bash -n scripts/2026-07/09_lora_seed42_capability_eval.sh
python -m pytest tests/test_seed42_capability_eval.py tests/test_seed42_capability_data.py -q
python -m py_compile experiments/lora_evq_v2/prepare_seed42_capability_data.py experiments/lora_evq_v2/eval_seed42_capability_three_arm.py
git diff --check
```

Expected: all checks pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/2026-07/08_prepare_lora_seed42_capability_data.sh scripts/2026-07/09_lora_seed42_capability_eval.sh tests/test_seed42_capability_eval.py
git commit -m "chore: launch seed42 capability evaluation"
```

---

### Task 4: Server preparation and final review

**Files:**
- No tracked file changes expected.

**Interfaces:**
- Consumes: the completed repository commits and the CPU-only server.
- Produces: prepared server-side evaluation data and a recorded dry-run status; it does not start GPU inference.

- [ ] **Step 1: Push the verified `main` commits and update the server code copy**

Use `git push origin main`, then update a dedicated server code directory without changing the historical checkpoint directories.

- [ ] **Step 2: Run CPU data preparation on the no-card server**

Run the CPU launcher with the existing LLaMA tokenizer/model path and LongBench v1 extraction. Save external benchmark sources and generated evaluation data under `/root/autodl-tmp/data/`.

- [ ] **Step 3: Run CPU validation and GPU-launcher dry run**

Validate file hashes, row counts, task/length/depth coverage, and that all prompts remain within their target token limit. Run the GPU launcher in a preflight-only mode that must stop before model loading because CUDA is absent.

- [ ] **Step 4: Request independent code review and fix all important findings**

Review the complete diff for scientific correctness, prompt/answer truncation, three-arm identity, cache/backend safety, data leakage, silent fallback, and GPU-cost hazards.

- [ ] **Step 5: Report ready-to-run paths**

Report the server code path, prepared-data manifest path, three adapter/model paths, exact GPU command, prepared example counts, disk use, and the fact that no GPU evaluation was started.
