# EVQ Seed-42 Retrieval Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, CPU-validate, and upload a gated EVQ-only retrieval continuation that starts from the completed seed-42 LongAlpaca adapter, trains at 8K and then 16K with pinned YaRN-derived scaling, and refuses unnecessary GPU work.

**Architecture:** A pure protocol module owns immutable stage budgets and gates. A CPU builder produces exact-token, split-disjoint retrieval bundles plus chat-templated passkey rows. The trainer reuses the tested answer-only/tail-logit machinery, validates the parent adapter and applies each YaRN factor afresh to canonical EVQ. The evaluator separates retrieval, formatting, stopping, counterfactual source dependence, and temporal retention; a shell launcher exposes explicit non-advancing commands.

**Tech Stack:** Python 3.12/3.13, PyTorch 2.x, Transformers, PEFT, shell, pytest; no new dependency.

## Global Constraints

- Start only from the completed step-300 EVQ-Cosh seed-42 LongAlpaca adapter.
- Canonical substrate is EVQ-Cosh with `tau=1.414`, `base=500000`, and `head_dim=128`.
- Preserve LoRA `r=64`, `alpha=128`, `dropout=0.05`, targeting `q_proj,k_proj,v_proj,o_proj` only.
- Use BF16, no quantization, and a single GPU process.
- R8 is 8,192 tokens, microbatch 1, accumulation 4; R16 is 16,384 tokens, microbatch 1, accumulation 2.
- Every segment is exactly 32 optimizer steps and 1,048,576 physical tokens; one fresh-optimizer rescue segment is the only extension per stage.
- Use learning rate `2e-5`, four warm-up steps, cosine decay, weight decay `0.01`, and max gradient norm `1.0`.
- R8 uses factor 1; R16 uses factor 2; final 32K evaluation uses factor 4 applied directly to canonical EVQ, never composed over factor 2.
- Label non-native EVQ scaling `YaRN-derived generalization on the EVQ substrate`; do not call it native-grid official YaRN.
- Train/validation/test nonce pools, filler regions, seeds, and instruction wording are disjoint.
- Train only on the 12 value tokens plus EOS; do not add LongAlpaca replay in this protocol.
- R16 requires a passing R8 gate; 32K and broad downstream evaluation require a passing R16 gate.
- Do not modify paper numbers, paper claims, or the existing matched Geo/EVQ frequency-adaptation protocol.
- CPU preparation and validation must finish before a CUDA model load.
- Never overwrite an output; use atomic temporary files and explicit SHA-256 provenance.

---

### Task 1: Pure stage, metric, and gate contracts

**Files:**
- Create: `rebuttal/evq_seed42_retrieval_repair/__init__.py`
- Create: `rebuttal/evq_seed42_retrieval_repair/protocol.py`
- Create: `tests/test_evq_seed42_retrieval_repair.py`

**Interfaces:**
- Consumes: no model or CUDA runtime.
- Produces: `StageSpec`, `get_stage(name)`, `segment_contract(stage, segment)`, `extract_first_passkey(text)`, `score_text_answer(prediction, gold, eos_terminated)`, and `decide_gate(stage, summary, parent_summary)`.

- [ ] **Step 1: Write failing tests for immutable budgets and factor identity**

```python
def test_stage_contract_has_fixed_physical_token_budget():
    r8 = get_stage("r8")
    r16 = get_stage("r16")
    assert (r8.seq_len, r8.accumulation, r8.factor) == (8192, 4, 1.0)
    assert (r16.seq_len, r16.accumulation, r16.factor) == (16384, 2, 2.0)
    assert r8.tokens_per_segment == r16.tokens_per_segment == 1_048_576
    assert segment_contract("r8", 2)["optimizer_state"] == "fresh"

def test_stage_contract_rejects_unregistered_segment():
    with pytest.raises(ValueError, match="segment must be 1 or 2"):
        segment_contract("r8", 3)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_evq_seed42_retrieval_repair.py -q
```

Expected: collection fails because `rebuttal.evq_seed42_retrieval_repair.protocol` does not exist.

- [ ] **Step 3: Implement stage and segment contracts**

```python
@dataclass(frozen=True)
class StageSpec:
    name: str
    seq_len: int
    min_distance: int
    max_distance: int
    accumulation: int
    factor: float
    pair_threshold: float

    @property
    def tokens_per_segment(self) -> int:
        return self.seq_len * self.accumulation * 32

STAGES = {
    "r8": StageSpec("r8", 8192, 2048, 6144, 4, 1.0, 0.80),
    "r16": StageSpec("r16", 16384, 6144, 14336, 2, 2.0, 0.50),
}

def get_stage(name: str) -> StageSpec:
    try:
        return STAGES[name]
    except KeyError as exc:
        raise ValueError(f"unknown stage {name!r}; expected r8 or r16") from exc

def segment_contract(stage: str, segment: int) -> dict[str, object]:
    spec = get_stage(stage)
    if int(segment) not in (1, 2):
        raise ValueError("segment must be 1 or 2")
    return {
        "stage": spec.name,
        "segment": int(segment),
        "steps": 32,
        "tokens": spec.tokens_per_segment,
        "optimizer_state": "fresh",
        "learning_rate": 2e-5,
        "warmup_steps": 4,
    }
```

- [ ] **Step 4: Add failing tests for answer metrics and gate outcomes**

```python
def test_text_metrics_separate_strict_extracted_containment_and_eos():
    score = score_text_answer("The key is 12345678. Extra.", "12345678", False)
    assert score == {
        "strict_exact": False,
        "first_value_exact": True,
        "gold_containment": True,
        "extracted_value": "12345678",
        "eos_terminated": False,
    }

def test_r8_gate_passes_only_all_registered_thresholds():
    summary = {
        "pair_consistency": 0.80,
        "source_removal_positive_fraction": 0.75,
        "passkey_containment": 0.52,
        "temporal_delta_nll": 0.20,
        "finite": True,
        "task_types": ["kv", "update"],
    }
    assert decide_gate("r8", summary, {"pair_consistency": 0.0})["status"] == "pass"
    summary["passkey_containment"] = 0.48
    failed = decide_gate("r8", summary, {"pair_consistency": 0.0})
    assert failed["status"] == "stop"
    assert "passkey_containment" in failed["failed_checks"]

def test_rescue_requires_ten_point_pair_gain_and_temporal_safety():
    summary = {
        "pair_consistency": 0.20,
        "source_removal_positive_fraction": 0.50,
        "passkey_containment": 0.20,
        "temporal_delta_nll": 0.10,
        "finite": True,
        "task_types": ["kv", "update"],
    }
    assert decide_gate("r16", summary, {"pair_consistency": 0.10})["status"] == "rescue_allowed"
    summary["temporal_delta_nll"] = 0.21
    assert decide_gate("r16", summary, {"pair_consistency": 0.10})["status"] == "stop"
```

- [ ] **Step 5: Run RED, then implement minimal metric and gate logic**

```python
PASSKEY = re.compile(r"(?<!\d)(\d{8})(?!\d)")

def extract_first_passkey(text: str) -> str | None:
    match = PASSKEY.search(str(text))
    return None if match is None else match.group(1)

def score_text_answer(prediction: str, gold: str, eos_terminated: bool) -> dict[str, object]:
    normalized = " ".join(str(prediction).strip().split())
    expected = " ".join(str(gold).strip().split())
    extracted = extract_first_passkey(normalized)
    return {
        "strict_exact": normalized == expected,
        "first_value_exact": extracted == expected,
        "gold_containment": expected in normalized,
        "extracted_value": extracted,
        "eos_terminated": bool(eos_terminated),
    }

def decide_gate(stage: str, summary: Mapping[str, object], parent: Mapping[str, object]) -> dict[str, object]:
    spec = get_stage(stage)
    checks = {
        "pair_consistency": float(summary["pair_consistency"]) >= spec.pair_threshold,
        "source_removal_positive_fraction": float(summary["source_removal_positive_fraction"]) >= 0.75,
        "passkey_containment": float(summary["passkey_containment"]) >= 0.50,
        "temporal_delta_nll": float(summary["temporal_delta_nll"]) <= 0.20,
        "finite": bool(summary["finite"]),
        "task_types": set(summary["task_types"]) == {"kv", "update"},
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if not failed:
        status = "pass"
    elif checks["finite"] and checks["temporal_delta_nll"] and (
        float(summary["pair_consistency"]) - float(parent["pair_consistency"]) >= 0.10
    ):
        status = "rescue_allowed"
    else:
        status = "stop"
    return {"stage": stage, "status": status, "checks": checks, "failed_checks": failed}
```

- [ ] **Step 6: Run GREEN and regressions**

Run:

```bash
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_evq_seed42_retrieval_repair.py \
  tests/test_frequency_adaptation_8b.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the protocol task**

```bash
git add \
  rebuttal/evq_seed42_retrieval_repair/__init__.py \
  rebuttal/evq_seed42_retrieval_repair/protocol.py \
  tests/test_evq_seed42_retrieval_repair.py
git commit -m "feat: define evq retrieval repair protocol"
```

---

### Task 2: Deterministic three-way CPU data builder

**Files:**
- Create: `rebuttal/evq_seed42_retrieval_repair/prepare_data.py`
- Modify: `tests/test_evq_seed42_retrieval_repair.py`

**Interfaces:**
- Consumes: a local LLaMA tokenizer and frozen document-disjoint filler directory.
- Produces: `build_template(tokenizer, split)`, `partition_nonce_pools(tokenizer, minimum)`, `prepare_bundles(tokenizer, train_filler, validation_filler, seed=42)`, `validate_manifest(path)`, and atomic `train/validation/test/passkey` artifacts.

- [ ] **Step 1: Write failing tests for disjoint identities and exact lengths**

```python
def test_nonce_partitions_are_pairwise_disjoint(fake_llama_tokenizer):
    pools = partition_nonce_pools(fake_llama_tokenizer, minimum=8)
    assert set(pools) == {"train", "validation", "test"}
    assert all(len(pool) == 8 for pool in pools.values())
    assert not (set(pools["train"]) & set(pools["validation"]))
    assert not (set(pools["train"]) & set(pools["test"]))
    assert not (set(pools["validation"]) & set(pools["test"]))

def test_complete_chat_render_matches_direct_template_tokenization(fake_llama_tokenizer):
    messages = build_messages("train", before_text="alpha", after_text="beta", key="Key", value="Value")
    rendered_ids, spans = render_complete_chat(fake_llama_tokenizer, messages)
    direct = fake_llama_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    assert rendered_ids == direct
    assert spans["answer_start"] < spans["answer_end"]

def test_repair_bundle_is_exact_length_and_answer_only(sample_bundle):
    validate_bundle(sample_bundle, stage="r8", split="train")
    assert sample_bundle["input_ids"].shape == (256, 8192)
    assert torch.all(sample_bundle["answer_end"] - sample_bundle["answer_start"] == 13)

def test_validation_and_test_have_registered_triplet_counts(prepared_manifest):
    files = prepared_manifest["files"]
    assert files["validation_r8.pt"]["rows"] == 48
    assert files["test_r8.pt"]["rows"] == 96
    assert files["passkey_r8.pt"]["rows"] == 25
```

- [ ] **Step 2: Run tests and verify RED**

Run the focused test file. Expected: import failures for the builder APIs.

- [ ] **Step 3: Implement full-message chat rendering and split-specific wording**

```python
WORDS = {
    "train": ("Read the records and return only the current stored value.", "Record", "stores", "Return Record"),
    "validation": ("Inspect the entries and answer with only the requested contents.", "Entry", "contains", "Contents of Entry"),
    "test": ("Use the document to recover exactly one requested value.", "Item", "currently holds", "Recover Item"),
}

def build_messages(split: str, before_text: str, after_text: str, key: str, value: str) -> list[dict[str, str]]:
    instruction, noun, verb, query = WORDS[split]
    content = (
        f"{instruction}\n\n{before_text}\n{noun} {key} {verb} {value}.\n"
        f"{after_text}\n{query} {key}."
    )
    return [{"role": "user", "content": content}, {"role": "assistant", "content": value}]

def render_complete_chat(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], dict[str, int]]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    direct = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if list(encoded["input_ids"]) != list(direct):
        raise RuntimeError("full chat-template string/token parity failed")
    return list(direct), locate_source_and_answer_spans(rendered, encoded["offset_mapping"], messages)
```

`locate_source_and_answer_spans` identifies the unique source occurrence and
the final assistant occurrence from character offsets.  Exact sequence length
and source-to-answer-predictor distance are solved by rendering the complete
message while monotonically adjusting before/after filler spans; every accepted
row is measured from the final full-message tokenization.  Do not call the
empty-user `_chat_boundaries` helper or concatenate separately encoded pieces.

- [ ] **Step 4: Implement nonce and filler partitioning**

Use the tested candidate-token policy from `frequency_adaptation_8b.prepare_data`. Select exactly `3 * minimum` ordinary non-special one-token words and slice them into train/validation/test blocks. Flatten the frozen train tensor for training; split the frozen validation tensor at its row midpoint so validation and test sample from non-overlapping regions. Reject a tensor with fewer than two validation rows.

- [ ] **Step 5: Implement exact registered bundles and passkeys**

For each stage:

```python
train_rows = 64 * stage.accumulation
validation_groups = 16
test_groups = 32
value_tokens = 12
task_counts = {"kv": 3, "update": 1}
```

Generate segment-1 and segment-2 training examples in one deterministic order; metadata records `segment = 1` for the first `32 * accumulation` rows and `segment = 2` for the remainder. Validation/test groups expand through `build_counterfactual_triplet`. Build 25 independent numeric passkeys per stage (five trials for each registered depth), with the actual LLaMA chat template included in the prompt token budget.

- [ ] **Step 6: Implement manifest and atomic validation**

Manifest requirements:

```python
{
    "format_version": 1,
    "purpose": "evq_seed42_retrieval_repair",
    "status": "prepared_no_results",
    "seed": 42,
    "split_policy": "disjoint_nonce_filler_seed_and_wording",
    "files": {name: {"sha256": digest, "rows": rows, "seq_len": seq_len}},
}
```

Write each tensor through a unique temporary file, load and validate it, then `os.replace`. Write `manifest.json` last. Refuse an existing output directory.

- [ ] **Step 7: Run focused tests and compile**

```bash
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_evq_seed42_retrieval_repair.py \
  tests/test_frequency_adaptation_8b.py -q
conda run --no-capture-output -n aidemo python -m py_compile \
  rebuttal/evq_seed42_retrieval_repair/prepare_data.py
```

Expected: all pass.

- [ ] **Step 8: Commit the data task**

```bash
git add \
  rebuttal/evq_seed42_retrieval_repair/prepare_data.py \
  tests/test_evq_seed42_retrieval_repair.py
git commit -m "feat: prepare evq retrieval repair data"
```

---

### Task 3: EVQ parent validation and continuation trainer

**Files:**
- Create: `rebuttal/evq_seed42_retrieval_repair/train.py`
- Modify: `tests/test_evq_seed42_retrieval_repair.py`

**Interfaces:**
- Consumes: model manifest, LongAlpaca manifest, prepared repair manifest, parent adapter, stage, and segment.
- Produces: `validate_parent_adapter(adapter_dir, longalpaca_manifest, model_manifest)`, `validate_stage_transition(stage, gate_path, parent_dir)`, `runtime_frequency_contract(canonical_evq, stage)`, `select_training_rows(bundle, stage, segment)`, `build_run_protocol(configuration)`, and one atomic continuation checkpoint.

- [ ] **Step 1: Write failing artifact and factor tests**

```python
def test_runtime_factor_is_always_derived_from_canonical_evq():
    contract = runtime_frequency_contract(canonical_evq, stage="r16")
    expected, mscale, _ = official_yarn_on_inv_freq(
        canonical_evq, head_dim=128, base=500000, scale=2.0,
        original_max_position_embeddings=8192,
    )
    assert torch.equal(contract["substrate_inv_freq"], canonical_evq)
    assert torch.allclose(contract["runtime_inv_freq"], expected)
    assert contract["mscale"] == pytest.approx(mscale)
    assert contract["label"] == "YaRN-derived generalization on the EVQ substrate"

def test_segment_two_selects_only_second_registered_slice(training_bundle):
    selected = select_training_rows(training_bundle, stage="r8", segment=2)
    assert selected["input_ids"].shape[0] == 128
    assert {row["segment"] for row in selected["metadata"]} == {2}

def test_r16_parent_must_be_passing_r8_gate(tmp_path):
    with pytest.raises(RuntimeError, match="passing r8 gate"):
        validate_stage_transition("r16", tmp_path / "missing-gate.json")
```

- [ ] **Step 2: Run tests and verify RED**

Expected: trainer module/API import failure.

- [ ] **Step 3: Implement strict parent validation**

For segment 1/R8, validate the existing adapter with the same seed-42,
LongAlpaca objective, completed step-300, model-manifest hash, training-manifest
hash, LoRA config, adapter file hash, and canonical EVQ frequency checks already
used by `eval_official_yarn_capability.py`. For segment 2, require a completed
segment-1 repair protocol. For R16 segment 1, require the selected passing R8
checkpoint and a gate JSON whose checkpoint SHA matches it.

- [ ] **Step 4: Implement factor-specific runtime contracts**

```python
def runtime_frequency_contract(canonical_evq: torch.Tensor, stage: str) -> dict[str, object]:
    spec = get_stage(stage)
    runtime, mscale, metadata = official_yarn_on_inv_freq(
        canonical_evq,
        head_dim=128,
        base=500000.0,
        scale=spec.factor,
        original_max_position_embeddings=8192,
        beta_fast=32.0,
        beta_slow=1.0,
        extrapolation_factor=1.0,
        attn_factor=1.0,
    )
    return {
        "substrate_inv_freq": canonical_evq.clone().to(torch.float64),
        "runtime_inv_freq": runtime.to(torch.float64),
        "factor": spec.factor,
        "mscale": float(mscale),
        "label": "identity EVQ substrate" if spec.factor == 1 else "YaRN-derived generalization on the EVQ substrate",
        "operator": metadata,
    }
```

Patch every rotary module's `inv_freq`, `original_inv_freq` when present,
`attention_scaling`, and caches. Verify all modules before training.

- [ ] **Step 5: Implement answer-only continuation**

Load the base in BF16 SDPA and the validated adapter using
`PeftModel.from_pretrained(model, str(parent_dir), is_trainable=True)`. Reuse
`TensorAnswerDataset`, `tail_answer_cross_entropy`, `LoraPairDiagnostics`, and
the `logits_to_keep` fail-closed check. Configure exactly 32 steps, microbatch 1,
stage accumulation, learning rate `2e-5`, warm-up 4, cosine schedule, weight
decay `0.01`, max norm `1.0`, gradient checkpointing, and no reporting service.

- [ ] **Step 6: Save complete provenance atomically**

Save adapter/tokenizer plus:

```python
{
    "format_version": 1,
    "purpose": "evq_seed42_retrieval_repair",
    "status": "complete_uninterpreted",
    "stage": stage,
    "segment": segment,
    "seed": 42,
    "parent": {"adapter_sha256": parent_sha, "protocol_sha256": protocol_sha},
    "training": segment_contract(stage, segment),
    "frequency": {
        "factor": factor,
        "label": label,
        "substrate_sha256": tensor_sha256(substrate_inv_freq),
        "runtime_sha256": tensor_sha256(runtime_inv_freq),
    },
}
```

Store canonical substrate and runtime tensors in `frequency_artifact.pt`; do not
reuse `custom_inv_freq.pt` for two different meanings. Mark status complete only
after adapter, frequency, diagnostics, and trainer state all validate.

- [ ] **Step 7: Run focused tests, help, dry-run, and compile**

```bash
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_evq_seed42_retrieval_repair.py \
  tests/test_frequency_adaptation_8b.py \
  tests/test_official_yarn_parity.py -q
conda run --no-capture-output -n aidemo python -m rebuttal.evq_seed42_retrieval_repair.train --help
conda run --no-capture-output -n aidemo python -m py_compile \
  rebuttal/evq_seed42_retrieval_repair/train.py
```

Expected: all pass without CUDA or model loading.

- [ ] **Step 8: Commit the trainer task**

```bash
git add \
  rebuttal/evq_seed42_retrieval_repair/train.py \
  tests/test_evq_seed42_retrieval_repair.py
git commit -m "feat: add evq retrieval continuation trainer"
```

---

### Task 4: Corrected evaluator and gate report

**Files:**
- Create: `rebuttal/evq_seed42_retrieval_repair/evaluate.py`
- Modify: `experiments/lora_evq_v2/eval_official_yarn_capability.py`
- Modify: `tests/test_evq_seed42_retrieval_repair.py`
- Modify: `tests/test_official_yarn_capability_eval.py`

**Interfaces:**
- Consumes: parent/repair adapter, factor-specific frequency artifact, validation/test/passkey bundle, parent temporal result, checkpoint temporal result.
- Produces: `score_generated_tokens(generated_ids, expected_ids, tokenizer)`, `summarize_repair_records(records, stage)`, `build_gate_report(stage, checkpoint_sha256, repair_summary, passkey_summary, parent_temporal, checkpoint_temporal)`, atomic raw JSON, and atomic gate JSON.

- [ ] **Step 1: Write failing metric regression tests**

```python
def test_correct_passkey_with_trailing_text_is_not_strict_but_is_retrieved():
    score = score_text_answer("12345678\n12345678", "12345678", False)
    assert score["strict_exact"] is False
    assert score["first_value_exact"] is True
    assert score["gold_containment"] is True

def test_repair_factor_parser_accepts_one_registered_factor():
    assert parse_registered_factor("1") == 1.0
    assert parse_registered_factor("2") == 2.0
    assert parse_registered_factor("4") == 4.0
    with pytest.raises(ValueError):
        parse_registered_factor("2,4")

def test_legacy_official_capability_parser_keeps_matched_pair():
    assert parse_yarn_factors("2,4") == (2.0, 4.0)

def test_factor_length_mapping_rejects_composed_or_wrong_context():
    assert registered_factor_for_length(8192) == 1.0
    assert registered_factor_for_length(16384) == 2.0
    assert registered_factor_for_length(32768) == 4.0
```

- [ ] **Step 2: Run tests and verify RED**

Expected: the new repair factor parser is missing and the existing evaluator
still reports only one exact metric.

- [ ] **Step 3: Correct official capability evaluation semantics**

Keep the existing evaluator's public `--yarn_factors 2,4` contract for the
matched Geo/EVQ runs. Add a repair-only parser in `evaluate.py` that accepts
exactly one factor from `{1,2,4}` and selects only rows whose target length
matches that factor. Always transform the canonical substrate, never the saved
factor-specific runtime tensor. Generation records prediction text, generated
token count, EOS termination, strict exact, first extracted value exact,
containment, and gold NLL.

- [ ] **Step 4: Implement controlled triplet evaluation**

Reuse tail-logit teacher-forced scoring and generation with an explicit
attention mask. Generate only original/swapped; source-removed uses NLL. Group
records by `group_id`, require all three variants, and calculate:

```python
{
    "pair_consistency": both_original_and_swapped_exact / groups,
    "source_removal_positive_fraction": positive_removal_delta / groups,
    "by_task": {"kv": kv_summary, "update": update_summary},
    "by_distance_bucket": distance_summaries,
}
```

- [ ] **Step 5: Implement temporal merge and gate output**

Read two frozen temporal result JSONs, verify identical examples/factor, compute
checkpoint mean NLL minus parent mean NLL, then call `decide_gate`. The gate JSON
contains checkpoint adapter SHA, evaluation/data hashes, every threshold/check,
`pass|rescue_allowed|stop`, and no interpretation beyond those registered rules.

- [ ] **Step 6: Run focused and regression tests**

```bash
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_evq_seed42_retrieval_repair.py \
  tests/test_official_yarn_capability_eval.py \
  tests/test_official_yarn_parity.py \
  tests/test_frequency_adaptation_8b.py -q
conda run --no-capture-output -n aidemo python -m py_compile \
  rebuttal/evq_seed42_retrieval_repair/evaluate.py \
  experiments/lora_evq_v2/eval_official_yarn_capability.py
```

Expected: all pass.

- [ ] **Step 7: Commit the evaluator task**

```bash
git add \
  rebuttal/evq_seed42_retrieval_repair/evaluate.py \
  experiments/lora_evq_v2/eval_official_yarn_capability.py \
  tests/test_evq_seed42_retrieval_repair.py \
  tests/test_official_yarn_capability_eval.py
git commit -m "fix: separate retrieval and output-format metrics"
```

---

### Task 5: Explicit cost-safe launcher

**Files:**
- Create: `rebuttal/evq_seed42_retrieval_repair/run_seed42.sh`
- Modify: `tests/test_evq_seed42_retrieval_repair.py`

**Interfaces:**
- Consumes: environment variables for local model, manifests, parent adapter, frozen filler, capability data, work directory, and Python.
- Produces: `prepare`, `preflight`, `baseline`, `train-r8`, `gate-r8`, `train-r16`, `gate-r16`, and `final` commands; none advances to another command automatically.

- [ ] **Step 1: Add failing launcher-contract tests**

```python
def test_launcher_has_explicit_non_advancing_commands():
    text = Path("rebuttal/evq_seed42_retrieval_repair/run_seed42.sh").read_text()
    for command in ("prepare", "preflight", "baseline", "train-r8", "gate-r8", "train-r16", "gate-r16", "final"):
        assert command in text
    assert "flock" in text
    assert "nvidia-smi" in text

def test_launcher_never_downloads_or_autostarts_next_stage():
    text = Path("rebuttal/evq_seed42_retrieval_repair/run_seed42.sh").read_text()
    assert "git clone" not in text
    assert "wget " not in text
    assert "curl " not in text
    train_r8_body = text.split("train-r8)", 1)[1].split(";;", 1)[0]
    assert "train-r16)" not in train_r8_body
```

- [ ] **Step 2: Run tests and verify RED**

Expected: launcher file is missing.

- [ ] **Step 3: Implement environment and preflight checks**

Require:

```text
EVQ_REPAIR_MODEL
EVQ_REPAIR_MODEL_MANIFEST
EVQ_REPAIR_LONGALPACA_MANIFEST
EVQ_REPAIR_PARENT_ADAPTER
EVQ_REPAIR_FILLER_DIR
EVQ_REPAIR_CAPABILITY_DIR
EVQ_REPAIR_WORK_DIR
PYTHON_BIN
```

`prepare` is CPU-only. `preflight` runs compile, pytest, shell syntax, manifest
validation, data validation, factor hashes, and trainer/evaluator dry-runs.
Training/evaluation commands require successful `nvidia-smi`, acquire one
global `flock`, reject existing outputs, print stage/factor/token budget, and
then launch exactly one requested operation.

- [ ] **Step 4: Implement gate transition validation**

Before R16, invoke a Python validation entrypoint that verifies a passing R8
gate and exact checkpoint SHA. Before final, do the same for R16. Do not parse
JSON with shell string matching. Segment 2 is a separate explicit argument and
requires `rescue_allowed` from segment 1.

- [ ] **Step 5: Run shell and focused tests**

```bash
bash -n rebuttal/evq_seed42_retrieval_repair/run_seed42.sh
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_evq_seed42_retrieval_repair.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit the launcher task**

```bash
git add \
  rebuttal/evq_seed42_retrieval_repair/run_seed42.sh \
  tests/test_evq_seed42_retrieval_repair.py
git commit -m "feat: add gated evq repair launcher"
```

---

### Task 6: Full verification, review, and selective server upload

**Files:**
- Verify all files from Tasks 1-5.
- Do not modify server data/checkpoints except to create the explicit repair work directory and prepared artifacts.

**Interfaces:**
- Consumes: reviewed local branch and no-GPU SSH server.
- Produces: local verification evidence, independent review, selected-file upload, and server CPU preflight evidence.

- [ ] **Step 1: Run the complete local verification**

```bash
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_evq_seed42_retrieval_repair.py \
  tests/test_frequency_adaptation_8b.py \
  tests/test_official_yarn_parity.py \
  tests/test_official_yarn_capability_eval.py -q

conda run --no-capture-output -n aidemo python -m py_compile \
  rebuttal/evq_seed42_retrieval_repair/protocol.py \
  rebuttal/evq_seed42_retrieval_repair/prepare_data.py \
  rebuttal/evq_seed42_retrieval_repair/train.py \
  rebuttal/evq_seed42_retrieval_repair/evaluate.py \
  experiments/lora_evq_v2/eval_official_yarn_capability.py

bash -n rebuttal/evq_seed42_retrieval_repair/run_seed42.sh
git diff --check
```

- [ ] **Step 2: Run an independent subagent review and fix findings**

Give the reviewer the approved design, this plan, branch diff, and test report.
Require findings ordered by severity with file/line evidence covering scientific
identity, factor composition, split leakage, adapter provenance, metric logic,
gate transitions, and GPU cost safety. Add regression tests before every fix.

- [ ] **Step 3: Upload only selected reviewed files**

Use `rsync -R` or explicit `scp` paths. Do not use `rsync --delete` and do not
copy the entire dirty repository. Upload the new package, its tests, the
official evaluator/operator files it imports, and launcher only.

- [ ] **Step 4: Run server CPU/no-GPU preflight**

Over SSH, identify the repository root, Python environment, model manifest,
parent adapter, frozen filler/capability roots, and available disk. Run the
focused pytest/compile/shell checks, then `run_seed42.sh prepare` and
`run_seed42.sh preflight`. Confirm no training/evaluation PID and no CUDA model
load. Record prepared manifest hashes and exact future GPU commands.

- [ ] **Step 5: Report actual state**

Report branch/head, changed files, local and server test counts, uploaded paths,
prepared data/manifest hashes, remaining server disk, and the exact first GPU
command. State explicitly that no GPU result exists until the user starts a GPU
and the baseline/training commands run.
