# LLaMA-3-8B LoRA Industrial Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed pipeline that first tests existing seed-42 LoRA artifacts for raw source-dependent capability, then conditionally supports a matched 8K-only continuation and a same-base midpoint-Geo/EVQ attribution experiment.

**Architecture:** A new isolated package owns immutable arm and gate contracts, deterministic chat-templated counterfactual data, one-process raw-frequency evaluation, clustered statistics, and answer-only continuation training. Existing temporal evaluators remain the source of artifact-validation patterns, while the YaRN capability evaluator remains a separate range-scaling tool and is not reused as the primary endpoint. A shell launcher exposes explicit non-advancing stages so no passing gate can automatically spend additional GPU time.

**Tech Stack:** Python 3.12, PyTorch 2.8+, Transformers 5.x, PEFT 0.19+, NumPy, shell, pytest; no new dependency.

## Global Constraints

- Use the same manifested LLaMA-3-8B-Instruct bytes and tokenizer as the completed LongAlpaca seed-42 pair.
- Existing parents are completed step-300 Geo+LoRA seed 42 and EVQ+LoRA seed 42 on the frozen LongAlpaca tensor.
- Preserve LoRA `q_proj,k_proj,v_proj,o_proj`, rank 64, alpha 128, dropout 0.05.
- Stage 0 and primary transfer evaluation use raw schedules only: no YaRN, NTK, PI, PoSE, LongRoPE, or other scaler.
- Treat native endpoint Geo, midpoint Geo, and midpoint EVQ-Cosh as three distinct frequency identities.
- `tau=1.414` and rank 64 are empirical fixed settings, not theoretical optima or rank/channel thresholds.
- Every task prompt uses the full LLaMA-3 chat template; all arms receive identical prompt and target token IDs.
- Teacher-forced NLL scores one frozen canonical target; do not choose a reference separately per arm.
- Primary lengths are 4K, 8K, 16K, and 32K. Diagnostic 12K/24K cannot alter the primary decision.
- Stage 1 physical length and every position ID are at most 8,192 and 8,191 respectively.
- Stage 1 resets identical optimizer and scheduler state for both parents and trains only existing LoRA parameters.
- Record physical and supervised tokens separately.
- Bootstrap semantic groups, not original/swapped/source-removed rows.
- Do not modify paper numbers, paper claims, historical result JSON, or `paper/main.pdf`.
- CPU preparation, hashes, tests, dry runs, output paths, and launch commands must be complete before CUDA model loading.
- Never overwrite an output, silently drop an arm/task, or automatically advance after a gate.

## File Structure

- Create `experiments/lora_evq_v2/industrial_capability/__init__.py`: public package exports.
- Create `experiments/lora_evq_v2/industrial_capability/protocol.py`: arm, length, training, metric, and gate contracts.
- Create `experiments/lora_evq_v2/industrial_capability/theory_diagnostics.py`: reproducible schedule-only diagnostics.
- Create `experiments/lora_evq_v2/industrial_capability/data.py`: deterministic controlled and external data freezing.
- Create `experiments/lora_evq_v2/industrial_capability/runtime.py`: model, adapter, and frequency activation.
- Create `experiments/lora_evq_v2/industrial_capability/evaluate.py`: answer NLL and generation for any registered arm.
- Create `experiments/lora_evq_v2/industrial_capability/summarize.py`: grouped metrics, clustered bootstrap, and gates.
- Create `experiments/lora_evq_v2/industrial_capability/train.py`: Stage 1 continuation and Stage 2 same-base training.
- Create `scripts/2026-07/14_lora_industrial_capability.sh`: explicit CPU/GPU orchestration.
- Create focused tests under `tests/test_lora_industrial_*.py`.
- Modify `rebuttal/README.md`: link the approved design and implementation plan as the authoritative LoRA capability path.

---

### Task 1: Immutable protocol and theory diagnostics

**Files:**
- Create: `experiments/lora_evq_v2/industrial_capability/__init__.py`
- Create: `experiments/lora_evq_v2/industrial_capability/protocol.py`
- Create: `experiments/lora_evq_v2/industrial_capability/theory_diagnostics.py`
- Create: `tests/test_lora_industrial_protocol.py`

**Interfaces:**
- Produces: `ArmSpec`, `ContinuationSpec`, `stage0_arms()`, `performance_arms()`, `diagnostic_arms()`, `continuation_spec()`, `validate_position_ids()`, `schedule_diagnostics()`, and `write_diagnostics()`.
- Consumes: canonical schedule helpers from `experiments.lora_evq_v2.train_evq_lora` and `scripts.lib.rope.schedules`.

- [ ] **Step 1: Write failing tests for the seven raw arms and training budget**

```python
from experiments.lora_evq_v2.industrial_capability.protocol import (
    continuation_spec,
    diagnostic_arms,
    performance_arms,
    stage0_arms,
    validate_position_ids,
)

def test_stage0_contract_has_seven_raw_arms_without_scaling():
    arms = stage0_arms()
    assert tuple(arms) == (
        "base_native",
        "base_midpoint",
        "base_evq",
        "geo_lora_native",
        "geo_lora_evq_cross",
        "evq_lora_native_cross",
        "evq_lora_evq",
    )
    assert all(arm.range_scaler is None for arm in arms.values())
    assert performance_arms() == ("base_native", "geo_lora_native", "evq_lora_evq")
    assert set(diagnostic_arms()) == set(arms) - set(performance_arms())

def test_continuation_budget_separates_physical_and_supervised_tokens():
    spec = continuation_spec()
    assert (spec.seq_len, spec.steps, spec.effective_batch) == (8192, 32, 4)
    assert spec.physical_tokens_per_step == 32768
    assert spec.physical_tokens_per_segment == 1_048_576
    assert spec.supervised_tokens_per_segment == 1664

def test_position_ids_fail_above_8k():
    validate_position_ids([0, 8191])
    with pytest.raises(ValueError, match="position_ids.max"):
        validate_position_ids([0, 8192])
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_lora_industrial_protocol.py -q
```

Expected: collection fails because `industrial_capability.protocol` does not exist.

- [ ] **Step 3: Implement the exact dataclasses and arm registry**

```python
from dataclasses import dataclass
from typing import Literal

Frequency = Literal["native_endpoint_geo", "midpoint_geo", "midpoint_evq"]
Adapter = Literal["none", "geo_s42", "evq_s42"]

@dataclass(frozen=True)
class ArmSpec:
    name: str
    adapter: Adapter
    frequency: Frequency
    tau: float | None
    fair_performance_arm: bool
    range_scaler: None = None

@dataclass(frozen=True)
class ContinuationSpec:
    seq_len: int = 8192
    steps: int = 32
    effective_batch: int = 4
    answer_tokens_per_example: int = 13
    learning_rate: float = 2e-5
    warmup_steps: int = 4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    @property
    def physical_tokens_per_step(self) -> int:
        return self.seq_len * self.effective_batch

    @property
    def physical_tokens_per_segment(self) -> int:
        return self.steps * self.physical_tokens_per_step

    @property
    def supervised_tokens_per_segment(self) -> int:
        return self.steps * self.effective_batch * self.answer_tokens_per_example

ARMS = {
    "base_native": ArmSpec("base_native", "none", "native_endpoint_geo", None, True),
    "base_midpoint": ArmSpec("base_midpoint", "none", "midpoint_geo", 0.0, False),
    "base_evq": ArmSpec("base_evq", "none", "midpoint_evq", 1.414, False),
    "geo_lora_native": ArmSpec("geo_lora_native", "geo_s42", "native_endpoint_geo", None, True),
    "geo_lora_evq_cross": ArmSpec("geo_lora_evq_cross", "geo_s42", "midpoint_evq", 1.414, False),
    "evq_lora_native_cross": ArmSpec("evq_lora_native_cross", "evq_s42", "native_endpoint_geo", None, False),
    "evq_lora_evq": ArmSpec("evq_lora_evq", "evq_s42", "midpoint_evq", 1.414, True),
}

def stage0_arms():
    return dict(ARMS)

def performance_arms():
    return ("base_native", "geo_lora_native", "evq_lora_evq")

def diagnostic_arms():
    keep = set(performance_arms())
    return tuple(name for name in ARMS if name not in keep)

def continuation_spec():
    return ContinuationSpec()

def validate_position_ids(position_ids) -> None:
    maximum = max(int(value) for value in position_ids)
    if maximum > 8191:
        raise ValueError(f"position_ids.max={maximum} exceeds 8191")
```

Expose defensive copies or immutable mappings; reject unknown arms and any non-null range-scaler field. `__init__.py` re-exports `ArmSpec`, `ContinuationSpec`, `stage0_arms`, `continuation_spec`, and `validate_position_ids` through an explicit `__all__` tuple.

- [ ] **Step 4: Add failing tests for independently reproducible theory rows**

```python
from experiments.lora_evq_v2.industrial_capability.theory_diagnostics import schedule_diagnostics

def test_llama8b_schedule_diagnostics_match_registered_values():
    out = schedule_diagnostics(head_dim=128, base=500000.0, train_length=8192, tau=1.414)
    assert out["dormant_omega_l_lt_1"] == {"native": 20, "midpoint": 20, "evq": 15}
    assert out["cross_8k_to_32k"] == {"native": 7, "midpoint": 7, "evq": 5}
    assert out["uniform_raw_entropy_rank"]["native"] == pytest.approx(23.60, abs=0.01)
    assert out["uniform_raw_entropy_rank"]["evq"] == pytest.approx(36.23, abs=0.01)
```

- [ ] **Step 5: Implement schedule diagnostics and atomic JSON output**

Use the canonical native endpoint, midpoint `tau=0`, and midpoint EVQ `tau=1.414` frequency constructors. Construct midpoint Geo explicitly with `compute_evq_cosh_inv_freq(head_dim, base, tau=0.0, midpoint=True)`; do not call `build_training_inv_freq("native_geo", ...)`, which intentionally returns the endpoint grid. Compute:

```python
def entropy_effective_rank(matrix: np.ndarray) -> float:
    eigenvalues = np.clip(np.linalg.eigvalsh((matrix + matrix.T) / 2.0), 0.0, None)
    probabilities = eigenvalues[eigenvalues > 1e-14]
    probabilities = probabilities / probabilities.sum()
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))
```

Use integer distances `0..L-1` for the causal-triangular prior with normalized weight `L-d`. Record the grid, centering, prior, thresholds, NumPy version, and schedule hashes. Write to `<output>.incomplete`, `fsync`, then rename; reject an existing output.

- [ ] **Step 6: Run focused tests and compile**

```bash
.venv/bin/python -m pytest tests/test_lora_industrial_protocol.py -q
.venv/bin/python -m py_compile \
  experiments/lora_evq_v2/industrial_capability/protocol.py \
  experiments/lora_evq_v2/industrial_capability/theory_diagnostics.py
```

Expected: all tests pass and compilation exits 0.

- [ ] **Step 7: Commit Task 1**

```bash
git add experiments/lora_evq_v2/industrial_capability tests/test_lora_industrial_protocol.py
git commit -m "feat: define lora capability protocol"
```

---

### Task 2: Frozen chat-templated counterfactual data

**Files:**
- Create: `experiments/lora_evq_v2/industrial_capability/data.py`
- Create: `tests/test_lora_industrial_data.py`

**Interfaces:**
- Consumes: LLaMA-3 tokenizer, a frozen natural-text filler tensor, pinned RULER JSONL, and explicit output directory.
- Produces: `CapabilityRecord`, `chat_prompt_ids()`, `build_counterfactual_group()`, `build_training_record()`, `prepare_bundle()`, `validate_bundle()`, `manifest.json`, `controlled.jsonl`, `training.pt`, `validation.jsonl`, and `external.jsonl`.

- [ ] **Step 1: Write failing tests for the chat, triplet, and canonical-answer contracts**

```python
def test_chat_prompt_uses_generation_prompt(fake_tokenizer):
    ids = chat_prompt_ids(fake_tokenizer, "read this", add_generation_prompt=True)
    assert ids == fake_tokenizer.expected_chat_ids
    assert fake_tokenizer.last_apply_chat_template == {
        "tokenize": True,
        "add_generation_prompt": True,
    }

def test_triplet_is_length_and_position_matched(fake_tokenizer):
    rows = build_counterfactual_group(fake_tokenizer, frozen_case(), target_length=8192)
    assert [row.variant for row in rows] == ["original", "swapped", "source_removed"]
    assert len({len(row.prompt_ids) for row in rows}) == 1
    assert len({row.answer_start for row in rows}) == 1
    assert rows[0].canonical_target_ids != rows[1].canonical_target_ids
    assert rows[2].canonical_target_ids == rows[0].canonical_target_ids

def test_canonical_target_is_fixed_before_model_scoring():
    row = build_record_with_aliases(["New York", "NYC"])
    assert row.canonical_reference_index == 0
    assert row.canonical_target_text == "New York"
```

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/test_lora_industrial_data.py -q
```

Expected: import failure for `industrial_capability.data`.

- [ ] **Step 3: Implement the record schema and chat helper**

```python
@dataclass(frozen=True)
class CapabilityRecord:
    schema: str
    example_id: str
    group_id: str
    split: str
    suite: str
    task: str
    variant: str
    target_length: int
    source_query_distance: int
    prompt_ids: tuple[int, ...]
    canonical_target_ids: tuple[int, ...]
    canonical_target_text: str
    references: tuple[str, ...]
    canonical_reference_index: int
    answer_start: int
    generation_tokens: int
    prompt_sha256: str

def chat_prompt_ids(tokenizer, text: str, *, add_generation_prompt: bool) -> tuple[int, ...]:
    values = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(text)}],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if not values:
        raise ValueError("chat template produced no token IDs")
    return tuple(int(value) for value in values)
```

The schema string is `evq_cosh.lora_industrial_capability.v1`. `validate_record` rejects a nonzero canonical reference index, empty target, duplicate ID, unknown variant, prompt over target length, or a hash mismatch.

- [ ] **Step 4: Implement deterministic split-disjoint controlled groups**

Use seed 42 for train, 4201 for validation, and 4202 for test. Build exactly three task families: `kv`, `last_write`, and `two_hop`. Use disjoint nonce pools and template IDs by split. Freeze primary test groups as:

```python
PRIMARY_LENGTHS = (4096, 8192, 16384, 32768)
GROUPS_PER_TASK_LENGTH = 16
CANARY_GROUPS_PER_TASK_LENGTH = 4
```

The source span and replacement span must have equal token length. `source_removed` draws a nonce-free span from held-out natural filler and verifies that neither the key nor either target occurs. Store `source_query_distance = answer_start - 1 - source_value_start` and fail if the requested distance falls outside the registered cell.

- [ ] **Step 5: Implement answer-only training records**

Create 128 deterministic 8K examples per 32-step segment with mix counts `45 kv`, `32 last_write`, `32 two_hop`, and `19 natural_qa`. Natural-QA rows select a deterministic 12-token extractive span from the evidence. Each example contains exactly 13 supervised tokens including EOS. Store `input_ids`, `labels`, `task`, `distance`, `template_id`, and `row_hash`; all non-answer labels are `-100`.

Allocate distances by exact row counts: 26 rows in `[256,2048]`, 38 rows in `(2048,5120]`, and 64 rows in `(5120,7680]`. Shuffle once with seed 42 and record the final row-order SHA-256.

- [ ] **Step 6: Import the fixed external guardrails**

Reuse the pinned source receipts already owned by `prepare_seed42_capability_data.py`. Wrap each unchanged RULER prompt string as one LLaMA-3 user message and label it `llama3_chat_adapted_ruler`. Preserve its references and scorer contract. For MCQA, freeze answer labels such as `" A"`, `" B"`, not option texts. Keep LongBench/NoLiMa records secondary and reject partial source preparation.

- [ ] **Step 7: Implement atomic bundle writing and validation**

Write every file as `.incomplete`, validate row counts, hashes, split disjointness, chat markers, task/length cells, canonical answers, and maximum position, then rename. Write `manifest.json` last. It records tokenizer files, source revisions, filler hash, code hash, group counts, row-order hash, and every output SHA-256.

- [ ] **Step 8: Run focused and regression tests**

```bash
.venv/bin/python -m pytest \
  tests/test_lora_industrial_data.py \
  tests/test_seed42_capability_data.py \
  tests/test_evq_seed42_retrieval_repair.py -q
.venv/bin/python -m py_compile experiments/lora_evq_v2/industrial_capability/data.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add experiments/lora_evq_v2/industrial_capability/data.py tests/test_lora_industrial_data.py
git commit -m "feat: freeze lora capability data"
```

---

### Task 3: One-process seven-arm raw evaluator

**Files:**
- Create: `experiments/lora_evq_v2/industrial_capability/runtime.py`
- Create: `experiments/lora_evq_v2/industrial_capability/evaluate.py`
- Create: `tests/test_lora_industrial_evaluator.py`

**Interfaces:**
- Consumes: Task 1 arms, Task 2 bundle, model/adapter/training manifests, and an explicit arm/subset list.
- Produces: `load_runtime()`, `activate_arm()`, `score_canonical_answer()`, `generate_answer()`, `evaluate_rows()`, and atomic per-arm JSONL files.

- [ ] **Step 1: Write failing tests for exact arm activation**

```python
def test_arm_activation_switches_adapter_and_frequency(fake_runtime):
    with activate_arm(fake_runtime, "base_midpoint"):
        assert fake_runtime.active_adapter is None
        assert fake_runtime.frequency_name == "midpoint_geo"
    with activate_arm(fake_runtime, "geo_lora_evq_cross"):
        assert fake_runtime.active_adapter == "geo"
        assert fake_runtime.frequency_name == "midpoint_evq"

def test_evaluator_rejects_range_scaling_argument():
    with pytest.raises(ValueError, match="range scaling is outside Stage 0"):
        parse_args(["--factor", "2"])
```

- [ ] **Step 2: Write failing tests for fixed-target NLL and MCQA labels**

```python
def test_nll_scores_only_the_frozen_canonical_target(fake_backbone, fake_head):
    score = score_canonical_answer(
        fake_backbone,
        fake_head,
        prompt_ids=[10, 11, 12],
        target_ids=[21, 22],
        device="cpu",
    )
    assert score["target_ids"] == [21, 22]
    assert score["target_tokens"] == 2
    assert score["nll"] == pytest.approx(score["nll_sum"] / 2)

def test_mcqa_scores_labels_not_choice_text():
    assert mcqa_targets(4) == (" A", " B", " C", " D")
```

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m pytest tests/test_lora_industrial_evaluator.py -q
```

Expected: import failures for `runtime` and `evaluate`.

- [ ] **Step 4: Implement strict runtime loading**

Follow the validated order in `eval_temporal_holdout_three_arm.py`:

1. validate the model manifest and tokenizer fingerprint before CUDA load;
2. validate Geo/EVQ adapter status, step 300, seed 42, model manifest, LongAlpaca manifest, LoRA config, and adapter hashes;
3. construct canonical native endpoint, midpoint Geo, and midpoint EVQ tensors;
4. verify the saved Geo and EVQ frequency artifacts against their canonical tensors;
5. load one BF16 base model with `attn_implementation="sdpa"`;
6. attach Geo as `geo` and EVQ as `evq`;
7. run base-disabled and active-adapter logit canaries;
8. expose a context manager that always restores the previous adapter and frequency.

Do not call `configure_packed_free_causal_sdpa` on the generation path. Under every arm, call `verify_model_inv_freq` before and after its row loop.

- [ ] **Step 5: Implement memory-bounded canonical answer NLL**

```python
def score_canonical_answer(backbone, lm_head, *, prompt_ids, target_ids, device):
    full = torch.tensor([list(prompt_ids) + list(target_ids)], device=device, dtype=torch.long)
    with torch.inference_mode():
        hidden = backbone(
            input_ids=full,
            attention_mask=None,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        start = len(prompt_ids) - 1
        stop = full.shape[1] - 1
        logits = lm_head(hidden[:, start:stop]).float()
        labels = full[:, len(prompt_ids):]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        )
    nll_sum = float(losses.double().sum().cpu())
    return {
        "target_ids": [int(value) for value in target_ids],
        "nll_sum": nll_sum,
        "target_tokens": int(labels.numel()),
        "nll": nll_sum / int(labels.numel()),
    }
```

Reject empty targets and prompts that exceed their frozen target length. For aliases, keep this canonical score fixed; generation may compare the decoded prediction against every frozen reference.

- [ ] **Step 6: Implement greedy generation and raw output**

Use the normal model forward with `use_cache=True`, `do_sample=False`, `temperature=None`, tokenizer EOS IDs, and task-specific `max_new_tokens`. Record raw token IDs, decoded text, strict exact, first-value exact, containment, EOS termination, and token count. Generate only original/swapped rows; source-removed rows receive NLL only.

Write one non-overwriting JSONL per arm. Each row contains model/adapter/frequency hashes, arm, prompt hash, group ID, variant, task, length, distance, canonical target, NLL sum/tokens, generation fields, runtime, and peak memory.

- [ ] **Step 7: Implement cost-gated subset selection**

Expose exact modes:

```python
SUBSETS = {
    "canary": {"arms": performance_arms(), "lengths": (4096, 8192), "groups_per_cell": 4},
    "core": {"arms": performance_arms(), "lengths": (4096, 8192, 16384, 32768), "groups_per_cell": 16},
    "mechanism": {"arms": diagnostic_arms(), "lengths": (8192, 16384, 32768), "groups_per_cell": 8},
}
```

Selection is by sorted frozen `group_id`; never select by model score. Require a non-existing output directory and a GPU lock before moving the model to CUDA.

- [ ] **Step 8: Run focused and regression tests**

```bash
.venv/bin/python -m pytest \
  tests/test_lora_industrial_evaluator.py \
  tests/test_temporal_three_arm_eval.py \
  tests/test_official_yarn_capability_eval.py -q
.venv/bin/python -m py_compile \
  experiments/lora_evq_v2/industrial_capability/runtime.py \
  experiments/lora_evq_v2/industrial_capability/evaluate.py
```

Expected: all tests pass; the new evaluator has no YaRN factor path.

- [ ] **Step 9: Commit Task 3**

```bash
git add \
  experiments/lora_evq_v2/industrial_capability/runtime.py \
  experiments/lora_evq_v2/industrial_capability/evaluate.py \
  tests/test_lora_industrial_evaluator.py
git commit -m "feat: evaluate raw lora capability arms"
```

---

### Task 4: Grouped summaries, clustered uncertainty, and Stage 0 gates

**Files:**
- Create: `experiments/lora_evq_v2/industrial_capability/summarize.py`
- Create: `tests/test_lora_industrial_summary.py`

**Interfaces:**
- Consumes: complete per-arm JSONL outputs and the Task 2 manifest.
- Produces: `summarize_groups()`, `paired_cluster_bootstrap()`, `stage0_validity_gate()`, `stage0_effect_gate()`, `summary.json`, and `gate.json`.

- [ ] **Step 1: Write failing tests for triplet pairing and clustered bootstrap**

```python
def test_group_summary_requires_all_three_variants():
    with pytest.raises(ValueError, match="original, source_removed, swapped"):
        summarize_groups(rows_missing_source_removed())

def test_cluster_bootstrap_resamples_groups_not_rows():
    result = paired_cluster_bootstrap(
        group_deltas={"g0": 1.0, "g1": 1.0, "g2": -1.0},
        resamples=10000,
        seed=20260714,
    )
    assert result["unit"] == "semantic_group"
    assert result["observed"] == pytest.approx(1 / 3)
```

- [ ] **Step 2: Write failing tests for invalid, null, and positive gates**

```python
def test_stage0_invalid_when_geo_and_base_fail_8k():
    gate = stage0_validity_gate(summary(base_pair=.4, geo_pair=.5, removal=.8))
    assert gate["status"] == "invalid"

def test_stage0_positive_requires_ci_effect_size_tasks_and_nll():
    gate = stage0_effect_gate(summary(
        ci_lower=.02,
        mean_points=6.0,
        positive_tasks=2,
        delta_answer_nll=-.2,
        removal_delta=.3,
    ))
    assert gate["status"] == "positive"
```

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m pytest tests/test_lora_industrial_summary.py -q
```

- [ ] **Step 4: Implement exact group metrics and bootstrap**

For each `(arm, task, length, group_id)`, require one row per variant. Define pair consistency as `original_exact and swapped_exact`. Define source-removal delta as `removed_nll - original_nll`; positive means the source helped. Preserve strict exact, extracted exact, NLL, EOS, and task fields separately.

```python
def paired_cluster_bootstrap(group_deltas, *, resamples=10000, seed=20260714):
    keys = tuple(sorted(group_deltas))
    values = np.asarray([float(group_deltas[key]) for key in keys], dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires at least two finite semantic groups")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[draws].mean(axis=1)
    return {
        "unit": "semantic_group",
        "seed": seed,
        "resamples": resamples,
        "observed": float(values.mean()),
        "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
    }
```

- [ ] **Step 5: Implement the registered gates**

Validity is `valid` only when adapter/frequency/data identities pass, every expected group exists, arm token IDs match, at least one of `base_native` or `geo_lora_native` reaches 0.60 pair consistency at 8K, and its source-removal positive fraction reaches 0.60. Otherwise emit `invalid` and list failed checks.

Effect status is `positive` only when the 16K/32K mean EVQ-minus-Geo pair-consistency effect is at least 5 points, the group-bootstrap lower bound is above zero, at least two task families are positive, EVQ canonical answer NLL is lower, and source-removal sensitivity is higher. A valid non-positive outcome is `negative`, never `invalid`.

- [ ] **Step 6: Run tests and compile**

```bash
.venv/bin/python -m pytest tests/test_lora_industrial_summary.py -q
.venv/bin/python -m py_compile experiments/lora_evq_v2/industrial_capability/summarize.py
```

- [ ] **Step 7: Commit Task 4**

```bash
git add experiments/lora_evq_v2/industrial_capability/summarize.py tests/test_lora_industrial_summary.py
git commit -m "feat: gate lora capability evidence"
```

---

### Task 5: Matched 8K-only continuation trainer

**Files:**
- Create: `experiments/lora_evq_v2/industrial_capability/train.py`
- Create: `tests/test_lora_industrial_training.py`

**Interfaces:**
- Consumes: Task 2 training tensor, one validated parent adapter or common initial adapter, one registered schedule, and an immutable run protocol.
- Produces: `answer_only_cross_entropy()`, `build_optimizer()`, `train_segment()`, adapter checkpoint, `custom_inv_freq.pt`, `run_protocol.json`, `train_log.jsonl`, and `mechanism_diagnostics.json`.

- [ ] **Step 1: Write failing tests for answer-only loss and position safety**

```python
def test_answer_only_loss_ignores_prompt_tokens():
    labels = torch.tensor([[-100, -100, 7, 8]])
    logits = fixed_logits(batch=1, length=4, vocab=16)
    loss = answer_only_cross_entropy(logits, labels)
    expected = F.cross_entropy(logits[:, 1:3].reshape(-1, 16), torch.tensor([7, 8]))
    assert float(loss) == pytest.approx(float(expected))

def test_trainer_rejects_long_position_ids():
    with pytest.raises(ValueError, match="position_ids.max"):
        validate_training_batch({"position_ids": torch.tensor([[0, 8192]])})
```

- [ ] **Step 2: Write failing tests for parent identity and fresh optimizer state**

```python
def test_stage1_requires_registered_parent_for_arm():
    with pytest.raises(ValueError, match="Geo parent"):
        validate_parent("geo", evq_parent_metadata())

def test_each_arm_builds_zero_step_optimizer():
    optimizer = build_optimizer(tiny_lora(), continuation_spec())
    assert optimizer.state == {}
```

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m pytest tests/test_lora_industrial_training.py -q
```

- [ ] **Step 4: Implement the shifted answer-only loss**

```python
def answer_only_cross_entropy(logits, labels):
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    if not bool(mask.any()):
        raise ValueError("batch has no supervised answer tokens")
    return F.cross_entropy(shift_logits[mask], shift_labels[mask], reduction="mean")
```

For the real 128K-vocabulary model, request only tail logits covering `answer_start - 1` through `answer_end - 1`; fail if installed Transformers lacks the registered `logits_to_keep` support. Verify the tail-logit result against full logits on a tiny CPU model.

- [ ] **Step 5: Implement immutable Stage 1 parent continuation**

Validate parent status, step, seed, model hash, LongAlpaca hash, LoRA config, and saved frequency. Load the parent with `is_trainable=True`, freeze every non-LoRA parameter, inject its exact saved frequency, and build a new fused AdamW with no loaded state. Use LR `2e-5`, four warmup steps, cosine decay, weight decay `0.01`, gradient clipping `1.0`, and exactly 32 optimizer steps.

The run protocol records arm, parent receipt, data/row hashes, `(microbatch, accumulation)`, effective batch, physical/supervised tokens, precision, attention backend, checkpointing, compile configuration, package versions, and code hash. It is written before CUDA training and refuses mutation on resume.

- [ ] **Step 6: Implement pre-result performance locking**

Expose a `benchmark` command that runs exactly two warmup and two measured non-claim steps for `(1,4)`, `(2,2)`, and `(4,1)` using the same frozen rows. Every candidate starts from cloned parent-adapter bytes and its updated clone is deleted after measurement. Require finite loss, identical supervised token count, no OOM, and at least 5% free device memory. Select the fastest eligible layout before task evaluation, write `execution_choice.json`, and require both arms to consume the same file.

Run eager/compiled parity on one update from identical cloned adapter bytes. Accept compile only when loss absolute difference is at most `1e-4` and maximum adapter-parameter update difference is at most `5e-4`; otherwise lock eager. Store the persistent Inductor cache path in metadata without treating it as a scientific factor.

- [ ] **Step 7: Implement task-weighted diagnostics**

Record per-layer, per-head, per-half-split-rotary-pair q/k activation energy, gradient energy, and effective LoRA update energy. Summarize each distribution by normalized mass, entropy effective rank, Gini concentration, and top-quartile mass. Do not emit `gradient > 0` as coverage.

- [ ] **Step 8: Save only segment-boundary artifacts**

At step 32, atomically save the adapter, tokenizer receipt, exact frequency artifact, optimizer-independent run metadata, training log, and mechanism diagnostics. Reject any global step other than 32. A second segment is a new output directory with a new zero-state optimizer and an explicit parent checkpoint.

- [ ] **Step 9: Run focused and regression tests**

```bash
.venv/bin/python -m pytest \
  tests/test_lora_industrial_training.py \
  tests/test_frequency_adaptation_8b.py \
  tests/test_legacy_lora_multiseed.py -q
.venv/bin/python -m py_compile experiments/lora_evq_v2/industrial_capability/train.py
```

- [ ] **Step 10: Commit Task 5**

```bash
git add experiments/lora_evq_v2/industrial_capability/train.py tests/test_lora_industrial_training.py
git commit -m "feat: train matched 8k lora continuations"
```

---

### Task 6: Parent/post transfer analysis and same-base attribution

**Files:**
- Modify: `experiments/lora_evq_v2/industrial_capability/protocol.py`
- Modify: `experiments/lora_evq_v2/industrial_capability/train.py`
- Modify: `experiments/lora_evq_v2/industrial_capability/evaluate.py`
- Modify: `experiments/lora_evq_v2/industrial_capability/summarize.py`
- Create: `tests/test_lora_industrial_transfer.py`

**Interfaces:**
- Produces: `transfer_difference_in_differences()`, `stage1_learning_gate()`, `stage1_transfer_gate()`, `create_common_adapter_init()`, and three registered Stage 2 arms.

- [ ] **Step 1: Write failing tests for the parent/post estimand**

```python
def test_difference_in_differences_uses_each_arms_parent():
    value = transfer_difference_in_differences(
        evq_parent=.20, evq_post=.60, geo_parent=.30, geo_post=.50
    )
    assert value == pytest.approx(.20)

def test_learning_gate_requires_both_arms_to_learn_8k():
    gate = stage1_learning_gate(evq=learning_summary(.85, .80), geo=learning_summary(.75, .80))
    assert gate["status"] == "stop"
    assert "geo_pair_consistency" in gate["failed_checks"]
```

- [ ] **Step 2: Write failing tests for same-base initialization identity**

```python
def test_stage2_arms_share_initial_adapter_hash(tmp_path):
    receipt = create_common_adapter_init(tiny_model(), tmp_path, seed=42)
    copies = materialize_stage2_arms(receipt, tmp_path / "arms")
    assert len({row["initial_adapter_sha256"] for row in copies.values()}) == 1
    assert tuple(copies) == ("native_geo", "midpoint_geo", "midpoint_evq")
```

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m pytest tests/test_lora_industrial_transfer.py -q
```

- [ ] **Step 4: Implement Stage 1 gates and DID output**

Both post arms must reach 8K pair consistency `>=0.80`, source-removal positive fraction `>=0.75`, finite non-empty task cells, temporal delta NLL `<=0.10`, and short-core accuracy drop `<=2.0` points. Only then compute parent/post endpoint and DID at 16K/32K using the same frozen semantic groups and the Task 4 clustered bootstrap.

If both arms already reach 0.80 pair consistency at step 32, do not run a second segment. Otherwise both must have pair consistency in `[0.50,0.80)`, improve pair consistency by at least 0.10 and canonical answer NLL by at least 0.15 nats/token relative to their own parent, reach source-removal positive fraction at least 0.60, and pass the 8K temporal guardrail. Only then may both arms receive the same second 32-step segment; one-arm extension is forbidden.

Status is `positive` only when endpoint and DID lower bounds exceed zero, the 16K/32K mean effect is at least 5 points, and two task families agree. A valid failure is `negative`; one-arm 8K learning is `stop_in_range_mismatch`.

- [ ] **Step 5: Implement common Stage 2 adapter initialization**

Set seed 42, attach rank-64 q/k/v/o LoRA to the manifested base, assert all LoRA-B matrices are zero, save one immutable `common_init` adapter, and hash it. Materialize three run protocols that all reference this same hash and differ only in frequency identity: native endpoint Geo, midpoint Geo, and midpoint EVQ `tau=1.414`.

Train each arm directly at its final fixed frequency using the Task 5 trainer and identical Task 2 rows. Do not introduce homotopy. Require all three arms to share one locked execution-choice file and row-order hash.

Run 32 steps for all three arms. A second 32-step segment is allowed only when all three independently satisfy the exact extension rule above; extend all three or none.

- [ ] **Step 6: Implement the Stage 2 shape gate**

Primary attribution is midpoint EVQ minus midpoint Geo. Native endpoint Geo is reported separately. Require both midpoint arms to pass the 8K learning gate before extrapolation. A positive single-seed shape result needs the same endpoint/CI/5-point/two-task rule as Stage 1. Label the output `single_seed_supporting_shape_attribution`.

- [ ] **Step 7: Run focused and all LoRA capability tests**

```bash
.venv/bin/python -m pytest tests/test_lora_industrial_*.py -q
.venv/bin/python -m py_compile experiments/lora_evq_v2/industrial_capability/*.py
```

- [ ] **Step 8: Commit Task 6**

```bash
git add experiments/lora_evq_v2/industrial_capability tests/test_lora_industrial_transfer.py
git commit -m "feat: analyze lora capability transfer"
```

---

### Task 7: Explicit non-advancing launcher and paid-GPU preflight

**Files:**
- Create: `scripts/2026-07/14_lora_industrial_capability.sh`
- Create: `tests/test_lora_industrial_launcher.py`

**Interfaces:**
- Consumes: all package CLIs plus explicit environment variables.
- Produces: shell modes `prepare`, `validate`, `benchmark`, `stage0-canary`, `stage0-core`, `stage0-mechanism`, `stage1-geo`, `stage1-evq`, `stage1-eval`, `stage2-native`, `stage2-midpoint`, `stage2-evq`, and `stage2-eval`.

- [ ] **Step 1: Write failing launcher tests**

```python
def test_gpu_modes_never_download_or_tokenize():
    script = SCRIPT.read_text()
    gpu_body = extract_case_arms(script, prefix="stage")
    assert "git clone" not in gpu_body
    assert "load_dataset" not in gpu_body
    assert "prepare_bundle" not in gpu_body

def test_launcher_does_not_auto_advance_after_gate():
    script = SCRIPT.read_text()
    assert "AUTO_ADVANCE" not in script
    assert "next_authorized_command" in script
```

- [ ] **Step 2: Implement environment and output checks**

Require explicit variables for Python, model, model manifest, LongAlpaca manifest, Geo adapter, EVQ adapter, filler data, capability data root, work root, compile cache, and GPU lock. `prepare` is CPU-only. Every GPU mode verifies CUDA, BF16 support, manifests, adapters, frozen data, a non-existing output, and the absence of `.incomplete` files before model loading.

- [ ] **Step 3: Implement explicit modes**

Each mode runs exactly one package command. After evaluation, call the summarizer and print `gate.json` plus a literal `next_authorized_command`; never execute that command. `stage1-geo` and `stage1-evq` require the same `execution_choice.json`. Stage 2 modes require a positive Stage 1 gate file and the common-init receipt.

- [ ] **Step 4: Add shell and dry-run validation**

```bash
bash -n scripts/2026-07/14_lora_industrial_capability.sh
.venv/bin/python -m pytest tests/test_lora_industrial_launcher.py -q
bash scripts/2026-07/14_lora_industrial_capability.sh validate
```

Expected: shell syntax and tests pass; `validate` confirms CPU artifacts and stops before CUDA.

- [ ] **Step 5: Commit Task 7**

```bash
git add scripts/2026-07/14_lora_industrial_capability.sh tests/test_lora_industrial_launcher.py
git commit -m "chore: gate lora capability execution"
```

---

### Task 8: Reviewer-facing registration and final verification

**Files:**
- Modify: `rebuttal/README.md`
- Modify: `docs/overview/REPRODUCE.md`
- Modify: `ai-handoff.md`

**Interfaces:**
- Consumes: completed Tasks 1-7.
- Produces: one authoritative path to the design, implementation plan, commands, evidence boundary, and current no-result status.

- [ ] **Step 1: Register the new path without changing evidence tier**

Add one README row naming the design and plan. State `implementation-ready / no result` until a valid artifact exists. Keep the existing temporal NLL as single-seed supporting evidence and the old YaRN evaluator as a separate range-scaling diagnostic.

- [ ] **Step 2: Document CPU and GPU commands**

Add exact `prepare`, `validate`, and explicit stage commands to `REPRODUCE.md`. Mark Stage 1 as conditional on a valid non-positive Stage 0 and Stage 2 as conditional on positive Stage 1. Do not include private server paths or hosts.

- [ ] **Step 3: Run the complete local gate**

```bash
.venv/bin/python -m pytest \
  tests/test_lora_industrial_*.py \
  tests/test_seed42_capability_data.py \
  tests/test_official_yarn_capability_eval.py \
  tests/test_temporal_three_arm_eval.py \
  tests/test_frequency_adaptation_8b.py \
  tests/test_evq_seed42_retrieval_repair.py \
  tests/test_legacy_lora_multiseed.py -q
.venv/bin/python -m py_compile \
  experiments/lora_evq_v2/industrial_capability/*.py
bash -n scripts/2026-07/14_lora_industrial_capability.sh
git diff --check
```

Expected: all tests, compilation, shell syntax, and whitespace checks pass.

- [ ] **Step 4: Run scientific and leak audits**

```bash
rg -n "1-r/K|r.?K.*phase|48 frozen|globally optimal|official YaRN" \
  experiments/lora_evq_v2/industrial_capability \
  docs/superpowers/specs/2026-07-14-lora-industrial-capability-design.md \
  docs/superpowers/plans/2026-07-14-lora-industrial-capability.md
deny_a='(mis''aya|hej''az|ssh''pass|seeta'
deny_b='cloud|/Use''rs/|/root/auto''dl-tmp|wandb''\.ai|BEGIN .*PRI''VATE|OPENAI_API_''KEY|HF_''TOKEN|GITHUB_''TOKEN|api[_-]?k''ey|pass''word|sec''ret)'
git diff --cached -U0 | rg -n "^\+.*${deny_a}${deny_b}" || true
```

Expected: the scientific scan only finds explicit rejection/boundary text; the staged leak scan finds no real private path, host, identity, credential, or credential material.

- [ ] **Step 5: Commit the completed implementation**

```bash
git add \
  experiments/lora_evq_v2/industrial_capability \
  scripts/2026-07/14_lora_industrial_capability.sh \
  tests/test_lora_industrial_*.py \
  rebuttal/README.md docs/overview/REPRODUCE.md ai-handoff.md
git diff --cached --stat
git diff --cached --check
git commit -m "feat: prepare lora capability conversion"
```

## Execution Order and Stop Conditions

1. Implement and CPU-validate Tasks 1-4.
2. Run Stage 0 canary. If invalid, fix the evaluator/data and repeat only the canary.
3. Run Stage 0 core. If positive, run the mechanism subset and external guardrails; do not train Stage 1.
4. If Stage 0 is valid but non-positive, implement/validate Task 5 and run both matched Stage 1 arms.
5. If either Stage 1 arm fails the 8K learning gate, stop.
6. If Stage 1 is transfer-positive, implement/run the seed-42 Stage 2 three-arm attribution.
7. Never start extra seeds, YaRN evaluation, 16K training, or a method redesign without a new explicit decision.

## Final Deliverables

- Reproducible schedule diagnostic JSON, not an untracked table.
- Frozen controlled/external capability manifest with canonical targets.
- Raw seven-arm Stage 0 per-example outputs and deterministic gate.
- If triggered, matched Stage 1 parent/post outputs with DID and 8K guardrails.
- If triggered, same-base native/midpoint/EVQ Stage 2 attribution.
- Raw negative and invalid artifacts retained with explicit status.
- No paper-number or claim-tier changes without a separate evidence audit.
