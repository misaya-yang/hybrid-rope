# LLaMA-3-8B Positional Distillation Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare, test, document, and publish the seed-42 q/k-only positional-distillation pilot without launching a GPU experiment.

**Architecture:** Freeze a deterministic LLaMA-tokenized FineWeb-Edu subset, then use one PEFT model as both native-Geo teacher and Geo/EVQ student by switching frequency artifacts around a no-grad teacher pass. Train only q/k LoRA against a bucket-normalized final-hidden-state loss and evaluate the four causal arms with explicit frequency provenance.

**Tech Stack:** Python 3, PyTorch, Hugging Face Transformers/Datasets/Trainer, PEFT, pytest, Bash, Git.

## Global Constraints

- Do not run a GPU experiment in this implementation pass.
- Keep all experimental numbers unchanged.
- Use seed 42, 8,192 tokens, 300 steps, rank 64, alpha 128, q/k-only targets, dropout 0.0, and learning rate 2e-5.
- Reject instruction/chat JSONL as distillation data; local fallback rows must contain plain `text`.
- Every trained/evaluated schedule must be backed by a validated frequency artifact.
- Runtime outputs remain outside the repository and must not be committed.

---

### Task 1: Freeze the approved protocol

**Files:**
- Create: `docs/superpowers/specs/2026-07-10-llama8b-positional-distillation-design.md`
- Create: `docs/superpowers/plans/2026-07-10-llama8b-positional-distillation-pilot.md`

**Interfaces:**
- Consumes: the approved A-stage design.
- Produces: exact scientific scope, defaults, decision gates, and runtime boundaries for all later tasks.

- [ ] **Step 1: Record the four arms and non-goals**

Write the Base-Geo, Base-EVQ, Geo-Distill, and EVQ-Distill arms exactly as
specified in the design. State that no GPU command is executed during code
preparation.

- [ ] **Step 2: Verify there are no placeholders**

Run:

```bash
rg -n 'TBD|TODO|FILL|PLACEHOLDER' \
  docs/superpowers/specs/2026-07-10-llama8b-positional-distillation-design.md \
  docs/superpowers/plans/2026-07-10-llama8b-positional-distillation-pilot.md
```

Expected: no output and exit status 1 from `rg`.

### Task 2: Add deterministic frozen-data preparation

**Files:**
- Create: `experiments/lora_evq_v2/prepare_positional_distill_data.py`
- Create: `tests/test_positional_distill.py`

**Interfaces:**
- Consumes: raw text iterables and a tokenizer exposing `__call__(text, add_special_tokens=False)`.
- Produces: `train.pt`, `validation.pt`, and `manifest.json`; public helpers `iter_plain_text_jsonl`, `pack_token_sequences`, and `sha256_file`.

- [ ] **Step 1: Write failing tests for plain-text enforcement and packing**

Add tests that require:

```python
rows = list(iter_plain_text_jsonl(path))
assert rows == ["alpha", "beta"]

packed, remainder = pack_token_sequences([[1, 2, 3], [4, 5, 6]], seq_len=4)
assert packed.tolist() == [[1, 2, 3, 4]]
assert remainder == [5, 6]
```

Also require a `ValueError` when a JSONL row contains `messages` without a
non-empty `text` field.

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```bash
python -m pytest tests/test_positional_distill.py -q
```

Expected: import failure because `prepare_positional_distill_data.py` does not
exist.

- [ ] **Step 3: Implement deterministic packing and manifests**

Implement CLI defaults:

```text
dataset=HuggingFaceFW/fineweb-edu
dataset_config=sample-10BT
split=train
text_field=text
seq_len=8192
train_sequences=2400
validation_sequences=128
seed=42
shuffle_buffer=10000
```

Save tensors as `torch.int32`, compute SHA-256 for both tensor files, and write
an atomic manifest only after sequence counts and hashes validate.

- [ ] **Step 4: Run the focused tests and observe GREEN**

Run:

```bash
python -m pytest tests/test_positional_distill.py -q
```

Expected: packing and JSONL tests pass.

### Task 3: Add q/k-only positional distillation

**Files:**
- Create: `experiments/lora_evq_v2/train_positional_distill.py`
- Modify: `tests/test_positional_distill.py`

**Interfaces:**
- Consumes: frozen `train.pt`, `manifest.json`, base model path, and `student_method` in `{native_geo, evq_cosh}`.
- Produces: PEFT adapter, tokenizer files, `custom_inv_freq.pt`, `experiment_meta.json`, and `trainer_state.json`.
- Public helpers: `position_bucket_ranges`, `normalized_bucket_hidden_mse`, `validate_distill_manifest`, and `build_distill_metadata`.

- [ ] **Step 1: Write failing loss and manifest tests**

Require zero loss for identical hidden states, equal weighting across non-empty
position buckets, rejection of non-8192 manifests, and metadata containing
exactly `q_proj,k_proj`, seed 42, teacher `native_geo`, and the frozen manifest
hash.

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```bash
python -m pytest tests/test_positional_distill.py -q
```

Expected: imports from `train_positional_distill` fail.

- [ ] **Step 3: Implement the minimal training path**

Use one PEFT model. Inside `Trainer.compute_loss`, install Geo and disable the
adapter for the no-grad teacher backbone pass, then install the student
schedule and run the adapter-enabled student pass. Return the mean normalized
MSE across `[0,2048)`, `[2048,4096)`, and `[4096,8192)`.

The parser defaults must be:

```text
student_method=evq_cosh
tau=1.414
lora_r=64
lora_alpha=128
lora_dropout=0.0
lora_targets=q_proj,k_proj
max_steps=300
per_device_batch_size=2
gradient_accumulation_steps=4
learning_rate=2e-5
warmup_steps=30
seed=42
```

- [ ] **Step 4: Run tests and static compilation**

Run:

```bash
python -m pytest tests/test_positional_distill.py -q
python -m py_compile \
  experiments/lora_evq_v2/prepare_positional_distill_data.py \
  experiments/lora_evq_v2/train_positional_distill.py
```

Expected: all focused tests pass and compilation exits 0.

### Task 4: Add four-arm evaluation and launch orchestration

**Files:**
- Create: `experiments/lora_evq_v2/eval_positional_distill.py`
- Create: `experiments/lora_evq_v2/summarize_positional_distill.py`
- Create: `scripts/2026-07/01_lora_positional_distill_seed42.sh`
- Modify: `tests/test_positional_distill.py`

**Interfaces:**
- Consumes: optional adapter, explicit candidate method, frozen validation data, and WikiText file.
- Produces: one JSON per arm with PPL@8K/16K/32K, hidden error, schedule provenance, and runtime metadata, plus `positional_distill_summary.json` with fixed pass/fail gates.

- [ ] **Step 1: Write failing variant and artifact tests**

Require stable filenames for all four variants, mandatory adapter frequency
artifacts, rejection of a method mismatch, and a recovery calculation:

```python
assert representation_recovery(injection_error=1.0, adapted_error=0.1) == 0.9
```

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```bash
python -m pytest tests/test_positional_distill.py -q
```

Expected: evaluation helper imports fail.

- [ ] **Step 3: Implement evaluation and a non-executing launcher**

The launcher accepts `prepare`, `train`, `eval`, and `all`. `all` composes the
three explicit phases but is never invoked during this implementation pass.
It records `nvidia-smi` identity/power/memory output before training and writes
all runtime artifacts under `$EVQ_POSITIONAL_DISTILL_DIR` (or the excluded
`experiments/lora_evq_v2/local/` default) by default.

- [ ] **Step 4: Verify shell and Python syntax**

Run:

```bash
bash -n scripts/2026-07/01_lora_positional_distill_seed42.sh
python -m py_compile experiments/lora_evq_v2/eval_positional_distill.py
python -m py_compile experiments/lora_evq_v2/summarize_positional_distill.py
python -m pytest tests/test_positional_distill.py -q
```

Expected: syntax checks exit 0 and focused tests pass.

### Task 5: Document the handoff and publish the branch

**Files:**
- Modify: `experiments/lora_evq_v2/README.md`
- Modify: `rebuttal/MINIMAL_EXPERIMENT_RUNBOOK.md`

**Interfaces:**
- Consumes: the implemented commands and decision gates.
- Produces: a reviewer-safe operator handoff that clearly says the experiment has not run.

- [ ] **Step 1: Add preparation, training, evaluation, and stop commands**

Document:

```bash
bash scripts/2026-07/01_lora_positional_distill_seed42.sh prepare
bash scripts/2026-07/01_lora_positional_distill_seed42.sh train
bash scripts/2026-07/01_lora_positional_distill_seed42.sh eval
```

State that only future verified JSON outputs may populate rebuttal numbers.

- [ ] **Step 2: Run final verification**

Run:

```bash
python -m pytest tests/test_positional_distill.py tests/test_rebuttal_protocol_regressions.py -q
python -m py_compile \
  experiments/lora_evq_v2/prepare_positional_distill_data.py \
  experiments/lora_evq_v2/train_positional_distill.py \
  experiments/lora_evq_v2/eval_positional_distill.py \
  experiments/lora_evq_v2/summarize_positional_distill.py
bash -n scripts/2026-07/01_lora_positional_distill_seed42.sh
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Stage only scoped files and scan for leaks**

Run:

```bash
git add \
  docs/superpowers/specs/2026-07-10-llama8b-positional-distillation-design.md \
  docs/superpowers/plans/2026-07-10-llama8b-positional-distillation-pilot.md \
  experiments/lora_evq_v2/prepare_positional_distill_data.py \
  experiments/lora_evq_v2/train_positional_distill.py \
  experiments/lora_evq_v2/eval_positional_distill.py \
  experiments/lora_evq_v2/summarize_positional_distill.py \
  experiments/lora_evq_v2/README.md \
  scripts/2026-07/01_lora_positional_distill_seed42.sh \
  tests/test_positional_distill.py \
  rebuttal/MINIMAL_EXPERIMENT_RUNBOOK.md
git diff --cached --check
git diff --cached -U0 > /tmp/evq-positional-distill-staged.patch
python - <<'PY'
from pathlib import Path

text = Path("/tmp/evq-positional-distill-staged.patch").read_text()
needles = [
    "/" + "Users/",
    "/" + "root/",
    "mi" + "saya",
    "he" + "jaz",
    "ssh" + "pass",
    "BE" + "GIN " + "PRI" + "VATE",
    "OPENAI_" + "API_KEY",
    "HF_" + "TOKEN",
    "GITHUB_" + "TOKEN",
    "pass" + "word=",
]
hits = [needle for needle in needles if needle.lower() in text.lower()]
if hits:
    raise SystemExit(f"staged leak scan found: {hits}")
PY
```

Expected: no whitespace errors and no real identity, credential, or private-host
leak.

- [ ] **Step 4: Commit and push the Codex branch**

Run:

```bash
git commit -m "add clean 8b positional distillation pilot"
git push -u origin codex/llama8b-positional-distill-pilot
```

Expected: the remote branch is updated without changing `main`.
