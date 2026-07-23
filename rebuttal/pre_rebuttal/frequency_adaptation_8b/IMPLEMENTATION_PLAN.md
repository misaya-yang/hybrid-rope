# LLaMA-3-8B Frequency Adaptation Implementation Plan

> Scope: add an isolated rebuttal research experiment. Do not modify paper text, historical experiment configs, or reported values.

**Goal:** Build a matched Geo/EVQ curriculum that tests whether task-directed, answer-only gradients can adapt a pretrained LLaMA-3-8B checkpoint to a changed RoPE frequency allocation.

**Architecture:** Reuse the canonical schedule and RoPE injection APIs. Prepare exact-length token tensors from the already frozen plain-text FineWeb-Edu packs. Train one common native-Geo LoRA warm-up, then branch into matched Geo and EVQ arms. EVQ moves along a smooth log-frequency path before fixed-endpoint 8K/16K phases. Evaluate only held-out counterfactual retrieval first; reuse existing PPL/RULER evaluators only after the capability gate.

**Stack:** Python, PyTorch, Transformers Trainer, PEFT, existing EVQ schedule helpers, pytest.

## Task 1: Correct rebuttal policy

Files:

- Modify: `rebuttal/pre_rebuttal/rebuttal_playbook.md`
- Modify: `rebuttal/README.md`

Actions:

1. Remove categorical `no-new-experiments` and blanket `no 8B/LoRA` rules.
2. Preserve reviewer-trigger, provenance, evidence-tier, and no-baseline-zoo gates.
3. Register this experiment as internal preparation that does not automatically enter the response.

Verification:

```bash
rg -n "no-new-experiments|不补充任何新实验|不启动 7B/8B实验|新 LoRA、video" rebuttal/pre_rebuttal/rebuttal_playbook.md rebuttal/README.md
```

Expected: no categorical ban remains.

## Task 2: Lock pure experiment contracts with tests

Files:

- Create: `tests/test_frequency_adaptation_8b.py`
- Create: `rebuttal/pre_rebuttal/frequency_adaptation_8b/__init__.py`
- Create: `rebuttal/pre_rebuttal/frequency_adaptation_8b/curriculum.py`

Test-first contracts:

1. Every default phase uses 32,768 physical sequence tokens per optimizer step.
2. Log-frequency homotopy returns the exact native tensor at progress 0 and exact EVQ tensor at progress 1, remains positive, and gives a geometric midpoint at progress 0.5.
3. Invalid/non-positive/mismatched frequency tensors fail closed.
4. Answer-only labels mask every prompt token and supervise exactly the answer span.
5. Rotary pair aggregation follows LLaMA half-split `(i, i + head_dim/2)` pairing.
6. Exact-distance examples have fixed sequence length and recorded source-query distance.
7. Counterfactual triplets keep length/positions fixed while changing only the intended source/target spans.

RED command:

```bash
python -m pytest tests/test_frequency_adaptation_8b.py -q
```

Expected before implementation: assertion failure because the experiment module does not exist.

GREEN command: same command, all tests pass.

## Task 3: Prepare frozen curriculum data

Files:

- Create: `rebuttal/pre_rebuttal/frequency_adaptation_8b/prepare_data.py`

Actions:

1. Validate the existing plain-text manifest and tensor hashes.
2. Keep train and validation filler tensors separate.
3. Derive deterministic, disjoint train/eval token-native nonce pools.
4. Build exact-length W/H/E8/E16 train tensors with exact distance metadata.
5. Build held-out original/swapped/source-removed triplets.
6. Save compact `input_ids`, `answer_start`, `answer_end`, metadata, protocol, and hashes; do not materialize full label tensors.

Verification:

```bash
python -m py_compile rebuttal/pre_rebuttal/frequency_adaptation_8b/prepare_data.py
python -m pytest tests/test_frequency_adaptation_8b.py -q
```

## Task 4: Implement phased LoRA training

Files:

- Create: `rebuttal/pre_rebuttal/frequency_adaptation_8b/train.py`

Actions:

1. Load the exact model/config and capture native `inv_freq` before any modification.
2. Compute canonical EVQ-Cosh midpoint frequencies with fixed `tau=1.414`.
3. Create r64 q/k/v/o LoRA for W, or load the specified prior-phase adapter as trainable.
4. Validate the prior adapter/frequency artifact and immutable phase protocol.
5. Inject fixed endpoint frequencies or install the per-step smooth log-frequency callback.
6. Generate answer-only labels on dataset access.
7. Register q/k LoRA-B gradient hooks and save half-split per-pair gradient/update energy.
8. Save adapter, tokenizer, exact frequency artifact, run protocol, trainer state, and diagnostics.

Verification:

```bash
python -m py_compile rebuttal/pre_rebuttal/frequency_adaptation_8b/train.py
python -m rebuttal.pre_rebuttal.frequency_adaptation_8b.train --help
```

## Task 5: Implement capability evaluation and gated launcher

Files:

- Create: `rebuttal/pre_rebuttal/frequency_adaptation_8b/evaluate.py`
- Create: `rebuttal/pre_rebuttal/frequency_adaptation_8b/run_seed42.sh`

Actions:

1. Score answer-token NLL and teacher-forced exact token predictions.
2. Aggregate original/swapped pair consistency and paired source-removal NLL deltas by task/distance.
3. Refuse adapter evaluation when its frequency artifact does not match the requested arm/phase.
4. Provide explicit `prepare`, `warmup`, `branch-geo`, `branch-evq`, and `eval` commands; do not auto-run past gates.

Verification:

```bash
python -m py_compile rebuttal/pre_rebuttal/frequency_adaptation_8b/evaluate.py
bash -n rebuttal/pre_rebuttal/frequency_adaptation_8b/run_seed42.sh
python -m rebuttal.pre_rebuttal.frequency_adaptation_8b.evaluate --help
```

## Task 6: Final repository gates

```bash
python -m pytest tests/test_frequency_adaptation_8b.py -q
python -m py_compile \
  rebuttal/pre_rebuttal/frequency_adaptation_8b/curriculum.py \
  rebuttal/pre_rebuttal/frequency_adaptation_8b/prepare_data.py \
  rebuttal/pre_rebuttal/frequency_adaptation_8b/train.py \
  rebuttal/pre_rebuttal/frequency_adaptation_8b/evaluate.py
bash -n rebuttal/pre_rebuttal/frequency_adaptation_8b/run_seed42.sh
git diff --check
git status --short
```

Review the final diff for private paths, result fabrication, changes to paper numbers, accidental edits outside rebuttal/tests, and any claim that rank or tau is theoretically optimal.
