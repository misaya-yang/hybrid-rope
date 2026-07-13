# Native-RoPE vs Endpoint-EVQ 150M Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare and upload a cost-safe, paired 151.9M native-RoPE versus endpoint-EVQ training and evaluation pipeline using the existing 500M FineWeb-Edu cache.

**Architecture:** A pure protocol module binds the model, schedules, hashes, and exact data counts. CPU preparation creates an independent validation tensor and a frozen legacy-format Passkey substitution cache. Training uses a self-contained, tied-embedding decoder and a fast fixed-shape compiled loss loop. Evaluation applies raw or pinned YaRN operators and reports natural-text NLL/PPL plus teacher-forced Passkey NLL gaps.

**Tech Stack:** Python 3.12, PyTorch 2.8, NumPy memmaps, Transformers tokenizer, shell, pytest; no new dependency.

## Global Constraints

- Do not use the legacy validation hash `85bfe9af77642d5e8995283e12645544c50b6233a33931fa2b1b37ec122e1b7e`.
- Keep model, seed, initialization, data order, optimizer, precision, and step count matched across arms.
- Only the endpoint frequency tensor differs between arms.
- Use `tau=1.5`, `base=500000`, `L_train=2048`, global batch 60 as
  micro-batch 12 x accumulation 5, and 4,069 optimizer steps.
- Use the old deterministic Passkey selector and marker schema at target ratio 0.02, preserving the historical approximately 10M-token absolute Passkey budget; never use validation filler for training.
- Label native scaling official YaRN and EVQ scaling YaRN-derived.
- Complete downloads, tokenization, hashes, tests, and dry runs before GPU launch.
- Do not modify paper numbers or report a result before both training and evaluation artifacts exist.

---

### Task 1: Protocol and frequency identities

**Files:**
- Create: `experiments/native_rope_evq_150m/__init__.py`
- Create: `experiments/native_rope_evq_150m/protocol.py`
- Create: `tests/test_native_rope_evq_150m.py`

**Interfaces:**
- Produces `ExperimentSpec`, `get_arm_inv_freq(arm)`, `legacy_passkey_indices(n_rows)`, `estimate_parameter_count()`, and `validate_training_manifest(manifest)`.

- [ ] Write tests asserting 151,898,880 parameters, 4,069 steps, 4,926 selected Passkey rows, endpoint native frequencies, and finite distinct EVQ frequencies.
- [ ] Run `python -m pytest tests/test_native_rope_evq_150m.py -q`; verify import failure because the package does not exist.
- [ ] Implement the immutable dataclass and schedule helpers using `geometric_inv_freq(..., midpoint=False)` semantics and `evq_cosh_inv_freq(..., midpoint=False)`.
- [ ] Run the focused tests and verify they pass.

### Task 2: CPU data preparation

**Files:**
- Create: `experiments/native_rope_evq_150m/prepare_data.py`
- Modify: `tests/test_native_rope_evq_150m.py`

**Interfaces:**
- Consumes the 500M `.npy`, its manifest, the local GPT-NeoX tokenizer, and FineWeb-Edu shard 004.
- Produces `passkey_train_2pct.npy`, `passkey_indices.npy`, `val_fineweb-edu_shard004_5000000.npy`, and `data_manifest.json`.

- [ ] Add failing tests for legacy selector parity, same-row training filler, atomic memmap output, forbidden legacy validation hash, and manifest rejection on source/hash mismatch.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement resumable mirror download, low-memory parquet tokenization, deterministic Passkey compilation, SHA-256 manifests, and validation guards.
- [ ] Run the tests and a tiny synthetic `prepare_data.py --self_test` end-to-end gate.

### Task 3: Matched compiled training

**Files:**
- Create: `experiments/native_rope_evq_150m/train.py`
- Modify: `tests/test_native_rope_evq_150m.py`

**Interfaces:**
- Consumes the validated data manifest and one arm name.
- Produces `<work_dir>/<arm>/model.pt`, `inv_freq.npy`, `train_meta.json`, and `train_log.jsonl`.

- [ ] Add failing tests that both arms generate equal trainable initialization hashes, share the same shuffled row order, refuse CPU/non-BF16 launch mode, and reject existing output directories.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement a memmap substitution Dataset, seed-isolated DataLoader, fixed-shape loss module, BF16 autocast, fused AdamW, cosine schedule, `torch.compile(dynamic=False)`, atomic checkpoint saving, and periodic throughput/memory logging.
- [ ] Run tests, `py_compile`, and `train.py --dry_run` against synthetic data.

### Task 4: Four-condition evaluation and launcher

**Files:**
- Create: `experiments/native_rope_evq_150m/evaluate.py`
- Create: `experiments/native_rope_evq_150m/run_seed42.sh`
- Modify: `tests/test_native_rope_evq_150m.py`

**Interfaces:**
- Consumes two checkpoints plus the validated held-out tensor.
- Produces `evaluation/raw_results.json`, `evaluation/summary.json`, and a single launch log.

- [ ] Add failing tests for target-matched factors, native official-YaRN identity, EVQ YaRN-derived identity, shared validation offsets, and NLL-to-PPL aggregation.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement raw/scaled operator injection, natural-text NLL/PPL, batched teacher-forced Passkey NLL gap, raw per-case retention, and official-source parity fail-closed behavior.
- [ ] Implement shell modes `prepare`, `preflight`, and `run`; `run` trains native then EVQ and evaluates automatically without downloading.
- [ ] Run `bash -n`, `py_compile`, focused pytest, official YaRN parity tests, and CPU preflight.
- [ ] Upload only the verified package/tests/docs to the server, run server-side tests and preflight, and record exact launch command without starting CUDA training.

## Final verification

Run on the server CPU environment:

```bash
/root/miniconda3/bin/python -m pytest \
  tests/test_native_rope_evq_150m.py \
  tests/test_official_yarn_parity.py -q
/root/miniconda3/bin/python -m py_compile \
  experiments/native_rope_evq_150m/*.py
bash -n experiments/native_rope_evq_150m/run_seed42.sh
bash experiments/native_rope_evq_150m/run_seed42.sh preflight
```

Expected: all tests pass, compilation and shell syntax exit 0, preflight
confirms CPU-only mode plus all immutable hashes, and no training process is
started.
