# Legacy LongAlign LoRA Multi-Seed Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development or superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Prepare a fail-closed, resumable, protocol-matched rerun of the
historical LLaMA-3-8B LongAlign LoRA experiment with fresh Geo/EVQ seeds
42/43/44 and two training-free baselines, without touching the running
positional-distillation experiment.

**Architecture:** Download and freeze one revision-pinned LongAlign-10k JSONL,
tokenize it once with the exact historical variable-length/full-token-loss
semantics, and bind all six training arms to the resulting manifest.  Reuse the
legacy trainer only through a strict protocol mode that adds recovery
checkpoints, automatic resume, immutable provenance, and final-artifact
validation.  Evaluate Base-Geo, Base-EVQ, and all six adapters on the same
frozen WikiText token offsets at 8K/16K/32K, then report paired seed statistics.

**Tech Stack:** Python 3.11, PyTorch 2.8, Transformers 4.57.6, PEFT 0.17.1,
Datasets 4.5, pytest, Bash, CUDA 12.8.

## Global Constraints

- Do not modify, pause, restart, or share output directories with the active
  positional-distillation run.
- Never reuse the historical `evq_r64_tau1414` seed-42 checkpoint in the new
  multi-seed aggregate.
- Train exactly six fresh arms: native Geo and EVQ-Cosh tau 1.414 for seeds
  42, 43, and 44.  YaRN is outside this fallback matrix.
- Preserve the historical science settings: q/k/v/o LoRA, r64, alpha128,
  dropout0.05, BF16, 300 steps, B2/GA4, LR 1e-4, warmup60, WD0.01, clip1,
  max length8192, first8000 accepted rows, split seed42, full-token causal loss.
- Call the result a protocol-matched rerun, not a bitwise reproduction, unless
  the original raw files and environment are recovered and hash-matched.
- Fail closed on unknown dataset identity, missing hashes, incomplete
  checkpoints, mixed protocol identities, or missing seed pairs.
- Runtime data, caches, checkpoints, logs, and results stay outside Git.

### Task 1: Define and test the frozen-data/protocol contract

**Files:**
- Create: `experiments/lora_evq_v2/legacy_lora_protocol.py`
- Create: `tests/test_legacy_lora_multiseed.py`

**Interfaces:**
- `sha256_file(path)`
- `validate_source_receipt(receipt)`
- `validate_legacy_protocol(protocol)`
- `legacy_run_name(method, seed)`
- `validate_complete_matrix(records)`

- [ ] Write failing tests for pinned source identity, six-arm names, exact
  scientific settings, 299-step rejection, q/k-only rejection, and incomplete
  matrix rejection.
- [ ] Run `python -m pytest tests/test_legacy_lora_multiseed.py -q` and observe
  import failures.
- [ ] Implement the pure validation helpers without importing Transformers.
- [ ] Re-run the focused tests and observe green.

### Task 2: Prepare one shared verified LongAlign artifact

**Files:**
- Create: `experiments/lora_evq_v2/prepare_legacy_longalign_data.py`
- Modify: `tests/test_legacy_lora_multiseed.py`

**Interfaces:**
- Consumes a local JSONL plus explicit source id, revision, split, URL, and
  expected SHA-256.
- Produces `train.pt`, `validation.pt`, and `manifest.json` atomically.
- Public helper `freeze_legacy_rows(rows, tokenizer, ...)` preserves the old
  first-8000, chat-template, truncation, minimum-length, fixed split behavior.

- [ ] Add failing tests for source-SHA mismatch, deterministic split, first-N
  selection before tokenization, and manifest hash binding.
- [ ] Implement the frozen-data CLI; prohibit network fallback and prohibit
  source ids other than the explicit receipt.
- [ ] Run focused tests and Python compilation.

### Task 3: Add strict recovery/resume to the legacy trainer

**Files:**
- Modify: `experiments/lora_evq_v2/train_evq_lora.py`
- Create: `experiments/lora_evq_v2/validate_legacy_lora_artifact.py`
- Modify: `tests/test_legacy_lora_multiseed.py`

**Interfaces:**
- New CLI: `--legacy_protocol_manifest`, `--resume_from_checkpoint auto|PATH`,
  `--compile`, and `--compile_mode`.
- Strict mode reads only the shared frozen tensors, saves every 100 steps,
  keeps two recovery checkpoints, records trainer state and hashes, and rejects
  any CLI value that differs from the manifest.
- Final validator requires global step 300, exact adapter config, exact
  frequency artifact, exact data/model/tokenizer/code identity, and adapter
  weight SHA.

- [ ] Add failing tests for immutable setting mismatch, resume mismatch,
  checkpoint-299 rejection, and exact q/k/v/o validation.
- [ ] Implement checkpointing, auto-resume, optional uniform compile, metadata,
  and the standalone validator.
- [ ] Run focused tests and static compilation.

### Task 4: Add matched baseline evaluation and paired summary

**Files:**
- Create: `experiments/lora_evq_v2/prepare_legacy_wikitext.py`
- Create: `experiments/lora_evq_v2/eval_legacy_lora_matched.py`
- Create: `experiments/lora_evq_v2/summarize_legacy_lora_matched.py`
- Modify: `tests/test_legacy_lora_multiseed.py`

**Interfaces:**
- Freeze WikiText-2 raw test at an explicit revision and store raw/token-id
  hashes plus five disjoint offsets for each of 8K/16K/32K.
- Evaluate `base_geo`, `base_evq_tau1414`, and six adapter variants using
  chunked LM-head NLL, distinct output filenames, and post-forward frequency
  verification.
- Summary requires all six adapters and reports raw seeds, mean, sample SD,
  range, and paired EVQ-minus-Geo deltas without significance claims.

- [ ] Add failing tests for Geo-adapter dispatch, Base-EVQ injection, unique
  filenames, mixed-eval-manifest rejection, and paired statistics.
- [ ] Implement preparation, evaluation, and summarization.
- [ ] Run focused tests and Python compilation.

### Task 5: Add the cost-gated operator launcher and documentation

**Files:**
- Create: `scripts/2026-07/03_lora_longalign_matched_multiseed.sh`
- Modify: `experiments/lora_evq_v2/README.md`

**Interfaces:**
- Phases: `preflight`, `prepare-data`, `baseline`, `seed42`, `remaining-seeds`,
  `eval`, and `summarize`.  There is deliberately no accidental all-expensive
  default.
- `set -Eeuo pipefail`; each arm validates before the next arm may start.
- Cost gate: Base-Geo provenance/eval first, then EVQ-s42, then Geo-s42; only
  an explicit `remaining-seeds` phase can launch seeds 43/44.

- [ ] Write shell assertions and document required environment variables.
- [ ] Run `bash -n` and verify that no private server path appears in tracked
  additions.

### Task 6: Final verification and publication

- [ ] Run focused tests, relevant existing LoRA tests, Python compilation, and
  shell syntax checks.
- [ ] Review `git diff --check`, final diff scope, and the staged secret/path
  scan.
- [ ] Commit the verified implementation on `main` and push `origin/main` as
  explicitly requested by the user.
