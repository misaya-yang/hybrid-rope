# EVQ-Cosh LoRA v2

This package is a supporting LoRA experiment for LLaMA-3-8B style checkpoints. It is not part of the primary NeurIPS claim tier.

Set local paths with environment variables instead of editing scripts:

```bash
export EVQ_LORA_BASE_DIR=/path/to/local/work
export EVQ_LORA_MODEL=$EVQ_LORA_BASE_DIR/models/Meta-Llama-3-8B-Instruct
export EVQ_LORA_TRAIN_DATA=$EVQ_LORA_BASE_DIR/data/longalign_10k/longalign_10k.jsonl
export EVQ_LORA_WIKITEXT=$EVQ_LORA_BASE_DIR/data/wikitext2/wikitext2_test.txt

python download_model_data.py --verify_only
bash run.sh dryrun
```

The default local cache is `experiments/lora_evq_v2/local/`, which is excluded from reviewer archives.

## Clean positional-distillation pilot (prepared, not run)

The seed-42 A-stage pilot separates frequency-injection shock from a cleaner
teacher-guided recovery protocol. It uses a frozen native-Geo view of
LLaMA-3-8B as teacher, an EVQ or Geo student, q/k-only LoRA (`r=64`,
`alpha=128`), and no token labels.
The loss matches final hidden states in equal-weight 0-2K, 2-4K, and 4-8K
position buckets.

The operator entrypoint is:

```bash
python --version  # the locked experiment stack requires Python 3.11+
export EVQ_LORA_BASE_DIR=/path/to/local/work
export EVQ_LORA_MODEL=$EVQ_LORA_BASE_DIR/models/Meta-Llama-3-8B-Instruct
export EVQ_LORA_WIKITEXT=$EVQ_LORA_BASE_DIR/data/wikitext2/wikitext2_test.txt
export EVQ_POSITIONAL_DISTILL_DIR=$EVQ_LORA_BASE_DIR/positional_distill_s42
export CUDA_VISIBLE_DEVICES=0

bash scripts/2026-07/01_lora_positional_distill_seed42.sh prepare
bash scripts/2026-07/01_lora_positional_distill_seed42.sh benchmark
bash scripts/2026-07/01_lora_positional_distill_seed42.sh train
bash scripts/2026-07/01_lora_positional_distill_seed42.sh eval
```

Use the repository lock with a PyTorch CUDA 12.8-or-newer build that contains
native `sm_120` kernels. The launcher fails closed on Python version, CUDA
architecture, BF16 support, single-GPU visibility, and a forced Flash-SDPA
smoke call before loading the claim run.

`prepare` freezes 2,400 train and 128 validation sequences from plain
FineWeb-Edu text at a pinned dataset revision, with a document-disjoint split.
If the server cannot stream Hugging Face data, set
`EVQ_POSITIONAL_TEXT_JSONL` to a local JSONL whose every row has a plain `text`
field. Chat/instruction rows are rejected.

`train` creates `evq_distill_s42` and a one-step `geo_distill_s42` null
checkpoint. The Geo arm is a pipeline sentinel, not a matched optimizer-drift
control: its teacher/student schedules are identical and the zero-output LoRA
starts at zero loss.
The implementation keeps one 8B model in memory, switches to native Geo with
the adapter disabled for the teacher pass, then switches to the student
schedule for backward. Defaults remain batch 2 and gradient accumulation 4 to
keep effective batch 8. The student backbone uses `torch.compile` by default,
the teacher remains eager, fused AdamW is used, recovery checkpoints are saved
every 100 EVQ steps, and `nvidia-smi` telemetry is sampled during training.
Every resume must match an immutable data/model/code/runtime/performance
protocol. A checkpoint becomes claim-ready only after its append-only
invocation ledger proves continuous step coverage and binds every invocation's
hardware record, training log, and GPU telemetry.

On the RTX PRO 6000, run `benchmark` before the claim run. It writes five
separate, immutable, non-claim 12-step probes (two warmup plus ten timed steps):
eager B2/GA4, compiled B2/GA4, compiled B4/GA2, compiled B8/GA1, and compiled
B2/GA4 without activation checkpointing. OOM/telemetry failures are excluded
from `benchmark_summary.json`; a candidate also needs at least 5% (minimum
4 GiB) VRAM headroom to be eligible. Probes never share the claim checkpoint
directory. Apply the fastest eligible configuration while keeping effective
batch 8. Disable compile or checkpointing only through the documented
environment flags:

```bash
export CUDA_VISIBLE_DEVICES=0
export EVQ_POSITIONAL_COMPILE=1
export EVQ_POSITIONAL_COMPILE_MODE=default
export EVQ_POSITIONAL_GRADIENT_CHECKPOINTING=1
```

`eval` evaluates Base-Geo, Base-EVQ, Geo-Null, and EVQ-Distill at
8K/16K/32K, measures all 128 held-out sequences, computes the LM head in
memory-bounded chunks, and writes
`positional_distill_summary.json`. The fixed gates are documented in
`docs/superpowers/specs/2026-07-10-llama8b-positional-distillation-design.md`.
Quick RULER is deliberately outside the A-stage pass/fail gates.

This pilot does not remove LoRA, does not reproduce the original LongAlign
protocol, and cannot by itself close the old Table 23 confound. A successful
single-seed result supports only teacher-guided positional recovery; an
industrial capability statement additionally requires a separately
pre-registered autoregressive/RULER evaluation and replicated seeds.

These commands have not been run as part of repository preparation. Do not
quote a result until the raw JSON, frozen-data manifest, checkpoint metadata,
GPU record, and seed scope have been verified.

## Protocol-matched legacy LongAlign multi-seed fallback

The fallback for the historical LoRA row is deliberately separate from the
positional-distillation pilot.  It runs six **fresh** adapters only:
native-Geo and EVQ-Cosh tau 1.414 at seeds 42/43/44.  It never reuses the
historical `evq_r64_tau1414` directory and does not mix YaRN into the matched
matrix.

The old downloader is not claim-safe: it can write LongAlpaca-12k under a
`longalign_10k.jsonl` filename and can write WikiText-103 under a WikiText-2
filename.  The new path therefore requires explicit source revisions and raw
SHA-256 values, freezes token IDs once, and rejects every silent fallback.

```bash
export EVQ_LORA_BASE_DIR=/path/to/external/runtime
export EVQ_LORA_MODEL=/path/to/Meta-Llama-3-8B-Instruct
export EVQ_LORA_PYTHON=/path/to/locked/python
export EVQ_LEGACY_LONGALIGN_JSONL=/path/to/verified/long.jsonl
export EVQ_LEGACY_LONGALIGN_SHA256=<sha256>
export EVQ_LEGACY_WIKITEXT_PARQUET=/path/to/verified/test.parquet
export EVQ_LEGACY_WIKITEXT_SHA256=<sha256>

bash scripts/2026-07/03_lora_longalign_matched_multiseed.sh preflight
bash scripts/2026-07/03_lora_longalign_matched_multiseed.sh prepare-data
bash scripts/2026-07/03_lora_longalign_matched_multiseed.sh baseline
```

`baseline` evaluates Base-Geo and Base-EVQ first.  After inspecting data and
metric compatibility with the historical row, explicitly unlock the fresh
seed-42 pair:

```bash
export EVQ_LEGACY_UNLOCK_SEED42=YES
bash scripts/2026-07/03_lora_longalign_matched_multiseed.sh seed42
```

Only after that matched pair is useful should the two remaining pairs run:

```bash
export EVQ_LEGACY_UNLOCK_REMAINING=YES
bash scripts/2026-07/03_lora_longalign_matched_multiseed.sh remaining-seeds
bash scripts/2026-07/03_lora_longalign_matched_multiseed.sh eval
bash scripts/2026-07/03_lora_longalign_matched_multiseed.sh summarize
```

All six arms are fixed to q/k/v/o LoRA r64, alpha128, dropout0.05, BF16,
300 steps, B2/GA4, LR 1e-4, warmup60, full-token causal loss, recovery saves
every 100 steps (latest two retained), and the same `torch.compile` mode. Automatic resume is allowed
only when the immutable data/model/protocol identities still match.  The final
summary reports raw seeds, mean, sample standard deviation, range, and paired
EVQ-minus-Geo deltas; n=3 is descriptive and is not a significance claim.

Because the original raw file hashes and runtime have not been recovered, the
honest label is **protocol-matched multi-seed rerun on verified LongAlign-10k**,
not bitwise reproduction of the historical single-seed row.
