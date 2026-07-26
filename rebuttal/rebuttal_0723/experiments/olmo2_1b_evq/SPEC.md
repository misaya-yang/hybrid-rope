# OLMo-2 1B EVQ controlled scale-transfer contract

## Reviewer or AC concern addressed

`R27bE.2`, `R27bE.5`, `AC.2`, and `AC.4`: evidence at a larger,
modern full-RoPE architecture with `base=500K`, `d_head=128`, and a
pre-specified intervention.

## Existing evidence

The 151.9M exact-range three-seed study isolates allocation shape and is already
the statistical evidence. The mature 8B LoRA result is supporting adaptation
evidence, not a from-step-zero allocation comparison.

## Smallest missing evidence

One EVQ branch from the released OLMo-2 step-0 initialization to the released
step-1000 token milestone, with the released Geo checkpoint as the control.

## Smallest executable plan

- Model repository: `allenai/OLMo-2-0425-1B-early-training`.
- Step-0 commit: `9f46fe81fa53e429771f051b36aa51c60f7e6c0f`.
- Released Geo step-1000 commit:
  `aeef9be8719cbdd31e3f89258e722360642afd2d`.
- Official recipe source: `allenai/OLMo` commit
  `090253dac6688f2532509daa7aa2eb5fae50e956`,
  `configs/official-0425/OLMo2-1B-stage1.yaml`.
- Sequence length 4096, global batch 512 sequences, seed 6198, BF16 autocast,
  official AdamW/warmup/z-loss settings.
- EVQ changes only the non-persistent native endpoint-grid `inv_freq` buffer;
  `tau=2`.
- First gate: 1000 optimizer steps =
  `2,097,152,000` counted input tokens.

The public model is branded “1B”, but the Hugging Face artifact contains
`1,484,916,736` parameters. Reports must use “OLMo-2 1B (1.485B actual
parameters)” when exact scale matters.

“Official FineWeb-Edu” is not interchangeable with this experiment's official
OLMo stream.  The admitted training artifact is the exact seed-6198 prefix
reconstructed from the 1,122 paths in the pinned OLMo stage-1 config, with the
official Dolma2 tokenizer, per-file 4,096-token chunking, duplicate path
weighting, repetition-instance masking, and NumPy PCG64 order.  A
GPT-NeoX-tokenized FineWeb-only tensor is rejected by preflight even when its
raw source was downloaded from an official Hugging Face repository.

On the high-memory preparation host, the final gate regenerates the full
PCG64(6198) permutation and compares the saved 512,000-instance prefix
element-for-element.  It also re-fetches 100 deterministic source ranges and
requires byte equality plus identical repetition-mask decisions.  The
hash-bound proof is copied with the stream; the 2 GiB target does not attempt
the multi-gigabyte regeneration itself.

## Execution package

`run_pro6000.sh` has separate no-GPU and paid-GPU modes.  The no-GPU sequence
is:

```bash
bash run_pro6000.sh prepare-env
bash run_pro6000.sh prepare-assets
bash run_pro6000.sh prepare-data
bash run_pro6000.sh prepare-eval
bash run_pro6000.sh prepare-retrieval
bash run_pro6000.sh cpu-preflight
```

The independent long-text anchor is the pinned PG-19 test/validation parquet
revision `b7bca68072ef1d86348f080bbda0996648d94315`.  It admits exactly one
16,384-token window from each of 128 distinct documents; it never concatenates
or duplicates documents.  The separate official OLMo validation anchor remains
256 rows at 4,096 tokens.  PG-19 is a fixed external evaluation source, not a
claim of corpus-level decontamination against every document in OLMo-mix.

If the target no-GPU container cannot hold the 1.485B FP32 model, run
`portable-preflight` on a high-memory no-GPU host with the same artifacts, copy
`portable_preflight.json`, and run `cpu-preflight-from-portable` on the target.
The target still revalidates code, every model/data/evaluation manifest and
hash, environment, output path, and storage.  Only the RAM-heavy
trainable-parameter/intervention check is reused, and any content drift makes
the target receipt fail closed.

The exact official permutation requires a multi-gigabyte full index array.
`prepare-data` checks both host and cgroup memory and must run on the
high-memory preparation host.  A 2 GiB no-GPU target validates and consumes
the transferred stream; it does not reconstruct it.

The CPU receipt records code, model, tokenizer, data-order, first-batch,
evaluation, retrieval, environment, disk, and exact launch-command hashes.  It
must be regenerated after any code or manifest change.

The checkpoint filesystem must still have at least 25 GiB free after all input
artifacts are present. On the fixed 50 GiB volume, step 500 is a logged and
validated gate but is not persisted as a full checkpoint. The retained
step-1000 FP32-model plus Adam state requires about 16.6 GiB, leaving explicit
headroom for logs, validation output, and the atomic save.

On an RTX Pro 6000 Blackwell, `gpu-sequence` first checks native-vs-Liger
loss/gradient parity at microbatch 4, safely probes microbatch 8, then admits
the selected runtime only after a 100-microstep warmup and 300 measured
microsteps with optimizer state resident.  Flash-only SDPA, BF16 autocast,
fused AdamW, TF32, persistent TorchInductor cache, and a 6% VRAM safety margin
are mandatory.

The scientific run is fail-closed:

1. 20-step native-Geo sentinel;
2. EVQ 0→500, with the first 20 steps checked against identical Geo batch
   hashes and comparable loss/gradient dynamics;
3. hash-verified full-state resume for EVQ 500→1000;
4. matched natural-text 2K/4K/8K/16K NLL evaluation;
5. teacher-forced retrieval NLL/rank/margin/source-deletion evaluation;
6. paired bootstrap comparison.

RULER/NIAH is not launched automatically.  The retrieval comparison emits the
capability gate; a bounded downstream subset is justified only if that gate
passes.

## Stop condition

Do not start paid training unless the no-GPU READY receipt proves:

1. both checkpoint revisions and weight-file hashes;
2. official Dolma2-tokenized data, not the repository's GPT-NeoX FineWeb
   tensors;
3. deterministic seed-6198 sample order and the first-global-batch hash;
4. native Geo equals endpoint EVQ at `tau=0`;
5. the only model-state intervention is the `inv_freq` buffer;
6. checkpoint/resume and disk-space capacity;
7. the exact launch command and output directory;
8. Flash-only SDPA eligibility and finite-loss GPU probe are still pending,
   never represented as CPU-verified.

At step 1000, stop expansion when 8K and 16K tail NLL are both stably worse and
the routing/rank probes show no positive signal. Do not search a replacement
`tau` after a negative result.

## Evidence boundary

The launcher uses the Hugging Face OLMo-2 implementation and a reviewed
single-GPU loop, while the released Geo checkpoint was produced by AI2's
distributed OLMo trainer.  Therefore the default label is
`same-initialization, same-recipe`, even after exact data-order reconstruction.
A finite Geo sentinel is a safety gate, not proof of a bitwise paired
trajectory.  Upgrade to `released matched trajectory` only if an official
same-stage reference loss or checkpoint establishes the registered tolerance.
If the data mix or tokenizer differs, the released Geo checkpoint is not an
admissible control at either level.
