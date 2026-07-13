# Native-RoPE vs Endpoint-EVQ 150M Design

Status: completed on 2026-07-13 as a single-seed diagnostic. See
`rebuttal/NATIVE_ROPE_EVQ_150M_500M_RESULT_20260713.md` and
`data/curated/native_rope_evq_150m_s42_500m_20260713.json`.

## Goal

Train one matched pair of approximately 150M-parameter decoder-only language
models at length 2,048 and measure whether endpoint EVQ-Cosh provides a better
substrate than native RoPE, both directly and under the pinned official YaRN
range operator.

This is a controlled mechanism experiment. It does not claim that 500M tokens
fully train a 150M model, that EVQ replaces YaRN, or that passkey retrieval
emerges from natural text without task exposure.

## Fixed paired protocol

Both arms use the existing `125m` architecture from `run_evq_sweep.py`:

- vocabulary 50,304; hidden width 768; 12 layers; 12 heads;
- rotary head dimension 64; SwiGLU intermediate width 3,072;
- tied token embedding / language-model head;
- exactly 151,898,880 parameters including embeddings;
- training length 2,048, RoPE base 500,000, seed 42;
- AdamW, learning rate `3e-4`, betas `(0.9, 0.95)`, weight decay `0.1`,
  2% warm-up, cosine decay to `3e-5`, and gradient norm 1.0;
- global batch 60 as micro-batch 12 times five gradient-accumulation steps,
  exactly 4,069 optimizer steps, BF16 autocast, fused AdamW, and no gradient
  checkpointing.

The arms differ only in the immutable training-time frequency tensor:

1. `native_rope`: endpoint grid `u=k/K`, identical to standard RoPE;
2. `endpoint_evq_tau1p5`: endpoint EVQ-Cosh with pre-registered `tau=1.5`.

`tau=1.5` is the historical Primary-I value and is close to the operating-rule
value `64/sqrt(2048)=1.414`. It is not tuned against the new validation set and
is not described as globally optimal.

The two arms receive identical trainable-parameter initialization, token order,
Passkey rows, optimizer, precision, and number of optimizer steps. Execution
details such as compilation and cache paths are recorded but are not scientific
variables.

## Training data

The natural-text source is the existing 499,998,720-token FineWeb-Edu tensor
with SHA-256 `115c5dca5c9023e5595fade1251cbc8d023edb8e0117abe5632aeb4218bcc2bf`.

Passkey exposure follows the prior repository mechanism:

- deterministic row selection by
  `Random(index * 6364136223846793005 + 1).random() < 0.02`;
- the same duplicated `<<PASS:d-d-d-d-d>>` marker and target format;
- deterministic secret and depth from the row index;
- selected rows replace natural rows, as in the previous `MixedDataset`;
- filler comes from the selected training row, never validation data.

For 244,140 rows this selects 4,926 rows, or 10,088,448 Passkey tokens
(2.0177%). This deliberately matches the approximately 10M synthetic-token
budget of the historical 100M-token, 10%-mix Primary-I run. Copying the old
percentage into a 500M-token run would silently multiply Passkey exposure to
50M tokens, even though historical 2K retrieval was already saturated.

The old helper used `val_data[:50000]` as training filler. That validation
contamination is not reproduced: the new cache uses the selected natural
training row as filler, and evaluation uses a disjoint held-out shard. The
prepared Passkey tensor and selected-index tensor are shared by both arms and
hash-bound in a manifest.

## Validation correction

The existing 5M validation tensor is invalid for this experiment: all of its
5,000,000 tokens equal the first 5,000,000 tokens of the 500M training tensor.
It must never be loaded by the new evaluator.

CPU preparation downloads FineWeb-Edu shard `004_00000.parquet`, which is not
listed in the 500M training manifest, and tokenizes a new 5M-token held-out
validation tensor. The manifest records repository revision, source shard,
tokenizer identity, output hash, and a guard proving that its prefix differs
from the training prefix. Evaluation refuses the legacy validation hash.

## Evaluation

The same frozen validation offsets and Passkey cases are used for every arm and
operator. Natural-text evaluation reports mean token NLL, PPL, individual
offset losses, and sample count at 2K, 4K, 8K, and 16K. The same forward pass
also reports NLL/PPL over only the final 2K prediction tokens so that easy
prefix predictions cannot hide an extrapolation failure near the context tail.

Each checkpoint is evaluated under:

- raw substrate;
- target-matched range scaling with factor `max(1, L/2048)`;
- the historical repository fixed-ramp operator, separately labeled and never
  described as official YaRN.

For `native_rope`, the scaled condition uses the exact equations pinned from
`jquesnelle/yarn@995db5b` and is labeled **official YaRN on native RoPE**. For
endpoint EVQ, the same equations use virtual frequency coordinates and are
labeled **YaRN-derived on endpoint EVQ**, not official native-grid YaRN.
These are inference-only, non-fine-tuned operator cells; they do not reproduce
the YaRN paper's long-context continuation-training recipe.

Passkey evaluation retains the old marker schema but uses unseen secrets and
held-out shard-004 filler at depths 10%, 25%, 50%, 75%, and 90%. The primary
metric is teacher-forced `NLL_wrong - NLL_correct`; positive values count as
retrieval. Because a single-negative sign test has a 50% chance boundary when
the gap is near zero, the continuous gap and raw per-case NLL are primary; the
aggregate sign rate is diagnostic only.
Autoregressive exact match is optional and disabled in the first cost-sensitive
run.

## Performance and launch behavior

Training uses a fixed-shape loss module compiled with PyTorch 2.8
`max-autotune-no-cudagraphs`, persistent `TORCHINDUCTOR_CACHE_DIR`, asynchronous
DataLoader workers, pinned host memory, non-blocking transfers, and fused AdamW.
Disabling CUDA graphs avoids retaining an additional fixed-shape workspace on
top of the 2K batch while retaining Inductor autotuning. Micro-batch 12 fits a
32GB RTX 5090; five equal micro-batches preserve the registered global batch 60,
and both 12 and 60 divide 244,140 rows exactly.

The launcher performs all CPU/hash/parity checks before CUDA model allocation,
then trains both arms sequentially and automatically evaluates the registered
arm/operator conditions. It never downloads or tokenizes data in GPU mode. An
existing output directory is not overwritten.

## Files

- `experiments/native_rope_evq_150m/protocol.py`: immutable identities and
  validation.
- `experiments/native_rope_evq_150m/model.py`: self-contained tied-embedding
  decoder architecture with BF16-preserving RoPE application.
- `experiments/native_rope_evq_150m/prepare_data.py`: held-out validation and
  frozen Passkey cache.
- `experiments/native_rope_evq_150m/train.py`: paired compiled training.
- `experiments/native_rope_evq_150m/evaluate.py`: PPL/NLL and Passkey scoring.
- `experiments/native_rope_evq_150m/run_seed42.sh`: CPU prepare, preflight,
  training, and automatic evaluation entrypoints.
- `tests/test_native_rope_evq_150m.py`: protocol, data, schedule, and operator
  regression tests.
