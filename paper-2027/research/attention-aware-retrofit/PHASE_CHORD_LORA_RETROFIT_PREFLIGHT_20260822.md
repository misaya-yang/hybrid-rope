# Static-table phase-chord LoRA retrofit preflight

- **Date:** 2026-08-22
- **Status:** **REVOKED BEFORE GPU** — the frozen source/continuation data are
  not identifiable from the model input; no GPU run is permitted
- **Role:** internal method protocol; not experiment evidence

## Revocation notice

This protocol must not be executed.  In every long example, one 1,024-token
block is selected uniformly as the source and placed among six natural
distractor blocks, but the input contains no query, anchor, block identifier,
or source-position cue that identifies the selected block.  The selected
source slot is written only to provenance and is never consumed by the model.
The appended target therefore does not define a learnable source-selection
problem: in the exchangeable idealisation its conditional distribution is a
mixture over the seven possible continuations.  Any apparent selection above
chance can come from block-length or concatenation-boundary leakage rather
than general source use.

The old train/validation views and their teacher/source-effect receipts are
invalid for method selection.  They may be retained only as an audit trail.
The replacement protocol must put an explicit unique natural anchor/query in
the input, keep that query fixed across correct/counterfactual variants, and
use strict answer-plus-EOS evaluation on document- and template-disjoint
rows.  The runner fails closed for every action that could consume the
revoked views.

## Decision

The next mature-model experiment is a three-arm static-table factorial, not a
new attention operator:

| Arm | Final RoPE table | Trainable model parameters |
| --- | --- | --- |
| Native | released Native | rank-64 Q/K/V/O LoRA |
| anchored EVQ-Cosh | endpoint-anchored EVQ-Cosh, tau 2 | same LoRA |
| phase-chord | true OLMo R0 phase-chord, lambda 0.1 | same LoRA |

All arms use the released OLMo-2-0425-1B-Instruct checkpoint, identical LoRA
initialization, serialized data order, optimizer, token budget, and evaluation
rows. The two non-Native arms use a training-only smooth log-frequency morph
from Native during the first 60 of 300 optimizer steps and finish at their
fixed target tables. Deployment uses one ordinary D128 RoPE table, standard
Flash SDPA, ordinary-width KV cache, and a standard PEFT adapter. There is no
length routing, second attention branch, headwise table, or learned frequency
search.

The endpoint-anchored comparator is not the historical mature EVQ table. Its
slow endpoint is locked to Native so that anchored EVQ-Cosh versus phase-chord
changes normalized interior allocation at fixed sampled support. It must be
named accordingly.

## Why this method

Three completed observations are combined without upgrading any of them:

1. fixed-support experiments identify normalized interior allocation as a
   causal training-time variable;
2. phase-chord gives a two-seed training-time Pareto improvement over the
   geometric fixed-support baseline, but has no mature-retrofit result;
3. mature EVQ LoRA proves that Q/K/V/O low-rank co-adaptation can convert a
   non-geometric table into task-family length transfer, while the later
   Native-residual CE chain proves that lower teacher-forced NLL alone does not
   create autoregressive source use.

The new training signal therefore supervises source use directly. For a
natural source block and its continuation, released Native supplies the short
context effect

\[
\Delta_T = \log p_N(y\mid x) - \log p_N(y\mid x_{\rm swap}).
\]

The student sees the same source/continuation relation at physical 8K and
optimizes its corresponding effect \(\Delta_S\) with positive-teacher-token
effect matching, a correct-over-swapped margin, and low-weight correct-token
CE. This is not RULER, NIAH, 2Wiki, or answer-template training.

The table-by-weights crossing and exact transplant obstruction rule out an
exact global Q/K compensation claim. Short retention is therefore an empirical
function constraint: every arm receives the same 4K Native replay target. No
result may be described as bitwise short-context preservation.

## Data and evidence contract

- Source rows are 1,024 distinct FineWeb-Edu documents tokenized by the
  released OLMo tokenizer; the last 128 documents are validation-only.
- Each source row yields four offsets. The 1,024-token source block is
  deterministically permuted among natural distractor blocks so physical
  source position cannot identify the correct branch.
- Training uses physical 8,192-token correct/source-swapped pairs. A separate
  validation-only 16,384-token view tests length extrapolation.
- The supervised natural continuation is 16 tokens and does not include an
  EOS contract. Greedy 16-token exact and absolute first-token top-1 are
  descriptive only because natural continuation is not a unique-answer task.
- Full answer-plus-EOS applies only to the registered downstream evaluators.
- RULER and 2Wiki are held out from method selection and training.

The frozen float32 frequency identities are:

| Table | SHA-256 |
| --- | --- |
| Native | `dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34` |
| endpoint-anchored EVQ-Cosh | `9e82c83312b6f5f44c63f6ad4fea8fbdd1256a64b3bc853ae8e52c5538a4d02d` |
| phase-chord OLMo R0 | `4d985cce3c47506079119d9d0454d02a49d753238bce86ea4bf0f0b2e398e931` |

The original frozen target manifest is the only table authority. Reconstructing
or approximating a missing table is forbidden.

Prepared data receipts:

| View | Shape | Distinct source documents | Manifest SHA-256 |
| --- | ---: | ---: | --- |
| train 8K | `3584 x 8192` | 896 | `4281f04358dc733ca27d1609ea6616839ea3eebe8928d99e8c2e7e7945c18afd` |
| validation 8K | `512 x 8192` | 128 | `eef4d2547bda30605f30239a7e2b1ea8f1c057d7250bfe501a11d77313c80dff` |
| validation 16K | `512 x 16384` | 128 | `74161ed4a81e0732560730b34ce737366ce60232a50e602190d164bc3330e248` |

Every source slot is represented in every view; correct and swapped branches
share the same slot and target. The train document set is disjoint from both
validation views, and the validation 8K/16K views use the same 128 source
documents. All NPY/provenance hashes were independently recomputed after the
atomic build.

The pinned RULER source is at commit
`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`. Its external essay corpus is
present with SHA-256
`2d3b4126795fd61f2cfb1fca0f831fdf7eb78e0905a50f4036cd5e0a3159c098`.
The 2 GiB no-GPU cgroup cannot hold the existing parent tokenizer plus its
generator child, so the full 13-family matrix is not yet a prepared asset. The
runner registers `prepare-ruler` as a CUDA-disabled action to execute alongside
the GPU teacher/training process once the full instance memory limit is
available. No RULER inference starts automatically.

## Decision endpoints

The first required evidence is the held-out natural matrix at 4K, 8K, and
16K. The matched advancement endpoints are Native-function/NLL retention,
correct-minus-swapped source effect, source-follow rate, and paired relative
first-token rank. Greedy 16-token exact and absolute first-token top-1 are
reported separately as descriptive diagnostics and cannot kill or advance an
arm.

Phase-chord advances only if its 8K and 16K matched natural endpoints improve
over anchored EVQ-Cosh while its 4K replay constraints pass. Only then may it
enter matched RULER-13 and 2Wiki evaluation. The causal contrasts are:

- anchored EVQ-Cosh minus Native: whether a fixed-support non-geometric table
  helps this mature retrofit protocol;
- phase-chord minus anchored EVQ-Cosh: whether the attention-aware interior
  allocation improves the retention/extrapolation trade-off.

A positive probability or source-effect result is probability/causal-source-use
evidence, not broad capability. Paper-level advancement requires held-out RULER
gain beyond NIAH-only families at 8K, measurable 16K capability, and no 4K
capability collapse. A single training seed remains a controlled pilot and does
not establish training-seed uncertainty.

## GPU boundary

No training or inference has run for this protocol. The prepared runtime uses
PyTorch 2.8.0+cu128, Transformers 5.15.1, PEFT 0.20.0, and Accelerate 1.14.0;
dependency import was verified without CUDA initialization. Before paid compute the
implementation must bind the released checkpoint, tokenizer, source tensor and
document receipt, three frequency tensors, serialized views, PEFT/runtime
versions, code hashes, output paths, and shutdown plan. GPU execution requires
an exact-shape Flash-only smoke with a real first optimizer step, finite loss
and gradient, and a throughput/memory receipt. The same grad-enabled compiled
B=2x8K graph must then read the exact final table, match an eager target-table
forward, produce finite nonzero LoRA gradients without an optimizer step, and
leave at least 1 GiB allocated-memory headroom. A fresh standard-PEFT load must
preserve the frozen table and pass full-forward versus prefill/decode
DynamicCache parity. Stages advance explicitly; no benchmark evaluator is
launched automatically.
