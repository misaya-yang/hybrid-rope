# OLMo-2 1.485B Native-preserving far-query EVQ residual

Status: `DESIGN_ONLY_RUNTIME_UNVERIFIED`

This is an EVQ-Cosh method: the added residual Q/K coordinates use the exact
fixed endpoint EVQ-Cosh non-geometric frequency tensor (`base=500000`,
`tau=2`, `head_dim=128`). The global inherited path remains Native. The
scientific identity is therefore **Native + EVQ-Cosh residual**, rather than
the submitted full-table EVQ replacement.

No GPU run or capability result exists yet.

## 1. Question

Can a mature OLMo-2 retain its inherited 4K function exactly for short
requests while learning an additional EVQ-Cosh Q/K route for queries beyond
4K, using only physically at-most-4K training rows?

This is a new post-hoc conversion experiment. It is not required to defend the
submitted training-time EVQ-Cosh result and must not be presented as submitted
evidence.

## 2. First-principles construction

For a long request, each attention layer computes one augmented score:

\[
s_{qt} =
\frac{
  \langle q_q^{N},k_t^{N}\rangle
g(p_q)\,\gamma\,
  \langle q_q^{E},k_t^{E}\rangle
}{\sqrt{d}},
\qquad
g(p_q)=\mathbf 1[p_q\ge4096].
\]

- `N`: the untouched inherited Native projection and Native RoPE.
- `E`: a new low-rank Q/K projection followed by the exact EVQ-Cosh RoPE.
- `gamma`: a positive learned per-layer logit gain.
- `d=128`: the Native head dimension and the residual head dimension.

The two Q/K blocks are concatenated before a **single** SDPA call and a
**single** softmax. Values are padded with 128 exact zeros; the residual value
coordinates therefore cannot introduce a second value path. After SDPA, the
zero-padded coordinates are removed and the untouched Native output projection
is used.

For a request whose total budget is at most 4096, the wrapper delegates to the
original OLMo attention module. It does not run the augmented kernel. This is
the exact inherited short-request code path.

For a request above 4096, the route must be selected before prefill. The cache
then has a fixed augmented width for both prefill and decode. Switching routes
with a non-empty cache is rejected.

This design uses query-position gating rather than a full pairwise
`g(p_q-p_k)` mask. A pairwise distance mask would require a materialized
quadratic mask or a new custom Flash kernel; it is deliberately excluded from
the first experiment.

## 3. Fixed first-run protocol

| Field | Registered value |
| --- | --- |
| Base model | OLMo-2 1.485B Instruct, untouched Native RoPE |
| Added path | low-rank Q/K only in all 16 layers |
| Residual Q/K dimension | 128 per head |
| Projection rank | 64 |
| Residual frequency | endpoint EVQ-Cosh, base 500K, tau 2 |
| Initial logit gain | 0.1 |
| Global Native Q/K/V/O | frozen |
| Physical training length | at most 4096 |
| Position offsets per optimizer step | 4096, 4096, 12288, 12288 |
| Objective | complete answer tokens plus immediate EOS |
| Steps | 100 |
| Micro batch / accumulation | 1 / 4 |
| Optimizer | fused AdamW, beta=(0.9,0.95), no weight decay |
| LR / warmup | 5e-5 / 10 steps |
| Gradient clipping | global norm 1.0 |
| Precision | BF16 autocast |
| Compile | disabled until the registered GPU smoke establishes eligibility |

Every micro-step exposes an answer query beyond the 4K threshold; no GPU
micro-step is spent on a structurally inactive short-request path. Short
retention is evaluated, not trained.

The training view is a single declared task-family view per run. A QA adapter
and a RULER-family adapter, if both are needed, are separate runs and separate
owners. They must not be spliced into one result.

## 4. GPU READY gate

The smoke is readiness only. It must establish:

1. the global model RoPE remains byte-identical Native;
2. every residual rotary buffer has the registered EVQ hash;
3. exactly 80 trainable tensors exist: Q-A/Q-B/K-A/K-B/gain in 16 layers;
4. the short route matches the pre-install Native logits bitwise;
5. augmented D256 Q/K/V Flash SDPA is eligible with math and
   memory-efficient fallbacks disabled;
6. active loss is finite and residual gradients are finite and nonzero;
7. augmented prefill and one-token decode use the same D256 cache;
8. the full registered input/output/code hashes match the prepared receipt.

Any failure stops the run. Do not silently use eager or math attention, reduce
the residual dimension, alter the route, or change the loss while paying for
the instance.

## 5. Capability evaluation

The same saved adapter is evaluated with:

- `frequency=native` for the global model;
- `adaptation=far_only_evq_residual`;
- route disabled at total budget 4096;
- route enabled before prefill at total budgets 8192 and 16384.

Required endpoints:

1. 4K Native-path bitwise smoke plus full 4K 2Wiki or RULER task endpoint;
2. 8K and 16K strict autoregressive task score;
3. complete generation text and terminal EOS;
4. matched untouched Native control on the identical rows.

Teacher-forced answer NLL in the trainer is diagnostic only.

## 6. Decision rule

- **Success:** 4K is identical to the untouched Native endpoint by route
  construction, while 8K/16K capability exceeds the matched Native control.
- **Partial:** the residual learns 8K but not 16K; report only the tested 8K
  endpoint.
- **Stop:** the GPU smoke cannot use Flash SDPA, residual gradients remain
  zero/non-finite, or the matched long-context control shows no gain.

Do not run a rank, gain, threshold, or step sweep in this experiment.

## 7. Entry points

- method and adapter contract: `far_only_evq_residual.py`
- no-GPU preparation: `preflight_4k_far_only_evq_residual.py`
- GPU smoke and training: `train_4k_far_only_evq_residual.py`
- strict 2Wiki evaluation: `evaluate_2wiki_phase_adaptation.py`
- strict RULER evaluation: `evaluate_instruct_ruler_transfer.py`

The preparation receipt prints the exact hash-bound smoke and training
commands for the machine on which the checkpoint and training view reside.
