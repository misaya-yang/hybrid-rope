# Cross-audit experiment preparation — 2026-09-07

> **2026-09-07 status / use:** Current use: historical E0/E1 preparation and reusable apparatus. E0/E1 completed; Z-only formal training did not start. The current Qwen work is recorded in the [pilot owner](../../docs/research/ROPE_SCALE_TRANSPORT_PILOT_20260907.md); the old stage plan below is not an execution queue.

> **Author scope correction, 2026-09-07 (current):** MrRoPE and YaRN are
> frozen references in this programme; adaptation is the project's own route.
> The proposed YaRN/MrPro/Z three-arm full training is withdrawn. Retain completed
> E0/E1 evidence. The reduced E2 plan adapts only the existing Z arm with the
> already prepared 1528-step recipe and final-only storage; new methods/LoRA are
> separate choices. Compare Z before/after and show extra training cost alongside
> frozen references; this does not isolate a frequency advantage under equal
> adaptation. The earlier three-arm rationale below is historical, not a queue.

- **Status:** implementation/preflight; no new model result or GPU authorization.
- **Source:** the author's explicitly selected 2026-09-06
  [cross-audit](external-reviews/ROPE_ICLR2027_CROSS_AUDIT_20260906.md), §§0, 9–10;
  SHA-256 `eade4043ca4481a0f2f7a59da9ec1f5e8172808ee39d3890c745273ff688824a`.
  The author reconfirmed this source on 2026-09-07. Another experiment's “v5”
  is not this run's specification.
- **Question:** does the existing allocation advantage survive strong deployment
  transforms, and does the mature-model advantage survive matched full adaptation?
- **Scope:** apparatus and CPU checks only; independent capability, throughput,
  memory feasibility and claim promotion remain unverified until their own runs.

> **Execution / retention amendment, 2026-09-07:** E0/E1 have since completed;
> see the [read-back](../../docs/research/ROPE_FREQUENCY_LUNA_ROI_20260907.md).
> Following the author's explicit correction, new full-adaptation runs retain
> per-step `steps.jsonl`, one final trained checkpoint and the necessary
> manifests/tokenizer. No intermediate weights or optimizer/RNG resume are saved.
> The original base and completed evidence stay intact. AdamW remains the
> optimizer; model, data, loss and 1528-step proposal are unchanged. Old E0 save
> timings include the former larger save and are conservative for this policy.
> Training-loss logs do not establish intermediate generation quality.

## Bounded implementation

1. E0: freeze corrected YaRN, paper-equation MrUni/MrPro and unchanged archived
   Z; preserve the old Y/Y2 outputs. Check exact arrays, amplitude, tokenizer,
   generation length accounting and complete-answer scoring. Rescore saved rows
   without generating new outputs. Test changed runtime with tiny CPU models.
2. E1 scratch: reuse paper-owned 32 anchors and exact Geo/Cosh checkpoint pairs,
   identity/YaRN-derived/shared-reference MrPro at 256/512/1024/2048. Keep fixed
   s4 and target-dependent transforms separately named. **The author selected
   existing seeds 137/256 for these new overlays on 2026-09-07.** Reuse historical
   seed42 results without searching for weights or retraining; weight availability
   is not a startup gate. The historical FMRoPE retarget remains a separate owner.
3. E1 mature: OLMo-2-0425-1B-Instruct, Native/YaRN/MrUni/MrPro/Z, each fixed s4
   table and declared amplitude across prefill, decode and Native requests.
   Use its actual chat template. Preserve complete near/far/compact/world groups.
   Existing Round12 training-instance outputs are diagnosis, not confirmation.
4. E2 preparation: full-parameter OLMo is the primary regime. Prepare a common
   training step and bounded cost probe for YaRN/MrPro/Z, including Native replay,
   backward, optimizer and save. Exact shared token budget is set from measured
   costs, reserving the main-pair second seed and evaluation. Do not default to
   500M tokens, revive a continuation, or substitute LoRA for full adaptation.
5. E3: all-linear r16 only for the later selected pair at common data/token
   milestones; not an automatic repeat of the old recipe.

## Frozen implementation distinctions

- YaRN uses the linear dimension ramp and cos/sin amplitude `1+0.1*ln(s)` from
  [upstream yarn@995db5b](https://github.com/jquesnelle/yarn/blob/995db5b/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py).
  The old smoothstep/square-root Y2 is a different operator.
- MrUni/MrPro use exclusive-prefix edge products, Eqs. 13–16 and Appendix B of
  [MrRoPE v1](https://arxiv.org/html/2601.22181v1). This is an equation
  reproduction with explicit 32/1-turn boundaries, not a claim of author-code
  identity. On non-native scratch tables the shared reference is named explicitly.
- Exact output accepts only declared full-answer aliases and EOS. Multi-key
  retrieval requires all keys; individual values are not full-answer aliases.
  Substring recall and normalized QA EM/F1 remain separate diagnostics, never
  mislabeled as a full official benchmark run.
- Missing worlds, duplicate row identities, truncated evidence/prompts, empty
  evaluation selections, hash/config drift and absent assets fail visibly.
- Full adaptation trains all weights with fixed RoPE; LoRA and any norm/embedding
  extras are distinct regimes. Native teacher targets must come from the original
  checkpoint/table, never a moving student or an unverified cache.

## First launch and decisions

The next purchasable step is a bounded E0 hardware/runtime qualification, then
E1. The audit's 6/12/38/12/14/12+6 GPU-hour split is a proposed ceiling, not
measured cost or permission. Freeze the exact machine and run budget after the
CPU receipt; require CUDA BF16 and Flash SDPA without math fallback.

An identity/scoring/data defect stops the affected assay for repair. OOM,
nonfinite loss, table drift or timeout stops the run without automatic retry.
No resolving positive/deleted-source controls means unresolved capability,
not a candidate/class verdict. Native damage and complete long-generation scores
are reported separately; final margins must be fixed before confirmation outcomes.

If strong transforms remove the scratch advantage, narrow the claim before new
scratch training. If a mature candidate loses under the matched qualified
protocol, preserve that candidate-scoped negative. If long performance improves
while Native fails, consider only a separately frozen Native-constraint increment.
Full/LoRA comparison is conditional on identical data/token milestones.

Runtime commands, actual asset locations and live readiness belong in HANDOFF
and private work-machine receipts; private paths and raw data are not published.

## CPU preparation findings

- The selected Downloads source and its repository copy are byte-identical.
- Pinned upstream YaRN code was run independently on CPU: the new table agrees
  within float32 rounding and amplitude is identical. Original M is byte-identical
  to the MrPro equation reproduction; original Y is also numerically faithful.
  Y2 differs materially. No broad invalidation of original Y/M is warranted.
- The corrected strict scorer rejects prefix-only answers and requires every
  multi-key value. Saved raw-mode N/M/Z counts change from 67/103/162 to
  64/102/146 of 512 rows; Y and Y2 counts do not change. Saved chat-mode N/Z
  counts remain 63/119 of 320. These are **row rescoring diagnostics**, not
  group-level capability comparisons or independently confirmed observations.
- 512 validation rows cover three task families, compact/near/far/deleted
  source conditions and paired worlds. Their semantic/source-lineage identities
  do not overlap the 512 prepared training views. Existing exposure elsewhere
  is not excluded, so the set remains development rather than confirmation.
- The actual available CPT corpus has 2048 × 16385 tokens (about 33.55M usable
  next-token targets); the proposed 500M v2 corpus was not completed. Repeated
  passes cannot be described as 500M unique tokens.
- The Native pool has 128 training rows in each of instruction, format,
  reasoning and text. Its positions are already **hidden-state indices**;
  shifting them again would be wrong. Teacher KL keeps the valid post-answer
  position; labeled NLL excludes positions without a next token in the row.
- The scratch model source agrees with the pinned archive architecture. The
  old aggregate training-code hash differs from the current evaluator bundle;
  the new loader checks weight/sidecar/table/config identities separately and
  requires replay of all historical Native anchors before new overlays.
- 29 affected CPU tests pass, including actual tiny OLMo/Qwen cached decoding,
  dense versus chunked CE values/gradients, KL position normalization, full/LoRA
  updates and execution guards. This does not validate real GPU throughput or
  long-context capability.

## Concrete training proposal and operator boundary

The supplied source does not fully specify a loss mixture. The prepared proposal
uses one alternating 8K/16K CPT prefix, one near/far answer pair and one Native
replay row per update; CPT mean CE, answer-token-weighted SFT CE and position-mean
forward KL have weights 1/1/1. Full adaptation uses FP32 master weights/Adam and
BF16 compute. The pure all-linear r16 bridge is distinct from historical recipes
that additionally unfreeze norms/embeddings. Freeze this choice with the exact
budget before training; no claim of historical-recipe equivalence is made.

The private job plan contains 40 concrete jobs. E0/E1 are ready for GPU
qualification; E2/E3 have executable entrypoints with explicit blocked budget/
opponent fields. E4/E5 remain conditional decisions, not speculative automatic
runs. A completed job is not a scientific pass. Stage batching stops on error,
writes a report and can power off only with explicit shutdown authorization.

The author prefers a 5090 for the compute-heavy training stages and a later
Luna session for execution/monitoring. At the author-quoted hourly prices, the
equal-cost speedup threshold is `2.78 / 1.58 = 1.76`. No same-recipe two-card
speedup has been measured in this preparation, and further hardware comparison
was explicitly excluded from scope.
