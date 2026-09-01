# P1: reference-correct frozen K128 coupling

## Entrance and hypothesis

P0 independently confirmed `L_ref=4096` for the exact Gemma-1.1 artifact under
natural continuation plus single-code retrieval. `L_config=8192` and the
documented family training context remain unchanged. This registration tests
whether reference/request-scale mismatch explains the earlier K128 negative.

No boundary, movement law or gain coefficient is fitted. Use
`x_H=.7382780681078285`, `x_L=.366403835112904`, `c=.074`, and the same runtime
Native frequency tensor. A confirmed reference receipt binds the checkpoint,
config, Native tensor, calibration decision and data identities.

## Stage 1: one-hop exact 2x

- Target 8192 gives the unique ratio `s=8192/4096=2`.
- Compare Native, frozen physical-x, frozen normalized-index and deterministic
  official-equation YaRN2. All three long profiles use the confirmed 4096
  reference in their construction. YaRN retains its published fixed gain;
  its comparison is combined-method, not isolated geometry attribution.
- Each profile is fixed before prefill for the entire request. Evaluate the
  same profile at both 4096 and 8192; never switch at a token boundary.
- Fresh official core-four RULER, seed `202609023`, 20 paired rows per task and
  length. These rows never supplied P0 length selection. Generate the 16384
  cells as future assets but do not open model outcomes before stage 2.
- Native 4096 must resolve nonzero capability; no candidate-vs-zero retention
  ratio is calculated. Report both absolute scores and 4096 retention.
- Report complete task vectors and paired row-bootstrap intervals. Differences
  of `.01--.03` do not identify the physical coordinate. The `.05` practical
  contrast is not a significance threshold.

Decision: single-key at most `1/20` or nonfinite likelihood rejects that frozen
profile at this endpoint; no rescue tuning. The stage-2 transport entrance is
physical single-key at least `16/20` and macro at least `.10` above Native at
8192. Separately record the `.875` 4096 macro-retention operating gate; a miss
cannot be relabeled a deployable pass even if a later scale-4 mechanism
diagnostic runs. If only YaRN resolves long behavior, record a method-
specific deficit. If all static long arms fail, retain an unresolved screen.

## Stage 2: direct 4x maximum profile, conditional

Target 16384 gives `s=4`, never a searched scale. Freeze new physical/index/
YaRN4 profiles with the **same** confirmed reference and boundaries. Evaluate
their 4096/8192/16384 curve, not just the extreme point. This separates a
successful one-hop extension from failure to generalize at larger requested
scale. Do not recompute Native slot labels from an already transformed table.

An old config-based profile may be retained as a labeled mis-reference control,
not as an additional selectable profile. No NTK, Resonance, PI, hierarchical
table, selector or broad SOTA sweep belongs to these stages.

The fixed mis-reference control is the already-existing physical table with
`L_ref=8192,s=2,c=.074`, tensor SHA-256
`fbd2f80a462f3271e65a8cdc9f3acb81b46c4afcf79c3be73509c36ac1f0bf0b`.
Evaluate only the same fresh 16384 core-four cells (80 generations). Compare
its original maximum-16K intention with the corrected `L_ref=4096,s=4` profile.
This is a joint reference/request-ratio control, not an isolated gain or
coordinate ablation, and it cannot be selected as a new candidate.

## Confounds and ceilings

Changing reference length changes both coordinate geometry and the correct
request ratio. Overall recovery would support the reference-mismatch
explanation; it would not assign a pure geometric effect independently of
gain/scale without a matched component control. This remains one exact K128
artifact and does not identify K causally or establish a universal law.

Fresh natural likelihood/continuation holdouts must be kept separate from P0
calibration/confirmation before making a final joint operating-point claim.
The core-four entrance is not itself full downstream or natural-task evidence.

## Fresh natural holdout (frozen before profile inference)

Select the first 32 unique FineWeb-Edu documents of at least 16384 tokenizer
tokens, from source row 20000 onward, excluding all 96 hash-bound P0 natural
documents. Selection reads input identities only, never model losses. Construct
4096/8192/16384 prefixes ending at the same 256 target tokens per document,
with one BOS and no appended EOS. Generate all lengths as assets; stage 1
opens only 4096/8192.

[`prepare_reference_coupling_nll.py`](../../../../scripts/data/prepare_reference_coupling_nll.py)
owns the input contract;
[`eval_reference_coupling_nll.py`](../../../../scripts/eval/eval_reference_coupling_nll.py)
loads the four fixed profiles before any model outputs. Each arm uses its
unchanging frequency tensor and gain at both lengths. Record per-document
NLL, paired changes from Native, PPL retention and runtime. The 4096 practical
PPL-retention margin remains `.875`; nonfinite loss rejects the arm. A finite
but failed Native gate is reported as such and cannot be rescued by long gains.
