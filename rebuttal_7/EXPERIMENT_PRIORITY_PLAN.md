# Rebuttal experiment priority plan

## Decision

The first server allocation should not go directly to a new 7B-family fine-tune. The simulated AC explicitly makes Q1–Q3, Q5, Q7, and Q9 acceptance-critical. Q1–Q2 are fixed locally and Q9 is closed from the recovered Primary-I raw payload; future server time should first close Primary II seeds, tuned-base controls, and the MLA convention ablation.

The proposed “7B experiment” remains important, but for rebuttal continuity it should be instantiated first as a corrected matched **LLaMA-3-8B-Instruct** LoRA experiment, because that is the model already reported. A new 7B model is useful as a later cross-model test of the rank-threshold hypothesis, not as a replacement for the broken matched control.

## Queue

### P0 — completed without a server

- Correct QuALITY figure/provenance and remove unsupported accuracy narrative (Q1–Q2).
- Prominent NLL-gap metric naming (labeling part of Q9).
- Paired-delta audit for existing three-seed primaries (Q15).
- NTK caveat, exponent/basin boundary, per-head boundary, terminology, worked example, decision guide, video/LoRA claim downgrades (Q10, Q13–Q18).
- Repair LoRA control/evaluation scripts and generic continued-pretraining geometry validation.
- Make all LoRA evidence paths fail closed on missing/mismatched frequency artifacts, validate every rotary module, record path-safe artifact hashes, and reject stale checkpoints before reuse.

### P1 — acceptance-critical server queue

#### P1.1 Primary II exact-protocol replication (Q3)

- Model/data: same 125M FineWeb-Edu `L_train=128` protocol as Table 4.
- Methods: Geo, DAPE, EVQ tau=5.0.
- Seeds: add 137 and 256 to retained seed 42.
- Outputs: PPL@128 and PPL@8K, per-seed paired deltas, mean/std, exact configs and checkpoint hashes.
- Decision rule: if direction is not stable, re-tier Primary II as supporting and remove primary/headline wording. Do not substitute the distinct L=256 sweep.

#### P1.2 Tuned-base geometric control and b=10K rule test (Q5–Q6)

- Anchor: the same 125M FineWeb-Edu, `L_train=128`, `128→8K` protocol used by Table 4; hold architecture, data order, token budget, optimizer, seeds, and evaluation examples fixed.
- Geo bases: 10K, 100K, 500K, 2M.
- EVQ controls: retain EVQ at b=500K; at b=10K compare the bare rule and explicit `c_pred(L,b)` correction against Geo b=10K.
- Seeds: 42/137/256, shared with P1.1 where the training stack permits reuse.
- Metrics: PPL@128 and PPL@8K, plus NLL-gap retrieval only if that exact protocol already defines it.
- Decision rule: report best tuned Geo, not only b=500K; if tuned Geo closes the gap, narrow the claim from shape advantage to regime-dependent allocation control.

#### P1.3 MLA convention screen (Q7)

- Same 432M MLA, `L_train=8192`, `d_rope=32`, `d_head=128` protocol.
- One-seed screen: tau=0.354 (`d_rope/sqrt(L)`), tau=0.707 intermediate, tau=1.414 reported convention.
- Replication: only after the screen, replicate the relevant comparison across seeds 42/43/88.
- Metrics: PPL at 8K/16K/24K/32K, both raw and matched YaRN scale.
- Decision rule: if tau=0.354 is competitive/better, revise the MLA operating convention; regardless of outcome, retain `K=d_rope/2` for quantization.

#### Completed locally: Primary I autoregressive exact match (Q9)

- The tracked raw payload already contains the separate AR exact field for the exact three Primary-I seeds per method; no rerun is required for the rebuttal.
- At 8K, Geo+YaRN is 0.0% AR exact in all seeds, while EVQ+YaRN is 58.0% mean (58/18/98%).
- Report NLL-gap retrieval and AR exact side-by-side. The abstract may retain `100%` only with the explicit teacher-forced NLL-gap label.

#### P1.5 L_eff^J measurement (Q8)

- Evaluation-only on existing MLA and any available 16K/32K checkpoints.
- Implement the stated Eq. 41/43 estimator, record sampling layers/heads/tokens, and report kappa_att plus uncertainty across sampled units.
- Decision rule: if substituting `L_eff^J` materially changes the exponent/rule, present the measured correction; otherwise report the tolerance rather than claiming exact confirmation.

### P2 — high-value robustness

#### P2.1 1B-token MLA replication (Q11)

- Add seeds to the exact single-seed schedule-sensitive run.
- Save progression checkpoints and evaluate raw and matched-YaRN variants at fixed intervals.
- Decision rule: if raw reversal replicates, scope EVQ-alone gains to the training-budget regime and center the substrate×rescaler interaction.

#### P2.2 Realistic distance-prior analysis (Q12)

- Priors: uniform baseline, preregistered power-law family, and empirical trained-attention distance histograms.
- Refit the surrogate coefficients and recompute exact-kernel collision metrics.
- Report deviation from cosh; do not force a cosh conclusion.

#### P2.3 Learnable-tau trajectory recovery/rerun (Q4)

- First recover existing per-step tau logs if available; rerun only if the artifact is absent.
- Plot tau, train loss, in-range validation loss, and extrapolation PPL at matched checkpoints.
- Treat the waterbed explanation as supported only if the trajectory/objective evidence agrees.

#### P2.4 Corrected 8B matched LoRA control (Q18; proposed large-model work)

- Model: LLaMA-3-8B-Instruct to match the submitted row.
- Training: identical LongAlign data, steps, LoRA targets, rank, alpha, batch/token budget, seed, and optimizer.
- Methods: native Geo+LoRA, EVQ+LoRA; optionally unmodified base as a no-adaptation reference.
- Seeds: start with one smoke seed after code validation, then use matched seeds for the result retained in rebuttal/paper.
- Evaluation: in-distribution/extrapolation PPL plus RULER/NIAH with unique variant names; load the exact saved training-time inverse-frequency tensor.
- Gate: reject any run whose metadata says native Geo while the tensor is midpoint-quantized, or whose evaluation result lacks adapter/frequency provenance.
- Interpretation: this isolates EVQ from LoRA capacity. It does not by itself validate the rank threshold across models.

### P3 — follow-up, not rebuttal-critical

- New 7B-family cross-model rank sweep and matched Geo control (Q18 generality).
- Per-head tau jitter or EVQ+CARoPE pilot (Q14).
- Evaluate gamma=0.465 versus 0.500 at matched checkpoints (Q13).
- RF-schedule derivation for the video correction (Q17).
- Additional downstream accuracy tasks only after selecting a model scale with non-floor task performance.

## Artifact contract for every new result

- Immutable config, seed, git commit, environment, checkpoint hash, training log, and evaluation JSON.
- Method label must be derived from saved metadata, not a directory name.
- Paired methods use identical evaluation examples and decoding settings.
- Summaries report per-seed values before mean/std.
- Any result used in a rebuttal gets a small curated provenance manifest; no private server paths enter paper-facing files.
