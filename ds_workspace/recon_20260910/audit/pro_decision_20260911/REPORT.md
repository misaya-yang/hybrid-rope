# Pro proposal and experiment audit — 2026-09-11

## Decision

Proceed with task-decision calibration as a research direction, after repairing
the execution path. It is not an already validated optimizer. Preserve b4wide's
conditional gain and the original task metric. Do not inherit the claim that
OLMo and Qwen exhibit an established opposite NLL response.

The author subsequently authorized ongoing experiment ownership and repairs;
see `../../RESEARCH_OWNER_20260911.md`. Audit snapshots here preserve the code
as inspected, and must not be mistaken for corrected production versions.

## Confirmed defects and evidence

### P1: 180 held-out rows contain only 120 unique prompts

Disambiguating row IDs did not create independent samples. Hashing the actual
token inputs reveals 60 duplicate groups: 60 short rows reduce to 40 unique
prompts, and 120 long rows reduce to 80. Duplicate groups have identical scores
for every audited arm. `holdout180.py:14-15` treats ID disambiguation as a valid
union, which inflated the nominal sample size and reweighted duplicate cases.

The historical numbers below faithfully describe the stored rows, not 180
independent examples. `check_results.json.unique_holdout` reports unique-prompt
contrasts, task cell counts, equal-task macro deltas and stratified SEs. The
simple unique-prompt mean gives b4wide +8.125pp at 16K (SE 4.211pp), -4.542pp
at 4K; whole-row long gain is +6.25pp. The useful direction remains; the old
significance does not. Step42's unique long mean is -3.667pp.

New calibration selection explicitly deduplicates actual prompt tokens. Fresh
evaluation must exclude all old prompt identities, not just renamed row IDs.

### P1: The YaRN coordinate used in the theory is not the official code's ramp

The pinned official YaRN implementation uses a linear ramp in **channel index**:
with t=(j-lo)/(hi-lo), `nu'/nu = 1-(1-1/s)t`, so
`m=-log_s(1-(1-1/s)t)`. Its first and second derivatives in t are positive;
interior m increments increase, not decrease.

MrRoPE Appendix A.2.1 instead substitutes a linear blend in **rotation count**
`r_j=W*nu_j/(2*pi)`. Its geometric r_j sequence yields a different frequency
table and the appendix's regressive argument. For OLMo 4x, official-code YaRN
has sum(m)=38.49176; the rotation-count blend has 42.91175. The corresponding
Qwen values are 30.10419 and 34.31387. See `formula_results.json`.

Therefore the prior “YaRN front-loaded vs MrRoPE back-loaded” narrative cannot
be copied onto the actual official-code table without qualification. This
does not disprove MrRoPE's reported task gains; it changes which actual
operators need comparison to explain them. The new probe names the two YaRN
operators separately and uses a common gain.

### P1: Old evq_shift combines midpoint displacement with endpoint native

`experiments/curvature_20260910/tables.py:53-55,134-144` subtracts
u=(j+.5)/K, while `m_to_inv_freq` reconstructs from native u=j/K. Consequently
the evaluated frequency is `theta**(1/(2*K))*theta**(-phi_midpoint)`, not the
claimed canonical midpoint EVQ grid. For OLMo the multiplier is 1.10795776;
for Qwen it is 1.11397386, at every slot and every tau.

This is a legitimate *midpoint displacement applied to endpoint native*
intervention if named as such; it is not a faithful literal canonical
replacement. The three old zero scores must retain that narrower identity.
New code distinguishes literal midpoint EVQ and endpoint-anchored EVQ, where
tau=0 exactly recovers the pretrained native index grid.

### P1: Qwen NLL winner is reversed in the readout

`qwen4x_power_read.py:122-128` prints MrRoPE as the winner when
`mean(BM - MrRoPE) < 0`. Lower NLL is better: this is a BM win. Both branches
are reversed. The script additionally promotes this mistake into a cross-model
mechanism conclusion. A six-sample CPU fixture where BM is better on every
sample by about .02 nats causes the reader to announce MrRoPE wins, t=-34.64.

This affects the verdict, not the stored NLL arrays. Correct the branch labels
and recompute the contrast; never rerun generation just to repair a sign label.

### P1: Cluster standard error belongs to a different estimator

`qwen4x_power_read.py:55-63,98-102` divides the piece-weighted mean by the SE of
equally weighted book means. Book chunk counts are unequal (one book supplies
8 chunks, another 4). The estimator and SE must use the same weighting.

For piece-weighted mean mu, use book residual sums S_b=sum_{i in b}(d_i-mu):
`SE = sqrt(B/(B-1) * sum_b S_b**2 / N**2)` (intercept-only cluster correction),
or report the mean and SE of book means together as a distinct estimand. A
fixture gives the old SE=.57735 versus piece-mean cluster SE=.320725.
No direction of actual significance change is assumed before reading the data.

### P1: Generic differentiable rotary is not an OLMo arithmetic twin

`experiments/curvature_20260910/model.py:127-151` unconditionally returns cos/sin
cast to the input dtype. The live Transformers OLMo2 rotary returns FP32 and
casts Q/K back only after FP32 rotary application. On CPU, with BF16 inputs,
the old patch changes cos dtype from FP32 to BF16; max cos difference is
.00195223 on positions 16000–16031. It also omits the stock disabled-autocast
region. The fixture tests rotary arithmetic, not full checkpoint accuracy.

Stock `Olmo2RotaryEmbedding.forward` has `@torch.no_grad()`, so simply enabling
gradients on inv_freq does not solve this. A faithful model-specific derivative
path is required. Existing `JointDesign` correctly carries full-network
derivatives in principle, but its wrapper inherits the arithmetic mismatch.
The direct `olmo_beta.py` task runner does not use this generic patch; this
finding does NOT invalidate all saved task scores or all finite-difference
diagnostics. Current remote fisher/marginal/kkt scripts install tables directly.

### P2: Qwen's “native” reference uses modified gain

`qwen_longnll.py:124-126,142` sets gain=1.138629436111989 even for the native
frequency grid. This is a valid *same-gain frequency control*, not the original
checkpoint at g=1. Consequently, claims about how much native ability needs
rescuing cannot use this difference as if only frequency had changed relative
to the untouched model. BM-versus-MrRoPE at the same gain remains interpretable.

The 131K job also receives `far=131073` for source arrays of length 131073,
while the runner requests `far+1` tokens by concatenating arrays. Thus each
sample's last target is borrowed from the following chunk, the last chunk is
lost, and some samples cross books. This is a mismatch to the declared
within-book tail-NLL experiment. Do not use it to settle tiny effects without
corrected same-book input/target alignment.

### P2: Resume and archive comparison do not enforce claimed identity

`olmo_beta.py:221-232,270-291` checks the archive row-id set, but not prompt
hashes as claimed in its header. Resume skips by row ID only, even after a
change of panel content, table, gain, or decoder under the same output name.
Records contain no generated token IDs, prompt hash, or run identity.

Confirmed anomaly: archived MrProBM and the later same-named BM/g=1.1386 run
differ on 24/350 scores and 210/350 texts, despite close means (.416714 vs
.419000). Both sets rescore correctly. Their frequency construction differs
at float32 rounding level; attribution of all output differences needs replay,
not speculation. `runtime.table_identity` hashes the full table dictionary,
whereas `tensor_sha256` hashes the frequency bytes; compare like with like.

For Pro's exact-output constraints, regenerate the selected targets under the
verified execution path and save token IDs, EOS/stopping metadata, table, model
and decoder identity. Text re-tokenization alone is not a proof of original
token-sequence identity.

### P2: Natural-task reference-count stratification is broken

`natural_read.py:133-134` reads references from generated-output records, which
do not store that field; fallback `[1]` assigns every row to a one-reference
group. Join references from the input panel. Reference count alone also does
not establish whether the scoring rule is binary, recall, or F1.

### Interpretation errors that are not failed experiments

- Step42's held-out failure is not solely short-context pooling: at 16K it is
  -1.597pp and -5.0pp whole-row against BM.
- b4wide is different: its real long-context gain survives in both metrics.
- Repeating one greedy arm establishes reproducibility for that path, not a
  zero floor for all numerical/execution differences or new-example uncertainty.
- Similar broken examples do not prove a unique common mechanism.
- Failing a collection of geometric objectives does not rule out every useful
  frequency rule; a fixed 4x table failing at 8x does not isolate a unique cause.

## Recomputed useful results

All 900 saved outputs below rescore exactly under the original panel and
RULER scorer; row-ID sets contain no duplicates or omissions. Actual prompts
do contain duplicates, as corrected above; this table is historical row-weighted
arithmetic, not the corrected independent-sample result.

| Arm vs BM | 4K task delta | 16K task delta | 16K whole-row delta |
|---|---:|---:|---:|
| b3 | -5.583pp | +2.181pp | -2.500pp |
| a1b64 | -4.111pp | +7.833pp | -0.833pp |
| b4wide | -6.417pp | +9.611pp | +6.667pp |
| step42 | -6.917pp | -1.597pp | -5.000pp |

b4wide repairs 11 and breaks 3 whole rows at 16K; at 4K it repairs 3 and breaks
10. These are usable calibration conflicts. Reusing them for method selection
makes them development evidence, not a fresh final test.

All 700 BM gain outputs also rescore exactly: g=1 gives .037143, g=1.1386 gives
.419000. The large gain response is real for these saved outputs.

## What to adopt and change in Pro's proposal

The greedy prefix induction is correct with identical inputs, token IDs,
positions, decoder processing and stopping behavior. Strict positive margins
are sufficient, not necessary: ties can select the desired token, and different
outputs can earn the same or better score. Constraints on every token of one
chosen output can be unnecessarily restrictive. Infeasibility of a local
linearization is not impossibility of task repair.

Use logits *after the configured processors* when processors affect selection.
In the audited OLMo config the ordinary greedy assumptions hold, but preserve
the config explicitly. Compare the full prefill plus teacher-forced output path
with actual cached greedy generation; floating-point kernels can change ties.

The 65-parameter count does not make the full Jacobian cheap: one backward
gives one scalar margin gradient, not all token-by-vocabulary margin rows.
Use the most competitive violating token at each output position, recheck all
vocabulary competitors after a proposed step, and add constraints as needed.
Keep full-model derivatives; use memory checkpointing/chunked vocabulary heads
without detaching prefix states. Trust-region acceptance uses true decoding.

Use b4wide as an OLMo starting point and comparable healthy MrRoPE/BM points on
Qwen. Fit a gain-only control on the same calibration examples as the frequency
calibration; otherwise any benefit might simply be gain fitting. Report raw
task-weighted short/long scores and retain useful tradeoffs. Weight-frozen
task calibration is not a data-free closed-form rule and must be labeled so.

## Reproduction and scope

`check_remote.py` runs on CPU via SSH stdin, uses existing raw artifacts and
isolated temporary synthetic fixtures, and writes `check_results.json` locally.
It does not allocate GPU memory or alter running jobs. The generic model source
is staged in `/tmp/rope_pro_audit_20260911/curvature_model.py` for its rotary test.
Stage the preserved `qwen4x_power_read.py` snapshot as
`/tmp/rope_pro_audit_20260911/qwen4x_power_read_before.py`; the live reader has
subsequently been corrected. Both originals are retained in this audit folder.

## Repairs and execution after author takeover authorization

- `FrozenRoPE` now removes the stock outer no-grad wrapper instead of replacing
  the arithmetic with a Qwen-specific copy. Seven CPU tests pass across OLMo2
  and Qwen2 (FP32/BF16, full forward/greedy equality, directional derivatives,
  next-token NLL gradient alignment).
- Corrected Qwen reader passes four additional CPU tests and is installed at
  the old reader path. Its outputs explicitly retain the legacy protocol limits.
- Misaligned Qwen 131K NLL job was stopped with identity-checked SIGTERM after
  detached SIGINT was ignored. Native and completed MrRoPE arrays are retained;
  no data was removed. This run cannot supply a complete BM comparison.
- New owner directory: `/root/autodl-tmp/rope_decision_20260911`. The first probe
  uses four unique observed conflicts, preserves token IDs and table identity,
  compares true YaRN definitions and EVQ grids, and checks full-path gradients.
- Initial probe launch rejected Transformers' `None` defaults as non-greedy;
  this guard was corrected to accept the effective default semantics. The
  failed attempt receipt is preserved; it generated no scores.

This audit verifies these code paths and stored outputs. It is not a claim
that every script from every historical campaign was audited or that all
aggregate score differences have a known cause.

Primary references checked:

- [PyTorch no_grad semantics](https://docs.pytorch.org/docs/stable/generated/torch.no_grad).
- [Transformers greedy generation](https://github.com/huggingface/transformers/blob/main/docs/source/en/generation_strategies.md).
- [MrRoPE v1 method and appendix](https://arxiv.org/html/2601.22181v1).
- [YaRN official implementation, pinned source](https://raw.githubusercontent.com/jquesnelle/yarn/995db5b/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py).
