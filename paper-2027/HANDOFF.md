# ICLR 2027 active handoff

- **Updated:** 2026-08-20
- **Target:** ICLR 2027
- **Active manuscript:** `paper-2027/`
- **Branch:** `main_0726`
- **Status:** result-first evidence-chain rewrite complete and verified;
  publication state is owned by Git history and the remote branch
- **Internal only:** exclude this file from the anonymous supplement

## 1. First principle

Maximise ICLR 2027 acceptance probability inside three hard constraints:
scientific truth, submission validity, and decision leverage. Add or retain
material only when it changes a likely reviewer ceiling, technical credibility,
human comprehension, or venue validity.

## 2. Current paper story

The paper separates sampled support from normalised interior allocation,

\[
x_k=-\log\omega_k=a+Rz_k.
\]

The outward evidence chain is now:

1. a raw-hash-receipted three-seed $151.9$M fixed-support control identifies
   $z$ independently of sampled support;
2. full sin/cos subspace geometry, the exact stable-rank identity, the exact
   transplant obstruction, and the $50$M table-by-weights crossing explain why
   the table is learned with the weights;
3. three-seed $432$M scarce-channel MLA, $750$M full-parameter continuation,
   and a $1.485$B same-initialisation/same-scientific-recipe comparison show
   training-stage and scale persistence;
4. matched $1.485$B Q/K-only adaptation supplies real-document QA and RULER;
   separate matched $8$B LoRA supplies mature-model probability and causal
   remote-source-use evidence;
5. target-aware retargeting is stronger in the target-matched exact-range
   condition, so support and allocation are distinct but interacting rather
   than additive coordinates.

The training-scale trend stops at $1.485$B. The $8$B results are adaptation
evidence and must never be described as pretraining-scale evidence.

Outward-facing text follows two locked presentation rules. It narrows claims
instead of volunteering internal negatives or speculative objections, while
retaining every disclosure required for scientific truth and venue validity.
It also keeps protocol identities distinct: `Geo` for geometric training
baselines, `Native` for unmodified pretrained-model baselines, `YaRN-style`
for the repository fixed-index range operator, `YaRN` for the cited method,
and `MLA wavelength-blend operator` for the run-specific MLA transform.

## 3. Implemented manuscript state

- Title: *RoPE Has a Spectral Budget*.
- Abstract: 184 words; one flagship comparison, $1.485$B PPL
  `182.73 -> 159.64` at $16$K; no compressed result ledger.
- Page 2 opens with `fig_evidence_overview.pdf`:
  - three-seed fixed-support per-seed curves and mean;
  - direct Geo/EVQ $1.485$B PPL crossover;
  - $1.485$B 2Wiki exact plus a separately labelled $8$B adapted
    remote-block-ablation callout.
- The existing method-overview and frequency-geometry figures remain.
- The old identification and mature-crossover figures and their generators are
  removed; the unused mature table is removed.
- The M4 table is in the identification appendix; the body retains its
  `10/12` and `9/12` cross-configuration/shape result.
- $432$M MLA and $750$M full-parameter continuation are visible in the body.
- Finite-$\tau$ and matched-exponential detail is compressed in the body and
  retained in the appendix.
- Discussion says support and allocation are distinct but interacting; it does
  not claim additive gains under range retargeting.

## 4. Canonical paper-facing values

### Three-seed fixed-support control

- Cosh minus uniform NLL at `256/512/1K/2K`:
  `+0.026/-0.281/-0.176/-0.146`.
- Every training seed favours Cosh at every OOD length.
- The $512$ magnitude is heterogeneous and remains visible as per-seed points.
- Target-matched means at `512/1K/2K`:
  `+0.060/+0.227/+0.460`; uniform FMRoPE is favoured by `3/3` seeds.
- Owner: `research/EXACT_RANGE_151M_3SEED_RESULT_20260820.{md,json}`.

### Scale and capability roles

- $432$M MLA: three-seed from scratch, $K=16$; PPL `35.4/35.8` at $8$K
  and `138.8/95.6` at $16$K.
- $750$M: single-seed full-parameter $2$K$\rightarrow4$K continuation from a
  shared Geo checkpoint; not from scratch. PPL `45.1/24.4` at $16$K and strict
  AR exact `0/77.5%` at $8$K.
- $1.485$B from initialisation: same initialisation, architecture, scientific
  recipe, reconstructed data-order prefix, counted-token budget, and evaluation
  rows; different trainer implementations. Geo/EVQ PPL is
  `177.99/191.36`, `161.19/167.45`, `163.88/156.87`, `182.73/159.64` at
  $2$/$4$/$8$/$16$K.
- $8$B: matched LoRA adaptation only. Keep its natural-text probability,
  RULER, attention-hit, and remote-block-ablation endpoints separate.

## 5. Internal negative results

- The LeRoPE structural-curvature oracle moves away from the published LeRoPE
  profile; do not promote it.
- The finite-swap $\kappa_{\mathrm{att}}$ probe fails the preregistered Tier-1
  ordering gate; do not promote the attractive first-order ranking or search
  subsets.
- Static rank remains positional-basis accounting, not an LM-quality or
  extrapolation predictor.

Owners:
`research/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md` and
`research/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md`.

## 6. Validation receipt

| Check | Result |
| --- | --- |
| Final PDF | SHA-256 `098eed6dc7bbf5606df99831be010a054d882eb6934f8226be1e1b789c855a05`; 683,696 bytes |
| Layout gates | 9 body pages, 29 total, US Letter, 0 undefined refs/cites, 0 pt overfull, anonymous |
| Fonts | Type-3 `0`; all fonts embedded |
| Outward terminology scan | no `GEO`, `RAMP`, `legacy scaler`, or `EVQ+YaRN` in manuscript/appendix sources; locked identities verified |
| Focused scientific tests | 243 passed |
| Paper-workspace tests | 3 passed |
| Anonymous package dry run | SHA-256 `9cc039bd88956bed2fce73abb02a810312aba9dda06c55f1e02cfcb2adfb1621`; 851,673 bytes; ZIP integrity clean |
| Isolated package | all three figures regenerated; paper rebuilt; 142 tests passed |
| Visual QA | all 9 body pages plus exact-range appendix tables inspected; Figure 1 legend/callout collision corrected and rechecked |
| Immutable NeurIPS PDF | SHA-256 `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772` |

The final package dry run is
`/tmp/rope-spectral-budget-iclr2027-reviewed.zip`. The existing root supplement
was not overwritten.

Use Conda `aidemo` for PyTorch/pytest checks.

## 7. Worktree and next action

- This handoff describes the validated publication state; verify the current
  local and remote SHA before any later release action.
- `paper/` remains immutable and unchanged.
- Do not start new training or GPU evaluation from this handoff.
- For the next independent review, extract only concrete score-ceiling,
  technical-credibility, or readability defects and verify them against the
  PDF and owners before editing.
- Before submission: author visual review, live policy/deadline recheck,
  OpenReview title/abstract equality, and durable root-supplement regeneration.
