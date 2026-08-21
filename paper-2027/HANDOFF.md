# ICLR 2027 active handoff

- **Updated:** 2026-08-21
- **Target:** ICLR 2027
- **Active manuscript:** `paper-2027/`
- **Branch:** `main_0726`
- **Status:** Opus-5 cross-review prose/flagship rewrite committed and pushed;
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
3. the three-seed $432$M scarce-channel MLA result is the systems flagship;
   $750$M full-parameter continuation and a $1.485$B
   same-initialisation/same-scientific-recipe comparison extend the
   training-stage and scale trend;
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
- Abstract: about 180 source words; one flagship effect size, the three-seed
  $432$M MLA $16$K PPL reduction of `31.1%`. The $1.485$B from-initialisation
  result remains trend evidence rather than the abstract headline.
- Page 2 opens with `fig_evidence_overview.pdf`:
  - three-seed fixed-support per-seed curves and mean;
  - three-seed $432$M, $K=16$ relative-PPL crossover;
  - $1.485$B 2Wiki token-F1 plus an $8$B adapted
    remote-block-ablation callout.
- The existing method-overview and frequency-geometry figures remain.
- The old identification and mature-crossover figures and their generators are
  removed; the unused mature table is removed.
- The M4 table is in the identification appendix; the body retains its
  `10/12` and `9/12` cross-configuration/shape result.
- $432$M MLA and $750$M full-parameter continuation are visible in the body.
- The prose declares endpoint/protocol roles once, then presents a hierarchy
  rather than repeating defensive `separate`/`matched` qualifiers. The main
  capability endpoint is 2Wiki token-F1; exact match remains in App. F.
- Finite-$\tau$ and matched-exponential detail is compressed in the body and
  retained in the appendix.
- Discussion says support and allocation are distinct but interacting; it does
  not claim additive gains under range retargeting. It now explains why scalar
  base search cannot reach non-geometric allocation and why the $K=16$ result
  makes the axis practically consequential.
- The AI-use statement remains unchanged. Its current wording was previously
  author-confirmed as complete and literally true; the external suggestion to
  assert author-only theorem statements/proof strategies was not adopted
  without a new factual confirmation.
- Related Work now closes the verified citation gaps without becoming an
  inventory: MrRoPE is positioned as training-free mixed-radix range
  conversion; Selective RoPE and Deconstructing Positional Information mark
  broader operator/logit analyses; RePo marks content-dependent position
  assignment; Kazemnejad et al. supplies the general length-generalization
  context. GRAPE, Urrutia et al., and xPos were already present.

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
| Final PDF | SHA-256 `29421c1678e797da0da4003c5692eeab9ea071b844aefc0062b9abacb0b7562d`; 688,299 bytes |
| Layout gates | 9 body pages, 29 total, US Letter, 0 undefined refs/cites, 0 pt overfull, anonymous |
| Fonts | Type-3 `0`; all fonts embedded |
| Outward terminology scan | no `GEO`, `RAMP`, `legacy scaler`, or `EVQ+YaRN` in manuscript/appendix sources; locked identities verified |
| Focused scientific tests | 243 passed |
| Paper-workspace tests | 3 passed |
| Anonymous package dry run | SHA-256 `46b72c3d7cbcfa68a248a06e8a5f38bf0a3ba376e20d420b323b934efaaf248b`; 851,750 bytes; ZIP integrity clean |
| Isolated package | all three figures regenerated; paper rebuilt; 142 tests passed |
| Visual QA | all 9 body pages inspected at rendered resolution; new Figure 1 labels and callout are readable |
| Immutable NeurIPS PDF | SHA-256 `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772` |

The final package dry run is
`/tmp/rope-spectral-budget-iclr2027-citation-repair.zip`. The existing root supplement
was not overwritten.

Use Conda `aidemo` for PyTorch/pytest checks.

## 7. Worktree and next action

- The Opus prose rewrite is local and uncommitted; do not publish it unless the
  user explicitly requests commit/push.
- `paper-2027/DOCUMENT_TEXT_MAP.md` is an unrelated untracked file that appeared
  during this turn. It was not created or modified by this work and must remain
  outside any future staging scope unless the user identifies its owner.
- `paper/` remains immutable and unchanged.
- Do not start new training or GPU evaluation from this handoff.
- For the next independent review, extract only concrete score-ceiling,
  technical-credibility, or readability defects and verify them against the
  PDF and owners before editing.
- Before submission: author visual review, live policy/deadline recheck,
  OpenReview title/abstract equality, and durable root-supplement regeneration.
