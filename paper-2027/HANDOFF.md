# ICLR 2027 active handoff

- **Updated:** 2026-08-21
- **Target:** ICLR 2027
- **Active manuscript:** `paper-2027/`
- **Branch:** `main_0726`
- **Status:** acceptance-oriented narrative rewrite integrated and validated;
  attention-aware method discovery is complete for two internal seeds and
  remains uncommitted
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
   the $454$M three-seed YaRN-style composition restores the strongest completed
   substrate result; $750$M full-parameter continuation and a $1.485$B
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
- Abstract: 186 source words. It opens with the geometric redundancy tax
  (`23` slow pairs, `46` nominal dimensions, $r_2=2.00$), then gives the
  three-seed fixed-support identification and the $432$M MLA $16$K PPL
  reduction of `31.1%`. The $1.485$B from-initialisation result remains trend
  evidence; $8$B is explicitly adaptation.
- Page 2 opens with `fig_evidence_overview.pdf`:
  - three-seed fixed-support per-seed curves and mean;
  - three-seed $432$M, $K=16$ relative-PPL crossover;
  - $1.485$B 2Wiki token-F1 plus an $8$B adapted
    remote-block-ablation callout.
- The existing method-overview and frequency-geometry figures remain.
- The frequency-geometry heatmap now uses a white-to-blue sequential palette;
  the previous black low-redundancy field is removed and the slow-pair block is
  highlighted in orange.
- The old identification and mature-crossover figures and their generators are
  removed; the unused mature table is removed.
- The M4 table is in the identification appendix; the body retains its
  `10/12` and `9/12` cross-configuration/shape result.
- $432$M MLA and $750$M full-parameter continuation are visible in the body.
- The prose declares endpoint/protocol roles once, then presents a hierarchy
  rather than repeating defensive `separate`/`matched` qualifiers. The main
  capability endpoint is 2Wiki token-F1; exact match remains in App. F.
- Finite-$\tau$ is presented as a zero-search operating prior, not a basin
  bound or point-optimum selector. The ratios $0.75\times$, $1.25\times$, and
  $1.5\times$ are discrete tested neighbours; they do not certify every value
  in that interval.
- The full theory appendix is retained: geometry,
  transplant obstruction, surrogate validation, waterbed/self-consistency,
  operating-rule scaling, stiffness, $L_{\mathrm{eff}}^J$, Fisher forcing,
  and discrete-channel transport. The $3/9$ neighbour comparison is retained
  as evidence that the rule is not a dependable per-configuration optimum;
  its separate value is avoiding a search while often improving on Geo.
- Discussion says support and allocation are distinct but interacting; it does
  not claim additive gains under range retargeting. It now explains why scalar
  base search cannot reach non-geometric allocation and why the $K=16$ result
  makes the axis practically consequential.
- The target-matched reversal is framed next to its mechanism: retargeting
  changes absolute phase coverage, so target-specific support and allocation
  should be selected jointly; the fixed condition remains the identification
  owner.
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
- The abstract separates the $1.485$B from-initialisation trajectory from its
  independent Q/K adaptation; EVQ-Cosh's fast-end allocation direction is
  stated before the theory.
- Exact-range naming is locked to FMRoPE versus anchored EVQ-Cosh; `Geo`
  remains reserved for geometric training baselines, and `Native` for
  unmodified pretrained-model baselines.
- Figures, axes, captions, and tables spell out `EVQ-Cosh`; the repository
  range operator is always `YaRN-style`, and the run-specific MLA operator is
  always `MLA wavelength-blend operator`. Deprecated descriptive aliases are
  absent from outward sources.
- Proposition~2 now defines its asymptotic variable and softmax-support
  condition. Theorem~4 is explicitly a continuum-surrogate result whose
  deployed table is a finite midpoint quantisation.
- Figure~2 is fixed in place immediately after the Theory opening rather than
  floating above the section; Figure~1 now separates the $8$B LoRA deletion
  callout from the $1.485$B QA panel.
- The reviewer supplement now has a public README, the machine-readable
  exact-range three-seed aggregate, a runnable exact-range entrypoint, and no
  stale figures, internal handoff links, or broken packaged CI commands.

## 4. Canonical paper-facing values

### Three-seed fixed-support control

- Anchored EVQ-Cosh minus FMRoPE NLL at `256/512/1K/2K`:
  `+0.026/-0.281/-0.176/-0.146`.
- Every training seed favours anchored EVQ-Cosh at every OOD length.
- The $512$ magnitude is heterogeneous and remains visible as per-seed points.
- Target-matched means at `512/1K/2K`:
  `+0.060/+0.227/+0.460`; FMRoPE is favoured by `3/3` seeds.
- Owner: `research/EXACT_RANGE_151M_3SEED_RESULT_20260820.{md,json}`.

### Scale and capability roles

- $432$M MLA: three-seed from scratch, $K=16$; PPL `35.4/35.8` at $8$K
  and `138.8/95.6` at $16$K.
- $750$M: single-seed full-parameter $2$K$\rightarrow4$K continuation from a
  shared Geo checkpoint; not from scratch. PPL `45.1/24.4` at $16$K and strict
  AR exact `0/77.5%` at $8$K.
- $1.485$B from initialisation: same initialisation, architecture, scientific
  recipe, reconstructed data-order prefix, counted-token budget, and evaluation
  rows; different trainer implementations. Geo/EVQ-Cosh PPL is
  `177.99/191.36`, `161.19/167.45`, `163.88/156.87`, `182.73/159.64` at
  $2$/$4$/$8$/$16$K.
- $8$B: matched LoRA adaptation only. Keep its natural-text probability,
  RULER, attention-hit, and remote-block-ablation endpoints separate.

Seed inventory for the mature/supporting arms:

- the $50$M weights-by-table co-adaptation crossing, $750$M continuation, and
  $1.485$B from-initialisation comparison each currently have one paired
  training trajectory;
- the $1.485$B selective Q/K, 2Wiki, and RULER results use one matched
  Native/EVQ-Cosh trained pair; routing conversion has a second independent
  EVQ-Cosh seed but not a second complete Native/EVQ-Cosh pair;
- the $8$B natural-LM/remote-deletion and RULER protocols use one seed per arm;
- the main Video-DiT comparison has two seeds, while the base-1000 diagnostic
  uses seed 42.

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

## 6. Attention-aware method discovery

This work is internal and does not change the current manuscript.

- R0 measured attention-distance occupancy from causally masked attention on
  the mature Native OLMo-2 1.485B Instruct checkpoint and paired seed-137
  151.9M FMRoPE and anchored EVQ-Cosh checkpoints. The profiles were
  non-uniform but did not pass the registered multi-peak gate.
- Mapping distance bins directly to frequency demand was falsified: the
  seed-137 candidate worsened tail NLL at 512 and 1K.
- The operator-aware replacement uses
  $m(\phi)=\mathbb E[1-\cos(\omega(\phi)\Delta)]$ and
  $\rho\propto(0.9m+0.1)^{1/3}$ at fixed endpoints.
- Phase-chord minus FMRoPE tail NLL at 256/512/1K/2K is
  `-0.008/-0.237/-0.152/-0.215` for method-selection seed 137 and
  `+0.010/-0.084/-0.160/-0.195` for the schedule-frozen seed-42 confirmation.
  The two-seed means are `+0.001/-0.161/-0.156/-0.205`.
- Both seeds improve all three OOD endpoints over FMRoPE, and the average
  in-window cost is nearly removed. The strict gate of retaining at least 80%
  of anchored EVQ-Cosh gain at every OOD length fails because the seed-42 512
  effect is smaller. Decision: `PROMISING_PARETO_SHIFT_NOT_PAPER_READY`.
- Canonical internal navigation:
  `research/attention-aware-retrofit/README.md`; completed report:
  `research/attention-aware-retrofit/EXPERIMENT_REPORT_20260821.md`; compact
  machine-path-free result receipt: `research/attention-aware-retrofit/evidence/RESULTS_20260821.json`.
- Seed 42 completed 7,629 steps and 499,974,144 tokens; its four-length
  evaluation completed. No seed 256, R1, R3, or mature-model adaptation was
  launched. The authorised GPU instance was shut down after artifacts were
  receipted locally; raw machine outputs remain on its stopped data volume and
  no remote run remains active.

## 7. Validation receipt

| Check | Result |
| --- | --- |
| Final PDF | SHA-256 `3de9a4c2cb9acb6d9135143ca6055335d35ca2b7619531027a4dda7eff17c4a7`; 691,054 bytes |
| Layout gates | 9 body pages, 30 total, US Letter, 0 undefined refs/cites, 0 pt overfull, anonymous |
| Fonts | Type-3 `0`; all fonts embedded |
| Outward terminology scan | no `uniform FMRoPE`, `anchored Cosh`, `smooth-ramp scaler`, `range scaler`, `MLA blend`, `RAMP`, or `EVQ+YaRN`; locked identities verified |
| Focused scientific/package/workspace tests | 223 passed |
| Anonymous root package | SHA-256 `00170422f0e4449103b4b1751e231736f77d0f167f2e2a480a14c4d27e580cef`; 842,953 bytes; ZIP integrity clean |
| Isolated package | exact-range entrypoint opened; all three figures regenerated in the dry run; paper rebuilt; 142 tests passed |
| Visual QA | all 9 body pages inspected; Figure 1 callout is legible, Figure 2 follows the Theory heading, and Figure 3 has no black heatmap field |
| Immutable NeurIPS PDF | SHA-256 `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772` |
| Attention-aware internal checks | 33 focused tests passed; phase-demand self-test and report-metric assertions passed |

The final reviewer package is
`rope-spectral-budget-iclr2027-supplement.zip` at the repository root.

Use Conda `aidemo` for PyTorch/pytest checks.

## 8. Worktree and next action

- The acceptance-audit integration is local and uncommitted; do not publish it
  unless the user explicitly requests commit/push.
- `AGENTS.md` contains the user-owned acceptance-first rules plus the corrected
  finite-tau, nomenclature, disclosure, and research-frontier guardrails.
- `paper/` remains immutable and unchanged.
- Do not start new training or GPU evaluation from this handoff.
- Immediate submission work must first make the completed evidence as strong
  and readable as possible without unsupported SOTA language.
- The next research question is mature-model preservation, not another blind
  whole-table swap. Reuse the existing Stage-D leave-one-pair-out attention-KL
  diagnostic to determine whether Native dependence is localised by pair,
  head, or layer; do not assume a 50/50 `d_head` split. Compare the smallest
  measured protected-complement design with a compact Native-plus-phase
  residual before matched 1.485B LoRA.
- If phase-chord is to become a paper-facing training-time method, seed 256 is
  its smallest missing replication. Do not launch it or mature-model GPU work
  without a frozen protocol and explicit authorisation.
- Treat the present $8$B results as evidence, not as the solved retrofit: the
  natural-LM arm improves long-position PPL and routing but does not establish
  simultaneous in-window retention and downstream improvement. Do not schedule
  $8$B multi-seed by default; reconsider it only after the new method passes
  smaller gates or suitable higher-memory hardware is explicitly authorised.
- For the next independent review, extract only concrete score-ceiling,
  technical-credibility, or readability defects and verify them against the
  PDF and owners before editing.
- Before submission: author visual review, live policy/deadline recheck, and
  OpenReview title/abstract equality.
