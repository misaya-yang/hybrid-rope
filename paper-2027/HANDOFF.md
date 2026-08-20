# ICLR 2027 active handoff

- **Updated:** 2026-08-20
- **Target:** ICLR 2027
- **Active manuscript:** `paper-2027/`
- **Branch:** `main_0726`
- **Status:** manuscript optimization complete for the current evidence set;
  waiting for user-supplied independent AI cross-reviews before another
  decision-relevant revision pass
- **Internal only:** this file must not enter the anonymous supplement

## 1. First principle

The first principle and highest priority is to maximize the probability of
ICLR 2027 acceptance. Scientific truth, submission validity, and anonymity are
hard feasibility constraints. Everything else—including completeness,
symmetry, more experiments, more caveats, and preserving earlier prose—is
subordinate.

The next revision is justified only if a verified issue can:

1. cap a positive reviewer below the next score;
2. keep a technical reviewer from moving up;
3. prevent a human reviewer from understanding or trusting the central claim;
4. invalidate the submission mechanically or ethically.

Generic novelty percentages, speculative alternative methods, and benchmark
wish lists without a score-changing path are noise.

## 2. Read order

1. `../AGENTS.md`
2. this handoff
3. `research/README.md`
4. `research/ICLR2027_RESEARCH_SYNTHESIS_20260819.md`
5. `research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`
6. the current manuscript section and the canonical owner for every number

The old `../rebuttal/rebuttal_0723/REBUTTAL_HANDOVER.md` is historical
NeurIPS material and is not the current queue.

## 3. Current paper in one paragraph

The paper now opens with `x_k = -log(omega_k) = a + R z_k`: sampled support
`(a,R)` and normalized interior allocation `z` are separate coordinates.
Exact-range and M4 show that changing only `z` changes trained behaviour.
Full sin/cos subspace geometry explains finite-table redundancy, while the
exact transplant obstruction and 50M 2x2 crossing show that weights co-adapt
to the training table. EVQ-Cosh is one closed-form,
zero-learned-parameter construction on this axis. The 1.485B and 8B studies
then show the same effective-context shift at mature scale.

### Evidence chain

| Layer | Owner | Paper role |
| --- | --- | --- |
| Pure allocation | 151.9M exact-range + 50.9M M4 factorial | Identifies `z` independently of sampled support |
| Static theory | full sin/cos Gram, canonical correlations, exact stable-rank identity, low-frequency collapse | Defines finite spectral-basis redundancy without claiming an LM predictor |
| Co-adaptation | exact fixed-Q/K obstruction + 50M table-by-weights crossing | Explains why a frozen table swap is not a pure geometry intervention |
| Mature scale | 1.485B from-initialization, OLMo Q/K-only, matched 8B 300-step LoRA, separate RULER adaptations | Establishes persistence and protocol-specific capability conversion |
| Range composition | same fixed-scale `YaRN-style` transform at 454M | Shows substrate-dependent leverage; not a tuned official-YaRN benchmark |

The strongest human-readable headlines currently in the manuscript are:

- exact-range OOD NLL delta
  `-0.478/-0.205/-0.113` at `512/1K/2K`;
- M4 non-uniform directions for Cosh and matched exponential in
  `10/12` and `9/12` configurations;
- self-consistent 50M weight/table PPL `7.14/7.16` versus post-hoc swaps
  `76.20/23.05`;
- matched 8B 300-step LoRA PPL `108.96 -> 24.07` at 16K;
- matched OLMo Q/K-only 2Wiki exact `22.0/21.5%` at 4K and
  `0/17.5%` at 8K.

Do not pool these protocols or imply that one result owns another result's
causal claim.

## 4. Implemented manuscript state

- Title: *RoPE Has a Spectral Budget*.
- Body: 9 pages; PDF: 29 pages total, US Letter.
- Four active figures:
  `fig_method_overview.pdf`, `fig_identification.pdf`,
  `fig_mature_crossover.pdf`, and `fig_frequency_geometry.pdf`.
- The first page explicitly separates support from allocation and shows that
  anchored/deployed Cosh have the same normalized `z`.
- Identification, co-adaptation, mature-scale, and range-composition evidence
  are separate causal layers.
- The learned inverse-frequency row is correctly treated as a learned table,
  not DAPE and not an allocation-shape owner.
- `YaRN-style` is used as the current name and scientific role. Do not spend
  a revision pass renaming it again unless a verified reviewer issue requires
  it.
- LeRoPE is cited and positioned as learned/fixed-table evidence, not as
  mechanism validation or a matched comparison.

## 5. Internal negative result that must not become a paper claim

The CPU-only LeRoPE profile-oracle test is complete and falsifies the proposed
shortcut. With A.15-compatible unsigned structural softmax curvature,
`rho proportional to w^(1/3)` moves farther toward fast bands rather than
reproducing LeRoPE's slow-tail profile:

- shape RMSE to EVQ: `0.17655`;
- shape RMSE to LeRoPE: `0.34686`;
- EVQ-to-LeRoPE projection: `-0.95657`.

Owner: `research/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md`.

Correct conclusion: this unsigned local curvature is not the required
trajectory-aware LM-risk object. Do not turn the negative audit into outward
mechanism prose or a new optimization campaign.

## 6. Current validation receipt

Current artifacts:

| Artifact | Receipt |
| --- | --- |
| `main.pdf` | SHA-256 `185af7984cd23fce6e118fb40c67f7e5ac9ee6f4d8d8dc9b135d05c63d9a28e8`; 736,159 bytes |
| Existing anonymous supplement | `rope-spectral-budget-iclr2027-supplement.zip`; SHA-256 `f2af442ac27ad283fff1d5791c28c61bbed55af3e142ce723db8fc26830f68e9`; ZIP integrity clean; predates this documentation refresh |
| Documentation-refresh package dry run | SHA-256 `dc093b9fc173eb4e46ac6723abd731b7c3055d1cb790506639a55010fe258388`; leak scan and ZIP integrity passed; internal `HANDOFF.md` excluded; existing user-owned archive was not overwritten |
| Build gates | 9-page body, 29 total, 0 undefined refs/cites, 0 pt overfull, anonymous, US Letter, Type-3=0, all fonts embedded |
| Focused scientific tests | 243 passed |
| Supplement tests | 25 passed |
| Isolated extracted package | paper build passed; 142 focused tests passed; all four figures regenerated |

The immutable NeurIPS baseline remains `../paper/main.pdf`, SHA-256
`fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`.

Use Conda `aidemo` for Python/PyTorch checks:

```bash
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_rope_core.py \
  tests/test_rebuttal_protocol_regressions.py \
  tests/test_rebuttal_evidence_bundle.py \
  tests/test_fmrope_125m_l256_500m.py \
  tests/test_frequency_adaptation_8b.py \
  tests/test_olmo2_1b_evq.py -q
```

Build and package:

```bash
(cd paper-2027 && ./compile.sh)
conda run --no-capture-output -n aidemo \
  python scripts/package_supplement.py --profile iclr2027
```

## 7. Worktree and authority boundary

At this snapshot, local `main_0726` and `origin/main_0726` point to
`db7e670490da096916d9bb02ba55c42f9b1109ca`, but the worktree intentionally
contains a large uncommitted manuscript/figure/package batch. No commit, push,
stage, reset, stash, branch switch, or cleanup is authorized by this handoff.

Preserve all untracked analysis outputs and LaTeX build products unless the
user explicitly asks to remove or package them. `paper/` must remain
byte-for-byte unchanged.

No GPU training, GPU evaluation, paid run, or remote action is currently
authorized.

## 8. Next action

Wait for the user to provide the independent AI cross-review reports. Then:

1. extract only concrete alleged defects or score ceilings;
2. verify each against the PDF, source, theorem, and owner;
3. rank verified issues by acceptance impact, especially the positive-reviewer
   ceiling and the technical-reviewer ceiling;
4. propose the smallest replacement-level change for each real issue;
5. do not modify the manuscript for noise or unverified speculation.

Before submission, still requires author action:

- final visual review of all PDF pages;
- regenerate the durable root supplement after the final documentation freeze;
- live recheck of ICLR deadlines/policies;
- OpenReview title/abstract equality with the PDF;
- if NeurIPS accepts, third-person citation and an explicit old/new
  contribution boundary before the ICLR full-paper upload.
