# ICLR 2027 active handoff

- **Updated:** 2026-08-20
- **Target:** ICLR 2027
- **Active manuscript:** `paper-2027/`
- **Branch:** `main_0726`
- **Status:** the decision-relevant manuscript pass is complete and the
  registered exact-range three-seed replication has finished successfully;
  its raw-hash-receipted internal owner is frozen, while manuscript integration
  remains the next authoring step
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

The first exact-range headline above is the current seed-42 manuscript text,
not the completed replication. The new three-seed owner reports
`-0.281/-0.176/-0.146` at `512/1K/2K`, with `3/3` training seeds favoring Cosh
at every OOD length and an in-domain cost of `+0.026` NLL. Use
`research/EXACT_RANGE_151M_3SEED_RESULT_20260820.md` and its companion JSON for
the next manuscript pass; do not retain or average the older author-confirmed
aggregate.

## 4. Implemented manuscript state

- Title: *RoPE Has a Spectral Budget*.
- Body: 9 pages; PDF: 29 pages total, US Letter.
- Four main-text figures:
  `fig_method_overview.pdf`, `fig_identification.pdf`,
  `fig_mature_crossover.pdf`, and `fig_frequency_geometry.pdf`.
- The frequency-geometry figure now appears beside the spectral-budget result;
  the redundant related-work taxonomy table was removed and replaced by an
  explicit three-part contribution statement.
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
- CoPE, RoPE-ID, and MHRoPE/MRoPE-I are now positioned against the exact
  fixed-support interior-allocation control.
- The 2026-08-20 optimization report's proposed protocol-invariant
  `1x--2x crossover law` was not adopted: it pools incompatible models,
  endpoints, metrics, and adaptations. Its proposed M4 redundancy--effect
  regression was independently checked and was flat: Spearman rho was
  approximately `0.05--0.08` with `p=0.81--0.87` for slow-pair count and
  canonical-rank deficit. Static geometry therefore remains a redundancy
  accounting, not a trained-effect predictor.

## 5. Internal negative results that must not become paper claims

### 5.1 LeRoPE profile oracle

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

### 5.2 Attention-measure kappa probe

The preregistered CPU-only Table 2 probe is complete on the original 1,920
head-query observations. The numerical direction `g` was frozen from the
Geo-weights/Geo-table reference cell; each of the four cells supplied only its
own attention distribution, and Eq. (37) used ratio-of-means aggregation.

- the first-order Eq. (35) direction gives Spearman `rho=+1.0` against log PPL
  and separates the two self-consistent cells from the two mismatches;
- the realised finite Geo-to-EVQ swap gives `rho=-0.2` and does not separate
  self-consistent from mismatched cells;
- the two rankings disagree, so the preregistered finite-tau rule selects
  Branch C: the attention-measure hypothesis fails its go/no-go gate.

The finite-swap expression matches direct rotated-logit subtraction to
`1.78e-15`, and a deterministic rerun reproduced the complete kappa payload
hash. The durable owner is
`research/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md`; raw receipts remain in
the ignored `../results/kappa_att/` layer. Do not promote the attractive
first-order ranking, search layer/head subsets, or tune a new direction.

Tier 2 was not executed: the canonical M4 owner records that the 12-config
weights were cleaned after evidence freezing, and no compatible checkpoints
remain locally. The surviving evidence JSON cannot recover attention
probabilities. The blocked audit is recorded in the durable owner; unrelated
weekend-sweep checkpoints must not be substituted.

## 6. Current validation receipt

Current artifacts:

| Artifact | Receipt |
| --- | --- |
| `main.pdf` | SHA-256 `ddc50f1b0af7546ada534cb9e358a5a40f6619de5f7b08f22e929408480bdda9`; 734,427 bytes |
| Existing anonymous supplement | `rope-spectral-budget-iclr2027-supplement.zip`; SHA-256 `f2af442ac27ad283fff1d5791c28c61bbed55af3e142ce723db8fc26830f68e9`; ZIP integrity clean; predates this documentation refresh |
| Current package dry run | SHA-256 `356806b4c2f74e32f797d58493ce600f50c8d3c39342b74a60f5d811e56aa20c`; 912,414 bytes; leak scan and ZIP integrity passed; internal `HANDOFF.md` excluded; existing user-owned archive was not overwritten |
| Build gates | 9-page body, 29 total, 0 undefined refs/cites, 0 pt overfull, anonymous, US Letter, Type-3=0, all fonts embedded |
| Focused scientific tests | 243 passed |
| Supplement tests | 25 passed |
| Isolated extracted package | paper build passed; 142 focused tests passed; all four figures regenerated |
| Exact-range three-seed run | wrapper exit `0`; seeds 137/256 both arms and evaluations complete; result JSON SHA-256 `6a5ab42b...` / `23a0dd06...`; no OOM/NaN/traceback |

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

The exact-range result-freeze batch is limited to the new internal report and
companion JSON plus this handoff and the research index. The user authorized
this documentation batch to be committed and pushed on `main_0726`; no raw
checkpoints, machine paths, ignored result trees, or `paper/` changes belong in
that commit.

Preserve all untracked analysis outputs and LaTeX build products unless the
user explicitly asks to remove or package them. `paper/` must remain
byte-for-byte unchanged.

The authorized 151.9M exact-range replication for seeds 137 and 256 is
complete. It used the shared prepared data, global batch 256, micro-batch 128,
accumulation 2, and `max-autotune-no-cudagraphs`; all four new arms ran at
approximately 183K token/s. Together with the historical seed-42 comparison,
the raw-hash-receipted aggregate is:

- fixed range, Cosh minus FMRoPE NLL:
  `-0.28073/-0.17599/-0.14571` at `512/1K/2K`, `3/3` seeds each;
- target-matched range: `+0.06032/+0.22720/+0.45959`, `0/3` Cosh wins;
- training length: `+0.02619` NLL for Cosh.

The whole-file manifest receipt differs from the historical seed-42 receipt;
the user confirmed the scientific experiment identity, and the owner records
the semantic protocol and paired receipts rather than requiring byte-identical
manifest metadata. Do not rerun for receipt-level symmetry.

## 8. Next action

For another independent AI cross-review:

1. extract only concrete alleged defects or score ceilings;
2. verify each against the PDF, source, theorem, and owner;
3. rank verified issues by acceptance impact, especially the positive-reviewer
   ceiling and the technical-reviewer ceiling;
4. propose the smallest replacement-level change for each real issue;
5. do not modify the manuscript for noise or unverified speculation.

For the next manuscript pass:

1. replace the seed-42 exact-range numbers in the abstract, introduction,
   identification figure, experiment paragraph, and appendix as one atomic
   change;
2. use `-0.281/-0.176/-0.146` and `3/3` direction consistency, retaining the
   `+0.026` in-domain cost and the target-matched boundary;
3. show per-seed points so the 512-length heterogeneity remains visible;
4. update the appendix batch geometry for seed 42 (`64x4`) versus seeds
   137/256 (`128x2`), all at global batch 256;
5. rebuild, package, and visually inspect after text and figure agree with the
   new owner.

Before submission, still requires author action:

- final visual review of all PDF pages;
- regenerate the durable root supplement after the final documentation freeze;
- live recheck of ICLR deadlines/policies;
- OpenReview title/abstract equality with the PDF;
- if NeurIPS accepts, third-person citation and an explicit old/new
  contribution boundary before the ICLR full-paper upload.
