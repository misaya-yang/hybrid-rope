# July rebuttal ignored-asset reconciliation and company handoff

## Executive decision

This document is the portable handoff for the July rebuttal audit. It reconciles the useful local files hidden by `.gitignore`, records what was promoted into tracked reviewer-safe artifacts, and maps the recovered evidence to all 18 Fable5 questions.

The central rule is simple: local existence is not the same as reviewer-grade provenance. Raw ignored files remain local; only anonymous, minimal, hash-identified evidence is tracked. A missing artifact means “not recovered in this checkout,” not “the experiment never ran.” No new experimental result is claimed here.

Current rebuttal readiness remains `draft_with_placeholders`. The corrected bundle now includes the full Primary I raw payload, the fixed-EVQ portion of the Primary II supplementary seeds, and a byte-exact reconstruction of the Primary III evaluator JSON. It does not close experiments requiring missing matched controls, checkpoints, or new evaluation.

## What was inspected

The inventory below is the first 2026-07-10 checkout snapshot. Sizes and counts are operational diagnostics, not scientific evidence. A second checkout contained an 86 GB `results/` tree with 1,736 files (1,700 ignored), dominated by checkpoints; it supplied the exact QuALITY, Phase11, Phase16, and base-pilot raw sources promoted below. The size difference does not change the policy: only minimal sanitized snapshots are tracked.

| Ignored family | Local snapshot | Rebuttal value | Decision |
| --- | ---: | --- | --- |
| `07 - rebuttal/` | 16 MB; 1,002 files; 57 JSON files | Copy-only audit workspace, archival branch material, code mirrors, copied reports/results | Keep ignored. Promote only selected result payloads with exact hashes. |
| `RESULT_PROVENANCE_MANIFEST.md` | 28 KB | Broad local forensic manifest | Keep ignored. Synchronize only sanitized conclusions into `docs/overview/RESULT_PROVENANCE_MANIFEST.md`. |
| `.codex_tmp/` | 123 MB; 1,200 files | Temporary launchers/scripts and a third-party VideoRoPE mirror | Keep ignored. Scripts/traces do not prove completed results; no third-party mirror is promoted. |
| `results/` | 39 MB; 368 files, of which 38 are tracked | Mixed current results, historical outputs, reports, logs, and ignored JSONs | Preserve locally. Promote exact minimal JSON snapshots only. |
| `.venv/` | 745 MB; 27,871 files | Local Python runtime | Exclude completely; it is reproducible environment noise, not evidence. |
| caches/build/checkpoints | variable | Bytecode, test caches, TeX intermediates, model weights | Exclude. Regenerate builds; recover checkpoints through sanitized manifests, never by committing them. |

The `07 - rebuttal/all_paper_experiment_code/` subtree contains 853 files and deliberately duplicates code and archival material. It is useful for recovery, but it must not become a second public source tree. The canonical implementation remains under `scripts/` and `experiments/`.

The ignored root `RESULT_PROVENANCE_MANIFEST.md` and tracked `docs/overview/RESULT_PROVENANCE_MANIFEST.md` are different artifacts. The former is a broad private audit aid; the latter is the reviewer-facing ledger.

## Portable evidence bundle

### Promoted and usable

| Tracked artifact | Evidence tier | Source identity | Safe use |
| --- | --- | --- | --- |
| `data/curated/primary1_evq_yarn_10pct_raw.json` | `raw-json-backed` | Archival raw SHA256 `1dbec88e...511c` | Primary I full six-run payload and recomputed Table 2 means; PK remains teacher-forced NLL-gap retrieval. |
| `data/curated/primary2_l128_fixed_tau5_3seed.json` | `raw-json-backed` | Seed-42 sweep SHA256 `980246a9...1fd`; extra-seed sweep SHA256 `4fd031f4...35d47` | Fixed EVQ tau=5 stability at L=128 only; not matched three-seed Geo/DAPE/EVQ evidence. |
| `data/curated/eval_3seeds_full_results.json` | byte-exact raw JSON | SHA256 `1e44d30...30953` | Reconstructed original Primary III evaluator output; no checkpoint-level provenance. |
| `data/curated/table18_mla_3seed_aggregate.json` | `raw-json-backed` | SHA256 `1e44d30...30953` | Primary III per-seed/aggregate MLA PPL and matched-scale YaRN analysis. |
| `data/curated/phase11_l256_3seed_recovered.json` | `raw-json-backed` | Raw SHA256 `6bdf9733...ffa30`; scaling SHA256 `1f9550c4...85321` | L=256 Geo/EVQ/YaRN/NTK archival analysis only; not a substitute for L=128 Primary II replication. |
| `data/curated/phase11b_125m_l256_3seed.json` | `raw-json-backed` | Scaling SHA256 `b8ae7170...de6d94`; DAPE SHA256 `44da360d...f70d` | L=256/100M three-seed plain scaling and DAPE compatibility; establishes a non-complementarity boundary, not a replacement for L=128 Primary II. |
| `data/curated/phase16_99run_manifest.csv` | `sanitized-run-manifest` | 99 rows; 45 pilot + 54 confirm; CSV hash in sidecar | Supports the reported basin/rank audit and run coverage, not checkpoint-level reproduction. |
| `data/curated/learnable_tau_128tok_evidence.json` | `report-backed` | Two tracked experiment reports | Final tau endpoints and reported PPL only; no per-step trajectory claim. |
| `data/curated/mla_channel_count_125m_pilot.json` | `report-backed` | Tracked 125M compression-ablation report | Single-seed qualitative scarce-channel support; not a d_eff/tau convention ablation. |
| `data/curated/quality_454m_full_eval.json` | `raw-json-backed` | Raw SHA256 `5fc3254c...e3caa` | Correct n=2,086 Table 21/Figure 8 values; accuracy remains inconclusive. |
| `data/curated/text_base_10k_500k_pilot.json` | `raw-json-backed` | Four Phase18 raw JSON hashes embedded in the snapshot | Single-seed trained-text support at base 10K/500K; not a tuned-base or `c_pred` control. |

The QuALITY aggregate and four base-pilot result JSONs were recovered in a second checkout after the first reconciliation snapshot. Their exact hashes are now enforced by the builder; machine fields and ignored source files remain excluded.

### Local-only source policy

- Keep `07 - rebuttal/`, broad `results/` outputs, checkpoints, copied repositories, logs, and machine-specific notes ignored.
- Never use the repository root as a supplement archive.
- Rebuild the recovered primaries with `python3 scripts/build_rebuttal_evidence_bundle.py --only primary1 --only primary2_tau5 --only mla_raw`. The Primary I and Primary II archival sources are tracked, and the MLA source is reconstructed from the tracked portable snapshot with an exact hash gate.
- Rebuild other locally available families with `python3 scripts/build_rebuttal_evidence_bundle.py --only phase11 --only phase11b --only quality --only base`; the script rejects unexpected source hashes.
- Validate the portable bundle with `python3 scripts/validate_rebuttal_evidence_bundle.py`.
- Treat a source mismatch as a new evidence-review event, not as permission to update the expected hash silently.

## Scientific impact of the recovery

The recovery improves the rebuttal in nine concrete ways.

1. Primary I now has its complete six-run archival payload, not only transcribed table values.
2. Primary I Table 2 means are recomputed from the raw PPL and teacher-forced PK cells.
3. The fixed EVQ tau=5 arm of Primary II is now raw-backed at seeds 42/137/256, while the missing matched Geo/DAPE seeds remain explicit.
4. Primary III is now portable at per-seed resolution and its original evaluator JSON is reconstructed byte-for-byte.
5. The 99-run formula-optimality evidence has a portable run manifest with configurations, metrics, and `inv_freq` hashes.
6. Phase11 archival records are no longer “missing,” but their protocol boundary is explicit: L=256 evidence cannot be repurposed as the requested L=128 Geo/DAPE/EVQ replication.
7. Phase11B preserves all 15 L=256/100M three-seed runs. Plain EVQ scales consistently, while EVQ+DAPE does not improve on Geo+DAPE, so this evidence limits rather than upgrades the complementarity claim.
8. QuALITY n=2,086 is raw-JSON-backed rather than report-only, while its near-random accuracy remains explicitly inconclusive.
9. The base 10K/500K pilot is source-hashed and portable, establishing only that the observed single-seed direction is not unique to base 500K.

It also prevents two overclaims.

1. Raw QuALITY provenance does not turn near-random accuracy into a positive downstream result.
2. Raw base-pilot provenance does not turn four single-seed arms into a tuned-base sweep or a `c_pred` control.

## Fable5 question-by-question response map

### F5-Q1 — QuALITY table/figure contradiction

**Status:** `DONE` for presentation integrity and aggregate provenance.

**Evidence:** `data/curated/quality_454m_full_eval.json` is rebuilt from the recovered n=2,086 aggregate with source SHA256 `5fc3254c...e3caa`. The obsolete n=200 accuracy pilot is not used as the source of the corrected NLL figure.

**Response:** We audited the QuALITY chain and replaced the contradictory pilot visualization with the four n=2,086 gold-answer-NLL rows retained in Table 21. The recovered aggregate reproduces every retained accuracy, count, and Gold-NLL value. The checkpoints were initialized at 2K, continued and fine-tuned at 4K, so 4K is in-distribution for this downstream protocol. Accuracy remains near random and is not used as positive evidence.

### F5-Q2 — QuALITY arithmetic and near-random accuracy

**Status:** `DONE` for correction and claim scope.

**Evidence:** Rounded table values give `26.8 - 24.6 = +2.2pp`, not `+0.2pp`. Across the four rows the rounded deltas are `+0.7/+2.2/+0.1/-0.4pp`.

**Response:** We corrected the arithmetic and removed accuracy as positive downstream evidence. All accuracies remain near the 25% random baseline with no stable direction. The retained supporting observation is probability-space: at 8K raw, reported gold-answer NLL changes from 3.202 to 2.239; this does not imply a reliable accuracy gain.

### F5-Q3 — Primary II additional seeds

**Status:** `PARTIAL`; fixed EVQ recovered, matched controls pending.

**Recovered but insufficient:** `primary2_l128_fixed_tau5_3seed.json` contains the exact fixed-EVQ tau=5 arm for seeds 42/137/256 under the submitted L_train=128/15M-token protocol. Mean PPL is 182.605 at 128 and 335.710 at 8K. Matched Geo and DAPE seeds 137/256 were not found. The L_train=256 Phase11/Phase11B payloads remain separate supporting protocols and cannot fill those controls.

**Response:** The submitted Geo/DAPE/EVQ comparison remains seed 42. The recovered fixed-EVQ arm shows that its direction is stable across two additional seeds, but it is not a matched three-seed comparison. The remaining closure is Geo and DAPE at seeds 137/256, with paired deltas; otherwise Primary II stays seed-scoped/supporting.

### F5-Q4 — Learnable-tau convergence

**Status:** `PARTIAL`.

**Evidence:** The tracked reports preserve final tau values 1.1391, 1.1445, and 1.1383 for seeds 42, 137, and 256, summarized in `learnable_tau_128tok_evidence.json`.

**Response:** The final endpoints are reproducible across the three reported seeds, but no per-step trajectory was recovered. We can answer that the learned parameter repeatedly settles near the in-range operating point while fixed larger tau improves extrapolation; we cannot claim trajectory-level convergence or optimizer independence without logs.

### F5-Q5 — Tuned geometric-base control

**Status:** `PENDING-EXP`.

**Evidence boundary:** The raw-backed 151.9M pilot compares Geo/EVQ at base 10K and 500K for seed 42. It shows the direction is not unique to base 500K, but it does not tune Geo over the reviewer-requested grid and is not the L=128 Primary-II protocol.

**Response:** We agree that base tuning is the nearest one-knob baseline. Run Geo at bases 10K, 100K, 500K, and 2M under the exact 125M/L=128 data, token, optimizer, and seed protocol, then compare the best Geo row against EVQ base=500K in-range and at 8K.

### F5-Q6 — Trained text at base 10K

**Status:** `PARTIAL`; raw single-seed support recovered, `c_pred` control pending.

**Evidence boundary:** The raw-backed base-10K arm improves PPL relative to Geo at 1K/2K/4K in the 151.9M, L=512, seed-42 pilot, but it does not compare the bare rule against `c_pred` and cannot establish generality by itself.

**Response:** The recovered pilot provides trained-text evidence that the direction survives at base 10K in one matched seed, so the effect is not observed only at base 500K. We still do not claim tuned-base dominance or general base-10K validity; the clean closure is matched Geo, EVQ bare-rule, and EVQ `c_pred(L,b)` with additional seeds and identical budget.

### F5-Q7 — MLA d_eff convention

**Status:** `PENDING-EXP`; provenance improved.

**Evidence:** The three-seed tau=1.414 MLA primary is raw-JSON-backed and portable. Its original 8,342-byte evaluator JSON is deterministically reconstructed with the exact recorded SHA256. The report-backed d_rope=32/16 pilot supports only the scarce-channel direction.

**Response:** The current experiment validates one operating convention; it does not derive the convention. Screen tau=0.354, 0.707, and 1.414 on the same MLA configuration and replicate the relevant comparison. Keep K=d_rope/2, d_head, and the empirical d_eff=128 transport convention conceptually separate.

### F5-Q8 — Measured effective attention length

**Status:** `PENDING-EXP/ANALYSIS`.

**Evidence boundary:** No trained-attention statistics or checkpoint-derived estimator output was recovered.

**Response:** `1/L` remains a falsifiable diffuse-attention approximation, not a measured property of the trained model. Closure requires a predeclared estimator, sampled layers/heads/token positions, effective distance distribution, kappa_att, L_eff^J, and uncertainty from the existing checkpoints.

### F5-Q9 — Autoregressive passkey exact match

**Status:** `DONE`.

**Evidence:** The paper now names PK as teacher-forced NLL-gap retrieval. We successfully recovered the AR exact match rates from the tracked `data/curated/primary1_evq_yarn_10pct_raw.json` payload. At 8K, EVQ+YaRN reaches 100.0% TF retrieval and 58.0% AR exact match, whereas Geo+YaRN reaches 61.3% TF retrieval but drops to 0.0% AR exact match across all seeds.

**Response:** We agree that the prior shorthand was ambiguous. We now label the primary metric as teacher-forced NLL-gap retrieval, and we report the recovered AR exact-match results side-by-side (EVQ+YaRN 58.0% vs Geo+YaRN 0.0% at 8K). This provides full transparency while retaining NLL-gap retrieval as the mechanism-sensitive diagnostic.


### F5-Q10 — NTK anti-composition

**Status:** `DONE` for scope/caveat.

**Evidence:** The paper already exposes the large-tau NTK counterexample; the recovered Phase11 payload preserves raw, YaRN, and NTK evaluator records separately.

**Response:** The positive composition claim is matched-scale YaRN-specific. NTK-aware scaling re-warps the frequency table and can compound a strong cosh warp, so we do not claim scaler-agnostic complementarity. Any rescaler must be validated jointly with the training-time substrate.

### F5-Q11 — 1B-token MLA reversal

**Status:** `PENDING-EXP`; scope correction done.

**Evidence boundary:** The current refs, archival branch, local result trees, and recoverable Git objects contain no reviewer-grade multi-seed 1B completion artifact. The reported row remains single seed.

**Response:** The 500M three-seed result is primary; the single-seed 1B reversal is schedule sensitivity, not durability evidence. Replicate the 1B setting before making any compute-budget claim and do not package the small composed advantage as robustness.

### F5-Q12 — Realistic distance prior

**Status:** `PENDING-EXP/ANALYSIS`; limitation stated.

**Response:** The exact theorem is conditional on its surrogate, and current functional validation uses a uniform distance prior. Refit under predeclared power-law priors and a checkpoint-derived empirical prior; report collision/objective changes rather than assuming the cosh family is invariant.

### F5-Q13 — Exponent 0.465 versus 0.500

**Status:** `PARTIAL`.

**Evidence:** The 99-run sanitized manifest makes the reported basin/rank audit portable. It supports the distinction between basin selection and exact optimum identification.

**Response:** The -1/2 exponent is a structural operating rule, not an exact identification claim; the sensitivity derivation gives 0.465. The 99-run evidence supports basin membership, but it does not measure the cost of replacing 0.500 with 0.465. Run that matched evaluation if feasible and do not use “top-3” as a substitute for the requested comparison.

### F5-Q14 — Shared versus per-head allocation

**Status:** `PENDING-EXP`; conceptual scope fixed.

**Response:** EVQ supplies a shared closed-form initialization substrate. It does not claim that shared allocation is globally optimal or that learned per-head specialization is unnecessary. A per-head jitter or learned per-head composition pilot is needed before claiming complementarity.

### F5-Q15 — Paired deltas and statistics

**Status:** `DONE` for existing three-seed primaries.

**Evidence:** Primary I now retains the complete six-run raw payload, including every PK cell and separate AR-exact field. The portable MLA JSON contains the complete seedwise Primary III results needed to recompute paired PPL changes.

**Response:** Report paired effects and ranges, not an unqualified significance claim at n=3. Keep one-seed and two-seed supporting rows explicitly scoped and do not upgrade them through aggregation in prose.

### F5-Q16 — Terminology, checkpoint description, figures, and guidance

**Status:** `DONE`.

**Response:** Internal phase terminology was removed from paper-facing claims, the QuALITY checkpoint/4K protocol was clarified, the tau-rank visualization and schedule example were improved, and practitioner guidance now distinguishes the bare rule, `c_pred`, MLA convention, and forcing diagnostic.

### F5-Q17 — Video correction factor

**Status:** `DONE` for scoping; derivation open.

**Response:** The 0.53 factor is a directional post-hoc decomposition, not an independently predicted universal constant. Modality-transfer evidence rests on controlled base comparisons that do not require this factor; a first-principles RF-schedule derivation remains open.

### F5-Q18 — LoRA threshold and matched Geo control

**Status:** `PARTIAL`; code/control repair done, empirical rerun pending.

**Evidence:** The LoRA stack now distinguishes true native geometric endpoints from the EVQ midpoint construction at tau=0, infers model geometry, verifies all rotary modules, records path-safe frequency provenance, and rejects stale/mismatched artifacts.

**Response:** The rank threshold is a one-model calibration hypothesis, not general rank guidance. The highest-trust closure is a fresh matched LLaMA-3-8B Geo/EVQ run because that is the submitted supporting setting and the old Geo control path was invalid. A 7B-family run is valuable as second-model generalization, but it introduces a model-family confound and must not replace the corrected matched 8B control.

## Experiment queue after server access returns

| Priority | Experiment | Questions | Minimum decision rule |
| --- | --- | --- | --- |
| P0 | L=128 Geo/DAPE/fixed-EVQ seeds 137/256 | Q3 | Report every seed; downgrade Primary II if direction is unstable. |
| P0 | Tuned Geo base + base=10K bare/`c_pred` | Q5–Q6 | Same data/tokens/optimizer/seeds; select Geo only on predeclared validation metric. |
| P0 | MLA tau convention screen, then replication | Q7 | Separate K, d_head, and d_eff; do not tune on final test only. |
| P0 | Primary I AR exact match | Q9 | Same examples/seeds/checkpoints; report beside NLL-gap retrieval. |
| P1 | Measured attention-distance/effective-length analysis | Q8 | Predeclare estimator and sampled model locations. |
| P1 | Corrected matched 8B Geo/EVQ LoRA rerun | Q18 | Fresh artifacts only; legacy affected Geo results remain invalid. |
| P1 | 7B second-model fine-tuning | Q18/generalization | Use after or alongside the matched 8B repair; treat as cross-model evidence. |
| P2 | 1B MLA replication, realistic priors, 0.465 comparison, per-head pilot | Q11–Q14 | Run only after acceptance-critical controls. |

The 7B experiment is important, but it is not the first scientific dependency. The smaller P0 controls directly answer the simulated reviewers and are cheaper. If only one large-model slot is available, the corrected matched 8B rerun has higher rebuttal value because it repairs the exact submitted control; use 7B to establish cross-model generality when an additional slot is available.

## Company-computer handoff

No ignored local source is required to read the responses, inspect the promoted evidence, or validate the bundle.

### Preferred path after the branch is merged

```bash
git fetch origin
git switch main
git pull --ff-only origin main
python3 scripts/validate_rebuttal_evidence_bundle.py
python3 -m unittest tests.test_rebuttal_evidence_bundle -v
```

### Direct feature-branch fallback

If the PR has not yet been merged, use the published feature branch directly:

```bash
git fetch origin
git switch --track origin/codex/llama8b-positional-distill-pilot
python3 scripts/validate_rebuttal_evidence_bundle.py
python3 -m unittest tests.test_rebuttal_evidence_bundle -v
```

If a local branch with that name already exists, replace the `--track` command with:

```bash
git switch codex/llama8b-positional-distill-pilot
git pull --ff-only
```

### Rebuilding from local ignored sources

This is optional and should only be done on a machine that possesses the exact ignored source files:

```bash
python3 scripts/build_rebuttal_evidence_bundle.py --only primary1 --only primary2_tau5 --only mla_raw
python3 scripts/build_rebuttal_evidence_bundle.py --only phase11 --only phase11b --only quality --only base
python3 scripts/validate_rebuttal_evidence_bundle.py
```

The builder intentionally stops if any raw source SHA256 differs. Do not edit expected hashes simply to make it pass; investigate the new source first.

## Release checklist

- [ ] Every numeric rebuttal insert points to a tracked artifact with an explicit evidence tier.
- [ ] No `trace-only` asset is used in final rebuttal claims.
- [ ] QuALITY is described as raw-JSON-backed and accuracy-inconclusive.
- [ ] The base pilot is described as raw-JSON-backed, single-seed, and not a tuned-base or `c_pred` control.
- [ ] L=256 Phase11 is not substituted for L=128 Primary II replication.
- [ ] Fixed-EVQ three-seed recovery is not described as matched Geo/DAPE/EVQ replication.
- [ ] PK is named teacher-forced NLL-gap retrieval unless AR exact match was actually run.
- [ ] No 2B/4B/7B/8B completion is claimed from scripts or historical traces alone.
- [ ] Bundle validator, unit tests, code smoke checks, leak scan, and curated supplement packager pass.
- [ ] Final push includes this document, all curated assets, their validators, and the updated provenance ledger.
