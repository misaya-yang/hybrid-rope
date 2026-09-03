# Research timeline

This is a chronological ledger for cold start. It answers **what we did, what
we observed, how our interpretation changed, and where the work stopped**.
It is not a fourth authority: rules remain in [`../../../AGENTS.md`](../../../AGENTS.md),
owner routing and durable agenda remain in [`../../../INDEX.md`](../../../INDEX.md),
and volatile state remains in [`../../HANDOFF.md`](../../HANDOFF.md). Every
number below defers to its linked owner.

The original technical question was method-centric: can a closed-form,
non-uniform finite RoPE allocation improve length extrapolation, and can a
static rule select its `tau`? Three later question changes matter more than the
experiment count:

`tau/method search -> fixed-support allocation identification ->
table/weight co-adaptation -> mature-checkpoint natural-generation transport`.

## 2026-02-24 to 2026-03-02 — early finite-frequency evidence

- **Did:** established short-context baselines, then ran 50M/125M finite
  EVQ-Cosh tau grids on from-scratch language modelling. Historical reports are
  grouped under [`docs/exp/2026-02/`](../../../docs/exp/2026-02/).
- **Observed:** the finite grid was non-monotone; the useful point was not a
  smooth monotone continuation from small tau. Cross-seed direction was more
  stable than effect size.
- **Changed our view:** tau became a fallible finite-grid operating prior, not a
  continuous optimum certificate.
- **Evidence status:** historical/system evidence; current causal ownership is
  later exact-range work.

## 2026-03 — scale, tasks, composition, and failure conversion

- **Did:** ran 454M passkey/PPL composition, L=256 extrapolation, 125M
  Kerple+MLP compatibility, 750M continuation, the 99-run formula sweep,
  QuALITY diagnostics, Video-DiT, and GQA/MLA channel-scarcity studies. Reports
  are under [`docs/exp/2026-03/`](../../../docs/exp/2026-03/).
- **Observed:** long PPL, teacher-forced retrieval, autoregressive exactness,
  and downstream QA did not convert automatically into one another. The same
  range operator had substrate-dependent leverage. The 750M run preserved a
  small in-window/large-OOD crossover, while the QuALITY pilot exposed protocol
  and scoring mismatch rather than a clean downstream verdict. Scarce-channel
  gains were large in the extreme MLA cell but not monotone across architecture
  cells.
- **Changed our view:** systems breadth is valuable, but each endpoint and
  operator identity must remain protocol-specific. Formula-centred grids show
  tested finite points, not a continuous basin or near-optimality.
- **Evidence status:** 750M persistence and 432M MLA breadth remain supporting
  owners; early claims and labels are bounded by current provenance audits.

## 2026-04 to 2026-06 — submission hardening and provenance repair

- **Did:** prepared the NeurIPS submission, tightened figures/tables, and built
  code/result/provenance manifests under [`docs/overview/`](../../../docs/overview/).
- **Observed:** several historical labels overstated implementation identity or
  seed scope; nearby scripts/results were not automatically matched evidence.
- **Changed our view:** a plan, script, checkpoint inventory, or paper row is
  not a result owner. Reviewer-safe numbers require explicit method, data,
  endpoint, seed, and hash/provenance boundaries.

## 2026-07 — rebuttal, mature models, and exact-range controls

- **Did:** audited 8B LoRA temporal NLL, 151.9M Native/EVQ controls, official
  versus repository YaRN-style components, M4 exact-range factorials, and the
  OLMo-2 1.485B mature-checkpoint route. Historical experiment reports are in
  [`docs/exp/2026-07/`](../../../docs/exp/2026-07/); rebuttal owners are under
  [`rebuttal/rebuttal_0723/theory_results/`](../../../rebuttal/rebuttal_0723/theory_results/).
- **Observed:** fixed-support interior allocation remained active in M4; a
  matched non-Cosh analytic shape was competitive, so Cosh uniqueness did not
  follow. Mature-model adaptation produced protocol-specific long capability.
  Exact position-independent invertible Q/K compensation is obstructed for
  unequal frequency multisets.
- **Corrected:** 28 historical direct-hybrid zero-score receipts were produced
  by an in-place Native/EVQ buffer alias. They are invalid evidence against
  partial-pair, blend, or per-head hybrids; no corrected GPU rerun was made.
- **Changed our view:** pure allocation identification, systems composition,
  mature adaptation, and frozen retrofit are separate estimands. EVQ-Cosh is a
  closed-form construction and intervention, not a universal optimum.

## 2026-08-19 to 2026-08-20 — ICLR causal core

- **Did:** completed the full-RoPE sin/cos geometry audit and the raw-hash-
  receipted 151.9M three-seed exact-range replication. Owners:
  [`foundations/FULL_ROPE...`](../foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)
  and [`evidence/EXACT_RANGE...`](../evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md).
- **Observed:** at fixed sampled support, moving only 30 interior frequencies
  changes trained behaviour with the same OOD direction in 3/3 seeds; when
  both grids are target-matched, the tested ordering reverses. Static
  collision/stable-rank objectives admit Fourier/ordering counterexamples and
  do not rank LM quality.
- **Changed our view:** support `(a,R)` and normalized allocation `z` are
  distinct but interacting causal coordinates. The paper's geometry claim is
  phase-invariant redundancy/effective dimension, not an extrapolation
  predictor.
- **Closed routes:** cosine-only collision ranking, collision/logdet as a
  behavioural target, attention-Fisher ordering, and the LeRoPE `w^(1/3)`
  profile oracle.

## 2026-08-21 to 2026-08-26 — mature-checkpoint intervention ladder

- **Did:** ran phase-chord from-training controls, released-model residual and
  table interventions, same-support Qwen/OLMo controls, fresh FineWeb NLL,
  static single-table gates, co-adaptive recovery, allocation dose response,
  and supporting Video-DiT confirmation. Owners are grouped under
  [`attention-aware-retrofit/results/`](../attention-aware-retrofit/results/).
- **Observed:** pure same-support `z` changes frozen mature behaviour; detailed
  derived profiles often did not separate from a coarse nearest ramp. Two
  analytic Native-support tables improved long NLL while failing Native-window
  retention. Co-adaptation recovered tail NLL but traded against full-sequence
  NLL and did not materially separate 2Wiki capability. The dose curve showed
  graded tail effects without a jointly passing point.
- **Changed our view:** table/weight compatibility and endpoint-specific
  readout dominate static geometry scores. NLL-only positives do not become
  capability or deployment positives.

## 2026-08-27 to 2026-08-31 — zero-training portfolio and ordered coupling

- **Did:** preregistered and closed a success-first portfolio, then evaluated a
  scale-consistent exponent-space table, gain controls, cross-checkpoint
  transport, and same-multiset permutations. The historical portfolio lives in
  [`preflights/ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md`](../attention-aware-retrofit/preflights/ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md);
  the result owner is
  [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`](../attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md).
- **Observed:** one static table could pass declared OLMo 1x gates and improve
  longer endpoints, and the construction transferred long capability to Qwen.
  Preserving the exact frequency multiset while permuting slot assignment
  caused decisive collapse.
- **Changed our view:** the mature object is the ordered pairing between learned
  rotary subspaces and frequencies/dilations, not an unordered spectrum. A
  successful table remains one checkpoint/protocol result, not a universal law.

## 2026-09-01 — cross-K, reference-length, and breadth confirmation

- **Did:** evaluated two-parameter coupling across Qwen K32/K64 and Gemma K128,
  repaired the Native reference-length identification protocol, completed
  reference-correct s2/s4 tables, ran fresh K32/K128 confirmations, and opened
  full RULER-13 only after registered gates. See
  [`attention-aware-retrofit/results/`](../attention-aware-retrofit/results/)
  and receipts under [`evidence/`](../attention-aware-retrofit/evidence/).
- **Observed:** physical/index ordering did not replicate at K32; normalized
  index was more Native-compatible and later won the independent K128 contrast.
  Reference- and request-scale correction recovered Gemma 8K/16K, while the old
  reference control remained zero. K and checkpoint still co-vary, so no K
  causality follows.
- **Corrected:** the earlier Gemma 16K zero used the configured 8K limit as the
  operating reference; the validated 4K reference changes the requested scale
  and supersedes that ceiling interpretation.
- **Changed our view:** normalized index is the best-supported tested transport
  coordinate, not a canonical or universal one. Cell averaging and conditional
  Native-Q/K diagnostics failed their entrance gates.

## 2026-09-02 — natural-QA validity correction, exploratory headwise work, benchmark

- **Did:** recorded Qwen packed-natural NLL and far-evidence QA/source-use
  session statistics, ran a headwise factorization development ladder, drafted
  a first-principles retrofit memo, and froze a 16-episode
  [`theory falsification benchmark`](../../../falsification_benchmark/README.md).
- **Validity audit:** the 38-row constructed “16K Hotpot” stress selected
  short-correct rows, used a mechanical prompt-tail boundary and non-official
  filler distribution, and has no recovered raw owner. It is invalid for
  benchmark, gate, task-radius, or route-closure claims. The separate
  Hotpot-200 headwise run used variable-length official rows capped at 16K,
  reused the development rows adaptively, and lacks a tracked executed/raw
  bundle; it is exploratory report-only, not exact-16K confirmation.
- **Theory correction:** the first-principles memo's exact-conditioning T4,
  unrestricted non-identifiability/off-arc T5, and novelty-ratio T7 do not
  survive audit. It remains working theory history, not a canonical proof owner.
- **Changed our view:** 9/2 did not establish QA closure, a disconnected basin,
  or a new law. It made configuration and gate validity mandatory preconditions
  for interpreting later mature-checkpoint outcomes; it did not replace the
  author-ordered static pure-`z` transport question. Tracked 9/1 NLL/RULER
  owners remain intact.
- **Lifecycle:** `FIXED_SUPPORT_TRAINING_CAUSAL_CORE_ESTABLISHED /
  STATIC_Z_NATIVE_TO_LONG_TRANSPORT_OPEN / ASSAY_PREFLIGHT_REQUIRED /
  NO_ACTIVE_GPU_RUN`.
