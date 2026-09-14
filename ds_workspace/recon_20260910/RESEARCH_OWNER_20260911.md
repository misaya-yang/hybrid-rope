# Research ownership and next decisions — 2026-09-11

## Latest author steering: continue thinking, no GPU experiments

The author requested continued pursuit of a reliably better method than MrRoPE,
then explicitly rejected restoring GPU access to spend more compute on ungrounded
experiments. Continue CPU derivations and analysis of saved evidence. Do not
launch/resume GPU jobs, ask again to enable the GPU, or treat a proposed comparison
as permission to spend compute. The `rope-2` and retired `rope` heartbeats were
both verified PAUSED; leave them paused.

The completed fresh-six-arm result is now collected and independently rescored:
[fresh 72-case result](FRESH72_COMPLETED_20260911.md). Calibration regresses
18.4375 pp against b4wide at 16K; b4wide's conditional OLMo win is retained and
does not settle the failed Qwen transfer. All 432 stored scores match the original
scorer and all 72 prompts are unique.

CPU-only YaRN/MrRoPE factorial arrays are implemented in
`experiments/rope_decision_20260911/operator_factorial.py`; three focused tests
pass. They separate the actual higher-frequency and lower-frequency interventions,
using supplied FP32 values and identical gain. These are untested constructions,
not a breakthrough or a queued experiment. At Qwen S4, the faster set is slots
24–37 and the slower set is slots 38–39 (zero based); both exterior plateaus match.

The failed calibrated table changes all 64 frequencies, but its maximum added
phase over a 16K relative distance is only 0.46496248 rad. Do not explain its
failure as multi-cycle *perturbation* wrapping without evidence. This arithmetic
does not establish the actual cause; full-model task transfer remains the test.

The saved LongBridge signed controls have now been read and rescored on CPU:
[signed-control result](SIGNED_CONTROL_RESULT_20260911.md). On 350 unique OLMo
prompts, slower-minus-faster is +11.8286 pp. About 65.2% of that score difference
comes from UUID completion: full UUID 38/50 versus 11/50, while correct first
eight characters occur 46/50 versus 44/50. In 25 of the 28 slower wins, faster
already emits that correct prefix. First-complete-UUID scoring preserves 38/11.
The signed holdout is still only 60/180 rows on the slower arm; long VT worsens
on the available overlap. This is a specific copying/completion phenomenon,
not a universal allocation rule or proof of an internal pathway. A common
additive frequency shift leaves the shifted band's complex Fourier Gram spectrum
unchanged, so improved conditioning of that object cannot explain this effect.

The SSH entry is unchanged. Local DNS failed; using the existing SOCKS proxy
with `ProxyCommand=nc -X 5 -x 127.0.0.1:7890 %h %p` reaches the original instance.
It currently has no GPU device and no live campaign workers. Raw output is intact.
Earlier launch/resume plans below are historical and subordinate to this section.

## Author's active objectives

The author requested continuous ownership of experiments and server monitoring on
2026-09-11. Continue until the research questions have evidence-backed answers:

1. Establish what changes from YaRN to MrRoPE, why those changes help or hurt real
   tasks, and use that evidence to construct better static frequency tables/gain.
2. Determine whether EVQ allocation can be improved for direct use with frozen
   language-model weights. Separate data-free construction from task-calibrated
   frequency parameters; do not silently substitute weight adaptation.

Use the author's Pro proposal (attachment
`/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/.codex/attachments/fdc2bdde-9706-42b7-b7f0-36eaf43c2151/pasted-text.txt`)
as guidance: preserve observed task repairs, reduce observed damage, compute
full-model decision responses, and validate actual outputs. Local margins and
geometric objectives are not substitutes for task results or generalization.
Conditional gains are useful; universal lossless wins were not required.

### Latest steering: priority correction

The author has now narrowed the active research question to YaRN→MrRoPE:
what exactly changes, what the authors claim explains the gains, and whether
the actual intervention supports that explanation. EVQ is outside the current
active scope. Do not automatically resume EVQ or extend the 65-parameter
decision calibration. Existing calibration results are evidence, not a mandate
to continue that method. Prioritize the verified operators and discriminating
task-level attribution, reusing comparable completed controls.

Fresh primary-source check: MrRoPE v1 sections 3.2 and 4.4 explain the method
through progressive intermediate-band radix expansion, preserving fine-scale
information, cosine-sum bounds, and intermediate attention behavior. These
motivate hypotheses; they do not establish an optimal task objective. The
official YaRN index-ramp discrepancy documented in REPORT.md remains material.

The author challenged why EVQ had been scheduled before the main Pro-guided
work was resolved. The EVQ scheduling was the assistant's choice, not an
explicit experiment assigned by Pro. Do not restart EVQ calibration or launch
additional EVQ jobs now. Finish the paired main-line validation and explain
the YaRN→MrRoPE operators and the failed/conditional task-calibration transfer
first. EVQ remains the second objective; it is deferred, not canceled.

`evq_calibration_01` failed from GPU contention before any calibrated result.
It is not a method failure and must not be automatically retried by the
heartbeat. Its development-only plan/code remain as unvalidated preparation.

## Live operations

- Host: `ssh -p 27741 [REDACTED_EMAIL]`.
- Existing raw campaign: `/root/autodl-tmp/phase1_20260910`.
- Runtime: `/root/miniconda3/bin/python`; 32 GiB GPU.
- Current task owns follow-up experiments and confirmed code repairs. Preserve
  raw artifacts and avoid duplicate launches. No shutdown request is active.
- Codex heartbeat `rope-2`, every 20 minutes, attached to this task. It should
  perform useful authorized follow-up and stay quiet on unchanged state.
- Retired heartbeat `rope` is PAUSED and targets another task. Leave it paused;
  its historical experiment and shutdown instructions are not this campaign.

## Audit findings and retained evidence

Read `audit/pro_decision_20260911/REPORT.md` and `check_results.json`.
Snapshot files in that directory are audit evidence, not corrected live code.

- Recomputed all 900 saved held-out rows across BM, b3, a1b64, b4wide, step42:
  zero score mismatches, BUT 180 row IDs represent only 120 actual prompts
  (40 short, 80 long). Old independent-sample claims are invalid.
- Deduplicated equal-task macro b4wide vs BM: 16K +7.917pp (stratified SE
  3.809pp), 4K -6.157pp (SE 3.447pp). Keep the conditional gain and uncertainty.
- Deduplicated equal-task macro step42 vs BM: 16K -2.963pp and 4K -7.546pp. Its failure is not
  explained solely by pooling short and long lengths.
- Qwen NLL reader reverses the sign of BM-minus-MrRoPE and mixes a piece-weighted
  mean with a book-weighted SE. Native control also uses gain 1.1386, not 1.
- Generic gradient rotary patch in `experiments/curvature_20260910/model.py`
  downcasts OLMo FP32 cos/sin to BF16. Stock rotary disables gradients. Reuse
  existing full-model derivative machinery only after model-specific parity.
- BM archive vs same-named rerun: 24/350 scores and 210/350 texts differ. The
  full archived table dictionary matches the original prepared dictionary;
  new reconstruction changes 20 FP32 slots by up to 5.96e-8. Output causation
  is not yet isolated; do not use old text as an exact replay target
  merely because aggregate accuracy is close.
- New runner stores text but not generated token IDs or sufficient run identity.
  Replay targets need verified token sequences and matching model/decoder/table.

## Immediate work, in dependency order

1. Finish audit report and resolve BM table identity; correct readout bugs with
   preserved originals. Distinguish data errors from interpretation errors.
2. Establish faithful differentiable OLMo and Qwen rotary paths. Check complete
   greedy-path parity, finite differences and exact token records on a minimal
   set of already-observed repairs/breaks. Do not fit through detached prefill.
3. Verify true YaRN, MrRoPE, BM/b4wide, EVQ formulas against primary methods and
   canonical project code. The campaign's uniform-increment arm is MrRoPE-Uni,
   not YaRN. Separate gain and frequency effects when explaining YaRN→MrRoPE.
4. Task-decision calibration: initialize at useful b4wide/BM, use one shared
   static table and gain, preserve selected long repairs and fix short damage.
   Include an equally calibrated gain-only control to attribute frequency value.
   Use trust regions/active competitor constraints and re-evaluate the entire
   original decoder; infeasible linear constraints are not proof no static
   solution exists. Save all candidate values, outputs and failures.
5. Evaluate frozen candidates on fresh rows; the repeatedly inspected 180-row
   panel is now development evidence if used to choose/calibrate a method.
   Reuse comparable MrRoPE caches and report original task-weighted scores by
   length. Scale a supported gain, not an NLL-only improvement.
6. EVQ: distinguish canonical span-preserving replacement (three tested OLMo
   failures) from bounded EVQ residuals around effective extrapolation tables.
   Test grounded corrections using the same decision evidence; an unconstrained
   calibrated table is not automatically an EVQ result.

## Live status after takeover

- New remote root: `/root/autodl-tmp/rope_decision_20260911`; code is under
  `code/experiments/rope_decision_20260911`, invoked as Python modules with
  `PYTHONPATH=<newroot>/code:/root/autodl-tmp/phase1_20260910/repoharness`.
- Corrected rotary/reader: 11 CPU tests pass. The legacy Qwen reader path now
  contains the corrected standalone reader. Misaligned Qwen NLL PID 70861 was
  stopped after native/MrRoPE arrays were saved; the BM arm did not finish.
- `probe_01` COMPLETE: four unique actual cases. Stock and differentiable
  full-output margins exactly agree; all own successful output margins >0.
  Complete 16K backward peak ~6.3 GiB. Repaired endpoint/midpoint EVQ at tau=1
  both score zero on these four selected cases; this is not a new whole-family
  impossibility result. Official-index and turn-count YaRN differ as operators.
- `calibration_01` COMPLETE: partial task-Pareto repair (first short fwe from
  1/3 to 1, second stays 2/3, both long examples stay 1). File status says
  LOCAL_CALIBRATION_UNRESOLVED because the stricter two-repair target was not
  met; do not erase this genuine partial repair.
- `gain_only_01` COMPLETE: same-calibration gain-only control, no task repair.
- `calibration_02` COMPLETE: continuing from candidate_07 with log-frequency
  ordering included inside the solver did not add another task repair.
  Reuse the eight identified BM/b4wide output records from `probe_01`.
  The goal is to fix two short fwe outputs while preserving two long outputs.
  Intermediate merit improvements or partial repairs are not the final result.
- `fresh_72` COMPLETE, seed 2026091101, QA offset 1000, six official
  RULER tasks, 4 short + 8 long rows per task. Before any GPU evaluation,
  token-prompt uniqueness verified: 72/72 unique, zero overlap against 2,227
  old rows / 1,337 unique old prompts.
  Primary readout is equal-task macro by length, original RULER scoring.
  Freeze a candidate based only on development outputs before opening fresh
  scores. Compare to BM/b4wide and equally calibrated gain; add true YaRN and
  MrRoPE on these new rows for the core method question if existing exact caches
  cannot cover them. This is independent sample validation, not a full benchmark.
- `validation_plan.json` frozen before fresh GPU outputs. Candidate chosen by
  real development scores then least scaled movement: calibration_01/candidate_02,
  not the last/largest change. Gain control is gain_only_01/candidate_02.
  Plan SHA256 `6e91bf13c022e4b25c0555070869c53840c88060337097cc76ab8a341b79c057`.
  `validation_reference`: bm,b4wide,gain_calibrated; `validation_methods`:
  decision_calibrated,yarn_index,mrpro. The latter MrRoPE uses the same native
  grid construction as YaRN, with bit-identical fast/slow plateaus. It is a
  fresh same-grid control, not a reused archived FP32 table.
  First complete independent readout: decision_calibrated vs BM is 79.9306%
  vs 77.5% at 4K (+2.4306pp), and 38.8542% vs 48.2292% at 16K (-9.375pp).
  Wait for b4wide/gain/YaRN/MrRoPE before attributing the loss to calibration
  versus the initializer. Do not promote this candidate or enlarge its fit
  merely because the four development examples improved.
- Old LongBridge/gain-sweep workers 73255/73320 remain active, finishing their
  current arm lists. Supervisors 68910/60221 were stopped without killing the
  workers: they would schedule duplicate-prompt follow-up or retry on a wrong
  completion filename (gain_bm_g1p2 vs actual gain_bm_g1p20). Current task owns
  subsequent decisions. Preserve raw files and do not repeat completed arms.
