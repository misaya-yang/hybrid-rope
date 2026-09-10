# Sol10 failure-transcript audit and constructive allocation rule

## Executive finding

The failure record does not support a universal frequency curve selected from geometry alone. It supports a narrower and more constructive principle:

> Allocate a fixed log-dilation budget over frequency **edges** according to the signed, full-model marginal value of moving that budget, while using EVQ-like geometry or MrRoPE only as a prior and feasibility structure. Train-time allocation and frozen deployment are two different optimizations because the former co-optimizes the weights and the latter must pay a checkpoint-compatibility cost.

This principle unifies the useful part of EVQ and MrRoPE without treating either shape as universally optimal. EVQ supplies a regularized prior over the interior log-frequency density. MrRoPE supplies the correct deployment scaffold: protect the high-frequency band, fully interpolate the slow band, and distribute a unit cumulative dilation through the transition. The missing object is the **signed task utility of each transition edge under the actual checkpoint and target-length distribution**.

The strongest obstruction is `Smooth_MrBudget`. Its certified minimum-roughness allocation improves several geometry quantities relative to MrPro, including shorter-lag distortion, yet its completed 36-row Qwen3B panel is worse at 128K by 9.79 points (3 wins, 5 losses; NLL deltas +0.00098/+0.00192/+0.00342). The complete archived tool record is SHA-256 `62570a5bc72e052efc9725cba74222d1cba68b68e672286b923a54332c37539f`, source session line 12870. Geometry is useful as a regularizer; it is not a selector.

## What the transcript actually establishes

### 1. The research object repeatedly changed without an explicit estimand change

The original active question moved among at least four different estimands:

1. training a model from scratch with a non-geometric allocation;
2. changing a frozen checkpoint's table with zero weight updates;
3. recovering a changed table through LoRA or broader adaptation;
4. studying positional encoding in sparse/hybrid attention.

The most severe substitution was dense per-token K/V dimensional compression. The implemented operator-family work retained all historical tokens and shrank each token's representation; it therefore did not test which tokens sparse attention selects or whether position changes selection. The dialogue explicitly corrects this at `sol10_full_dialogue.jsonl:241-243` and again at `:277-280`. The operator-family result remains a useful side experiment: output distillation improved held-out NLL from the common initialization's 5.30 to 3.94, while score fitting worsened it to 9.99 and neither recovered the 8K retrieval answer (`:260-279`). It cannot validate or falsify sparse-selection claims.

The user correction is broader: the recurring error was treating an executable next step as the right next step (`sol10_full_dialogue.jsonl:74-80`). This is not merely a workflow complaint. It explains why proxy diagnostics, code preparation, and extra candidates accumulated while the causal question changed.

### 2. Proposed, implemented, tested, and established were repeatedly conflated

The dialogue and project files contain all four states:

- **Proposed only:** many preflights and candidate grids, including some later retired before execution.
- **Implemented/prepared:** the visibility replay and several long-training paths passed mechanical checks but were not model results. `ROPE_FIXED_POSITION_VISIBILITY_PROTOCOL_20260908.md:1-8` says `PREPARED_CODE / RUNTIME_UNQUALIFIED / NOT_RUN`.
- **Partially tested:** the 64K LoRA run stopped at 59/128 steps and produced no final adapter; it is an interrupted run, not a LoRA result (`ROPE_OVERNIGHT_EXPERIMENT_REVIEW_20260908.md:87-99`).
- **Established in a bounded protocol:** the three-seed fixed-support training comparison and the two-seed weights-by-table crossing.

A recurring failure was to promote the state too early. Examples include calling a completed implementation a prepared experiment, treating a small development panel as a method, describing a partial RULER matrix as a macro, and treating a formula's mathematical optimum as a behavioral optimum.

### 3. Proxy-to-capability jumps are the central technical failure

The repository itself summarizes the pattern: a local quantity was improved, renamed, or given more freedom, then treated as a bridge to capability without proving the bridge (`docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md:33-52`). The following are distinct counterexamples:

- Cosh is the exact optimizer of a chosen convex surrogate, but the surrogate is not derived from language-model loss. Matched exponential and Cosh were not separated in the factorial comparison.
- A phase-isotropy profile improved neither the registered small-model screen nor all long endpoints; its receipt is explicitly `SCREEN_UNRESOLVED`.
- Carrier lowered its background-response objective but degraded generation; the overnight report rejects “background target decreases, therefore capability improves” (`ROPE_OVERNIGHT_EXPERIMENT_REVIEW_20260908.md:122-145`).
- E8 had larger predicted local attention gain than E1 but lost 13.89 points at 128K, showing that moving mass toward the answer position does not necessarily improve the content computation required for the answer (`sol10_full_dialogue.jsonl:369-370`).
- `Smooth_MrBudget` is smoother and has lower short-lag distortion, yet is materially worse on long tasks.
- A frequency multiset is not sufficient: permuting the same frequencies across learned slots can collapse performance. Therefore unordered spectral entropy, density, or frame-potential metrics cannot determine frozen-checkpoint behavior.

The correct conclusion is not that geometry has no value. Geometry defines admissible supports, phase scales, and useful priors. It cannot supply the missing signs of Q/K content coefficients, V transport, softmax competition, and downstream readout.

### 4. Training-time and frozen-deployment evidence must not be merged

The strongest training result holds support/endpoints fixed and changes only the interior allocation. The paper-level owner shows consistent OOD direction across three training seeds. Separately, the 151.9M crossing shows that FMRoPE-trained weights prefer the FMRoPE-derived runtime table while Cosh-trained weights prefer the Cosh-derived table; the crossover interaction is 3.400 and 3.251 NLL in two seeds (`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md:12-39`, `:191-226`).

That crossing is the mathematical reason the two regimes require different objectives:

- From scratch, weights can rotate, rescale, and repurpose channels while the allocation is learned or fixed during training.
- Frozen deployment changes phase trajectories under weights that already assigned functions to slots. It must preserve useful existing computation while improving target-length behavior.

Support retargeting adds a second distinction. The same three seeds that favor Cosh at fixed support reverse ordering when the support is retargeted. The failure synthesis correctly states that fixed-support superiority cannot be extrapolated to a different deployment strategy (`ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md:101-104`).

### 5. Strong baselines and full tasks arrived too late

Several apparent gains disappeared when the comparison moved from a weak or incomplete reference to a strong one:

- BM was spectacular against MrPro on OLMo 16K but did not transfer as a general ordering to Qwen3B/7B 128K. Qwen7B BM is 71.11 versus MrPro 84.44 on the six-task development subset (`ROPE_QWEN7_BM_RESULT_20260908.json:1-91`).
- In the sparse-position branch, candidates that beat a weak mean baseline remained below Quest32; RoPE-before/after clustering was essentially tied. The transcript's postmortem identifies the late strong baseline as avoidable rework (`sol10_full_dialogue.jsonl:68-72`).
- A local metric improvement often preceded generation. Once complete output was checked, the gain disappeared or changed task.

This does not justify requiring a huge benchmark before every idea. It justifies using the closest strong baseline and the smallest target-level comparison at the start.

### 6. Evaluation and data mistakes produced false scientific impressions

The failure ledger includes several protocol errors that materially changed interpretation:

- FineWeb-trained models were once evaluated on Proof-Pile while the cache retained a FineWeb-looking filename; absolute PPL above 600 created a false negative. Strict FineWeb evaluation restored the expected long-range direction (`2026-03-14_staged_diagnostic_report.md:42-70`, `:224-278`).
- An assistant-header newline error changed an exact-generation result from the correct 4/8 to 0/8 (`sol10_full_dialogue.jsonl:71-72`).
- Changing a random seed did not change QA question order; later work explicitly offset question IDs to avoid fake replication (`:12-16`).
- A later QA sample concentrated on one “steam engine” article and had to be diversified before new-model evaluation (`:364-365`).
- Several reports mixed official task recall, substring/NLL-gap retrieval, full-string exact match, and EOS completion. These measure different failures.

These errors motivate input/source identity and output inspection where they answer a concrete risk. They do not motivate repeated hash and smoke rituals after the path is already stable.

### 7. Small positive results should be preserved without being made universal

Two examples matter for the next allocation rule:

- FullLagP2 has genuine conditional benefit: Qwen1.5B at 64K improves multikey retrieval and FWE, and at 128K improves mean VT while multikey remains at floor. Transfer to Qwen3B yields one retrieval gain and regressions in VT/FWE (`ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json`).
- E1 `s28_less` changes one slot and scores +5.21 points at 128K on a 36-row development panel with two wins and no losses; the reverse-matched perturbation is flat. It is a development signal, not a transferable law. The two-slot `pair28_29` result loses 4.17 points. This establishes failure to compose in thresholded task accuracy. It does **not** establish strongly nonadditive underlying logits: the inspected continuous multikey cross-cell margins are close to additive, with only a small interaction, so threshold crossing is a sufficient explanation.

This distinction matters. A valid allocation method should optimize a continuous target/distractor margin or answer loss, then validate task accuracy. Binary wins alone do not identify the local algebra.

## Constructive allocation framework

### 1. Parameterize allocation by edge increments

Let native frequencies be ordered from fast to slow, \(\omega_0>\cdots>\omega_{K-1}>0\), and let \(s=L_{\rm target}/W\). Write a deployed table as

\[
\nu_j=\omega_j s^{-m_j},\qquad 0\le m_j\le 1.
\]

Use the rotation-count statistic \(c_j=W\omega_j/(2\pi)\) to define a fast anchor \(\ell\) and slow anchor \(h\), as YaRN/MrRoPE do. Fix \(m_j=0\) for \(j\le\ell\), \(m_j=1\) for \(j\ge h\). In the transition define nonnegative edge increments

\[
\epsilon_r=m_r-m_{r-1}\ge0,\qquad
\sum_{r=\ell+1}^{h}\epsilon_r=1.
\]

This is the common finite allocation object. MrRoPE-Pro chooses a deterministic increasing sequence of increments. A Cosh/EVQ density, after anchoring and discretization, also induces an increment sequence. `Smooth_MrBudget` solves a roughness-regularized version. The question is therefore not “which named curve?” but “where should the one unit of cumulative dilation be placed?”

An edge increment at \(r\) changes every slower slot \(j\ge r\). For any differentiable full-model loss \(L\),

\[
\frac{\partial L}{\partial\epsilon_r}
=\sum_{j=r}^{h-1}\frac{\partial L}{\partial m_j}.
\]

This cumulative edge gradient is old local calculus, not by itself a selection theorem. Its value is that it matches the mixed-radix construction exactly and exposes what the selector must measure.

### 2. Frozen deployment: compatibility-constrained signed utility

For a fixed checkpoint \(\theta_0\), define a target-level calibration objective

\[
L_F(\epsilon)=
\mathbb E_{x\sim D_{\rm long}}
\big[L_{\rm answer}(x;\theta_0,\epsilon)
+\alpha L_{\rm bind}(x;\theta_0,\epsilon)\big]
+\lambda\,mathbb E_{x\sim D_{\rm short}}L_{\rm preserve}(x;\theta_0,\epsilon).
\]

`L_answer` is conditional answer NLL or another task-appropriate differentiable loss. `L_bind` is a target-versus-competing-record margin such as \(-[z_* - \log\sum_{i\ne *}e^{z_i}]\), evaluated through the actual full prefill and read path. `L_preserve` is short/native answer loss or a full-model behavior-preservation loss. It is not a frequency-only distance.

Around a reference allocation \(q\) (MrPro is the natural frozen reference), estimate the signed edge utility \(u_r=-\partial L_F/\partial\epsilon_r\) and a local empirical curvature/trust metric \(H\) from full-model finite differences or gradients. Then solve the constrained projection

\[
\epsilon^*=arg\max_{\epsilon\in\Delta}
u^\top(\epsilon-q)-\tfrac12(\epsilon-q)^\top H(\epsilon-q)
-\gamma R(\epsilon),
\]

where \(\Delta=\{\epsilon\ge0,\sum\epsilon=1\}\) and \(R\) is a modest roughness or EVQ prior. For diagonal \(H={\rm diag}(h_r)\) and \(R=0\), the KKT solution is explicit:

\[
\epsilon_r^*=\max\{0,q_r+(u_r-\nu)/h_r\},
\quad \sum_r\epsilon_r^*=1,
\]

with \(\nu\) found by one-dimensional water filling. For banded \(H\), this is a small convex QP.

The reference/prior and curvature do not make this sufficient. `Smooth_MrBudget` proves that. The essential new input is signed, checkpoint-specific, target-level utility. Full prefill is required because successful E1 cross-cache behavior depended on both prefix formation and readout. A one-query frozen-state gradient may propose a direction but cannot certify it.

### 3. Training from scratch: remove frozen compatibility and co-optimize weights

For training from scratch, optimize

\[
\min_{\theta,\epsilon\in\Delta}
\mathbb E_{(x,y)\sim D_{\rm train}}L_{\rm CE}(x,y;\theta,\epsilon)
+\beta R_{\rm geom}(\epsilon)
+\eta R_{\rm deployment}(\epsilon).
\]

Here the weights \(\theta\) are allowed to co-adapt. There is no penalty for staying close to a pre-existing checkpoint's MrPro table. EVQ-Cosh is recovered only in the special continuum limit where the behavioral utility is omitted and `R_geom` is exactly the historical Brownian/min-kernel surrogate. That is a valid analytic prior, not a theorem that Cosh minimizes CE. A realistic train-time rule either learns \(\epsilon\) jointly or alternates weight updates with low-dimensional edge updates under actual CE and target-length exposure.

This also clarifies why in-window LoRA was an inadequate test of train-time allocation. The failed route lacked target-length exposure, had a mismatched supervision objective, was undertrained, and in some versions did not update all computations needed to adapt. PPL improvements cannot substitute for task recovery.

### 4. Why the rule naturally retains low/mid/high bands

The three-band scaffold has a sound but limited basis. High frequencies experience many rotations in the training window, so scaling them changes learned short-distance phase relations strongly. Very slow frequencies cover little phase during training; interpolation keeps target-window phase within a familiar range at low short-distance cost. The transition is where these costs compete. Exactly three segments, the 32/1-turn thresholds, and a particular middle curve are empirical approximations, not consequences of a universal theorem.

The proposed edge optimization preserves this structure while making the transition checkpoint- and task-aware. If utilities are monotone and smooth, the solution reduces to a familiar MrRoPE/EVQ-like ramp. If utilities have a knee near slots 28/29, budget concentrates there. If no stable signed utility transfers across held-out rows, the result is a model-specific calibrated table rather than a general allocation law.

## Counterexample checks and outcome interpretation

1. **Smoothness counterexample:** `Smooth_MrBudget` lowers geometry costs and fails tasks. Therefore setting \(u=0\) and optimizing only \(R\) is rejected.
2. **Permutation counterexample:** the same multiset in different slots can collapse. Therefore allocation is indexed and checkpoint-relative.
3. **Support-retargeting reversal:** changing support can reverse the trained comparison. Therefore \(s\), support, and allocation must be explicit inputs.
4. **Threshold non-composition:** E1 one-slot accuracy gains do not compose. Continuous margin evidence can still be near-additive; task thresholds and small interactions must be separated.
5. **Frozen cross-model transfer:** P2/BM/E1 directions can help one model or task and hurt another. Therefore the frozen objective is conditional on \(\theta_0,D_{\rm long},D_{\rm short}\).
6. **Proxy success with behavior failure:** local attention/KL/NLL can improve while generation remains wrong. Therefore the final decision is made on the task endpoint that defines the claim.

## Minimal decision-sufficient next test

For Qwen2.5-3B, W=32768 to 128K, do not launch another curve grid. Use MrPro as \(q\), estimate signed utilities for transition edges on a small calibration set containing multikey binding plus one preservation task, solve the single convex simplex problem above, freeze one table, and compare it with reused MrPro on held-out rows. Report continuous target/competitor margins and exact task outcomes. One magnitude-matched redistribution in the opposite edge-utility direction is the sufficient attribution control if the proposed table wins.

If it fails while the calibration objective improves, that specifically falsifies transfer of the chosen full-model utility estimator. If it fails already on calibration, the local/QP approximation or objective is wrong. If it wins only one task family, preserve that conditional result and formulate the claim around that computation. If it transfers across new inputs and a second checkpoint after freezing the edge rule, the evidence supports a general allocation principle.

## Failure ledger condensed

| Failure/correction | Verified result | What it rules out | Constructive constraint |
|---|---|---|---|
| Cosh surrogate treated as behavioral derivation | Cosh exact for chosen functional; matched exponential unresolved | surrogate optimum = LM optimum | use Cosh as prior, measure signed task utility |
| Smooth geometry treated as selector | Smooth_MrBudget 128K -9.79 pp | roughness/local distortion selects capability | geometry only regularizes a target-level objective |
| E8 local answer attention | proxy stronger, task -13.89 pp | answer-mass increase = content computation improvement | use target-vs-distractor/full-output loss and validate generation |
| E1 single-slot development gains | s28_less +5.21 pp; pair result loses | binary single-slot wins necessarily compose | optimize continuous margins; test frozen joint table |
| Fixed-support training merged with deployment | weights-by-table crossing 3.400/3.251 | one allocation objective covers both regimes unchanged | separate joint-training and frozen-compatibility objectives |
| Support ignored | three-seed retarget reversal | fixed-support ordering transfers to new support | make support/scale explicit |
| Dense KV compression substituted for sparse attention | all-token operator-family experiments | validates sparse selection/topology claim | label side investigation; test selection and readout directly |
| NLL/proxy substituted for ability | operator score fit 9.99 NLL; KD 3.94; both retrieval wrong | lower local loss proves task recovery | claim endpoint decides |
| Weak baseline entered first | Quest/strong comparisons erased apparent gains | local improvement over weak control is method evidence | use closest strong baseline initially |
| Evaluation source/template errors | Proof-Pile fallback; header newline; clustered QA | observed score always reflects model/method | inspect source, prompt, scoring at first discriminating sample |
| Partial run promoted | LoRA stopped 59/128, no final adapter | LoRA succeeded/failed | report interrupted state only |
| Over-defensive paper framing | user repeatedly corrected invented stronger claims | every condition must be written as a weakness | state positive tested claim; keep essential conditions in protocol |

## Evidence limits

All 397 dialogue records and all 115 additional assigned project files were read in full. Relevant archived tool outputs were inspected for the completed non-geometric result table and the E1 cross-cache row. I did not read every archived tool-output record in `tool_outputs_000..086`; the assignment required opening entire relevant records when a claim needed confirmation, not claiming the whole tool archive. I did not inspect unassigned full Pro files or other agents' reports. No GPU job or extra agent was used.
