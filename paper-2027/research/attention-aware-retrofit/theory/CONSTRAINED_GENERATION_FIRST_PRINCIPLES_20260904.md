# Native-constrained generation: a first-principles experimental contract

> **2026-09-05 decision amendment:** Read the [Pro audit reconciliation](../analysis/PRO_REPORT_AUDIT_RECONCILIATION_20260905.md) before acting on older priorities below. Double qualification limits that family, not all valid single-evidence/restoration comparisons; the old joint unresolved verdict remains. Next proposed training is matched N_compact, then fixed-recipe Qwen Z/Y, with no prefix change. Unit-amplitude diagnostics and N128 aggregate confirmation have completed; the latter passes with a format/indexing regression. Its exposed confirmation pool cannot tune or independently confirm new variants. No new GPU run was launched by this amendment.


- **Date/status:** 2026-09-04 v3; prospective algebra/dossier review. Subsequent GPU observations and execution corrections belong to the [execution owner](../results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md), not the predictions below.
- **Current decision:** confirm the existing Z witness; no new table/gain search.
  Main adaptation is all-linear r16 with actual-deployment Native constraints,
  not the earlier QK/source-contrast prototype. See §8 for what changed and why.
- **Questions:** Z: maximize measured physical useful reach with frozen weights,
  one table and one gain, subject to separate Native retention constraints.
  F: learn only from physical 1x replay/2x/4x, then measure untouched 8x/16x/32x
  complete generation under the same constraints.
- **Object:** OLMo-2-0425-1B-Instruct first; full legacy-u p2 is the incumbent,
  not an optimum. Standard stationary RoPE, all-layer global table/gain.
- **Artifacts:** independent implementation
  [`single_table_generation.py`](../../../../scripts/experiments/single_table_generation.py),
  [`generation_contract.py`](../../../../scripts/lib/rope/generation_contract.py),
  and its tests. Execution hashes belong in each generated run receipt.
- **Supported:** identities and conditional implications proved below; a new
  discriminating protocol, not its success. **Unsupported:** universal 8x bound,
  guaranteed 32x transfer, unique movement profile, V/O causal attribution,
  or natural-QA capability inferred from synthetic diagnosis.
- **Correction:** replaces use of the mixed C2-s2/full-p2-s4 quadratic as a
  same-path prediction. Old scores remain valid for their actual tensors.
  The earlier conditional phase-cost ceiling is not promoted as a theorem.

## 1. Define the actual optimization problem

Let W be the checkpoint, T=(omega_1,...,omega_K,a) the static table and rotary
amplitude, and A the adapter (A=0 on route Z). Define measured Native ratios

    R_P = exp(NLL_Native - NLL_(T,A))
    R_task = macro_task_(T,A) / macro_task_Native.

R_P is a PPL ratio, not relative NLL. The strict constraints are separately
R_P >= .88 and R_task >= .88. Keep .875 only as a labelled historical sensitivity
analysis; never average the two. Also report task score multiplied by normal
EOS completion separately; a model that runs into its decode cap is not certified
by a partial-match score. A zero Native denominator makes that assay unresolved.

For a fixed task distribution and predeclared utility threshold u, define

    Reach(T,A) = max L such that C(T,A,l) >= u for all registered l <= L,

with C the full generated answer plus EOS, plus working controls. It is a
finite tested reach, not a continuous guarantee. Evaluate each task as well as
the macro. Table factor s is an input to construction, not Reach/N_native.
The new synthetic screen uses nonzero joint exact+EOS to permit investigation;
it is not a product-level definition of useful natural performance.

For Z: maximize tested Reach over a finite frozen candidate set satisfying both
Native constraints. For F: compare adapted minus frozen at an identical table,
then candidate versus matched Native/standard-extension adapters if successful.
Selection and confirmation groups remain separate. Previously exposed assets
cannot become fresh by renaming their split.

## 2. What static geometry can and cannot determine

For a frozen query/key content pair, put d_k=log(omega_k/omega'_k). The rotary
logit contribution has the form

    ell(Delta) = a^2 sum_k [c_k cos(omega_k exp(-d_k) Delta)
                           + s_k sin(omega_k exp(-d_k) Delta)].

**Derived identity.** Hugging Face's rotary `attention_scaling=a` multiplies
both sin and cos used on Q and K; the logit multiplier is a^2, not a. All new
receipts record both. This is not a second independently tuned temperature.
The implementation convention was checked against the
[Transformers v4.57.6 OLMo2 source](https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/olmo2/modeling_olmo2.py);
the actual installed version is recorded and still needs a work-machine smoke.
At a fixed content pair,

    d ell / d d_k = a^2 omega'_k Delta
                   [c_k sin(omega'_k Delta) - s_k cos(omega'_k Delta)].

This derivative depends on content, distance, and phase. No scalar frequency
count alone can determine its sign or downstream usefulness. Earlier failures
of orbit counts, displacement, and D* as selectors are therefore relevant
counterexamples, not invitations to fit another scalar after outcomes.

For Native risk D(u), u=(d,log a), the local expansion is

    D(u)-D(0) = b^T u + (1/2)u^T H u + remainder.

Native need not be stationary in these coordinates: b need not vanish. H need
not be positive definite. Even a valid local lower curvature is not a finite-8x
bound without a remainder through that displacement and a proven link from
extension to u. Separate task risks need separate treatment. We retain this as
a local sensitivity interpretation, not an LM ceiling or candidate ranker.

**Correction to earlier ceiling construction.** A requirement that a fraction
of readout mass move by a threshold does not, by itself, establish the proposed
box/isotonic quadratic lower bound: the actual displacement must lie in the
optimization set, or a quadratic-form-preserving domination argument must be
proved. Off-diagonal curvature terms can invalidate coordinatewise clipping.
Also Native stationarity and finite-region lower curvature were assumptions,
not measured facts. No numerical ceiling is inferred from that construction.

## 3. Dilution: exact identity, limited extrapolation

For relevant keys R and other unmasked keys D, define

    G = logsumexp(ell_R) - logsumexp(ell_D).
    attention_mass(R) = sigmoid(G).

If the relevant exponential sum is held fixed and the distractor exponential
sum is multiplied by q, G decreases exactly by log q. Nominal sequence-length
ratio is not necessarily q; multilayer hidden states and both sums can change.
Thus a 4x margin reserve of log 2/log 4/log 8 is only a conditional diagnostic
for 8x/16x/32x, never a guarantee of answer accuracy.

The independent assay keeps distractor-record density fixed as physical length
grows. Its local-oracle condition moves all required records near the query,
removes the old copies, and preserves length/number of records. This holds
competitor count approximately fixed while changing source distance. It is a
controlled distance/content relocation; it does not isolate a single head or
keep all activations/logits constant.

## 4. Why NLL repair failed to guarantee generation

Let y_1,...,y_T include the intended terminal EOS and z_t the vocabulary logits
on the GOLD prefix. Define

    M_t = z_t(y_t) - max_(v != y_t) z_t(v).

**Derived sufficient condition.** If every M_t>0, greedy decoding returns the
entire gold string and immediate EOS. Proof by induction: the first argmax is
gold; each next evaluated prefix is therefore gold; the final argmax is EOS.
With a specified tie-breaking rule this can be extended to ties. A single
negative M_t can destroy complete generation despite a lower mean CE.

A second sufficient condition is sum_t CE_t < log 2: each nonnegative CE_t is
then <log 2, so each gold conditional probability exceeds 1/2. This condition
is sufficient, very strong, and not necessary. Mean NLL without answer length
or the worst step loses this information. Neither implication allows corpus
PG-19 NLL to stand in for task-specific gold-prefix margins.

Source credit is a different axis:

    E_t = log p(y_t | correct_source, gold_prefix)
          - log p(y_t | swapped_source, SAME gold_prefix).

E_t>0 need not make M_t>0: corrupting the source can merely worsen an already
wrong prediction. Conversely a high M_t without source dependence can reflect
memorization or query leakage. This explains why improving one old proxy left
the requested endpoint unresolved.

The v2 objective uses each legitimate world's own answer/EOS target:

    L_task = mean_t CE_t + .25 max_t [min(m_teacher_compact,t,1) - M_t]_+.

**Why lawful paired CE supplies source pressure (derived).** For equiprobable
worlds with the same question, let t be the first differing canonical answer
token. Their preceding answer prefix is identical. A source-insensitive model
has the same distribution q in both worlds, so

    -(log q(y0_t) + log q(y1_t))/2 >= log 2,

because q(y0_t)+q(y1_t)<=1 and their product is at most 1/4. Thus balanced lawful
world supervision penalizes ignoring the source without an extra wrong-context
training reward. Sequence averaging can dilute that one-step penalty; the worst
trajectory margin directly targets the remaining greedy failure. This is a
conditional objective argument, not an optimization or generalization guarantee.

Only truth-verified, Native-compact-correct worlds qualify for transport training.
The same-target source effect remains a symmetric diagnostic, not a training
reward for merely lowering wrong-context likelihood. This replaces the earlier
source-contrast/worst-step QK prototype before any GPU execution.

For preservation, every student replay forward retains its DEPLOYED table:

    D_j(phi) = E_(c in Native stratum j) KL(p_original_Native || p_student,T,phi).
    L = L_task + sum_(observed j) lambda_j (D_j/.02 - 1).

The .02 KL target is a prospective optimization setting, not a replacement for
the author's .88 endpoint-retention gate. The zero-adapter starting point may
have D_j(0)>0 because its table already differs. Measure it; neither zero LoRA
output nor PG-19 delta determines this KL. Never add KLs from different stages
and call their sum total Native damage. Independent endpoint retention remains
necessary even when sampled-prefix KL is small.

## 5. Observable task decomposition and falsification

Each example asks for exactly `route|value` then EOS. Lookup needs one remote
record; chain needs key->route and route->value records. Every field is freshly
randomized by semantic group. The two source twins have identical query,
distractors, physical length, and nonsource tokens, but require different outputs.
All length renderings of one group stay in the same split.

Three conditions are mandatory on each candidate:

| Condition | Intervention | Interpretation |
| --- | --- | --- |
| Remote twins | target records at physical remote positions | actual joint task |
| Local oracle twins | move required records near query; erase old copies | can this model solve/read out the same task with nearby evidence? |
| Deleted | replace only target records with their original background | source-independent answer/leakage control |

Primary exact+EOS compares the ENTIRE decoded raw generation with the reference,
with special tokens visible and tokenizer cleanup disabled, plus terminal EOS.
Equivalent token segmentations of the same complete string are accepted; literal
canonical-token equality is an additional diagnostic, not a false-negative gate.
Score complete route, complete value, canonical token equality, normal EOS, and
BOTH twins exact+EOS. Component scores are diagnosis only.
A printed route is observable source-following behavior, not an attention-route
measurement. A route-correct/value-wrong case does not prove V/O is the cause.

The automatic report uses prospective screen thresholds: at least eight
independent groups per task/length, local-oracle exact+EOS >=.8, deleted target
exact+EOS <=.05. These are operational gates, not natural discontinuities.

- Oracle fails: local task/format/readout or long-context systems issue remains;
  no remote-mechanism verdict. Check Native 1x and cached/full decoding parity.
  Do not teach the diagnostic format and then count that as transfer of an
  already existing Native capability. Native-compact qualification is explicit.
- Deleted succeeds: inspect leakage; do not train against a non-resolving assay.
- Oracle resolves, remote is zero: close that candidate/protocol and longer cells.
- Route recovered, answer weak: test a parameter-matched QKVO placement; any gain
  is placement evidence, not direct causal identification of V/O transport.
- Answer recovered, EOS weak: the failure is visible termination behavior;
  compare the frozen EOS-aware objective, not a new rank sweep.
- Both twins succeed: extend blinded lengths and confirm Native retention;
  natural tasks and standard baselines remain separate admission gates.

## 6. Fixed witness and constrained computation repair

E0 first uses a small synthetic diagnostic to qualify scorer/cache/source logic.
The NEW compact condition distinguishes task acquisition from background burden;
near/far swap exactly equal token blocks, preserving the block multiset. Full
canonical gold-prefix traces include every answer token and EOS; positive margins
with a different cached trajectory trigger a numerical/implementation audit.

E1 confirms N / fixed Z / matched-support G / official fixed YaRN. No outcome
selects a new movement profile, gain or scale. MrRoPE is a related practical
comparator to add only with a verified official implementation, not invented code.

E2 uses qualified NATURAL compact/near/far worlds, not the synthetic source pack.
Fixed all-linear LoRA covers Q/K/V/O and gate/up/down projections, r=16, alpha=16;
norms, embeddings, head, base weights, table and gain remain frozen. In the pinned
OLMo shapes this is 12,058,624 parameters. Attention-only r46 matches that count
and is an explanatory ablation after feasibility, not another selection sweep.
All-linear r16 does not contain every attention-only r64 update: it distributes
a finite rank budget differently. Broader module coverage is a hypothesis about
where useful directions lie, not a strict expressivity dominance theorem.

First seed N/Z/Y uses the same qualified data/order/budget. Native restoration:
32 steps, batch8 from four separate strata; student always uses the arm table.
Transfer: 96 steps, task batch8 (768 views once), plus two Native replay examples
per step with observed-group dual updates. Save 0/32/64/96/128; select only on
independent <=16K validation after Native feasibility. Never use blind longer
outcomes to select a checkpoint. Further seeds 43/44 are conditional replication,
not an automatic spending commitment. The original .88 Native rule remains.

The [constrained engine](../../../../scripts/train/train_single_table_native_constrained.py)
implements the schedule and full-vocabulary original-teacher cache. It requires
qualified task and Native replay manifests; these assets have NOT been recovered
or generated locally. Its mechanical preflight does not independently prove
natural counterfactual truth. Do not call E2 ready until source qualification,
actual runtime smoke and held-out validation/selection assets pass.

## 7. Highest-ROI additional openings

1. **Natural transport confirmation:** condition training on Native compact
   success in both worlds; report the unfiltered final population AND the
   Native-short-correct subgroup. All worlds/lengths/templates stay group-split.
2. **Pointwise decision stability:** for a unique teacher top-1 probability p_a
   and top-2 p_b, the nearest competing-argmax forward-KL boundary is

       kappa(p) = p_a log(2p_a/(p_a+p_b)) + p_b log(2p_b/(p_a+p_b)).

   Minimizing KL(p||q) over q_b>=q_a gives q_a=q_b=(p_a+p_b)/2 and q_j=p_j
   elsewhere. Differentiating the expression with respect to p_b gives
   log(2p_b/(p_a+p_b))<=0, hence top-2 is the nearest boundary. Pointwise
   KL(p||q)<kappa preserves this teacher argmax; apply along its entire greedy
   trajectory for a trajectory certificate. It does not prove teacher truth,
   population retention, or stability from average KL. Ties have radius zero;
   tiny radii require a numerical uncertainty label, not epsilon division.
   The dependency-light implementation and boundary tests check this identity.
3. **Decision-direction transport:** for frozen linear head W and post-norm
   hidden difference delta_h, delta_z=W delta_h exactly. For every rival v,
   m_far(y,v)=m_compact(y,v)+(w_y-w_v)^T delta_h. A small shift in one critical
   direction can matter more than a large hidden norm in harmless directions.
   This diagnoses realized states; it supplies no bound on unseen 32x delta_h.
4. **Symmetry audit when needed:** joint within-head pair relabeling must also
   permute Q/K output rows, biases and coordinatewise norm weights. Frequency-
   only permutations are different interventions. The linear-algebra identity
   is known; no new model-level parity run is claimed here.

The earlier m^1.25 direction, gain/factor grid, and QK/source-margin training
are not the default research queue. No result was produced by those prototypes.

## 8. Fine-grained audit of the supplied GPT-6 Pro dossier

Source: `hybrid_rope_iclr2027_theory_experiment_dossier_20260904.md`, all 1,734
lines read. SHA-256:
`2e9408d72783d672c39331d027d56d96ccc2150189961929197ecdaec821f0de`.
This is user-supplied analysis, not an experimental authority. Its Library IDs
are not local artifacts, and its uploaded manuscript is older than current TeX.

| Dossier point | Decision after current-repository verification |
| --- | --- |
| Stop frequency search and confirm one existing witness | Adopt for the next run; freeze exact full-p2/c=.074. G is a controlled comparator, not another candidate. |
| Earlier failures included QKVO and sparse KL | Verified in `OLMO2_1B_NON_RULER_ADAPTATION_SEARCH_20260731`: seven QKVO r64/alpha128, 1500-step 4K arms. Retract any suggestion that simply opening V/O is a new sufficient fix. |
| Compact/near/far with valid paired worlds | Adopt; add compact and exact near/far block exchange to the independent diagnostic. Natural-world truth still needs raw/source validation. |
| All-linear r16 + original-Native functional constraint | Adopt as a bounded main candidate, not a minimal-rank theorem or a success probability. Attention-only parameter matching is an explanatory follow-up. |
| Drop wrong-context likelihood contrast from the primary loss | Adopt. Balanced lawful-world CE + full-trajectory margin; source contrast is diagnostic only. |
| Two RULER vectors are an unexplained conflict | Correct the dossier: current owner §5 is log-L1/c=.10; §6 is c=.074, with distinct result hashes. Do not merge them or downgrade valid owners because the uploaded copy lacked later context. Raw bundles were not rehashed in this session. |
| 9/2 readout/rerank numbers show definitive failures | Do not promote: current summary explicitly marks unrecovered raw/config evidence as unresolved/report-only. The older seven-arm owner is stronger evidence. |
| Native thresholds .03 NLL / -2 points replace current tolerance | Do not silently adopt. Retain the author's approx 12%/.88 primary convention; report tighter thresholds as optional diagnostics. |
| PPL retention .875302 is about 14.25% PPL increase | Correct: retention loss and PPL relative increase are different quantities. Delta NLL approx .1332 does NOT by itself prove a particular teacher-KL budget infeasible. Measure total KL separately. |
| 72 GPU-h, nine main runs, new figures and three contributions | Not an authorization or automatic manuscript redesign. Stage by evidence/ROI and keep current owner-backed paper identity; scale only after feasibility. |
| Qwen copy of normalized-index m confirms full-p2 | Not established: repository C2/index transport and full legacy-p2 are distinct. Require an exact transfer owner; no silent substitution. |

Native-teacher restoration is not itself novel: primary sources
[LongReD](https://arxiv.org/abs/2502.07365) and
[LinearARD v2](https://arxiv.org/html/2604.00004v2) already study restoration.
Their methods/conditions are not identical to the proposed objective, and this
inspection is not an exhaustive novelty audit. The paper's core remains the
controlled support/allocation decomposition and its lifecycle consequences.

## 9. FFN review: learning under a Native constraint

**Status, 2026-09-04:** local derivation and prospective review; no FFN training
result exists. This amends the unexecuted v2 recipe's runtime/validation handling.
The dossier and author's FFN suggestion identify a useful hypothesis, not the
established cause of earlier failures. The pinned OLMo MLP uses gated SiLU:

    f(h) = W_down [s(W_gate h) * (W_up h)].

**Derived first-order result, h fixed.** Write g=W_gate h and u=W_up h. Then

    delta f = delta W_down [s(g)*u]
            + W_down [(s'(g)*(delta W_gate h))*u
                      + s(g)*(delta W_up h)] + O(||delta W||^2).

Opening gate/up/down therefore supplies direct directions to change gating,
feature composition and the residual write. Attention updates instead change
the MLP input h as well as later attention computations. They can already alter
f(h); a frozen FFN is not categorically unable to use new information. Whether
the direct FFN directions are necessary/useful for these tasks is empirical.
For LoRA at zero B, A's initial gradient can be zero while B's is nonzero;
requiring every individual factor to have a nonzero first-step gradient is a
wrong smoke criterion. A and B both zero would be a genuine disconnection.

**Working hypothesis.** Fixed-table long inputs may preserve some evidence but
move intermediate features outside combinations decoded reliably by the frozen
short-trained computation. Joint attention/FFN adaptation can repair that
composition. This is stronger than observing changed attention and weaker than
claiming a uniquely identified FFN bottleneck. Old QKVO r64 failures make this
module expansion worth testing; their different data/length/regularization also
prevent assigning their failure uniquely to frozen FFNs.

**Preservation is a function constraint.** Locally, Native and long logit changes
are J_N delta_phi and J_L delta_phi. A useful direction would reduce long answer
loss while staying within sampled Native constraints. Extra FFN coordinates may
enlarge the available directions; they may also create new forgetting directions.
No supplied evidence establishes a useful nullspace of J_N that remains useful
under J_L, and finite low-rank updates need not obey this local approximation.
Neither low rank nor small parameter norm guarantees absence of forgetting.

The reviewed design addresses four distinct risks:

| Risk | Mechanism in the fixed protocol | Remaining limit |
| --- | --- | --- |
| Opening FFN changes old short computation | Frozen base/norm/embedding/head plus original-teacher full-vocabulary KL, with the student always on its deployed table | Low rank and sampled KL do not guarantee Native task retention |
| Retention regularization prevents learning | Record separate task and weighted-Native gradients by attention/FFN, ratios/cosines before clipping, actual LoRA-factor updates, answer/EOS loss and margins | Gradients are local diagnostics; Adam transforms them and no automatic regularizer relaxation is justified |
| Replay looks healthy but unseen Native fails | Independent calibration plus fresh held-out text NLL and full-generation instruction/reasoning/position-format tests; paired source-group retention intervals | Coverage and finite-sample precision remain explicit; mean KL is not a certificate |
| Model memorizes 128 worlds or learns the format only | One fixed pass over 768 lawful views, source/template/semantic separation, Native-compact qualification and held-out C/N/F plus unseen-family test | The trained families remain family-overlap, not unseen-task transfer |

In the pinned architecture, r16 allocates 4,194,304 parameters to attention and
7,864,320 to FFN (65.2% of the total). Attention-only r46 has the same total count;
the contrast tests allocation of adaptation capacity, not a strict superset or
a universal FFN-necessity theorem. Hold data/order/optimizer/budget/checkpoint
selection fixed. Run this explanatory control only after the all-linear main
candidate establishes Native-feasible generated transfer. A second FFN-only
or layer/rank/learning-rate grid is not the first response to failure.

Step32 supplies a restoration-only control for free. Report step0, step32 and
the independently selected later checkpoint; otherwise a repair caused entirely
by Native restoration could be misattributed to long labels or FFN task learning.
If all-linear improves all N/Z/Y similarly, the constrained training worked but
Z's special advantage was not shown. If it improves only training loss, neither
capability migration nor the FFN mechanism is established.

The v3 implementation fixes the one-step smoke LR of zero, journals reusable
teacher cache shards, checks complete held-out matrices before CUDA, saves
optimizer/RNG/dual/exposure state, and defaults to a review after 32 R + 32 T. The
[execution contract §§7–8](../preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md#7-reviewed-first-boot-sequence-and-result-driven-continuation)
defines evidence-to-action routing. These changes improve diagnosability and
bounded execution; CPU tests do not establish GPU resume parity or scientific
success. This historical readiness statement is superseded by the execution
amendment: assets are prepared, N128 completed, and natural double controls
remain unresolved. Read the current execution owner before another launch.


## 10. YaRN correspondence, supervision density, and a smaller next claim

**Status/date:** 2026-09-04; checked algebra plus primary-source comparison and
prospective experiments. GPU observations remain in the [execution owner](../results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md).
This section neither turns a YaRN result into our result nor changes old scores.
Sources: the [ICLR2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf),
[original launcher](https://github.com/jquesnelle/yarn/blob/a18e54988639030e706f6ff9db224cf74e08a65a/train.sh),
and [official training implementation](https://github.com/jquesnelle/yarn/blob/ce518ee3b6427a010cc732bef6b07011c721055d/finetune.py).

### Shared structure does not imply the same experiment

For a frozen scale s>1, write the YaRN Native-preservation coefficient as gamma_k:

    omega_Y,k = omega_N,k * [gamma_k + (1-gamma_k)/s]
    omega_Z,k = omega_N,k * s^(-m_k)
    m_Y,k = -log[gamma_k + (1-gamma_k)/s] / log(s).

**Derived:** this is an exact coordinate conversion. gamma=1 leaves the high-
frequency band unchanged; gamma=0 interpolates by1/s. It puts both methods in
one movement coordinate for comparison; it does not imply the learned Q/K or
behaviour is invariant to movement. The official static implementation ramps
across dimension indices with rounded boundaries; the article's rotation-count
ramp is not numerically identical. Specify implementation and realized tensors.

The [author's static implementation](https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py)
also changes rotary amplitude. In today's s4 controls, Y's amplitude1.13862944
means attention-logit multiplier1.29647699; Z's1.10258578 means1.21569541.
Y is6.6449% larger in this multiplier. Hence Y versus Z is a practical-method
comparison; G versus Z holds support/gain fixed for the allocation comparison.
Positive scaling preserves a fixed head's logit ordering but can change the
weighted mixture; lower entropy alone does not imply improved evidence use.

**Working hypothesis, not a launched candidate:** if fixed Y empirically protects
Native better, retain its high/low bands and put one normalized interior movement
only in the transition band. Hold its endpoints, gain, data and budget fixed.
Do not multiply two independently compressed tables: low frequencies could
otherwise be compressed twice. No new curve/temperature sweep follows from this
idea. Dynamic-YaRN changes with request length and cannot certify the global
static-table route; a changing table would also require explicit KV-cache handling.

### The crucial training comparison

| Quantity | Original YaRN Llama2 main recipe | Today's Qwen N experiment |
| --- | --- | --- |
| Trainable parameters | Whole model; main launch does not enable optional LoRA | 18,464,768 all-linear adapter parameters |
| Supervision | Dense next-token LM on PG19 | Answer+EOS CE/margin, plus separate sampled Native KL |
| Physical training length | 64K in both extension stages | Compact/8K/16K |
| Budget | 400 steps, then200; global batch64 | 32 identity/restoration +96 task steps; batch8 |
| Input tokens | Derived1.678B then cumulative2.517B | Measured6,328,481 actual forward input tokens |
| Direct task label positions | Almost every next-token position | Measured3,642 answer/EOS labels over768 views |
| Beyond-training endpoint | 128K,2x maximum training length | Blind beyond16K not yet established |

The input-volume ratio is about398x, not a GPU-time ratio. The old N receipt
counted6,329,249 prompt+target tokens, including the terminal EOS not fed into
the student; its actual-forward field was768 too high and is corrected here. The nominal dense-
label count is vastly larger still, but repeated and correlated positions are
not independent information samples. Full-vocabulary Native KL is additional
supervision, so the3,642 count must never be described as all learning signal.
N's task-learning phase constrains192 replay rows; the256 restoration rows were
used while N was still unchanged. Current truth-prefix KL is not a uniform
constraint over the teacher's actual generated trajectories.

The paper reports small short-benchmark losses in its Llama2 regime: MMLU
43.8→41.7(7B) and55.8→51.9(13B), relative4.79%/6.99%. These are benchmark-specific,
not a universal loss bound. The paper's64K→128K mean0.49 figure is an incremental
score change, not Native→128K complete-generation retention. Its ordinary
[passkey code](https://github.com/jquesnelle/yarn/blob/master/eval/passkey.py)
extracts a first number. Keep that comparison separate from our strict full
string+EOS endpoint and its newly observed formatting failure mode.

### Two bounded discriminating experiments, not a larger sweep

1. On one checkpoint with resolving simple/natural controls, compare existing
   fixed Y versus Z using the same current all-linear budget. This answers whether
   the method base is easier to adapt under this low budget; it is not a YaRN
   training reproduction. If independent Native or complete generation fails,
   stop the candidate. Do not infer parameter- or temperature-specific causality.
2. On the best **validation-feasible** fixed base, compare current answer-only
   task supervision with one predeclared added prefix-LM term. Keep inputs,
   answer term, Native constraint, trainable modules, schedule and token budget
   fixed. Use a frozen weight and no coefficient sweep. This tests distributed
   supervision rather than simultaneously changing model size, data and rank.
   The implemented low-cost companion samples at most128 uniform prefix
   positions per view with fixed weight0.1. It estimates the per-view dense LM
   mean in expectation; the realized subset and optimization trajectory differ.
   One forward jointly projects answer and sampled-prefix logits, avoiding the
   full16K-by-vocabulary tensor. Work-machine CPU gradient/boundary tests passed;
   an actual16K-containing two-step smoke subsequently passed (see execution owner). No full training completion
   or language/task improvement from this variant is claimed. NLL alone is insufficient.

For the next manuscript increment, the author narrowed scope to2–3 benchmark
families: NIAH retrieval, one natural QA endpoint, and one standardized short
retention endpoint (ARC-Challenge is a prospective choice, not yet measured).
Keep official task metrics alongside the declared full-output/EOS operational
metric. Existing failure rows are retained; a new endpoint gets frozen fresh
cases before model selection. We seek demonstrated transfer across tasks and
checkpoints, not universal correctness on every benchmark.

Only if these instruments resolve should a larger8B Native-only pilot or more
VRAM be purchased for a specific bottleneck. OLMo1.485B's4K Native makes16K a
real4x test; Qwen1.5B's32K Native makes16K0.5xNative. Choose the question's length
reference before choosing a model. Keep existing32GB hardware for short pilots;
upgrade according to measured BF16 peak, activation recomputation and cost to a
resolved experiment, not because a bigger model sounds more publishable.
