# Astra09: a finite-softmax frequency step that preserves learned channel roles

2026-09-10. All six assigned full texts, 1,120,284 bytes, were ingested literally; receipt: `astra09_coverage.json`. Reference: `astra09_rule.py`. No GPU work, manuscript edits, runtime edits, or extra agents. Independent CPU derivation/tests only. Parent owns full-model testing.

The constructive rule is **one exact finite-softmax gradient step in frequency space, constrained by existing frequency gaps, and weighted by downstream output sensitivity**. It changes the actual frequency of each original rotary pair. It neither changes amplitudes nor reallocates abstract counts while ignoring the frozen channel labels. An oracle-support loss is a secondary diagnostic restricted to retrieval-qualified heads; applying it indiscriminately to every head is invalid.

## 1. Exact finite loss and the cumulant correction

For one query/head and each causally visible key i, capture unrotated Q/K and retain all original pair coefficients. With d_i = key_position − query_position and split-half planes j,

s_i(theta) = c_i + sum_j [a_ij cos(d_i theta_j) + b_ij sin(d_i theta_j)],

a_ij=(qx_j kx_ij+qy_j ky_ij)/sqrt(head_dim),
b_ij=(qy_j kx_ij−qx_j ky_ij)/sqrt(head_dim).

c_i is any fixed native additive bias; Qwen normally has none. Native attention gain must be included in a,b exactly as used by the real implementation. The reference accepts optional `bias` per key, default zero; adding fixed native bias does not change any derivative or bound below. It must be supplied when applying this construction to NOSA.

For input-located support S, L_support = LSE_i(s_i) − LSE_{i in S}(s_i) is the exact negative log attention mass on S. Equivalently L_support=softplus(log D−log S), where D and S are the *finite actual* distractor and support exponential sums.

For either finite set A,

log Z_A = log |A| + mu_A + var_A/2 + R_A,

R_A := log mean_{i in A} exp(s_i) − mu_A − var_A/2.

R_A is an exact, finite quantity. Calling it an exact remainder does not require convergence of an infinite cumulant series. Thus the correction inside the support/distractor competition is R_D−R_S, not just a skewness term at zero tilt. Both signs are possible. For nonuniform native bias one may use native-bias tilted weights and retain log Z_bias, or simply include bias in each exact score.

CPU counterexample: A=(10,0,...,0), 64 keys; B=(1,...,1). mu_A+var_A/2<1, but LSE(A)>LSE(B). Every covariance approximation, including complete covariance, that stops at second order can mis-rank these blocks. Evaluating actual scores and stable LSE eliminates this approximation error without claiming statistical tail generalization.

## 2. Head roles: primary output-aware objective

Maximizing support mass on all heads trains syntax/sink/local heads to become retrieval heads. It is not a valid general target. We instead use downstream cotangents from a separate calibration example's answer loss:

u_hq = d L_answer / d y_hq,  y_hq = sum_i p_hqi v_hi.

Define c_hqi = u_hq dot v_hi and freeze a,b,v,u while constructing the table. The scalar calibration objective is

F(theta) = sum_hq sum_i softmax(s_hq(theta))_i c_hqi.

Its exact finite gradient is

dF/dx_j = sum_hqi p_hqi [c_hqi−E_p c_hq] (d_i/T)
          [−a_hqij sin(d_i theta_j)+b_hqij cos(d_i theta_j)],

where x=T(theta−theta0). This automatically handles positive, negative, and zero output influence. Constant c across keys makes the whole row's frequency gradient zero; a head whose value mixture is irrelevant to the output is not forced toward any source. This avoids estimating a universal retrieval-head label.

**Useful exact first-order connection.** If cotangents and coefficients are captured at *every* attention application affected by a globally shared theta, summing these local direct parameter derivatives equals the full network's gradient dL_answer/dtheta at theta0. Upstream and downstream effects are already represented by reverse-mode cotangents. Freezing activations does not invalidate this derivative at the capture point. Capturing only late queries/layers gives a partial gradient, however; it is then only a local intervention surrogate.

The finite F(theta) away from theta0 still holds cotangents, hidden Q/K and V fixed, so its second-order/global improvement is not a theorem about full-model answer loss. Applying the exported table requires fresh prefill and fresh generation; decode-only key re-rotation does not recreate altered hidden states.

Calibration answer targets are allowed training information, separate from held-out evaluation. If only input-located record spans are available and no backward cotangents exist, the reference's support objective may be used on independently retrieval-qualified heads, but that qualification must be supplied and checked; all-head averaging is not a fallback. Native support-removal versus matched-wrong-block removal is a useful output-level qualification. Enrichment of attention alone is weaker and must be labelled accordingly.

## 3. Closed explicit step and its certificate

Let r_ij=hypot(a_ij,b_ij), delta_i=d_i/T, with T bounding all absolute distances. For each query/head row define

G²=max_i sum_j delta_i² r_ij²,
H=max_ij delta_i² r_ij.

These globally bound the squared score-Jacobian norm and score-Hessian operator norm with respect to x. For the support loss, a valid gradient-Lipschitz constant is 2(G²+H), because each of the two LSE Hessians has norm at most G²+H.

For the output-aware row F=E_p c, let C=max_i c_i−min_i c_i. Its Hessian is exactly

Hess F = E_p[(c−E_p c) Hess s]
       + E_p[(c−E_p c)(grad s−E_p grad s)(grad s−E_p grad s)^T].

Therefore ||Hess F|| ≤ C(H+G²). Sum/average constants with exactly the same row weights as F to get L. No Gaussianity, pair independence or low-rank closure enters this bound.

For descending positive theta0, set g=grad_x F(0) and gap radius

r_j^box = min{B, 0.49 T theta0_j,
                   0.49 T min(theta0_{j−1}−theta0_j,
                              theta0_j−theta0_{j+1})},

omitting absent endpoint gaps. B is an explicit maximum phase displacement at target length, default one radian; it is a trust budget, not a derived optimal universal constant. The nearest-gap box guarantees no pair-order crossing and positivity. This requirement is conservative, not a mathematical necessity for label preservation: keeping slot indices fixed already preserves labels, even if numerical frequencies cross.

**Frequency rule:**

x*_j = clip(−g_j/L, −r_j^box, +r_j^box),
theta*_j = theta0_j + x*_j/T.

The step minimizes g·x+(L/2)||x||² on this box. Since zero is feasible,

F(theta*)−F(theta0) ≤ g·x*+(L/2)||x*||² ≤ 0.

The reference returns both actual captured-objective change and this certified upper bound. Zero gradient returns the unchanged table. A pair with a=b=0 is unchanged. If the single step is too small to survive model float32 table export, that is an informative lack of actionable evidence at this conservative resolution; do not relabel a rounded no-op as a new method.

Under the one-radian default, the maximum added phase at W=32768 when T=131072 is 0.25 radians. This is an operator perturbation bound, not a short-task retention guarantee. The acceptance comparison must measure native-window tasks directly. No fixed positive B can guarantee retention for all frozen models: arbitrarily small logit margins give counterexamples.

## 4. In what sense this joins EVQ, MrRoPE, P2 and E1

The common object is finite signal-versus-distractor competition under a frequency table, rather than a requirement that every setting move frequencies the same direction.

- From scratch, weights and channel roles can co-adapt. After exchangeability, constant signal, and Gaussian distractor closure, log-MGF reduces to mean plus variance/2. With the additional Brownian/continuum kernel and boundary assumptions, this can yield the EVQ variational allocation. These assumptions are supplied by the respective derivation, not guaranteed by RoPE. I have not independently reread the EVQ full paper in this assignment and do not claim an exact finite-loss EVQ theorem.
- Frozen deployment must keep the learned a,b attached to original pair labels. Remote useful signal can favor smaller theta to preserve alignment over longer d, while distractor interference and local useful signals can favor the opposite motion. This gives the MrRoPE rationale as a regime, not a universal compression rule.
- P2 and the reported E1 slot28 decompression can be accommodated: the exact signed gradient is free to raise or lower theta_j. A prior that forces every middle frequency down excludes potentially useful decompression before checking evidence.
- Conditional Brownian gains and GapCapped failures show why smoothness or collision improvement alone cannot choose the sign. The common finite objective retains higher-order tails and original coefficients; allocation shape is an output, not imposed first.

This is a practical local construction, not a novel universal closed-form schedule, and not a proof that existing published EVQ and MrRoPE are exact minimizers of one realistic network objective.

## 5. Smallest useful full-model test for parent

Use the already selected deployment baseline table as theta0 and the same frozen model, gain, masks, runtime, prompts and output budget. Form one candidate table from a calibration subset separate from final evaluation. Use actual target-length calibration inputs; multiplying positions in native captures tests a fixed-feature stretch surrogate and must not be called native 128K calibration.

Primary path: one native baseline backward supplies output cotangents. Full affected-site derivatives give exact first-order answer-loss direction; if storage/cost restricts capture to late retrieval queries, declare that restriction and use the output-weighted local surrogate. Export exactly one candidate from the rule above. Parent can use input-only record spans to diagnose whether changed attention moves toward the queried records, without using gold values to choose support.

Run fresh prefill/full generation for baseline and candidate on the fixed held-out 32K and128K inputs. Reuse valid baseline outputs when prompt/runtime/table identities match. Report full-answer accuracy/binding, raw outputs and EOS, plus support diagnostics and captured-loss change separately. The construction is useful only if actual held-out behavior improves under the user's objective; the curvature certificate does not replace that test. A third amplitude/count variant is unnecessary because neither changes here.

A particularly informative outcome is lower captured F but worse answer loss: it falsifies finite-step transfer from the frozen activation/cotangent approximation. Lower answer NLL but unchanged/worse full answers separates smooth calibration objective from the requested generation endpoint. A positive result with all-site gradient requires comparison to ordinary same-budget frequency-only gradient calibration before novelty claims.

## 6. Assigned evidence and implementation cautions

1. `ROPE_GAP_CAPPED_RESULT_20260908.json`: 36 rows; 32K macro delta −0.02778, 128K −0.15972; paired wins0/losses7. Multikey 128K row0 changes correct3377632 to3377226, showing meaningful digit errors rather than merely EOS change.
2. `ROPE_MRPRO_BM_CANDIDATE_20260908.json`: exact parabolic increment minimizer of the *stated roughness objective*. OLMo middle exponents increase versus Mr by a total2.83333; outer bands/gain unchanged. Its own review notes compression differs from P2 in earlier middle slots. It does not assert task optimality.
3. `ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json`: 778 rows/arm; extended task-equal macro0.216243→0.254437, delta0.038194, with all five extended task deltas positive. Within-native-length refers to two S4 deployments, not Native original-table retention. Full read identifies7 duplicate prompt-hash groups,14 rows,771 unique hashes; one pair has differing scores despite identical output. NarrativeQA rows192 and84 at lines16105/17433 share SHA07ccfa... but have baseline.13333/.15385 and candidate0/.22222. Do not infer a dataset bug or truncation solely from this; it limits row-independence interpretation. The recorded F1 improvement remains a finite selected-pool observation.
4. `TWO_CORE_ROUTE_EVALUATION_20260909.md` is a design/failure review, not GPU results. It explicitly derives higher-cumulant and cross-pair counterexamples, identifies NOSA's existing post-RoPE summary, and separates attention mass from output utility. Its historical execution plans are not active instructions.
5. Archived review log lines1184–1210 explicitly catches the wrong claim that all successful routes move budget toward long scales. It records EVQ fast-density increase versus frozen-profile frequency lowering. Other claims in the archive are historical and not independently current-verified.
6. `eval_longbench.py` full read: fallback RoPE type can change yarn→dynamic/linear; a requested label alone is insufficient implementation identity. OOM single-example fallback truncates input (`fallback_used`, line1349). Custom frequency argument is passed only for adapter-backed entries (line2215), so this CLI should not be assumed to run a frozen custom-table-only baseline correctly. Its trace uses prompt_sha1 at line1604, whereas the OLMo result artifact uses SHA256, so this script alone does not identify the duplicate artifact's producing path or prove truncation. Use the established parent runtime for the new full-model comparison.

## 7. CPU receipt

`python .agents/rope_unification_20260910/reports/astra09_rule.py` passes:

- Rare-key full-covariance counterexample reverses the exact ranking.
- Exact frequency gradient matches central finite differences for support and downstream-weighted objectives.
- Projected table stays positive/strictly descending and preserves slot labels.
- Zero coefficient plane and constant-value-output-utility row leave frequencies unchanged.
- Actual captured objective decrease satisfies analytic quadratic upper bound.

Example support loss3.09644258→3.02863919, actual−0.06780340≤bound−0.03648199. Output-aware toy−0.08884738→−0.08983000, actual−0.00098262≤bound−0.00049039. These verify algebra/reference only, not Qwen generation or performance.
