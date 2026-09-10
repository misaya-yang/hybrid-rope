# Astra04: corrected full-row calibration and concrete allocator

## Main result

The 6Pro objective is a valid fixed-state position-stretch distillation loss, but it contains no extra-key competition. A mathematically exact correction under an explicit distractor-invariance hypothesis is

\[
L_{\rm full}=D_{KL}(T\Vert Q_O)+\log(1+Z_D/Z_O),
\]

where T is the native teacher on original keys O, Q_O is the candidate softmax conditional on O, and Z_D,Z_O are candidate exponential-logit sums. This equals KL of the teacher padded by zeros against the augmented student. It is not a theorem that unrelated natural-text blocks deserve zero teacher mass.

## Source facts and map correction

Full 6Pro attachment was read, together with the complete capture implementation. `experiments/nongeometric_screen/pro_block_calibration.py:22-30` sets native frequencies with the common MrPro gain, documents 0:4 for fitting and 4:6 for validation, M=4096 and S=4. Lines43-46 select four query heads and only q=32767. Lines53-55 retain all raw keys and values for two KV heads; lines60-77 use key-minus-query signed separation, exact sin/cos, and the score multiplier including gain squared. Lines82-83 then run the actual original layer, so the stored states are native/common-gain states, not recomputed candidate states.

With W=BM=32768, B=8, the written map f(bM+r)=SbM+r ends at118783, not131071. To fill exactly128K causal slots, use f(bM+r)=(S-1)M+SbM+r. This inserts a leading12288-token gap and seven inter-block12288-token gaps:98304 distractor slots. Common translation preserves every original query-key displacement. This correction does not turn stitched raw states into an actual128K forward pass.

## Exact decomposition and local dominance

Partition original keys into the eight source blocks. Let p_b=sum_{k in b}T_k and q_b=sum_{k in b}Q_{O,k}; T_b,Q_b denote normalized conditionals. Then

\[
D_{KL}(T\Vert Q_O)=D_{KL}(p\Vert q)+\sum_b p_bD_{KL}(T_b\Vert Q_b).
\]

Thus a teacher with p_local=0.999 can allow arbitrarily bad far-block conditional preservation at negligible average loss. Equalizing conditional block terms is an explicit scale-priority choice, not a correction to the native probability distribution.

A concrete answer-free objective is

\[
L_{bal}=D_{KL}(p\Vert q)+B^{-1}\sum_b D_{KL}(T_b\Vert Q_b)+\log(1+Z_D/Z_O).
\]

Compute each conditional from block log-softmax directly, including tiny-mass blocks, to avoid underflow. Equal block weights define the intervention's uniform source-block evaluation prior. Retain native block-mass KL: globally flattening T would actively destroy local-attention structure. This objective is zero exactly when original-key probabilities match T and distractor mass vanishes (finite logits cannot attain the latter). Report ordinary full KL, every p_b, conditional block KL, and distractor mass separately; balanced-loss improvement alone cannot establish useful far relations. Nearly uniform low-mass block teachers might just encode noise; this is a real limitation, not justification to select semantic labels from evaluation answers.

Identity-map guard: calculate the same block-balanced original-support loss at f(p)=p with D empty. Require its fitting aggregate no worse than MrPro's value. This concretely preserves an existing reference rather than introducing a free weight between short and long losses. The source native teacher gain is common gain, and is not operational Native gain1.

## Extra distractors: what can be fit now

Use each fit document's Q with its own32K K for O. Fill the98304 gaps with the other three fit documents' K, independently at each same layer and correct KV-head index. Keep their internal token order in contiguous chunks. Existing full K on both KV heads makes mismatched selected Q-head coverage irrelevant for donor keys. Average over the three cyclic donor orders so donor identity is not confounded with distance. This is a small exact symmetry average, not a map grid. Do not use validation documents as fit donors. No additional GPU capture is required.

Holdout has only two documents, so the same nonrepeated three-donor construction is impossible using holdout alone. Honest options are (a) validate original-support position stretch using docs4,5 and separately report augmentation using repeated chunks of the other heldout doc, explicitly as a changed donor distribution; or (b) use the already-frozen fit donor bank with heldout queries and disclose that only source queries/original docs are held out. Option(b) most directly checks query generalization under the calibrated interference distribution. It is not a wholly new-document distractor holdout. Do not silently count reused donor blocks as independent observations.

Every stitched donor K was generated from a different prefix under native RoPE. Therefore query/donor compatibility and earlier-layer states differ from real insertion. Zero-padding the teacher assumes the inserts should be ignored. A counterexample is a donor with K identical to a desired key: it necessarily competes, and if its V is also identical then attention-output behavior can be unchanged while the penalty is positive. Hence this penalty tests original-key identity retention, a stronger criterion than functional retention. Existing V can provide an auxiliary head-output difference diagnostic, but may not be used to turn this into end-to-end task evidence. Conversely, copying original K/V four times makes the marginal softmax over duplicates exactly T when duplicate logits match: no factor4 distractor penalty is justified in that case. The semantics of augmentation determine the correct teacher.

## Concrete constrained allocator

Parameterize x_j=-log(nu_j/nu_ref), x_0 fixed and x_63 fixed to MrPro endpoints. Use x itself, with linear inequalities x_{j+1}-x_j>=epsilon, where epsilon is a numerical order tolerance rather than a frequency prior. For the causal comparison also fix sum_j x_j=sum_j x_j^Mr. This leaves61 dimensions; no prescribed knees, signed per-slot labels, or geometry roughness objective.

Start at MrPro. Minimize mean fit L_bal for the one declared block-stretch intervention with distractor fill, subject to the identity-map guard above. Average documents equally, then layers equally, then saved heads equally. Stream full keys and logsumexp without storing per-pair coefficients across all docs. Use the exact finite-phase objective in a constrained optimizer (e.g. SLSQP with analytic first derivatives), not a one-shot Taylor step. Freeze one final table using fitting objective convergence; evaluate docs4,5 once. Do not search optimizer checkpoints using task answers. If the solver cannot improve while respecting constraints, keep MrPro and report that the declared calibration problem did not supply a direction.

For row r and key k, with z=sigma*sum_j(C_jk cos(d'_k nu_j)+D_jk sin(d'_k nu_j)),

\[
\partial z_k/\partial x_j=\sigma d'_k\nu_j[C_{jk}\sin(d'_k\nu_j)-D_{jk}\cos(d'_k\nu_j)].
\]

For the balanced original-support terms, derivative w.r.t original z_k is

\[
(q_b-p_b)Q_b(k)+B^{-1}(Q_b(k)-T_b(k)).
\]

For distractor penalty its derivative is Q_aug(k)-Q_O(k) on O, and Q_aug(k) on D. Sum, then contract with the finite-phase derivative. This retains cross-pair effects through the full summed logits and softmax, without a pair-additive loss assumption.

After the table is frozen, use v=x_fit-x_Mr; scale both +v and-v by the same largest factor<=1 preserving order if a reverse control is needed. Both directions keep endpoints and cumulative compression. Evaluate positive, reverse, and MrPro with identical gain on newly generated independent full-model long tasks. Native identity, sparse-position stretch, and dense inserted128K are three distinct evidence conditions.

## Mathematical checks and obstructions

A CPU scalar test checked padded KL against KL(T||Q_O)+log1p(Z_D/Z_O), with T=[.7,.2,.1], original logits[1,-.5,.2], donor logits[.4,-.7]; both0.45529644748142606, difference0. Map endpoints and98304-slot count also checked. No GPU jobs were launched.

The single-query-per-document capture cannot identify behavior across query positions or generalize query-local versus block-local effects. Equalizing layers or blocks expresses a test priority; it supplies no causal task-importance labels. Native/common-gain states may already differ from native/gain1. Full-prefill candidate states must be evaluated separately. Historical cross-cache file `docs/research/ROPE_BM_CROSS_CACHE_RESULT_20260908.json` documents source-cache-dependent MK2 outcomes and a VT original-versus-cached continuation mismatch; the two-case diagnostic directly cautions against equating fixed-cache improvements with generated-answer improvements.

## Scope of unification

The allocation coordinates unify EVQ and MrRoPE exactly as table coordinates. This loss is a proposed frozen-compatibility estimator. It does not derive EVQ's analytic spacing cost from real task loss, and does not prove a scratch-trained optimum. The constructive rule is falsifiable: it predicts an allocation before target answers are inspected. Failure at dense128K after success on frozen stitched rows specifically falsifies transfer of this estimator, not frequency-allocation degrees of freedom in general.

## Completed audit addendum

All102assigned files (1,107,653bytes) were subsequently read in contiguous pages, including all result JSON rows and historical failure/decision appendices; coverage receipt is `astra04_coverage.json`. No assigned text omission remains. Oversized initial reads were discarded as coverage evidence and redone. The additional full6Pro attachment and entire capture script were also read. Remote binary captures were not inspected by this agent.

The analytic combined balanced-loss + distractor-loss gradient was checked against centered finite differences on a seeded3-block,21-key,5-frequency CPU fixture; maximum absolute error2.3480405640652346e-10. This validates the displayed derivative, not the model-quality hypothesis.

Additional source-specific limitations found during full ingestion:

- `paper-2027/claude_code_workspace/reports/ROUND11_OLMO_RESULTS_20260905.md`, section9, explicitly downgrades earlier mechanism narratives to hypotheses: improved termination and answer-containing substrings were insufficient for exact generation, and the claim that LoRA only repaired formatting conflicted with improved strict answers. Apply the same restraint to calibrated attention KL.
- `experiments/rope_operator_family/results/20260909_gpu/REPORT.md` records output-KD natural NLL3.942717 versus score-fit9.989853 while score fitting had smaller position-response error than output-KD. Both missed the frozen retrieval answer. This is a different channel-compression experiment, and is used only as an objective-transfer warning.
- `experiments/rotary_budget/DESIGN_SOURCE.md`, appendixB, reports a controlled full48-condition softmax fit where E16 wins narrow responses against G16 but loses wide responses at4096; U16 beats E16 in the declared equal-width mixture. It illustrates why a declared calibration prior must remain explicit. It does not determine this Qwen3B deployment table.
- Cross-document donor raw states include artificial native document-start/sink states. Repeating such states in gaps can create additional sink competition absent from an actual uninterrupted128K document. Report donor sink behavior if augmented fits look anomalous; do not silently delete difficult donor keys after observing losses.

Memory registry was consulted for objective fidelity only (`MEMORY.md:41`); all concrete source facts above were freshly read.
