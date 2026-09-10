# Sol18 — operator map, label oracle, and the exact Qwen model-level test

Status: all 101 assigned files (1,107,673 bytes) were read in full. The 1,536-line development JSONL was read in contiguous bounded line pages; every truncated batch was superseded by smaller complete reads. Astra01–08 and Astra09's executable rule were additionally read. No GPU/model job, runtime edit, paper edit, or extra agent was used.

## Result

The current repository contains four different computational objects that must not share an unlabeled proxy:

1. **Static dense RoPE deployment.** Qwen keeps every key and changes the labeled per-pair frequencies used by every attention layer. This is the actual EVQ/MrRoPE allocation question.
2. **Sparse block selection with an unchanged reader.** The native sparse experiments rank physical pages, then read selected original post-RoPE K/V with one softmax. Exact-mass variants scan all keys and are diagnostic oracles, not efficient sparse methods.
3. **Prefix-only KV retention.** PM-Keep scores a question-blind prefix, physically gathers a fixed number of original K/V positions per KV head, then appends question/answer K/V at their original logical positions. Its future-query sampler is a selection proxy, not a frequency allocator.
4. **Dense low-rank operator replacement.** `rope_operator_family` replaces full Q/K/V attention with a learned compact content-plus-rotary operator. It still reads every token and changes representation capacity, projections, values, and cache width. Its score/output distillation results cannot be relabeled as sparse selection or static-table evidence.

The usable exact model-level test for the root's direction is therefore **teacher-forced whole-model target CE on source-controlled paired record tasks under a single shared static Qwen table**, followed by frozen family-disjoint full generation. It directly optimizes the requested output, keeps slot identity, and needs no invented head-importance weights. Parser-derived record roles are retained as diagnostics and counterfactual controls rather than substituted for target CE.

## 1. Actual operator map and what transfers to a static table

### Dense static RoPE

For Qwen split-half pair (j), signed separation (d=p_k-p_q), and pre-RoPE coordinates after the checkpoint's actual Q/K normalization, the contribution is

\[
z_j(\nu_j)=\gamma\{C_j\cos(d\nu_j)+S_j\sin(d\nu_j)\},
\quad C_j=q_jk_j+q_{j+64}k_{j+64},
\quad S_j=q_{j+64}k_j-q_jk_{j+64}.
\]

This exact labeled finite-phase object transfers. The slot index, layer, query head, KV head, key identity, target/distractor role, signed distance, table gain, and complete normalizer must stay attached. A frequency multiset or projection norm loses the learned association.

The existing natural capture confirms the arithmetic but cannot supply role supervision: it uses Native frequencies with MrPro's common gain, only the final natural-text query, four sampled heads, and all keys (`experiments/nongeometric_screen/pro_block_calibration.py:20-33,43-77`). It has no question query or requested-record label. It is useful as a compatibility control, not as the allocation target.

### Sparse/native readers

The Qwen sparse reference selects B64 pages per query head, retains sink/local keys, and uses the original native K/V reader. Support repair locates record spans from query keys without reading answers, but then restricts them to eligible remote B64 pages (`experiments/native_sparse_position/prepare_support_oracle.py:11-43`). That remote-page restriction belongs to the sparse selector. A static-table calibration must keep every causal key, including local, sink, question, self, and background tokens, in the normalizer.

The NOSA reference is another operator: 32/16 mean windows, per-head softmax, shared-GQA aggregation, block max pooling, and a QK/CIS quota union. PC2 changes its cached block statistic while preserving raw K/V and a learned per-key CIS bias. Neither its shared selector score nor CIS has an analogue in ordinary Qwen static RoPE. The transferable lesson is negative: head-dependent normalization and GQA aggregation can change routing, so a per-head mass proxy is not a whole-model label.

### PM-Keep

PM-Keep's prefix boundary is sound: offset mapping assigns a straddling BPE token to the future suffix and verifies lossless prefix/suffix reconstruction (`experiments/pm_keep/prepare.py:39-74`). Its role locator maps query key plus ordinal to an exact source record; this semantic contract is reusable. Its scoring operation is not: sampled future positions rotate prefix Q against original post-RoPE K, average normalized attention, and select a fixed KV budget. The actual task question is intentionally invisible. That is appropriate for retention, but insufficient to tell a frequency allocator which record supports the requested answer.

The assigned 1,536-row development file makes the label problem concrete. Every pair has two `relation` rows and one `content` row. Relation rows are essentially ceiling/high-margin, while content rows are usually wrong/negative-margin. Pooling them as generic correct-record evidence would let the easy relation role dominate the content-binding failure the allocation is meant to improve.

### Operator-family compression

The compact operator stores content (c) plus rotated (k_R), reconstructs a different key/value operator, and reads every cached token. The observed full-model result is decisive about proxy transfer: native validation NLL is 2.6514, score-fit NLL 9.9899, and output-KD NLL 3.9427; none of the compact fits restored the retrieval answer. At layer 27, attention-KD has much lower local response errors than progressive KD, yet both fail the task. This is evidence that an operator-fit metric is not the static-table target, and no sparse-attention claim follows because no key selection occurs.

## 2. Correct source-controlled labels

Use a four-way family: `content_swap` by `query_ordinal_swap`. The repository already checks the defining invariants (`experiments/nosa_position/test_data.py:18-34,59-83`): each history contains the same record inventory, requested ordinal and value assignment change independently, opposite cells can share the same answer, and dev/test token content is disjoint.

For each rendered prompt define, from input text and tokenizer offsets:

- (T): the requested key+ordinal record span;
- (H): other occurrences of the same key;
- (O): other explicit records;
- (B): remaining background/structural tokens;
- (Q): question tokens, including copies of the queried key.

These are lawful relation labels, not claims that every token in (T) or every attention head is causally useful. Keep key-token and value-token subspans separately. The PM locator's ordinal logic is preferable to the RULER unique-key locator for repeated keys. Do not select only rows where Full/MrPro is already correct; include complete predetermined families. Otherwise calibration becomes outcome-conditioned headroom selection like the deliberately narrow two-row target-record oracle.

Choose value strings that are distinct single tokens under the actual Qwen tokenizer and native chat template. The primary target is the full-vocabulary next-token log probability of that value. Also teacher-force the terminal EOS and, if multi-token values are later used, every answer token. This prevents a target-vs-one-wrong-value margin from hiding premature EOS, formatting, or another vocabulary error.

Role diagnostics should report exact full-row log masses

\[
\log Z_T-\log Z_{\bar T},\quad \log Z_T-\log Z_H,
\]

plus sensitivity-weighted value contributions. They must not replace target CE. Query and local keys stay in (\bar T); excluding them creates a conditional mass that can improve while whole-model CE worsens.

## 3. Exact model-level calibration test

### Frozen inputs and table

Build actual contiguous 128K prompts with the four-way record families, family-disjoint calibration/validation/test splits, dense distractor backgrounds, and the question at the end. Use actual logical positions (0,\ldots,L-1). Do not create 128K by stretching/stitching 32K captured K/V: that preserves neither added-key competition nor upstream state formation.

Use one shared 64-pair table in every Qwen attention layer. Parameterize

\[
\nu_j=\omega_j\exp(-x_j).
\]

Keep labeled slot order. For the evidenced MrPro face, fix (x_j=0) for (j\le23), (x_j=\log4) for (j\ge40), and optimize only slots 24–39 subject to monotone frequencies. Keep the common MrPro gain fixed. If endpoints are opened, name that a range-plus-allocation experiment rather than a pure middle-allocation test.

### Primary objective

With all model weights frozen, minimize the family-balanced teacher-forced loss

\[
L_{\rm target}(x)=\frac1{|\mathcal F|}\sum_{f\in\mathcal F}
 \frac14\sum_{c,q\in\{0,1\}}
[-\log P_{x}(y_{fcq}\mid prompt_{fcq})-\log P_x(EOS\mid prompt_{fcq},y_{fcq})].
\]

This is exact whole-model CE under the installed finite table: every layer, value path, residual, MLP, decoder vocabulary competitor, and all 128K keys participate. It avoids assigning head weights. Use natural 32K full-row KL and the paired source-world CE as hard/reference constraints no worse than MrPro on calibration; report each family, rather than hiding a failed role in one mean.

Generate one candidate by constrained projected optimization from MrPro with exact forward/backward CE and backtracking on the exact loss. Freeze it before validation. This is supervised frequency calibration, not a universal analytic formula. If memory makes full differentiable 128K training impossible, finite differences or checkpointed exact recomputation are valid engineering alternatives; a detached Q/K replay is only a proposal filter and must not be called the model-level test.

### Minimal decision comparison

On family-disjoint held-out actual 128K rows, compare:

1. MrPro, fixed common gain;
2. the one frozen CE-calibrated table;
3. the equal-norm mirror direction about MrPro, scaled only as needed to keep order/endpoints.

Reuse P2/E1 as contextual baselines if their exact prompt/gain/position contracts match; otherwise do not mix their percentages into this paired test. Primary readouts are target-token CE, teacher-forced answer+EOS CE, and complete greedy value+EOS. The mirror distinguishes a useful direction from generic perturbation/fitting. Report paired family outcomes and all regressions. A candidate that lowers detached attention loss but not target CE has failed the requested mechanism.

## 4. Hidden implementation traps

1. **Gain double counting.** The natural capture applies Native at MrPro's comparison gain, then reconstructs logits with `module.scaling * common_gain**2` (`pro_block_calibration.py:22-30,56-77`). If candidate cosine/sine tensors already include the attention factor, multiplying gain again changes logits by (g^4). Record whether gain multiplies Q and K tables, logits, or both; apply it exactly once per Q and K.
2. **Operational Native is a different control.** Native/gain1 and Native-frequencies/MrPro-gain are not the same model. A result at common gain cannot be reported as operational Native retention.
3. **Sign/layout mismatch.** Repository Qwen code uses split-half pairs, not adjacent dimensions, and the capture records (d=key-query). Reversing (d) flips the sine term. Qualify exact replay against runtime log probabilities before optimization.
4. **Target-token boundary.** Text decode/encode can alter whitespace and assistant-header tokens. Derive target IDs by applying the actual chat template and concatenating the answer once; verify the prompt is a token prefix of prompt+answer. Include terminal EOS explicitly.
5. **Gradient silently detached.** Existing table installers copy NumPy/FP32 values into rotary buffers, many runners use `inference_mode`, and cached cos/sin may be detached or rebuilt outside autograd. A CE optimizer must prove nonzero finite gradients for the table and match centered finite differences on a tiny complete model before trusting the 128K path.
6. **Stale cache after table changes.** Post-RoPE cached keys belong to the table that formed them. Changing frequencies between steps requires a full fresh prefill (or exact raw-K rerotation plus upstream-state recomputation, which is still a full-model problem). Never reuse candidate-A K/V for candidate-B CE.
7. **Fake long positions.** Sparse position IDs or a block-stretch map test phase transport with a small key set. Actual 128K introduces roughly four times the keys and changes every upstream state. Keep these evidence conditions separately named.
8. **Mask/normalizer truncation.** Source-only, T-vs-H, selected-page, and original-key KL objectives omit competitors. The target CE test needs the real causal mask and all keys. A B64 oracle that scans all keys to select pages is still a sparse diagnostic, not the dense table test.
9. **Outcome-selected rows.** The two-row target oracle deliberately chooses Full-correct examples. Frequency calibration must use predetermined complete four-way families and split at family/material level.
10. **Argmax has no training gradient.** Optimize teacher-forced CE; use greedy generation only after freezing the table. Do not backpropagate through token selection or constrain decoding to the gold vocabulary.

## 5. What the mathematical unification can honestly claim

Astra01/03/06 provide compatible conditional laws: role-conditioned signed means/covariances, exponential-moment softmax bounds including finite distractor count, and exact finite labeled allocation under special covariance structure. They explain why EVQ can arise from constant protected signal plus coherent nuisance and why Mr-like positive remote signal enters with the opposite sign. They do not supply Qwen's role means, covariance, MGF, or downstream head/value labels.

Astra09's executable step improves an exact captured-attention support or frozen output-cotangent objective, with labels and finite-phase normalization retained. It is a useful proposal filter. Its output cotangents and Q/K/V are frozen, so it cannot substitute for the whole-model CE calibration above when all-layer frequency changes alter upstream states. The correct empirical bridge is to test whether its predicted direction agrees with the exact CE gradient on the same source-controlled calibration families. If they disagree, use the model-level target and diagnose which frozen-state assumption failed.

For scratch training, exchangeable count-density models can justify an EVQ-like initializer under explicit equal-loading/shared-noise assumptions, then the model must be trained and evaluated. For frozen deployment, the finite labeled table and learned slot associations are the object. The common principle is signal-versus-interference under the actual adaptation lifecycle, not a theorem that Cosh and MrPro are the same optimizer.

## Decision

Proceed with the source-controlled whole-model CE calibration only after its target/gain/position/gradient qualification passes. The existing natural captures cannot be relabeled into this evidence because they lack question queries and correct-source roles. The result should be one static table plus its mirror, not a curve grid. If the candidate improves calibration CE but loses family-disjoint 128K generation or source constraints, reject the frozen transfer claim; do not rescue it with head mass, covariance fit, geometry distortion, or the dense operator-family results.
