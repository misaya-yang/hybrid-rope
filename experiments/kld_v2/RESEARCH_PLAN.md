# KLD v2 — next experiment, separate from rotary-budget paper

Status: independent mathematical reproduction and implementation feasibility review.
No checkpoint adaptation or full-model KLD performance is claimed.
Source: `DESIGN_SOURCE.md`, supplied by the user on 2026-09-08.

## Judgment

The derivative interpretation is precise and useful. It does not establish a
new universally superior memory family: an additional ordinary delta state
with independently controlled erase/write can approximate the same derivative
by finite differences. The next experiment must distinguish a usable conditional
advantage (information, conditioning, or compute) from simply doubling memory.
There is no present basis for treating KLD as semantic event order.

The independent `audit.py` verifies derivative equivalence and chunk/serial
agreement below 5e-16 in float64, and observes first-order finite-difference
convergence. It also checks the fixed-feature linear-capacity bound and computes
the interference counterexample. These are mathematical tests, not checkpoint
results. The user's original audit script was not supplied beside the document;
this is a new implementation, not a claim to have rerun that exact script.

## Concrete checkpoint entry point

The official Qwen3.5-0.8B config was inspected on 2026-09-08:
24 layers, 18 linear / 6 full attention; key and value heads 16 each, both width
128; recurrent state FP32. Last linear layer is **zero-based index 22**; layer23
is full attention. Pin a model revision and all files before any runtime study.
[Official config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json).

The server's installed Transformers 5.15.1
`transformers/models/qwen3_5/modeling_qwen3_5.py` was read. Its reference recurrence:
1. normalizes q/k, scales q by 1/sqrt(128), uses FP32 recurrence;
2. sets beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias), D=exp(g);
3. decays S, reads the pre-update state, writes beta*k*(v-k^T*Sbar), then reads S;
4. applies the original gated RMSNorm and `out_proj` after the recurrent read.
Its scalar per-head D fits the proposed shared-decay derivative. L2 normalization
with epsilon yields norm(k) at most one rather than an exact symbolic unit vector;
the nonexpansive argument extends with beta*||k||² in [0,1]. Actual loaded kernels
can be replaced by dispatch, so capture which fused/reference implementation ran.

The fixed-driver causal diagnostic should intercept q/k/v/g/beta **after** the
original convolution/projection and normalization conventions, not raw embeddings.
Capture the scaled query convention as well; otherwise auxiliary read magnitude
will be wrong by sqrt(128). Keep original S/cache updates and output gate intact.
The reader residual belongs after the base module output, with its last projection
factor zero-initialized. Initial-output parity needs full prefill and cached decode.

One extra FP32 state at this layer costs 16*128*128*4 = **1,048,576 bytes per
sequence**, before any reader/temporary storage. This is a computed state budget,
not an observed end-to-end memory or throughput saving.

## Stage A: decide whether useful information is present

Use complete tokenized records at 4K, 8K, 16K, 32K, plus short task checks. Freeze
64 independent document/entity families for development and 128 for untouched
mechanism evaluation. Balance old/current value, repeated mention, cross-attribute
update, distractor, and absent-answer questions within each family. Freeze each
whole counterfactual family to one split. Values are random strings with an exact
canonical answer and a declared abstention token. No token-local writer masking.
Do not tune on LongMemEval or use public-test has_answer to select context.

At layer22, capture complete true model drivers. Track source coefficients for
selected source writes, with the entire competing-source covariance included.
Compare current read S, [S,H], and [S,X] using the same query and equal extra
FP32 state. Use linear-MMSE diagnostics under independently injected payloads,
report SVD cutoff sensitivity and condition number, and separately repeat at FP32.
A source-norm increase without better recoverability is not a pass. The sum-capacity
bound applies only while the feature matrix/query is fixed across source targets.

X dynamics: X_t=(I-e_t beta_t P_t)D_t X_(t-1)+w_t beta_t k_t v_t^T.
Include the finite-difference neighborhood e≈1 and slower erasure, with write
scaling independent. First diagnostic compares a declared small development grid,
then locks the rule for the held-out families. Do not choose a duplicate-state-only
baseline. A successful H diagnostic must survive a well-conditioned X comparison.

Stop this entry point if H has no added recoverability or X matches it within
measurement uncertainty. This does not prove every layer or learned architecture
impossible. No reader training is justified solely by a large H norm.

## Stage B: reader-only adaptation if Stage A supports it

Keep original q/k/v/beta/D and every backbone weight frozen. Preserve full attention,
convolutions, FFNs, residual paths and the native cache. Compare:
- C: current-read residual adapter, with duplicated current feature slots so its
  reader tensor shapes match the two-input reader.
- X: current plus an equal-sized extra ordinary delta memory.
- H: current plus derivative state.

Use the same positionwise reader architecture, low-rank output shape, optimizer,
training examples/order, answer+EOS targets, and three paired training seeds.
Zero the output factor only; retain a nonzero input factor so gradients can start.
Current-only is the adaptation control, not the equal-state efficiency comparator.
For the strict initial reader-only stage, freeze the X erase/write settings chosen
on development data; give all arms the same number of development trials. If X
rates are optimized jointly later, disclose that extension and account for its
additional parameters; do not claim exact parameter equality by ignoring gates.
No test-driven layer, rate, or rank choice. Reader rank and adaptation token budget
must be frozen after a short resource probe, before comparing held-out outcomes.

Measure complete generated-answer correctness, terminal EOS and format separately
from answer NLL. Report each task/length, including current-state regressions and
absent-answer false positives. Add independent ordinary-text NLL. Only if complete
text answers improve against X should the study expand or consider from-scratch
architecture training. There is no automatic launch of a second large training
program while the rotary-budget core is running.

## Causal path confirmation

For a fixed driver sequence, inject two payloads only in a chosen old-write value
channel. Propagate H for each payload. At the receiver query, replace H alone;
keep S, query, all other caches and residuals fixed. Include S-only replacement,
archive permutation, and matched-norm nuisance interventions. Trace whether the
reader output follows the injected payload. This proves use of that injected
path, not that natural text was originally encoded in it. Whole hidden/cache swaps
between two texts are unsuitable for archive-specific attribution.

## Close baselines and novelty boundary

- [MDN](https://arxiv.org/abs/2605.05838) already has momentum-like multiple states
  and a chunkwise algorithm. A second state and a parallel solve are insufficient
  novelty on their own.
- [GDN2](https://arxiv.org/html/2605.22791v1) separates erase/write controls; its
  update must not inherit the symmetric stability statement without rederivation.
- [SFDA](https://arxiv.org/abs/2607.11897) requires distinguishing the default
  phase mechanism from its partial-permutation extension.
- [EDA](https://arxiv.org/abs/2606.26560) separates erase/write addresses, a nearby
  alternative explanation that an eventual architecture claim must address.

These pretrained architecture results are not directly comparable scores for the
small proposed adapter study. An initial success means conditional quality/cost
value for this preserved checkpoint and entry point; it does not mean KLD beats
all these architectures. External validation can then use original-history
LongMemEval cleaned v1, with any shortened adaptation explicitly renamed.
