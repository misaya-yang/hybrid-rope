# A concrete composition to test: read-conditioned relative bias

Status update, 2026-09-09: this proposal has been withdrawn from the current
main experiment line. Its CPU implementation is retained as an untrained
exploration. The current source-position-before-compression candidate and its
evidence requirements are described in
[the literature and evaluation audit](SPARSE_POSITION_EVALUATION_LITERATURE_20260909.md).

2026-09-09. Work remains theory, experiment design and CPU code preparation.
GPU is intentionally off. This is an unvalidated candidate, not a claim of
improved language-model quality or an established new positional-encoding family.

## Intended contribution and reason for this construction

The user permits combinations of existing components. The intended contribution
is a useful way to connect positional computation across sparse/global readers
in a hybrid model, with measured capability and cost. It is not a new frequency
curve, a claim that recurrent layers erase every reference, or a requirement that
every model must use RoPE.

The prior RefCarry proposal used the average of reference rotations before one
softmax. Its exact operation is a geometric pool of reference-conditioned
distributions. This can suppress alternative references. If a reference denotes
uncertainty about WHICH occurrence to read, the natural operation is instead an
arithmetic mixture of separately normalized reads. If the intended semantics is
agreement among several constraints, a geometric pool may be appropriate. Neither
is universally superior. This distinction determines the counterfactual experiment.

TAPE already contextualizes position fields using attention; RePo predicts
positions from hidden states; NTM already combines content addressing with a
relative location operation. These are reusable components and mandatory nearby
comparisons, not reasons to invent another supposedly first-ever addressing theory.

## Operator

Let S be the existing selected-key set for a reader head, with original logical
addresses p_j, original native logits s_j, and its original values v_j. A prior
writer for the SAME query token has actual attention probabilities over addresses.
A learned row-stochastic head mixture aligns writer and reader heads. Keep a small
set of reference addresses a_r and their actual masses mu_r, with sum(mu_r)<=1.
Do not renormalize away the omitted probability. Set nu_0=1-sum(mu_r).

With an explicitly supplied fixed Fourier feature basis,

    phi(d) = [cos(omega_1 d),...,cos(omega_F d),
              sin(omega_1 d),...,sin(omega_F d)],
    u_i = W_ref h_i / sqrt(2F),
    b_r(j) = u_i^T phi(p_j-a_r),
    P_0 = softmax(s),
    P_r = softmax(s+b_r),
    P = nu_0 P_0 + sum_r mu_r P_r,
    o = sum_{j in S} P(j) v_j.

All branches have exactly the same existing sparse support and causal mask.
A native sink, if present, is included in each normalizer with its original logit
and zero value. Its mass is not silently redistributed to visible tokens.

W_ref is zero initialized. The whole reader therefore starts as the native
reader, while gradients into W_ref are nonzero in general. Head-mixture gradients
can begin after W_ref moves from zero; there is no requirement that both factors
receive a gradient at the first step. This is a learned retrofit, not a
training-free claim. Training data and opportunity must match relevant controls.

We ADD a relative bias to the native scores. We do not reinterpret a pretrained
query as if it were expressed relative to a new origin. This avoids assuming that
the native hidden state has not already encoded its needed reference. For NoPE,
s_j is its native content score. For RoPE, s_j includes the original rotation.
For a relative-value/shared-KV reader, values and the original output-frame
conversion stay untouched. This changes selection probabilities, not the value
coordinate frame, so it does not require the query-only/full-frame equivalence
that failed in the prior V4 analysis. Compression of payloads is not repaired by
this definition, and excluded sources remain excluded.

References denote original token addresses (or the architecture's explicitly
declared compressed-entry addresses). They never denote ranks in a changing top-k
list. A compressed block address is not falsely interpreted as every source token
inside that block. The first implementation supports one query at a time; batched
prefill and a production sparse kernel are not implemented.

## Direct consequences and their limits

1. **Native initialization.** With W_ref=0, every branch equals P_0. The CPU
   implementation accumulates differences from P_0 and reproduces it exactly.
   A real native-kernel parity check remains necessary before model experiments.
2. **No premature reference averaging.** For any retained reference, P(j)>=mu_r
   P_r(j). A second reference cannot negate that contribution by cancelling its
   phase. This does not prove the reference or its predicted answer is correct.
3. **Truncation bound.** If the full writer mixture is Q and omitted reference
   mass is epsilon, replacing just those branches by P_0 gives TV(P,Q)<=epsilon.
   Proof: P-Q is the sum of omitted masses times (P_0-P_r); apply the triangle
   inequality. With a common value set of diameter D (including zero if there is
   a sink), ||o_P-o_Q||<=epsilon D. This bounds this particular truncation only;
   it does not bound the error from a wrong writer, sparse exclusion, or the LLM.
4. **Relative and incremental features.** Shifting both key and reference logical
   addresses by the same amount leaves b_r unchanged. Appending later positions
   does not rewrite old position features. These are structural properties, not
   claims of task-level translation invariance or unknown-length extrapolation.
5. **No free uncertainty benefit.** If references give identical P_r, or a single
   reference carries all probability, arithmetic and geometric pooling coincide.
   At zero/small bias the two also agree through first order. A pooling advantage
   should emerge, if at all, when the reference-conditioned reads genuinely
   disagree. A point-reference baseline is therefore especially strong.

## Computation and memory

The usual Fourier addition identity factors b_r(j): rotate u_i by a_r, then dot
it with the immutable phi(p_j). This avoids an R by S by 2F feature tensor. Native
content QK scores are computed ONCE. After branch normalizations, combine their
key probabilities and perform ONE value aggregation, since every branch reads
the same values in the same native frame. This is ordinary distributivity, not
a claim that a specific fused implementation already achieves this traffic.

Potential overhead is O(R S F) bias work, O(R S) normalizers, and a position
feature cache of O(L F) if used, shared across layers with the same basis. At
1M tokens, 64 real BF16 features alone use 128,000,000 bytes; they are not free.
The current reference computes selected features on demand instead. Per-query
writer state stores a bounded number of integer addresses and masses per head.
Extracting those masses from an actual Flash/sparse writer is additional work
that must be implemented and timed; native indexer scores are not automatically
the writer's attention probabilities.

R=2 is the smallest diagnostic setting that can represent two alternatives; it
is not a theoretically optimal constant. The basis is supplied explicitly and
must be shared with the matched geometric/current-position controls. There is no
frequency search proposed here. Native NoPE does not provide a native frequency
table; that case requires declaring a chosen fixed feature basis, not pretending
to reuse nonexistent parameters.

## One discriminating experiment, with useful counterfactual outcomes

Use a capable existing checkpoint, a fixed native context length, and a paired
occurrence-retrieval task derived from the original public MRCR conversations.
Do not train a tiny model from scratch again: the previous scratch assay did not
establish query-dependent ability. Preserve the original MRCR rows separately
from our diagnostic transformations and use an untouched held-out set later.

Each family should include: the original question, a swap of two assistant
responses to identical requests (same text inventory, changed correct answer),
and a different ordinal query on the unchanged history. Labels must be checked
against actual message content, not trusted from a file field alone. An ordinary
content query and a compact-context control separate copying/format competence
from long-range occurrence resolution. Fixed external histories are shared;
generated answers never alter subsequent paired inputs.

The focused learned comparison shares the backbone, adapter parameterization,
reference candidates, sparse support, feature basis and training data:

- Arithmetic normalized composition above.
- Geometric/mean-bias composition with the same parameters and references.
- Single-reference variant and current-query-origin relative bias.

The native model is the quality anchor; a comparably budgeted ordinary residual
adapter is an additional capacity control. These are not six arbitrary candidate
methods: each removes one explicit explanation of a possible gain. First decide
the central arithmetic-vs-geometric/point contrast; broader attribution follows
only if there is useful answer-level signal. A full/restricted TAPE comparison is
required before presenting deployment cost as an advantage over contextual PE.

Measure complete output, EOS, the official MRCR sequence-match metric, and strict
whole-string-plus-EOS separately. Score all frozen examples. Do not rename
SequenceMatcher as exact correctness, replace failed generation with answer NLL,
or interpret a format-only difference as repaired reference selection.

| Observation | Consequence |
| --- | --- |
| Arithmetic improves answers where retained references disagree, and the relation-swap answers track the requested occurrence | Evidence for the proposed composition; confirm independently and measure cost |
| Both pools improve equally over native/current-origin control | Read-conditioned position may help; no support for a special uncertainty-pooling benefit |
| Point reference matches or beats the mixture | Prefer the point mechanism if cheaper; do not insist on a distributional state |
| Ordinary residual adapter matches the gain | No demonstrated positional-interface advantage at that budget |
| Correct source is excluded from S | This reader-only construction cannot repair the access failure; do not call that a pooling failure |
| Correct source is available but every trained variant remains poor | This recipe has no useful repair; inspect whether references are informative before proposing another gate |
| Compact/native controls cannot resolve different ordinal queries | The assay does not test the proposed mechanism; do not repeat the failed scratch-model pattern |

This experiment cannot establish a universal PE across all sparse models. That
requires a second distinct interface, independent natural tasks, and measured
quality/cost gains. Nor does modern architecture motivation turn a toy proof or
CPU implementation into an acceptance-level paper result.

## Prepared code and verified boundary

- `experiments/refcarry_audit/normalized_reference.py`: one-row additive bias,
  immutable feature factorization, normalized mixture and retained-mass handling.
- `experiments/refcarry_audit/reference_bridge.py`: writer-to-reader head mixing,
  reference merging, causal same-token handoff and unchanged reader support.
- CPU tests verify 13 numerical/semantic properties, including nonzero adapter
  gradients at native initialization. No model hook, trained adapter, long-context
  generation result or performance kernel is claimed complete.

Sources: [TAPE](https://arxiv.org/html/2501.00712v1),
[RePo](https://arxiv.org/html/2512.14391v1),
[RoVE](https://arxiv.org/html/2606.11275v1),
[NTM](https://arxiv.org/pdf/1410.5401),
[official MRCR data and scoring](https://huggingface.co/datasets/openai/mrcr).
