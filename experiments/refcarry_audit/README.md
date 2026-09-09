# RefCarry review: what is valid and what needs revision

Reviewed from the user's pasted text on2026-09-09. The linked sandbox ZIP and
long-form plan were not supplied here; this review does not claim to have read
those files. GPU remains off. These are operator-level CPU checks and a primary
source review, not trained-model results.

## Decision

Keep **read-conditioned reference information** as a possible inductive bias.
Do not accept the current proposal as an established new theory family or commit
to its thirteen-GPU-hour plan. Its nearest work is substantially closer than its
comparison table suggests, and the stated oracle gate can misdiagnose failure.

The surviving engineering hypothesis is narrow but coherent: a constrained,
query-only contextual-position side channel may improve adaptation or inference
cost when a writer actually identifies a reference and a later reader cannot
use it effectively. Neither that bottleneck nor a capability-cost advantage is
currently demonstrated.

## Reproduction

Run from the repository root:

```bash
.venv/bin/python experiments/refcarry_audit/checks.py
```

`results.json` contains all CPU outcomes. The checks include:

| Check | Result | Interpretation |
|---|---|---|
| Group-moment logit identity,100 random examples | max error2.11e-15 | Valid for mean logits |
| Softmax before/after averaging | TV0.333288 | Distinct read operators |
| Equal moments, different reader mixtures | Exact equal matrices, different distributions | Moments do not determine general marginalized reads |
| Fixed-key TAPE-style operator mapping | error0 | Core weighted-matrix algebra overlaps |
| Probability-simplex sketch | rankPhi2, sufficient sketch1 | Universal uncentered rank statement needs conditions |
| One-hot reference, opposite native/reference phases, g=.5 | query norm6.12e-17 | Gate can cancel a perfectly certain reference |
| Pretrained query already encodes the target offset | native target5, gold-rebased target1 | Fixed-query phase surgery is not a recovery upper bound |
| Shift reference with fixed heterogeneous keys | selected position7 for all four shifts | Unit spatial slope does not follow from the group law |
| Ordinary nonlinear two-address switch | max error2.22e-16 | One-affine-map lower bound does not cover an MLP bridge |

These examples invalidate broad assertions, not the possibility of a useful
learned adapter under additional conditions.

## The precise operator

For fixed q,K and mu, M_mu=sum_a mu_a rho(a) gives

`E_mu[q^T rho(j-a) k_j] = (M_mu q)^T rho(j) k_j`.

After softmax, the result is the normalized **geometric pool** of the individual
anchor-conditioned read distributions. It is generally different from their
arithmetic mixture. With two opposite anchors, each favoring a different target,
the moment read can be uniform over those targets and an irrelevant key. In the
CPU example the irrelevant key receives1/3 instead of0.0000454.

More strongly, on four cyclic addresses with one rotary pair, distributions
`(.5,0,.5,0)` and `(0,.5,0,.5)` have the same moment matrix but different actual
mixtures of reader distributions. A first-order group moment is sufficient for
the proposed average-logit operator, not for every subsequent addressing operation.
If the intended semantics is a single anchor or a consensus score, say so. If it
is uncertainty marginalization, this formula is not an exact implementation.

With known total mass1 and an affine/general decoder, the precise sketch dimension
is `rank([ones;Phi])-1`. Proof: any zero-sum v killed by S must also be killed by
Phi, because an interior distribution can be perturbed by ±epsilon*v. Conversely,
a basis of the centered feature row space plus the known constant reconstructs
the moment. For sufficiently many generic positions this can still equal2K.
The correction does not imply a generic one-dimension saving. For a point-address
family, one integer pointer suffices; the full-simplex bound cannot establish a
need for2K floating coordinates on a single-anchor task.

## Closest work, at the operator level

[TAPE §3.2](https://arxiv.org/html/2501.00712v1) updates positional matrices with
attention-weighted sums (Eq.7) and uses their bilinear product in the next read
(Eq.6). Set E_a=rho(a)^T, share a writer distribution, update only the query field
E'_i=sum_a mu_a E_a, and retain E_j=rho(j)^T on keys. Its score becomes exactly
q^T E'_i E_j^T k=(M_mu q)^T rho(j)k. This is a restricted operator mapping:
default TAPE has per-group weights, updates both fields, and includes positional
MLPs/residuals. Do not claim default implementations are identical, but do not
describe the core attention-to-position update as absent from TAPE either.

[RoVE §3](https://arxiv.org/html/2606.11275v1) transports values through relative
rotations. A constant payload in a rotary pair yields the relative-frame version
of this Fourier moment. [RePo §3](https://arxiv.org/html/2512.14391v3) predicts scalar
positions; a point-reference query rebase is a single-sided dynamic-position
case. The precise cross-layer bypass remains a deployment choice to evaluate.
[Neural Turing Machines §3.3.2](https://arxiv.org/html/1410.5401) already retains a
previous address weighting and composes content addressing with location shifts.
It is not a pretrained-hybrid baseline, but the generic read-address-state concept
cannot be claimed as new.

## Causal and architectural corrections

1. **Hybrid layers do not erase the residual stream by definition.** Official
   Qwen3.5 code preserves `residual + update` after GDN/full attention and after
   the MLP. Its sequential recurrent state and the query token's depth-wise
   residual stream are different objects. A layer19→23 side channel can help,
   but three intervening GDN layers do not establish the claimed bottleneck.
   [Official source](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
2. **Gold-reference is a specified intervention, not an optimal oracle.** For a
   native query q=rho(a+r-i)u, the unchanged reader already targets a+r. Replacing
   its origin i with a moves the peak to2a+r-i. The CPU example moves a correct
   target5 to1 using the correct anchor6. A trained adapted reader could behave
   differently. Zero-shot failure can reject this exact surgery, not the entire
   reference-information hypothesis.
3. **The proposed ordering is not leakage-proof.** A target before an anchor
   prevents the target state from seeing the future anchor, but lets the anchor
   state contain the earlier target. Earlier layers and residuals provide further
   legal pathways. Clearing one writer output or deleting one raw target KV does
   not remove all of them. These are normal model computations, not benchmark
   cheating; a diagnostic must explicitly control the information it attributes.
4. **The collision theorem needs the whole downstream information set fixed.**
   Equal C alone is insufficient if the reader still sees different K/V or caches.
5. **Shift prediction needs a content condition and a coordinate definition.**
   RoPE encodes token indices. A requested number of semantic slots is not a fixed
   token distance without a specified tokenization contract. Even with token
   coordinates, moving only the query phase does not translate arbitrary key
   content. A slope1 theorem requires corresponding translated keys or another
   justified structural condition.
6. **Norm controls must match the actual gated multiplier.** It is
   `z_k=(1-g)exp(i omega_k i)+g*m_k`, not merely m_k. Its norm can vanish at g=.5
   even for a certain point reference. A faithful norm-only comparison uses
   `abs(z_k)*exp(i omega_k i)` with the same q. Equal |m| for a wrong reference
   does not guarantee equal |z| after interpolation.

## A useful revision, before any new GPU commitment

- State which operation is required: point-reference addressing, consensus of
  reference-conditioned scores, or a mixture over uncertain references. Separate
  their theory and tests.
- Treat the method as a constrained query-only contextual-position adapter.
  Include an actual scalar-pointer/query-rebase control for point references and
  a function-matched fixed-key TAPE variant. The full TAPE/RePo comparisons remain
  necessary for an eventual capability-cost claim.
- Give matched controls the same auxiliary reference information. Compare the
  structured rotary use against a small nonlinear/multiplicative adapter, not just
  an affine bridge. Otherwise extra information is confounded with the operator.
- Separate finding a reference, retaining it, and using it. A gold address skips
  the first two and cannot by itself locate which stage failed.
- For value-path attribution, calibrated donor/recipient target-V swaps with Q,K,
  non-target values and upstream states fixed are more specific than deletion.
  They test that value path only, not the entire model's information bottleneck.
- A late-bound-payload CPU memory experiment can forbid answer transport before
  the reference is read. Label it an operator diagnostic. Natural text generation
  must still be evaluated separately; the artificial barrier is not a fact about
  a pretrained hybrid model.
- Keep complete answer+EOS as the capability endpoint. Answer NLL and arbitrary
  proposed effect-size thresholds cannot substitute for successful generation or
  establish an acceptance level.

Current recommendation: retain the question, revise the formal target and
identification strategy, and do not yet designate RefCarry the winning method.
