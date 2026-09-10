# RoPE extrapolation: failure mechanisms, limits, and the remaining design problem

2026-09-10. Theory investigation alongside the active frozen-weight experiments.
This document develops the user's question about why YaRN/MrRoPE use frequency
bands, why vanilla RoPE fails beyond training, and what an improved method must
actually overcome. It does not declare a new method or a universal context limit.

## 1. The computational object

For a fixed head, let the raw queries and keys generated under a table \(\nu\)
be \(q_i^\nu,k_j^\nu\). Its attention logit is

\[
z_{ij}^\nu=\frac{g^2}{\sqrt{d_h}}
(q_i^\nu)^\top R_\nu(i-j)k_j^\nu.
\]

There are two table dependencies: the explicit relative rotation and the
queries/keys formed by earlier computation. Values and later residual states
also change. A theory of the rotation alone is not yet a theory of the output.

Every block rotation is orthogonal:
\(\|R_\nu(d)x\|=\|x\|\) and
\(|q^\top R_\nu(d)k|\leq\|q\|\|k\|\).
The sine/cosine functions remain defined and bounded at any distance.
Consequently, extrapolation does not intrinsically cause unbounded vector
norms. A much larger logit than the training distribution is possible through
changed alignment even at fixed norms; upstream states can also change.

In exact arithmetic, translating **all** position IDs by the same constant
leaves static RoPE's relative rotation unchanged. A corresponding full-model
invariance holds when the causal mask and every other operation are unchanged.
BF16 rotation/reduction may violate the realized invariance. This gives a
specific numerical diagnostic without changing tokens or relative distances.

## 2. Why bands are reasonable, and why three bands are not a theorem

Define the training-reference rotation count
\(r_j=W\omega_j/(2\pi)\).
For a small local displacement \(d\), uniform interpolation produces the
operator change

\[
\|R(d\omega_j)-R(d\omega_j/S)\|_2
=2\left|\sin\frac{d\omega_j(1-1/S)}2\right|.
\]

For small angular change this is approximately
\(|d\omega_j(1-1/S)|\). Preserving high frequencies therefore has a concrete
local-fidelity motivation. At the other extreme, a slow pair with
\(W\omega_j\ll1\) contributes approximately constant/linear positional
functions, and changing it causes smaller absolute local rotations.
Choosing \(\nu_j=\omega_j/S\) restores its maximum phase span because
\((SW)\nu_j=W\omega_j\).

These are different asymptotic regimes of a continuous spectrum. A middle band
is a practical transition between them; it is not a third mathematically
distinct kind of information. The empirical 1/32-cycle thresholds do not prove
three universal functional modules. Furthermore, the effective distance
distribution seen in training need not be uniform over the configured window.

[YaRN, Sections 3.1–3.2](https://arxiv.org/html/2309.00071v2#S3) motivates
frequency-dependent interpolation through high-frequency detail and local
distance preservation. [MrRoPE, Section 3.2 and Appendix B](https://arxiv.org/html/2601.22181v1#S3.SS2)
explicitly starts from YaRN's empirical evidence, retains the band principle,
and changes the middle-band radix conversion. Its arithmetic progression of
radix exponents is a construction choice, not a proof of task optimality.

## 3. Why seeing a circle is not sufficient

The low-frequency coverage hypothesis identifies a plausible source of
distribution shift: a learned near-constant content channel can rotate into
an unfamiliar alignment at long distance. It does not show that higher
frequencies are automatically safe once one circle is covered.

The vector of phases, its content-conditioned coefficients, and the competing
keys are learned jointly. Marginal coverage of each circle does not imply
coverage of that joint object. Likewise, a relevant key need not lose score
monotonically with distance: the score contains signed sine and cosine terms.

[Round and Round We Go, Sections 3–6](https://arxiv.org/html/2410.06205v1)
provides counterexamples to universal attention decay and analyzes positional
heads and low-frequency information channels in Gemma. Its single-frequency
semantic-channel theorem is informative but is not a quantitative bound on
all heads/layers of our Qwen models. Partial-RoPE training results also do not
justify cold-removing frequencies from an already trained checkpoint.

The current E1 slots illustrate the gap: native slots 28 and 29 rotate 12.37
and 9.97 times within the Qwen reference window. Their conditional successes
cannot be explained solely by an unvisited fraction of an individual circle.

## 4. A joint finite-window quantity worth testing

For complex Fourier columns on integer distances \(d=0,\ldots,W-1\), the
normalized uniform-window inner product at frequency difference \(\delta\) is

\[
G_W(\delta)=\frac1W\sum_{d=0}^{W-1}e^{id\delta}
=e^{i(W-1)\delta/2}\frac{\sin(W\delta/2)}{W\sin(\delta/2)},
\]

with the continuous limit at denominator zeros. Thus nearby frequencies with
\(|\delta|W\ll1\) produce nearly collinear positional columns. This depends on
frequency **spacing** and the distance measure, not only each frequency's
individual circle coverage. Correlation against the full set of columns is
more informative than one nearest-neighbor gap. It still does not measure
the independent content coordinates attached to those frequencies.

FullLagP2 uses a related, causal-lag-weighted conditional residual. Retrospective
comparison reveals agreement with E1 at the two selected slots:

| Slot | Native cycles in W | MrPro m | FullLagP2 m | E1 m |
|---|---:|---:|---:|---:|
| 28 | 12.367 | .098039 | .074131 | .065359 |
| 29 | 9.966 | .137255 | .221149 | .183007 |

Both constructions change from weaker to stronger compression between these
slots. This is a concrete hypothesis-generating agreement, not independent
validation of P2's residual-to-exponent mapping. P2 changes many additional
slots, has gain confounding in some comparisons, and shows task/model tradeoffs.
The queued two-slot experiment measures whether the agreement is useful.

## 5. More keys create a second, non-positional difficulty

In a simple read with one relevant key having logit advantage \(\Delta\) over
\(N-1\) equal-scoring distractors,

\[
p_* = \frac{e^\Delta}{e^\Delta+N-1},\qquad
\Delta=\log(N-1)+\log\frac{p_*}{1-p_*}.
\]

Maintaining a fixed target mass after multiplying the number of distractors by
four requires approximately \(\log4\) more logit advantage. More generally,
the relevant quantity is the target logit minus the competitors' logsumexp.
This pressure exists without RoPE. A temperature change can alter concentration
but cannot change key ordering at fixed raw states; it can change later states.

For uniformly bounded logits, \(z_*-z_j\leq C\) for all distractors implies
\(p_*\leq[1+(N-1)e^{-C}]^{-1}\). A fixed bounded-logit read cannot retain a
constant mass on one unique item as \(N\to\infty\). This is a conditional
softmax limit, not proof that an entire model or an aggregation task must fail.
Distributed evidence, other information paths, and changes in logit scale
matter. Our output-token margins are also not numerically interchangeable with
attention-logit margins.

This is why natural-text NLL can remain good while binding fails, and why
gain must be tested as its own variable. The currently queued MrPro/gain074
arm completes the existing table-by-gain comparison without a broad grid.

## 6. What an actual limit can and cannot say

**Exact local preservation fixes the static operator.** If, at a fixed pair
identity, all Q/K dot products at displacement one must be preserved, then
\(R_\nu(1)=R_\omega(1)\), hence \(\nu=\omega\pmod{2\pi}\).
A nontrivial static reallocation cannot preserve every old local operation
exactly while changing the long-distance operation. This does not forbid
better task performance: preserving every possible Q/K is stronger than
preserving the computations used by a model on a task distribution.

**Finite-dimensional bounded phase codes have no uniform separation at
unbounded length.** Divide each of K phase circles into M bins. Among
\(M^K+1\) positions, two lie in the same product bin. Their positional feature
vectors differ by at most \(2\pi\sqrt K/M\) in Euclidean norm. Arbitrarily
close positional recurrences therefore occur as length grows, even without
exactly commensurate frequencies. The bound is extremely loose at K=64 and
does not establish a 128K or 1M practical ceiling. Different content and other
heads/layers can still distinguish tokens at close positional codes.

**A scalar proxy's first zero is not a universal language-model ceiling.**
For example, [Base of RoPE Bounds Context Length, Theorem 1](https://arxiv.org/html/2405.14591v1#S4)
derives its cosine-sum signal under a specified random-query/key model and a
particular similar-key construction. These assumptions must accompany its
interpretation. [RoPE Distinguishes Neither Positions Nor Tokens, Limitations](https://arxiv.org/html/2605.15514v1#Sx1)
also states a regular-amplitude assumption and uses a distributional
approximation. Such results inform failure hypotheses; they do not establish
an unconditional bound for the nonuniform learned bands in our models.

**Freezing weights adds a compatibility problem.** Better positional basis
geometry need not be accessible to the already learned Q/K projections and
prefix computation. Our manuscript's exact transplant obstruction concerns
equality of operators for all content/distance under invertible linear maps;
it does not imply that beneficial finite-distribution cold swaps are impossible.
BM, P2, and E1's measured conditional gains must be retained as counterweights
to any overly broad reading of that obstruction.

## 7. Consequences for the active research

The target is a rule that preserves useful learned positional operations while
improving content-conditioned competition at long distances. A geometric
criterion is useful only if its relationship to that computation can be
checked on the model. The current program separates the following questions:

1. **Does a frozen rule transfer?** Completed E1 transfer is mixed: Qwen7B
   32K tie / 128K -1.667 pp; OLMo 4K -3.333 / 16K +3.819 pp. The OLMo result
   remains substantially below its existing BM comparison. Independent new
   Qwen3B task inputs are queued. The two models' panels and floors differ.
2. **Is the local transition directional and composable?** Equal absolute
   frequency-step reversal, adjacent-slot control, and the s28/s29 pair are
   fixed before their evaluation. Initial 12-row ties are ceiling-limited;
   complete historical rows remain necessary before interpreting these arms.
3. **Which part of the computation changes?** Positive-case prefill/readout
   crossover already separates several pathways. The multikey success mostly
   reflects additive answer-margin improvements crossing argmax. It is not
   evidence of a universal hidden-state interaction mechanism.
4. **Is the gain stable to an exact symmetry's numerical realization?** A
   single global position-ID shift of +1 is queued for three existing positive
   cases and their MrPro controls. Tokens, causal order and relative distances
   remain unchanged. A flip would identify arithmetic sensitivity, not a new
   task failure or proof that all frequency improvements are numerical artifacts.
5. **Does a larger kernel help because of its frequencies?** E10's first 12
   RULER rows tie MrPro while NLL improves. A same-clock expanded-kernel NLL
   control separates frequency mixing from the changed BF16 execution path.

These tests are evidence for choosing the next method. They are not a demand
for a complete theory of all failures before using a meaningful positive result.
The current narrow candidate remains E1; the deeper hypothesis is a learned,
finite-window transition in usable positional responses, with explicit tests
against geometry-only, gain-only, and numerical explanations.
