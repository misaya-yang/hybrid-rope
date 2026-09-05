# MaxEnt dilation allocation

- **Date:** 2026-09-01
- **Status:** deterministic candidate derived; CPU identities verified; no LM result
- **Code owner:** `scripts/lib/rope/schedules.py::maxent_dilation_inv_freq`
- **CPU audit:** `scripts/analysis/maxent_dilation_allocation.py`
- **Compute:** no GPU authorization

## 1. Question

Can the native geometric RoPE lattice itself induce a deterministic non-affine
table for a declared extension factor `s`, without learning frequencies or
selecting a formula from long-context outcomes?

## 2. Dilation coordinate

For native

\[
z_i=-2i/d,\qquad \omega_i=b^{z_i}=e^{-hi},\qquad
h=2\log b/d,
\]

a context dilation `r=e^{mh}` translates the phase lattice:

\[
r\omega_i=\omega_{i-m},\qquad \omega_i/r=\omega_{i+m}.
\]

The first identity describes the phase acceleration caused by evaluating a
distance `r` times larger. The second is the compensating frequency move used
by a long-context table. Learned Q/K coefficients are not generally equivariant
to translation across rotary-pair index, which is the frozen-readout obstacle.

For any proposed table define its per-pair dilation coordinate

\[
r_i=\frac{\omega_i}{\omega_i'}=b^{z_i-z_i'},\qquad
z_i'=z_i-\log_b r_i.
\]

No hard routing is implied: the finite set `{log r_i}` is a basis dictionary
covering the continuous deployment-scale interval `[0, log s]`.

## 3. Scale-neutral measure and the missing preference

The invariant measure on positive dilation is Haar measure

\[
d\mu(r)=dr/r=d\tau,\qquad \tau=\log r.
\]

Uniform mass in `tau` gives `r_i=s^{q_i}`. Because `q_i` is affine in channel
index, this produces an affine exponent map. Therefore group structure plus
`s` alone cannot yield a non-affine `z`; an additional Native/long preference
is mathematically necessary.

## 4. Maximum-entropy preference

Maximize entropy relative to `d tau` on `[0, log s]` subject to normalization
and a declared mean log-dilation. Strict entropy concavity gives the unique
truncated exponential family

\[
p_\lambda(\tau)=\frac{e^{\lambda\tau}}{
\int_0^{\log s}e^{\lambda u}\,du}.
\]

Its midpoint quantiles `q_i=(i+1/2)/K` are

\[
r_i(\lambda)=
\begin{cases}
\left[1+q_i(s^\lambda-1)\right]^{1/\lambda},&\lambda\ne0,\\
s^{q_i},&\lambda=0,
\end{cases}
\]

and the candidate table is

\[
\boxed{
z_i^{(s,\lambda)}=-\frac{2i}{d}-\log_b r_i(\lambda),\qquad
\omega_i^{(s,\lambda)}=\omega_i/r_i(\lambda).
}
\]

The limits are

\[
\lambda\to-\infty:r_i\to1,\qquad
\lambda\to0:r_i=s^{q_i},\qquad
\lambda\to+\infty:r_i\to s.
\]

Thus negative `lambda` leaves most fast/mid pairs near Native and concentrates
dilation in the slow tail; positive `lambda` assigns near-maximal dilation to
most pairs. No endpoint is pinned because midpoint quantiles lie strictly
inside `(0,1)`.

## 5. Why larger dilation is paired with slower channels

MaxEnt determines a dilation multiset, not its channel assignment. The missing
coupling is fixed by Native-readout preservation. At the edge of the training
window, a per-pair phase displacement is bounded by

\[
L\,\omega_i\left(1-\frac1{r_i}\right).
\]

Native frequencies decrease with `i`, while `1-1/r_i` increases with dilation.
The rearrangement inequality therefore assigns increasing `r_i` to decreasing
`omega_i` to minimize the summed phase displacement. The same opposite-order
pairing solves the bottleneck assignment for the maximum product. This supplies
the monotone `q_i` coupling; it is not a hard-coded frequency cutoff.

## 6. What is and is not established

Established analytically and by CPU contract:

- every table is positive and strictly decreasing;
- dilation factors lie strictly inside `(1,s)` and increase with pair index;
- `lambda=0` is affine in exponent space;
- every finite nonzero `lambda` in the registered grid is non-affine;
- mean log dilation increases monotonically with `lambda`;
- monotone channel coupling has no larger summed Native phase displacement than
  reverse coupling;
- both physical endpoints may move.

Not established:

- any NLL, PPL, downstream, or SOTA improvement;
- that maximum entropy is the preference frozen Transformers ultimately need;
- a universal `lambda` or a zero-search method.

Choosing `lambda` from Native retention is a seven-point inference-time
hyperparameter selection with zero weight/frequency optimization. It must not
be described as zero-search.

## 7. First falsification gate

The candidate grid is

\[
\lambda\in\{-4,-2,-1,0,1,2,4\},\qquad s=4.
\]

Stage A opens only multiplier `1` from the receipt-bound formal token manifest
owned by
[`SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823`](../results/zero-training-deployment/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md)
(`token_manifest_sha256=74022bf36d444a1735baab72bda0312b9867dd38c9f85ece376049b5f35f66f3`).
It has two separate gates:

1. **Likelihood retention.** Let `N_native` and `N_lambda` be paired 4K PG-19
   final-tail NLL. Define

   \[
   R_{\rm PPL}=\frac{\exp N_{\rm native}}{\exp N_\lambda}
   =\exp(N_{\rm native}-N_\lambda).
   \]

   Pass iff `R_PPL >= 0.875`, equivalently
   `N_lambda-N_native <= log(8/7) = 0.1335313926`.
2. **Actual downstream retention.** For Qasper, MultiFieldQA-en, HotpotQA,
   2WikiMQA, and GovReport, compute each task's existing official-style score
   (`QA-F1` for the first four, `ROUGE-L-F1` for GovReport), then take the
   equal-task macro `M`. Pass iff

   \[
   R_{\rm task}=M_\lambda/M_{\rm native}\ge0.875.
   \]

Both inequalities must pass. They are not averaged and one cannot compensate
for the other. NarrativeQA is absent from the existing 1x owner and is not
silently inserted. RULER is reported later as a capability diagnostic, not used
to select `lambda`.

The executable gate uses an unrounded Native rerun on the identical rows,
checkpoint, precision, decoding, and scorer. Historical rounded values provide
only a receipt sanity check: Native PG-19 NLL `2.9712` implies candidate NLL
about `<=3.1047` (PPL `<=22.3032`), and Native task macro `0.3424` implies
candidate macro about `>=0.2996`.

Among nonzero grid points that pass both gates, select the largest `lambda`;
`lambda=0` remains the affine control and is not eligible to become the claimed
non-affine method. If no nonzero point passes, the family fails and Stage B is
not opened. Otherwise the selected table and its float32 hash are frozen before
Stage B opens `2x` and `4x` likelihood and outcome-unseen capability rows once.

Controls: Native, official YaRN, the affine `lambda=0` table, the historical
successful m64/ramp table, and the selected non-affine MaxEnt table. PPL and NLL
are two representations of the same likelihood endpoint. The `87.5%` value is
an author-chosen deployment tolerance, not a theorem-derived constant.

The existing formal evaluator is reused with `external_table_static`,
`--table-support explicit`, `--long-attention-scaling 1.0`, and
`--multipliers 1` during Stage A. This is a protocol identity, not GPU
authorization.

## 8. CPU receipt

```bash
conda run --no-capture-output -n aidemo \
  python scripts/analysis/maxent_dilation_allocation.py --self-test
```

Observed: `MAXENT_DILATION_CPU_CONTRACT_OK`; `140` focused schedule/core tests
passed on 2026-09-01. This verifies identities and implementation only.
