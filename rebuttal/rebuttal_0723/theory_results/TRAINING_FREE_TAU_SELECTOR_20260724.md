# Training-free finite-\(K\) tau selector — method and failed historical gate

Date: 2026-07-24

Status: **historical gate failed; no prospective training authorized**

Scope: supporting/mechanistic analysis. This does not change a paper table or
replace the submitted operating rule.

## 1. Method

For nominal base \(B\), \(K\) rotary frequency pairs, midpoint or endpoint
quantiles \(u_k\), and Cosh parameter \(\tau\), use the actual deployed
finite-\(K\) table

\[
\phi_\tau(u_k)
=
1-\frac{1}{\tau}
\operatorname{asinh}\!\left((1-u_k)\sinh\tau\right),
\qquad
\omega_k(\tau)=B^{-\phi_\tau(u_k)}.
\]

The implementation evaluates the continuous \(\tau=0\) limit exactly and uses
the repository's midpoint grid for the Phase16 audit.

Define the normalized RoPE distance feature

\[
z_\tau(\delta)
=
\frac{1}{\sqrt K}
\left[
\cos(\omega_0\delta),\sin(\omega_0\delta),\ldots,
\cos(\omega_{K-1}\delta),\sin(\omega_{K-1}\delta)
\right],
\]

and its Gram kernel

\[
G_\tau(\delta,\delta')
=
z_\tau(\delta)^\top z_\tau(\delta')
=
\frac1K\sum_{k=0}^{K-1}
\cos\!\left(\omega_k(\delta-\delta')\right).
\]

The training-distance law is not fitted. For a causal window of length \(L\),
uniformly sampling a distinct query/key token pair gives

\[
\Pi_L(\delta)=\frac{2(L-\delta)}{L(L-1)},
\qquad \delta=1,\ldots,L-1.
\]

The user must declare a deployment distribution
\(\Pi_{\mathrm{target}}\). For a workload containing target window lengths
\(\{T_j\}\) with declared occurrence weights \(\{w_j\}\), the implementation
uses

\[
\Pi_{\mathrm{target}}
=
\frac{\sum_j w_j\Pi_{T_j}}{\sum_j w_j}.
\]

For two distance laws \(P,Q\), define their distinct-offset squared collision

\[
\mathcal C_\tau(P,Q)
=
\mathbb E\!\left[
G_\tau(\delta,\delta')^2
\mid \delta\ne\delta'
\right],
\quad \delta\sim P,\ \delta'\sim Q.
\]

The repaired risk is

\[
R(\tau)
=
\max\left\{
\mathcal C_\tau(\Pi_{\mathrm{train}},\Pi_{\mathrm{train}}),
\mathcal C_\tau(\Pi_{\mathrm{target}},\Pi_{\mathrm{target}}),
\mathcal C_\tau(\Pi_{\mathrm{train}},\Pi_{\mathrm{target}})
\right\}.
\]

The maximum has a deployment meaning: the selector protects the worst of
training-range resolution, target-range resolution, and train-to-target
aliasing, without introducing a fitted trade-off coefficient.

The unique output contract is

\[
\tau^\star
=
\min\operatorname*{arg\,min}_{\tau\in[0,\infty]}R(\tau),
\]

where the smallest minimizer breaks ties. The implementation performs a
deterministic one-dimensional global grid in
\(q=\tau/(1+\tau)\in[0,1]\), followed by fixed-tolerance golden refinement.
It reads no PPL, checkpoint, attention statistic, or fitted cross-configuration
coefficient.

## 2. Inputs and output

Required inputs:

- \(B\): nominal RoPE base;
- \(K\): actual number of rotary frequency pairs;
- grid identity: endpoint or midpoint;
- \(L_{\mathrm{train}}\);
- explicit target lengths and their deployment weights, or an equivalent
  discrete \(\Pi_{\mathrm{target}}\).

Unique output: one scalar \(\tau^\star\) and its three auditable collision-risk
components.

## 3. Theoretical properties

1. The method uses the actual finite-\(K\) frequencies, so changing \(B\),
   \(K\), the grid, \(L_{\mathrm{train}}\), or the declared target workload can
   change the result.
2. \(G_\tau\) is positive semidefinite, stationary in
   \(\delta-\delta'\), and satisfies \(G_\tau(\delta,\delta)=1\).
3. Each conditional collision and \(R(\tau)\) lies in \([0,1]\).
4. With the continuous limits at \(\tau=0\) and \(\tau=\infty\),
   \(R\) is continuous on the compactified search interval, so a minimizer
   exists.
5. These properties make the selector well-defined; they do not prove that
   squared feature collision is the risk optimized by a trained transformer.

## 4. Historical configuration-level validation

Source: the retained Phase16 99-run manifest. The selector and all weights were
fixed without using PPL from any configuration, so every configuration is an
unseen configuration for the selector rather than a fold used to fit it.

Audit protocol:

- base \(B=500{,}000\);
- \(K\in\{16,32,64\}\);
- \(L_{\mathrm{train}}\in\{256,512,1024\}\);
- target windows \(\{2L,4L,8L\}\);
- target-window weights \(\log_2(r+1)\), exactly matching the retained
  Phase16 extrapolation-NLL metric;
- five seed-42 pilot taus per configuration;
- reported selector regret uses the nearest actually trained tau because the
  historical sweep did not train the new continuous \(\tau^\star\);
- oracle is the best of the five trained taus in the same configuration.

The selector returned \(\tau^\star\in[13.131,13.344]\) for all nine
configurations. It therefore mostly selected the largest available historical
tau, rather than adapting reliably to \(K\) or \(L_{\mathrm{train}}\).

| Metric across 9 configurations | repaired selector | old formula | fixed \(\tau=0\) |
| --- | ---: | ---: | ---: |
| Mean NLL regret | 0.05219 | **0.04501** | 0.07723 |
| Mean relative PPL regret | 5.44% | **4.69%** | 8.08% |
| Median relative PPL regret | 4.49% | **3.11%** | 7.63% |
| Top-2 basin entries | 1/9 | **5/9** | not a selector |

The repaired selector has lower regret than the old formula in only 4/9
configurations and higher regret in 5/9. A common nonzero fixed tau was not
trained in all nine configurations, so the only exact all-configuration fixed
tau comparison is \(\tau=0\); no interpolation or synthetic PPL was used.

This fails both historical requirements: it is less stable at entering the
best basin and its average PPL regret is higher, not lower, than the old rule.

## 5. Prospective experiment decision

No new RTX 5090 configurations were trained. The prompt authorized the three
prospective Geo/old/repaired comparisons only after a clear historical win.
That gate failed, so running them would be post hoc GPU spending rather than a
registered prospective validation.

## 6. Decision

This Gram-collision construction is a complete, deterministic,
training-free calibration rule, but it does **not** replace
\(\tau=d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\). On the retained historical
curves it is slightly worse in average PPL regret and substantially worse at
entering the top-two basin.

The falsifiable conclusion is therefore:

> Cosh can be derived as a schedule family under the stated surrogate, but the
> trained-model operating basin is not determined by finite-frequency
> positional Gram collision alone. A static geometry-only objective, even when
> it uses the exact base, finite channel count, grid, training range, and
> declared target-distance distribution, is insufficient in this test.

No replacement empirical regression is proposed.

## 7. Reproduction

```bash
python rebuttal/rebuttal_0723/experiments/repaired_tau_selector.py \
  validate-phase16 \
  --manifest data/curated/phase16_99run_manifest.csv \
  --output results/repaired_tau_selector_20260724/phase16_validation.json
```

Validation JSON SHA256:
`483eeba8d59e97a15434a005f62d8083d058d382a4f423705bcdf0d5b32d016f`.
Source manifest SHA256:
`39ce676ca26967434c0091e09d36824cd16d1a1a204ad464dad0a33aef7b18d5`.
