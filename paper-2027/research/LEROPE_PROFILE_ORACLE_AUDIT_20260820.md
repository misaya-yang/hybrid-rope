# LeRoPE profile-oracle audit

- **Date:** 2026-08-20
- **Status:** `CPU_ONLY_COMPLETE`; internal falsification pilot, not a manuscript claim
- **Question:** does the owner-backed positional-utility choice
  \(\rho^*(\phi)\propto w(\phi)^{1/3}\) recover the profile reported by LeRoPE?
- **Answer:** **No.** Under the only existing A.15-compatible scalar with a
  validated implementation, the predicted table moves farther toward fast
  frequencies than EVQ-Cosh and away from the published LeRoPE profile.

## 1. Primary-source facts

Primary source: [Karypis et al., *LeRoPE: Learnable RoPE Frequencies Improve
Language Modeling*, arXiv:2607.10134v1](https://arxiv.org/abs/2607.10134),
local PDF `rebuttal/rebuttal_0723/LeRoPE_2607.10134v1.pdf`, SHA-256
`a5ff7cdcbde2196dce3919659ca5252e0f9246e657b0fb47affc74ab13d05c1e`.

The following are stated directly in the paper:

1. LeRoPE learns one log-scale \(\alpha_m\) per frequency band,
   \(\hat\theta_m=e^{\alpha_m}\theta_m\), initialized at zero and shared
   across all layers and heads (Section 3.1). With the paper's fixed
   head dimension 64, this is 32 learned scalars.
2. Table 1 reports lower validation PPL than RoPE and p-RoPE at every point of
   the 52M, 217M, 608M, 1.34B, and 2.52B ladder. Figure 1 reports a 3.4% RoPE
   compute multiplier at 2.52B. These are performance facts, not profile
   measurements at every scale.
3. Figure 3 explicitly plots similar learned profiles for 217M, 608M, 1.34B,
   and 2.52B models, three 217M seeds, and 217M runs at training lengths
   2048/4096/8192. Fast bands remain near RoPE, middle bands slow, and the
   slowest bands are driven to much lower frequencies.
4. Table 2 reports PPL `18.2377` for RoPE, `18.1714` for Fixed LeRoPE, and
   `18.1335` for learned LeRoPE at 217M. Thus the independently learned and
   then frozen table retains 63.6% of the full PPL gain; this supports value in
   the fixed table while also leaving a joint-training contribution.
5. Table 10 reports six dominant-band wavelengths near
   \(2.205L_{\rm train}\). Figure 8 reports a `0.762`-nat loss increase when
   band 17 is zeroed, versus `0.069` for the next-largest band ablation.
6. Section 7 reports that unmodified LeRoPE extrapolates more sharply downward
   than RoPE/p-RoPE and attributes the rapid explosion to the dominant band's
   sign change beyond trained offsets. The paper does **not** establish that
   slow-tail suppression itself causes this extrapolation failure; joining
   those two observations is an interpretation, not a primary-source fact.

The v1 PDF contains no LeRoPE code/checkpoint URL. The numeric public profile
used here is therefore transcribed from the 32 frequency labels printed in
Figure 6 at \(L=2048\), not digitized from pixels. Those labels have only
roughly two to three significant digits. The audit propagates half of the last
displayed digit as a rounding envelope; this is not statistical uncertainty.

## 2. Why the proposed experiment was under-specified

The high-rate formula

\[
D_K[\rho]\simeq \frac{1}{12K^2}\int_0^1
\frac{w(\phi)}{\rho(\phi)^2}\,d\phi,
\qquad \rho^*(\phi)\propto w(\phi)^{1/3}
\]

does not define \(w\). A.15 supplies a Fréchet derivative of attention logits
under a schedule perturbation, but it does not prove that its square is the
distortion weight governing LeRoPE's jointly trained LM optimum.

This audit uses the narrowest owner-backed definition already implemented and
finite-difference checked in
`scripts/analysis/attention_fisher_50m_probe.py`:

\[
J_{i,k}(j)=\frac{\partial z_{ij}}{\partial\log\omega_k},
\qquad
w_k=\mathbb E_{\ell,h,i}\left[
J_{i,k}^{\mathsf T}\bigl(\operatorname{diag}p_i-p_ip_i^{\mathsf T}\bigr)
J_{i,k}\right].
\]

This is the diagonal per-band **structural softmax curvature**. Multiplication
by \((\log b)^2\) converts the derivative from \(\log\omega\) to \(\phi\), but
the common factor cancels when \(\rho\) is normalized. It is not the signed LM
gradient used by LeRoPE training, not an LM Hessian, and not a training-free
estimate of the learned optimum.

## 3. Measurement and quantization

The primary run reuses the canonical completed CPU probe with SHA-256
`32770ef134a457de8f4878f6d669f499defedafd079f96da625503b2660bc2c9`:

- frozen 50.9M geometric-table checkpoint, seed 42;
- TinyStories validation, \(L=512\), base 500K, \(K=32\);
- 8 windows, query positions `63/127/255/383/511`, 6 layers, 8 heads;
- 1,920 head-query observations;
- manual attention vs SDPA max absolute error `6.56e-7`;
- log-frequency derivative finite-difference max error `1.89e-7`;
- checkpoint SHA-256
  `4a96f99f4abbc5d2a1a6d5b7725e07ab3f3fe9743eeee385658556bc2df1204b`;
- validation SHA-256
  `afa2a5ca8fc9757b65e88cc4b9978f4c0d929459c0cd4035ab8f5a036a65d920`.

No parameter was updated. A direct two-window checkpoint recomputation was
also run on CPU to exercise the portable path independently of the saved JSON;
it produced the same qualitative result and nearly identical distances.

For a reproducible finite table:

1. interpolate \(\log w\) piecewise linearly over \(\phi\in[0,1]\), with no
   fitted smoothing parameter;
2. set \(\rho(\phi)=w(\phi)^{1/3}/\int w^{1/3}\);
3. take 32 inverse-CDF midpoint quantiles
   \(F^{-1}((k+1/2)/32)\).

For a common public coordinate, the resulting normalized positions are mapped
to the LeRoPE configuration's standard base 10K. Geo uses
\(\phi_k=k/32\). EVQ-Cosh uses its deployed midpoint rule with
\(\tau=64/\sqrt{2048}=\sqrt2\). Two metrics are reported:

- **support-aware:** RMSE in natural-log wavelength after this shared mapping;
- **shape-only:** RMSE after independently min-max normalizing each profile's
  log-wavelength, removing its observed support.

The mapping from the measured 50M/base-500K/L512 checkpoint into the
base-10K/L2048 public coordinate is a diagnostic transport assumption. It is
not a matched prediction of LeRoPE.

## 4. Result

The measured utility falls by roughly 12 orders of magnitude from the fastest
to the slowest band. Cube rooting reduces but does not reverse that bias. The
raw midpoint oracle places its slowest quantile at only \(\phi=0.433\); on the
base-10K comparison support, that corresponds to a wavelength of only about
340 tokens. LeRoPE instead reports its slow tail near \(10^{-5}\) radians per
token, or several hundred thousand tokens in wavelength.

| Comparison | Geo | EVQ-Cosh | \(\rho^*\) oracle |
| --- | ---: | ---: | ---: |
| support-aware log-wavelength RMSE to LeRoPE | 2.623 | 3.050 | **6.244** |
| shape-only RMSE to LeRoPE | 0.118 | 0.175 | **0.347** |

The oracle's shape-only RMSE to EVQ-Cosh is `0.177`, about half its `0.347`
distance to LeRoPE. Projecting the oracle onto the EVQ-to-LeRoPE direction
gives \(\alpha=-0.957\), where \(0\) is EVQ and \(1\) is LeRoPE, with residual
RMSE `0.056`. Thus it is not between the two profiles: **it lies beyond EVQ in
the direction away from LeRoPE**.

The LeRoPE display-rounding envelope cannot change this verdict. The
support-aware RMSE interval is `6.224–6.265` for the oracle, compared with
`3.031–3.069` for EVQ-Cosh. The direct two-window recomputation gave
\(\alpha=-0.944\), oracle-to-EVQ shape RMSE `0.174`, and oracle-to-LeRoPE
shape RMSE `0.345`.

![LeRoPE profile oracle](../figs/fig_lerope_profile_oracle.pdf)

## 5. Scientific verdict

This pilot **falsifies the proposed oracle under its most defensible existing
operationalization**. It does not derive LeRoPE's learned shape and does not
produce a coexistence point. The A.15 structural curvature rewards bands whose
position-dependent logits change strongly when frequency moves; at finite
window, that mechanically emphasizes faster bands. LeRoPE's LM objective can
instead benefit from suppressing rotation in slow bands and from a signed,
jointly trained dominant-band trajectory. An unsigned local positional
curvature cannot represent either choice.

Consequences:

- do not claim \(\rho^*\propto w^{1/3}\) predicts LeRoPE;
- do not use profile opposition alone as proof that LeRoPE's entire learned
  table is the in-window optimum antagonistic to extrapolation;
- retain LeRoPE's verified value as independent learned/fixed-table evidence;
- if profile prediction is pursued later, the missing object is a signed,
  training-trajectory-aware LM risk derivative on a matched base/data/model,
  not another positive density heuristic.

This result should remain internal unless a later matched experiment repairs
the configuration and objective mismatch. It does not alter the paper's
fixed-range allocation or co-adaptation claims.

## 6. Reproduction

```bash
conda run --no-capture-output -n aidemo python -m py_compile \
  scripts/analysis/lerope_profile_oracle.py

conda run --no-capture-output -n aidemo python \
  scripts/analysis/lerope_profile_oracle.py \
  --probe-json /tmp/attention_fisher_50m_probe_20260819.json

# Portable CPU-only recomputation of one checkpoint/table cell:
CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n aidemo python \
  scripts/analysis/lerope_profile_oracle.py --windows 8
```

Produced files:

| File | SHA-256 |
| --- | --- |
| `scripts/analysis/lerope_profile_oracle.py` | `2885d3a565d1d55540157168c7ac61e7b5f6f62df3dfa42b0da5bd57086f1a33` |
| `paper-2027/figs/fig_lerope_profile_oracle.pdf` | `0e525c8a65fbc7e8443ccb323f18e47d42d72c99d181367843a2e55c4981c3c5` |

Validation passed: Python compilation, canonical-probe hash check, figure
generation, visual inspection, and direct two-window CPU recomputation. GPU,
training, remote jobs, manuscript TeX edits, paper compilation, Git staging,
commit, and push were not performed. `paper/` remains untouched.
