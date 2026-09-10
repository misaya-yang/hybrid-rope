# Sol 04 — constructive allocation surviving the proposal failures

## Bottom-line result

The defensible unification is **not one universal exponent curve**. It is one allocation problem with a stage-dependent compatibility term:

\[
\omega_k=\exp[-(a+Rz_k)],\qquad 0=z_0<z_1<\cdots<z_{K-1}=1.
\]

The endpoints \((a,R)\) define the sampled range and \(z\) allocates the finite rotary pairs inside it. This normalization is necessary because base and unnormalised exponents are otherwise non-identifiable (`RoPE_Exponent_Allocation_Unified_Plan_20260905.md:86-112`; `RoPE_ICLR2027_Research_Guidance_20260905.md:61-77`). The useful common object is

\[
z^*_{\mathcal S}=\arg\min_{z\in\mathcal Z}
\underbrace{D_{\rm demand}(z;\mu_{\Delta})}_{\text{distance/radix coverage}}
+\lambda_{\mathcal S}\underbrace{D_{\rm compat}(W_{\mathcal S},z;\mu_N)}_{\text{learned Q/K compatibility}}
+\gamma\,\Omega(z),
\tag{1}
\]

where \(\mathcal S\in\{\text{scratch},\text{frozen},\text{LoRA}\}\). EVQ supplies a continuous demand-density construction; MrRoPE supplies discrete mixed-radix target scales. They are unified by representing either as a demand measure over log phase scales, then allocating the finite \(K\) slots to that measure. The compatibility weight is approximately zero at initialization, large for frozen deployment, and reduced only in directions the adapter can repair for LoRA. The proposal itself says that the same curve need not serve all three installation stages (`RoPE_Exponent_Allocation_Unified_Plan_20260905.md:450-496`).

## Surviving mathematics

For a fixed layer input and \(\Delta=p_j-p_i\), the actual RoPE score is

\[
s_{ij}=\frac{\alpha}{\sqrt d}\sum_k
\{C_{ij,k}\cos(\omega_k\Delta)+D_{ij,k}\sin(\omega_k\Delta)\}.
\tag{2}
\]

Thus allocation changes a content-conditioned Fourier sum, not a position-only Gram matrix (`Unified_Plan:121-158`). The exponent derivative is

\[
\frac{\partial s_{ij}}{\partial z_k}=
\frac{\alpha R\omega_k\Delta}{\sqrt d}
\{C_{ij,k}\sin(\omega_k\Delta)-D_{ij,k}\cos(\omega_k\Delta)\}.
\tag{3}
\]

Contributions can change sign across heads, tokens, and distances, so a small development set cannot reliably estimate a universal slot direction (`Unified_Plan:192-205`). This is consistent with the recorded failure of a 64-dimensional frequency gradient to transfer from development to holdout (`Unified_Plan:71-77`).

For a finite frozen change, let \(s'=s+\eta\), \(p=\operatorname{softmax}s\). Then

\[
\operatorname{KL}(p\|p')=\log \mathbb E_p e^\eta-\mathbb E_p\eta.
\tag{4}
\]

This exact identity is the correct local-layer compatibility readout; it remains valid across many phase wraps (`Unified_Plan:225-246`). It must be measured on native activations and followed by complete-model output and task checks. A scalar gain can change concentration but cannot generally repair candidate ordering at fixed Q/K (`Unified_Plan:248-277`).

For small changes under LoRA, let \(A\delta z\) be the Fisher-whitened native output perturbation from the table and \(Bv\) the allowed adapter perturbation. The irreducible local native error is

\[
\min_v\|A\delta z+Bv\|^2
=\delta z^\top A^\top(I-BB^\dagger)A\delta z.
\tag{5}
\]

This gives the promised stage distinction: frozen uses \(B=0\); LoRA discounts only the component in \(\operatorname{col}(B)\); scratch allows the entire learned representation to co-adapt. It is a local repairability calculation, not a task-success theorem (`Unified_Plan:323-345`).

The finite four-cell decomposition is also exact. With weights learned under allocation \(a\) and evaluated under allocation \(b\),

\[
I=L_{ZZ}-L_{ZG}-L_{GZ}+L_{GG}.
\tag{6}
\]

It separates frozen table incompatibility from the training response without requiring a Jacobian or assuming the path is unique (`Unified_Plan:466-496`). This is the cleanest empirical bridge between EVQ-style joint training and MrRoPE-style frozen installation.

## Constructive allocation rule

### A. From-scratch: demand-quantile EVQ/Mr allocation

Let \(x=-\log\omega\in[a,a+R]\). Build a predeclared demand density

\[
q(x)=\sum_{r\in\mathcal R}\pi_r\,K_h(x-\log \Delta_r),
\tag{7}
\]

where \(\mathcal R\) contains the required relation scales. For a mixed-radix design, \(\Delta_r\) are the radix place values and their within-place neighborhoods; for EVQ, \(q\) can be the Cosh-derived continuous density. The kernel bandwidth \(h\) reflects a phase band around \(\omega\Delta\asymp1\), and \(\pi_r\) comes from the declared task/data mixture, not downstream benchmark outcomes.

For squared log-scale quantization error, the high-rate one-dimensional optimum has point density

\[
\rho_{\rm scratch}(x)\propto q(x)^{1/3}.
\tag{8}
\]

Derivation: a cell of width \(h_x\) contributes approximately \(q(x)h_x^3/12\); with point density \(\rho=1/h_x\), total distortion is \(\int q/(12\rho^2)\,dx\). Minimising under \(\int\rho=K\) gives \(\rho\propto q^{1/3}\). Allocate anchored quantiles

\[
\int_a^{x_k}\rho_{\rm scratch}(u)du
=\frac{k}{K-1}\int_a^{a+R}\rho_{\rm scratch}(u)du,
\qquad z_k=(x_k-a)/R.
\tag{9}
\]

This recovers EVQ as continuous quantisation and smooths MrRoPE's discrete radix demands into a finite ordered table. It does **not** claim that squared log-scale distortion predicts task success; it is an initialization rule whose behavioral value must be tested by matched training. Existing anchored Cosh results support an intervention effect at fixed endpoints but not a universal optimum (`Research_Guidance:30-34, 204-228`).

### B. Frozen Qwen2.5-3B, 32K to 128K: compatibility-constrained radix motion

Start from the actual native tensor \(\omega_k^0\) and retain slot identity. For scale \(s=4\), parameterise the mixed-radix motion as

\[
\omega_k(q_k)=\omega_k^0s^{-q_k},\qquad 0\le q_k\le1.
\tag{10}
\]

Here \(q_k=0\) preserves the native slot and \(q_k=1\) applies the full 4x slow-down. Define a predeclared radix benefit \(b_k\) from coverage of the 32K-to-128K relation-scale demand in (7). Define native compatibility curvature \(f_k\) from the exact attention KL (4), using symmetric small perturbations of slot \(k\) on fixed native Q/K and fixed sampled layers/queries. If LoRA is allowed, replace \(f_k\) by the residual curvature implied by (5).

Choose the single table by the convex program

\[
q^*=\arg\max_{q\in\mathcal Q}
\left[b^\top q-\frac{\tau}{2}\sum_k(q_{k+1}-q_k)^2\right]
\quad\text{s.t.}\quad
\frac12\sum_k f_kq_k^2\le\varepsilon_N,
\tag{11}
\]

\[
\mathcal Q=\{0\le q_k\le1:\;x^0_{k+1}+q_{k+1}\log4\ge x^0_k+q_k\log4\}.
\]

The last constraint preserves frequency order. Solve (11) by bisection on the native-budget multiplier and weighted isotonic projection. With \(\tau=0\) and before projection, the water-filling form is

\[
q_k(\lambda)=\operatorname{clip}_{[0,1]}\!\left(\frac{b_k}{\lambda f_k}\right),
\tag{12}
\]

and \(\lambda\) is the unique value meeting the native budget when it is active. This is a concrete allocation, not a candidate grid. The full table is then frozen, hashed, and used for every sequence length.

The native budget \(\varepsilon_N\) is fixed from a deployment contract before long-task evaluation. After solving, verify exact fixed-Q/K attention KL, complete-model native output KL/task deltas, and only then the untouched 128K generation set. If no feasible allocation yields useful 128K generation, the conclusion is a failure of this demand/cost model and budget at this checkpoint, not a theorem against static tables.

### C. Minimal empirical comparator

The decisive comparison is not EVQ versus MrRoPE labels. It is:

1. Native table, no motion.
2. Standard full MrRoPE motion \(q^{MR}\) under its published rule.
3. Compatibility-constrained motion (11), with the same endpoints, gain, decoder, and frozen weights.
4. If training is allowed, matched geometric and allocated arms with identical LoRA budget, plus the four weight-by-table cells in (6).

Do not retune gain while evaluating allocation. The FFN audit shows that historical Z/Y used different amplitudes, so that comparison was not pure allocation (`SINGLE_TABLE_FFN_REPORT...:166-187`). It also shows that joint full-linear adaptation cannot be interpreted as FFN necessity (`SINGLE_TABLE_FFN_REPORT...:491-507`).

## Counterexample checks that kill stronger rhetoric

1. **Geometry can improve while behavior worsens.** If a candidate lowers positional Gram distortion but changes a heavily used content slot, (2) can worsen task decisions. This is exactly the logical pattern of the supplied Smooth_MrBudget failure. It rules out selecting by geometry alone.
2. **Low-frequency positional redundancy is not content redundancy.** When \(\omega_k\Delta\ll1\), the term is \(C_{ij,k}+D_{ij,k}\omega_k\Delta+O((\omega_k\Delta)^2)\); \(C_{ij,k}\) still varies across candidate tokens. Softmax does not cancel it as a row-constant (`Unified_Plan:160-177`). Therefore “move all slow slots” is invalid.
3. **Permutation counterexample.** Permuting frequencies alone changes which \((C,D)\) pair receives which phase. Jointly permuting the Q/K rotary coordinates and frequencies preserves the bilinear calculation. Allocation is an ordered coupling, not an unordered spectrum (`Unified_Plan:181-189`).
4. **Training and frozen signs can differ.** A table can help when Q/K co-adapt yet catastrophically fail when installed on Geo-trained weights; the guidance reports PPL 7.14 to 76.20 despite effective rank increasing 4.57 to 12.54 (`Research_Guidance:30-34`).
5. **Task outcomes must remain the endpoint.** The FFN audit separates semantic correctness, format compliance, and termination and warns that contains-answer and EOS cannot stand in for correct generation (`SINGLE_TABLE_FFN_REPORT...:108-151`). Any allocator validated only by KL, effective rank, attention mass, or EOS has not met the requested outcome.
6. **A single-slot positive is a seed, not a law.** The supplied E1 slot-28 slight decompression result on tiny development samples is best used to initialise/check the sign of \(b_{28}/f_{28}\), then frozen and tested on independent examples. It cannot justify a global monotone decompression profile.

## What survives, stated narrowly

- Fixed-endpoint interior allocation is a real intervention variable; anchored EVQ-Cosh has multi-seed from-scratch evidence for conditional NLL effects, including a reversal when the runtime range changes (`Unified_Plan:23-35`).
- Mature-model allocation must include learned slot compatibility. Position-only effective rank, average frequency error, or smoothness can diagnose a table but cannot rank its task behavior.
- EVQ and MrRoPE can share the demand-measure/finite-quantisation framework in (7)-(9). They should not be forced to share the same optimum after training.
- For frozen deployment, (11) is the constructive next rule: allocate 4x radix motion where declared long-range demand is high and measured native compatibility cost is low, subject to one predeclared native budget and order preservation.
- For LoRA, use the residual compatibility cost in (5) and matched controls. A smaller local residual does not promise 128K task success.

## Coverage receipt

All three assigned files were read in full as contiguous line ranges: Unified Plan 1-897, Research Guidance 1-579, and FFN Audit 1-699. No assigned lines or files were omitted. The machine-readable receipt is `sol04_coverage.json`.
