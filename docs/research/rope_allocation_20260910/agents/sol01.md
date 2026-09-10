# Sol 01 — what survives of MrRoPE as mathematics, and an EVQ-compatible allocation rule

## Decision

MrRoPE supplies a useful **cumulative frequency-allocation coordinate system**, but it does not supply an exact mixed-radix encoding theorem or a task-optimal allocation rule. The mathematically sound object is

\[
\nu_j=\omega_j S^{-m_j},\qquad
m_j=\sum_{d<j}\epsilon_d,\qquad
\lambda_d=S^{\epsilon_d},
\]

with anchored endpoints and, when desired, nonnegative increments \(\epsilon_d\). MrRoPE-Pro chooses arithmetic increments and therefore a quadratic cumulative exponent. That construction rigorously delays compression relative to MrUni, but the arithmetic progression is an assumption, not an optimizer derived from attention or task loss.

For frozen deployment, the constructive next rule is: optimize the **exact finite-window, source-weighted EVQ collision functional** over these cumulative exponents, subject to an explicit measured budget on damage to the pretrained model's native-window logits. This turns MrRoPE into the feasible coordinate system and EVQ into one long-range objective, while refusing to treat either geometry alone as accuracy. For training from scratch, the pretrained-damage constraint has no meaning; frequencies and network weights must be optimized jointly under train and target distance distributions.

## 1. Exact mixed-radix arithmetic versus the RoPE analogy

Let radices \(r_1,\ldots,r_D\) be positive integers and define place weights

\[
P_1=1,\qquad P_j=\prod_{d<j}r_d.
\]

The exact digit and reconstruction for an integer \(n\in[0,P_{D+1})\) are

\[
a_j(n)=\left\lfloor\frac{n}{P_j}\right\rfloor\bmod r_j,
\qquad n=\sum_{j=1}^D a_jP_j.
\]

A carry occurs because incrementing \(n\) changes the floored quotient, and overflow modulo \(r_j\) propagates to \(j+1\). If \(r_j=\beta\lambda_j\), then \(P_j=\beta^{j-1}\prod_{d<j}\lambda_d\). This is the legitimate source of MrRoPE's cumulative products.

RoPE instead observes continuous circular phases

\[
\varphi_j(n)=\left(\frac{n}{P_j}\right)\bmod 2\pi.
\]

This is not the digit \(a_j\): it has no floor, its modulus is \(2\pi\) rather than \(r_j\), and the usual \(\beta=b^{1/D_r}\) and \(\beta\lambda_j\) are generally nonintegers. The paper itself acknowledges discarding the floor and the difference in modulus before calling RoPE radix-like (primary paper lines 174–203), then calls a weighted sum of wrapped phases a biased estimate (lines 205–249). Reintroducing the phases does not reintroduce the missing floor or make the reconstruction exact.

Thus the strongest correct statement is:

> Any positive per-slot RoPE frequency table can be parameterized by adjacent cumulative ratios analogous to mixed-radix place weights. This is an algebraic frequency parameterization, not a digit representation with carries.

This distinction matters because no collision-free range \(\prod r_j\), unique decoding theorem, or allocation optimality transfers from radix arithmetic to RoPE phases.

## 2. Finite-index ambiguities and an off-by-one convention that must be fixed

The paper defines

\[
\nu_j=\frac{\omega_j}{\prod_{d=1}^{j-1}\lambda_d}
\]

(primary paper lines 305–324 and 441–456). Therefore:

1. \(\lambda_{D_r}\) affects no frequency at all. The claim that the RoPE table's range is enlarged by \(\prod_{d=1}^{D_r}\lambda_d\) contains an unidentifiable last factor. Observable frequencies determine only \(\lambda_1,\ldots,\lambda_{D_r-1}\).
2. If \(\lambda_d\ne1\) for \(d_l\le d<d_h\), then \(\lambda_{d_l}\) first changes frequency slot \(d_l+1\), and \(\lambda_{d_h-1}\) changes slot \(d_h\). This conflicts with prose saying the middle frequencies themselves are exactly \([d_l,d_h)\) (lines 464–485).
3. A table should therefore specify **frequency anchors**, not vaguely reuse the same indices for frequencies and gaps. Let transition frequencies be \(j=l,l+1,\ldots,h\), set \(m_l=0,m_h=1\), and define the \(N=h-l\) gap increments by \(\epsilon_q=m_{l+q}-m_{l+q-1}\), \(q=1,\ldots,N\). Then there is no ambiguity:

\[
\nu_{l+q}=\omega_{l+q}S^{-m_{l+q}},\quad
m_{l+q}=\sum_{i=1}^q\epsilon_i,\quad
\sum_{i=1}^N\epsilon_i=1.
\]

This convention also makes explicit that the endpoint slot \(h\) is fully scaled and that the \(N\) allocation variables live on gaps.

## 3. Complete MrRoPE-Pro construction and what it proves

MrRoPE-Pro assumes increasing arithmetic gap increments (primary paper lines 412–433):

\[
\epsilon_q=cq,qquad
1=\sum_{q=1}^N cq=c\frac{N(N+1)}2,qquad
c=\frac{2}{N(N+1)}.
\]

Hence

\[
\boxed{\epsilon_q^{\rm Pro}=\frac{2q}{N(N+1)}},\qquad
\boxed{m_q^{\rm Pro}=\frac{q(q+1)}{N(N+1)}}.
\]

The CPU exact-fraction check reproduced \(\sum_q\epsilon_q=1\) for \(N=2,3,17,18\), including target-style \(N=17\) endpoints \(\epsilon_1=1/153\), \(\epsilon_{17}=1/9\), and \(\sum_{q=1}^{16}m_q=16/3\). These match the project reconstruction at `ROPE_MRPRO_BM_CONSTRUCTION_ANALYSIS_20260908.md`, lines 7–20 and 36–45.

Compared with MrUni, \(m_q^{\rm Uni}=q/N\),

\[
m_q^{\rm Uni}-m_q^{\rm Pro}
=\frac{q(N-q)}{N(N+1)}>0\quad(0<q<N).
\]

So Pro strictly delays all internal cumulative compression while preserving endpoints. This is the rigorous allocation principle that follows from the chosen arithmetic sequence. Nothing in the derivation establishes that arithmetic increments minimize a model-relevant loss. The project evidence explicitly reaches the same boundary: progressive delay is interpretable, but a completed single-slot cycle does not imply joint phase/content coverage (construction analysis lines 18–20).

The BM counterexample blocks upgrading spectral smoothness into the missing theorem. BM lowers several geometric roughness/distortion quantities yet moves every internal slot slower than MrPro and adds 50% to the sum of internal cumulative exponents (construction analysis lines 22–63); it then loses 7.29 points on the paired Qwen 3B 128K six-task aggregate, with heterogeneous task effects and prefix-formation evidence (lines 67–81). Conversely, OLMo BM strongly beats MrPro on its tested input (line 65). Therefore neither smaller geometric distortion, delayed compression, nor total exponent mass is a universal task utility.

## 4. Scope of the YaRN “regressive” proof

The appendix writes the cumulative YaRN factor as

\[
A(r_j)=c+(S-1)r_j,\qquad c=\beta-S\alpha,
\]

and compares adjacent ratios using \(r_{j+1}+r_{j-1}\ge2r_j\) (primary paper lines 1114–1242). The stated direction needs \((S-1)c\ge0\), hence for \(S>1\), \(\beta-S\alpha\ge0\). It holds in the target deployment regime \(S=4,\alpha=1,\beta=32\). It is not valid for every \(S\) or every \(b\ne1\), as the paper claims at lines 1240–1242: when \(c<0\), multiplying the AM–GM gap by \(c\) reverses its contribution. This does not invalidate the target \(S=4\) classification; it invalidates the universal quantifier.

## 5. A concrete EVQ–MrRoPE rule

Use the anchored cumulative variables above and log-frequency positions

\[
\phi_j(m)= -\log_b\nu_j
=\phi_j^0+\frac{\log S}{\log b}m_j.
\]

Let \(K_L(\phi_a,\phi_b)\) be the exact finite-window EVQ kernel. The supplemental EVQ note gives it in closed form via cosine integrals (`EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md`, lines 8–43). It also proves that the local delta approximation overpenalizes nonzero-wavenumber variation and is unsafe for narrow slot reallocations (lines 45–91), exactly the regime of MrRoPE transition editing.

For a frozen pretrained model define two empirically estimable quantities on frozen calibration examples:

\[
D_0(m)=\mathbb E_{x,\ell,h,n\le L_0}
\left[\bigl(z_{\ell h n}(m;x)-z_{\ell h n}(0;x)\bigr)^2\right],
\]

where \(z\) is the actual pre-softmax attention logit, and

\[
C_T(m)=\sum_{a,b}W_{ab}\,K_{L_T}(\phi_a(m),\phi_b(m)).
\]

Here \(W\succeq0\) must come from frozen-model slot/source statistics (including cross-slot structure), not equal slot weights. The allocation rule is

\[
\boxed{
m^*=\arg\min_{m}\ C_T(m)
\quad\text{s.t.}\quad
m_l=0,\ m_h=1,\ 0\le m_{j+1}-m_j,\ D_0(m)\le\delta .}
\]

Choose \(\delta\) from an already accepted baseline's measured native-window distortion (for example, MrPro), rather than inventing a scalar tradeoff weight. This is decision-ready: it returns one Pareto allocation; it preserves the required total extension and ordering; it uses exact nonlocal EVQ rather than a smoothness proxy; and it rejects a candidate whose long-range collision gain requires more pretrained-logit damage than the baseline budget.

For small changes around a feasible baseline \(m^0\), let \(g=\nabla C_T(m^0)\), \(D_0(m^0+u)-D_0(m^0)\approx\tfrac12u^THu\), and project onto the endpoint-preserving tangent cone. Ignoring temporarily active monotonicity faces, the trust-region direction is

\[
u^*=-\sqrt{\frac{2\Delta}{g^TH^{-1}g}}\,H^{-1}g.
\]

With active faces, solve the corresponding convex quadratic subproblem. This gives a precise marginal rule: spend extension displacement where exact EVQ benefit per unit of measured native-logit damage is largest, including cross-slot couplings. It also explains why a per-slot independent argmin is unjustified: the project review explicitly retains cross terms in expected logit error (transition review lines 107–130).

This is a **candidate-generation rule**, not a success theorem. Full nonlinear recomputation and the paired long-task evaluation remain necessary because softmax, V mixing, and layerwise prefix formation are outside the cosine kernel; the BM cross-cache evidence shows that prefix formation matters.

## 6. Training from scratch is a different optimization problem

For frozen deployment, \(D_0\) encodes a real causal asymmetry: Q/K/V weights learned under \(\omega\) are held fixed, so changing frequency can damage learned native relations. MrPro's delayed compression is a reasonable hand-designed point in this feasible set.

For training from scratch, there is no pretrained \(m=0\) behavior to preserve. Using the frozen objective would privilege an arbitrary initialization. The correct formulation is joint:

\[
(\theta^*,m^*)=\arg\min_{\theta,m}
\mathbb E_{(x,y),\,L\sim p_{\rm train}}
[\mathcal L(f_{\theta,m}(x_{1:L}),y)]
+\gamma C_{p_{\rm target}}(m),
\]

with the same identifiability anchors and any hardware/ordering constraints, followed by held-out target-length evaluation. EVQ may regularize frequency crowding under the target distance prior, but its weight and source matrix must be learned/validated jointly; native-logit preservation is removed. A scratch-trained optimum can legitimately compress early channels more than MrPro because the content projections can adapt. Consequently, evidence for frozen MrPro does not establish a scratch allocation, and a scratch result does not isolate training-free frequency substitution.

## 7. Counterexamples and falsifiable checks

1. **Exact-radix counterexample:** with one frequency and any claimed last radix \(\lambda_1\), Eq. 10 contains an empty product, so the RoPE phase is unchanged while the claimed representable mixed-radix range changes. Therefore range scaling cannot be a property of the observable RoPE table.
2. **Index counterexample:** for \(D_r=4\), set only \(\lambda_2=S\). Frequencies 1 and 2 remain native; frequencies 3 and 4 scale by \(1/S\). Calling dimension 2 the scaled middle dimension is false under the paper's own product convention.
3. **Progressive-limit checks:** \(N=1\) gives \(\epsilon_1=1\), so Pro=Uni=one endpoint jump. For every \(N>1\), Pro delays every internal point; it does not minimize total compression under an increment cap. The project-derived coordinatewise-delay solution under a cap is instead a narrowed MrUni identity (construction analysis lines 83–113).
4. **EVQ proxy check:** if exact \(C_T\) improves while paired task outcomes worsen, the collision objective lacks necessary source/task structure; do not rescue it by claiming geometry success. Smooth/BM history already makes this a live possibility.
5. **Frozen-versus-scratch discriminator:** if the same table ranks differently after end-to-end retraining, that supports the learned-native-damage mechanism rather than a universal encoding geometry.

## Recommended next comparison

Do not launch a curve grid. Starting from the deployed MrPro table for Qwen2.5-3B, estimate \(g\) from the exact finite EVQ kernel with frozen source weights and estimate an HVP-based \(H^{-1}g\) direction from native-window conditional logits. Generate one endpoint-preserving, monotone trust-region candidate at the already measured MrPro native-distortion budget. Reject it before GPU work if the nonlinear exact-kernel recomputation loses its predicted improvement or if the source-weighted benefit is dominated by the unweighted proxy. If it survives, compare only this candidate with the reused MrPro baseline on the already paired 128K tasks; interpret the outcome as one frozen-model allocation test.

## Evidence boundary

All three assigned files were ingested in full. The EVQ nonlocal-kernel note was additionally ingested in full because the assignment explicitly required an EVQ connection. No GPU or model job was launched, and no paper/runtime source was changed. CPU work was limited to exact-fraction construction checks and source/hash verification.
