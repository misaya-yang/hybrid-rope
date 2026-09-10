# Sol16 report: finite frequency allocation and frozen-model calibration

## Decisive result

The clean unification is an **allocation manifold**, not a universal score-optimal schedule.  Write every RoPE table in log-frequency coordinates

\[
x_j=-\log \omega_j=x_0+A z_j,\qquad 0=z_0<z_1<\cdots<z_{K-1}=1.
\]

Equivalently, the normalized positive gaps \(a_j=(x_j-x_{j-1})/A\) lie on a simplex. EVQ specifies a continuum density and quantizes its inverse CDF into such gaps; MrRoPE specifies additive dilation gaps over a native grid. This coordinate system correctly separates fast endpoint \(x_0\), support span \(A\), and internal allocation \(a/A\), matching the 6Pro decomposition (`docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:40-55`).

The training regime changes the optimization problem:

\[
\text{scratch: }\min_{W,a}\;\mathbb E_{(x,y)\sim P_{\rm train}}[-\log p_{W,a}(y\mid x)],
\]

\[
\text{frozen deployment: }\min_a\;\mathbb E_{(x,y,p)\sim P_{\rm deploy}}[-\log p_{W_0,a}(y\mid x,p)],\quad W_0\text{ fixed}.
\]

These are not interchangeable. The co-adaptation table gives direct counterevidence: matched Geo/Geo and EVQ/EVQ are PPL 7.14/7.16, while cross-swaps are 76.20 and 23.05 even though one swap improves static rank (`paper-2027/tables/table_coadapt.tex:6-21`). Thus a static phase metric can be a construction prior, but cannot select a frozen deployment table.

## Exact finite conventions

For Qwen2.5-3B, use the model's native HF convention, not an EVQ midpoint surrogate:

* head dimension \(d=128\), pair count \(K=64\), base \(\theta=10^6\);
* zero-based pair index \(j=0,\ldots,63\);
* \(\omega_j^{\rm native}=\theta^{-2j/d}=\theta^{-j/64}\). The fast endpoint is 1; the last pair is \(\theta^{-63/64}\), not \(\theta^{-1}\). This is the installed Qwen construction (`transformers/models/qwen2/modeling_qwen2.py:89-98`) and the canonical local geometric builder (`scripts/lib/rope/schedules.py:86-91`).

The current MrPro transition is exactly zero-based `low=23`, `high=40`, so \(N=17\) transition gaps. For \(q=j-23\):

\[
m_j^{\rm Mr}=\begin{cases}
0,&j\le23,\\
q(q+1)/(17\cdot18),&24\le j\le40,\\
1,&j\ge40,
\end{cases}
\qquad \omega_j=\omega_j^{\rm native}4^{-m_j}.
\]

The 17 positive radix increments are \(\epsilon_i^{\rm Mr}=2i/[17\cdot18]\), \(i=1,\ldots,17\), and sum to one. Hence the constrained transition has 16 effective allocation degrees of freedom. This also explains why “16 learned transition parameters” must not mean 16 unconstrained frequencies: that would lose the exact total scale or ordering.

Use an identifiable 16-vector \(\eta\), a fixed \(17\times16\) orthonormal Helmert basis \(B\) for the zero-sum subspace, and

\[
\epsilon(\eta)=\operatorname{softmax}(\log\epsilon^{\rm Mr}+B\eta),\qquad
m_{23+q}=\sum_{i=1}^{q}\epsilon_i(\eta).
\]

At \(\eta=0\), this is exactly MrPro in float64. Every finite \(\eta\) keeps increments positive, the cumulative exponent strictly increasing, the transition total exactly one, the outer bands fixed, and the final table decreasing. All 16 stored scalars are identifiable; using 17 free logits would introduce a useless constant-shift null direction.

The CPU reference is `.agents/rope_unification_20260910/code/sol16_frequency_calibration_reference.py`. It checks initialization, endpoints, strict order, all-parameter gradient flow through the actual FP32 rotary computation, and autograd finite differences. It passes under the local `aidemo` environment.

### EVQ convention warning

There are three distinct conventions in this repository and they must not be mixed:

1. Native/HF geometric uses \(u_j=j/K\) (`schedules.py:86-91`).
2. Canonical EVQ defaults to midpoint \(u_j=(j+1/2)/K\); at \(\tau=0\) this is midpoint geometric, not the native HF table (`schedules.py:162-208`).
3. Some experiments take midpoint EVQ then affinely re-anchor its first and last sampled \(\phi\) values to the native endpoints (`scripts/core_text_phases/phase16_phase_allocation_budget_matrix_m4.py:84-93`).

The theory note uses \(u_k=k/N\) while discussing the continuous \(u=1\) endpoint (`docs/theory/EVQ_COSH_THEORY.tex:49-66,277-292`). With \(k=0,\ldots,N-1\), the sampled slow endpoint is not \(u=1\). `LearnableEVQRoPE` also uses midpoint points (`scripts/lib/rope/learnable_evq.py:82-84`) while its prose says endpoints do not move. For frozen Qwen calibration, deriving frequencies as a multiplicative dilation of the actual native FP32 table avoids all three ambiguities.

## Smallest full-model frequency-only calibration

This is a calibration experiment, not a new allocation theory.

1. Freeze the exact Qwen2.5-3B checkpoint and hash all weight files plus the in-memory non-rotary state. Keep MrPro's gain fixed in every arm; do not learn gain, position remapping, adapters, or model weights.
2. Replace only `model.model.rotary_emb` with the differentiable shared module above, initialized at \(\eta=0\). Set `use_cache=False`, use SDPA, disable dropout through `model.eval()` while leaving autograd enabled, and request only the last answer-span logits if the answer is at the end.
3. Freeze a small set of 32K record tasks before optimization and split by record identity into fit and calibration-selection subsets. The prompt must not contain the answer. Standard teacher forcing may expose earlier answer tokens only when predicting later answer tokens. Labels are active only on the exact correct-answer span.
4. Use the exact stretched position convention \(p'_t=4p_t\), \(p_t=0,\ldots,L-1\), with \(L\le32768\). This reaches position 131068 and keeps the token/attention graph at 32K while exercising the 128K frequency phases. It is a position-stretch calibration distribution, not evidence for real 128K-token behavior. A later full 128K-token evaluation remains required.
5. Optimize mean correct-answer token CE over complete model forwards. Every layer, attention head, and causally visible key participates. No selected-key replay, selected-head objective, attention geometry surrogate, or correct-count discontinuity is used for gradients.
6. Do not introduce a penalty-weight grid. Save each optimizer step. Among steps that satisfy `native-position answer CE <= initial MrPro native-position answer CE + numerical tolerance` on the held-out calibration subset, select the step with lowest stretched-position answer CE. If no non-initial step is feasible, the calibrated answer is MrPro and the result is negative.
7. Evaluate the selected table once against Native, MrPro, and the existing strongest relevant frozen baseline on untouched real 128K tasks. Report correct-answer CE first, then exact-match/task scores. A 32K stretched-position gain alone is not promotion evidence.

Adam on 16 scalars is sufficient. Start with one fixed optimizer setting and an explicit maximum displacement bound chosen before task labels are opened, rather than a candidate grid. Because the parameterization already enforces order and total scale, projection is unnecessary. If a displacement bound is needed, bound \(\|B\eta\|_\infty\) symmetrically around MrPro and treat hitting the bound as a diagnostic.

The 6Pro review already establishes why old replay data cannot substitute for this: the stored rows retained only limited keys and sometimes included the answer prefix (`ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:57-66`). Full-model CE removes that selected-key/head approximation, although stretched 32K positions still remain a synthetic calibration distribution.

## Mandatory implementation fixes and controls

The installed `Qwen2RotaryEmbedding.forward` is decorated with `@torch.no_grad()` (`modeling_qwen2.py:100-113`). Therefore all of these silently fail to learn frequency allocation: turning `inv_freq` into a parameter, copying a differentiable table into the buffer, or reusing the stock forward. The shared rotary module must be replaced by a forward without `no_grad`. Preserve the stock numerical convention: cast inverse frequencies and position IDs to FP32, disable autocast for phase matmul/cos/sin, multiply the fixed gain, and cast cos/sin to the hidden dtype.

Gradient checkpointing is compatible only after that replacement. Use non-reentrant checkpointing (`use_reentrant=False`) and `enable_input_require_grads()` as the existing recovery runtime does. Position embeddings are computed once before the decoder-layer loop in current Qwen (`modeling_qwen2.py:384-400`), so their graph can accumulate gradient from every checkpointed layer. Disable `torch.compile` until checkpointing-on/off parity is proven.

Run these failure controls before any calibration result is interpreted:

* **MrPro parity:** \(\eta=0\) table matches the deployed MrPro FP32 tensor and whole-model CE/logits match buffer injection within fixed tolerance.
* **Gradient scope:** the optimizer has exactly one 16-vector; every model weight has `requires_grad=False`; all 16 gradients are finite and nonzero on an ordinary stretched batch; non-rotary state hashes are unchanged after a step.
* **Zero-effect controls:** with all position IDs zero, or with scale \(S=1\), the gradient to \(\eta\) must be exactly zero because phase no longer depends on the allocation.
* **Directional derivative:** for a random unit direction, autograd \(g^Tv\) agrees with central finite differences of whole-model answer CE on a short CPU/tiny-model contract and on one GPU batch at a practical epsilon.
* **Checkpoint parity:** one batch with checkpointing disabled and enabled gives matching loss and eta gradient. A one-step tiny-subset overfit lowers CE in both modes.
* **Precision parity:** phase stays FP32 under BF16 model autocast. Record loss/gradient differences against an FP32 rotary reference.
* **No cache/compile mutation:** `use_cache=False`; no dynamic-RoPE decorator; the active table hash and fixed gain are checked before and after every forward.
* **Answer hygiene:** decode and inspect label boundaries; verify answer tokens are absent from the prompt; validate that only shifted answer labels contribute to CE.

The local learnable EVQ module is useful evidence that a differentiable scalar schedule is straightforward, but its `tau.item()` data-dependent branch (`learnable_evq.py:95-112`) is hostile to compilation and its midpoint/endpoints do not match native Qwen. It should not be inserted directly into this calibration.

## Evidence boundaries and novelty

The assigned 6,144-row artifact contains 2,048 paired worlds: relation accuracy is 1.0 (4,096/4,096; mean NLL 0.000773), while content accuracy is only 0.1948 (399/2,048; mean NLL 2.2525). The associated 512-pair development status is similarly content-limited (0.2012) despite perfect relation EM. This is evidence that proxy or floor-limited task panels can mis-rank tables; it is not evidence for a frequency allocation law.

Boundary-matched MrPro is the unique minimizer of its stated adjacent-gap roughness objective, but that theorem has no task-score consequence (`scripts/analysis/build_boundary_matched_mrpro.py:15-43,97-105`). The existing source itself states that limitation. Smooth_MrBudget's better geometry and worse task behavior is the decisive counterexample class: smoothing or distortion reduction cannot replace model likelihood.

Likewise, the proposed 16-DOF correct-answer-CE optimization is methodologically close to learnable RoPE / LeRoPE-like task calibration. A successful calibrated table would establish: “on this checkpoint and frozen calibration/validation distribution, moving within the MrPro transition simplex improves likelihood/task behavior.” It would not establish a new universal EVQ–MrRoPE theory.

A theory contribution would need an ex-ante rule that predicts allocation from declared model/data/topology observables, fixes the finite convention without task-label fitting, and survives counterexamples across checkpoints and scales. The safe current claim is:

* EVQ supplies a continuum allocation prior for joint training.
* MrPro supplies a deployment support/band structure and a sensible initialization.
* Full-model frozen CE supplies the correct checkpoint-specific selector.
* The log-gap simplex is the shared mathematical coordinate system.

This is a constructive unification of parameterization and regime, with calibration explicitly separated from novelty.

## Coverage

All 15 assigned files were read in full. `rows.jsonl` was read as contiguous bounded pages covering lines 1-6144. No assigned-file omissions. I additionally read the complete EVQ theory note, 6Pro review, MrPro boundary source, canonical schedule/learnable/knot code, and the complete core training runner. No GPU jobs were launched.
