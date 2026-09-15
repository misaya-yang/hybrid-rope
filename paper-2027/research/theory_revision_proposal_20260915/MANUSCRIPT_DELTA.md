# 精确改稿增量（V2已应用）

以下由当前稿件生成。每节给出实际删除与插入的LaTeX，新增附录另列完整文件。
本文件保留应用前的V2差异；后续R08仅进一步澄清图1c的派生表列标签。

阅读说明与取舍见[审核入口](README.md)，全部理论身份见[理论汇总](THEORY_SYNTHESIS.md)。


## sections/00_abstract.tex

```diff
--- 
+++ 
@@ -1 +1 @@
-Rotary position embeddings (RoPE) typically tie their frequency placement to a single base. We study frequency allocation as a way to improve model quality within a chosen context range. Controlled experiments show that redistributing interior frequencies improves model performance even when the frequency range is fixed. Full sine--cosine geometry characterizes positional overlap and exposes allocation rankings missed by cosine-only proxies, while coordinate interventions distinguish this supplied geometry from its learned use. We introduce TailSpline, an explicit allocation that smooths the transition into the extended low-frequency tail. It retains the standard rotary operator and requires neither weight updates nor calibration. TailSpline substantially outperforms MrRoPE-Pro on RULER at both intermediate and target extension lengths, with a small observed native-window trade-off. Supporting experiments with Cosh, a different frequency transport for extrapolation, establish that useful allocation extends beyond this construction. Together, the analysis and experiments make internal frequency placement a practical dimension of RoPE design.
+Rotary position embeddings (RoPE) typically tie their frequency placement to a single base. We study frequency allocation as a way to improve model quality within a chosen context range. Controlled experiments show that redistributing interior frequencies improves performance even when the frequency range is fixed. Full sine--cosine geometry characterizes positional overlap and exposes rankings missed by cosine-only proxies. Coordinate interventions distinguish these positional directions from the model's learned use of them. We introduce TailSpline by formulating and solving a discrete allocation problem for the transition into the extended low-frequency tail. The resulting closed-form rule redistributes distance scaling while retaining the standard rotary operator, with no weight updates or calibration. TailSpline substantially outperforms MrRoPE-Pro on RULER at both intermediate and target extension lengths, with a small observed native-window trade-off. A complementary Cosh transport demonstrates allocation benefits through paired learning and extrapolation experiments. Together, the analysis and experiments make internal frequency placement a practical dimension of RoPE design.
```


## title_abstract.txt

```diff
--- 
+++ 
@@ -1,3 +1,3 @@
 Beyond the Base: Frequency Allocation in RoPE
 
-Rotary position embeddings (RoPE) typically tie their frequency placement to a single base. We study frequency allocation as a way to improve model quality within a chosen context range. Controlled experiments show that redistributing interior frequencies improves model performance even when the frequency range is fixed. Full sine–cosine geometry characterizes positional overlap and exposes allocation rankings missed by cosine-only proxies, while coordinate interventions distinguish this supplied geometry from its learned use. We introduce TailSpline, an explicit allocation that smooths the transition into the extended low-frequency tail. It retains the standard rotary operator and requires neither weight updates nor calibration. TailSpline substantially outperforms MrRoPE-Pro on RULER at both intermediate and target extension lengths, with a small observed native-window trade-off. Supporting experiments with Cosh, a different frequency transport for extrapolation, establish that useful allocation extends beyond this construction. Together, the analysis and experiments make internal frequency placement a practical dimension of RoPE design.
+Rotary position embeddings (RoPE) typically tie their frequency placement to a single base. We study frequency allocation as a way to improve model quality within a chosen context range. Controlled experiments show that redistributing interior frequencies improves performance even when the frequency range is fixed. Full sine–cosine geometry characterizes positional overlap and exposes rankings missed by cosine-only proxies. Coordinate interventions distinguish these positional directions from the model's learned use of them. We introduce TailSpline by formulating and solving a discrete allocation problem for the transition into the extended low-frequency tail. The resulting closed-form rule redistributes distance scaling while retaining the standard rotary operator, with no weight updates or calibration. TailSpline substantially outperforms MrRoPE-Pro on RULER at both intermediate and target extension lengths, with a small observed native-window trade-off. A complementary Cosh transport demonstrates allocation benefits through paired learning and extrapolation experiments. Together, the analysis and experiments make internal frequency placement a practical dimension of RoPE design.
```


## sections/01_intro.tex

```diff
--- 
+++ 
@@ -28,8 +28,9 @@
 At fixed endpoints, internal placement improves model performance;
 changing the evaluation range can reverse the preferred allocation.
 \item \textbf{Positional structure and learned use are distinct.}
-Complete rotary pairs expose finite-window direction overlap, while
-coordinate and table interventions show how models use the supplied frequencies.
+Complete rotary pairs expose finite-window direction overlap. Task gains
+with lower positional effective rank, together with coordinate and table
+interventions, distinguish this structure from its learned use.
 \item \textbf{Analytic allocation improves task quality.}
 TailSpline gives a closed-form static extension with clean RULER gains at
 both $2L$ and $4L$. Cosh provides a supporting extrapolation transport.
```


## sections/03_theory.tex

```diff
--- 
+++ 
@@ -1,4 +1,4 @@
-\section{How Allocation Changes Positional Structure}
+\section{Positional Structure and Content Use}
 \label{sec:theory}
 
 The controlled gains raise a structural question: what changes when the frequency range stays fixed? We examine the positional directions supplied by complete rotary pairs. Their overlap connects frequency spacing to redundancy within a finite window; the model learns how to combine these directions with content.
@@ -55,31 +55,28 @@
 carry positional overlap (Appendix~\ref{sec:static-counterexamples}).
 The complete-pair measure captures this structure independently of phase.
 
-\paragraph{Design implication.}
-The same number of rotary pairs can supply very different positional diversity.
-Reallocating frequencies changes this overlap, providing a concrete reason to
-consider their spacing during learning. Longer-range coverage also matters:
-a full-rank parity lattice can repeat outside its reference window
-(Appendix~\ref{sec:rank-coverage}). We therefore design allocations for a stated
-operating setting, balancing finite-window resolution with frequency coverage.
+\subsection{Shared positional dependence and content use}
+Each rotary block remains invertible even when its positional functions
+overlap. Shared dependence on distance can therefore coexist with distinct
+content comparisons. A slow-block construction makes this distinction
+explicit (Appendix~\ref{sec:content-coordinate-retention}), complementing
+prior observations of positional and semantic frequency use
+\citep{barbero2025round}.
 
-\subsection{Changing frequencies changes the positional kernel}
-The learned-compatibility experiments raise a second question: can a fixed
-change of Q/K coordinates absorb a frequency change? For the exact bilinear
-rotary kernel, the rotation spectra give a simple criterion.
+The complete frozen Llama tables provide a concrete example. At $16/32$K,
+TailSpline has full-pair effective ranks $8.28/10.08$, compared with
+MrPro's $8.74/10.21$ under uniform separations, while improving clean
+RULER by $3.39/11.72$ points (Table~\ref{tab:clean-length-main}).
+The formulas, full precision and alternative separation measures are in
+Appendix~\ref{sec:allocation-rank-quality}.
+These gains establish the value of controlling distance dependence through
+allocation even when normalized positional diversity decreases. The model's
+use of the supplied frequencies determines their practical value.
 
-\begin{corollary}[Integer-position equivalence]\label{cor:discrete-kernel}
-Let $\Omega,\Lambda$ contain $K$ frequencies in $(0,\pi)$ and
-$\mathcal R_\Omega(d)=\operatorname{diag}_k R(\omega_kd)$.
-Position-independent invertible $A,B$ satisfy
-$A^\top\mathcal R_\Omega(d)B=\mathcal R_\Lambda(d)$ for $d=0,1$
-if and only if the frequency multisets agree.
-\end{corollary}
-At $d=0$, $B=(A^\top)^{-1}$; at $d=1$, similarity preserves
-$\{e^{\pm i\omega_k}\}$. In $(0,\pi)$ this identifies the multiset.
-A block permutation proves the converse for every integer $d$.
-This spectral criterion separates changing the supplied frequencies from
-reassigning an unchanged spectrum, complementing STRING's structural
-characterization \citep{schenck2025string}. The assumptions concern exact
-kernels for all content vectors; the full proof and aliasing cases are in
-Appendix~\ref{sec:discrete-kernel-proof}.
+\paragraph{Frequency changes and coordinate assignment.}
+Changing frequencies also differs from reassigning a fixed spectrum.
+In the no-alias interval $(0,\pi)$, position-independent invertible Q/K
+maps preserve the full integer-position rotary kernel exactly if and only
+if the frequency multisets agree (Appendix~\ref{sec:discrete-kernel-proof}).
+This criterion complements the coordinate interventions and STRING's
+structural characterization \citep{schenck2025string}.
```


## appendix/a1_proofs.tex

```diff
--- 
+++ 
@@ -346,6 +346,14 @@
 
 \subsection{Integer-position kernel equivalence and its boundaries}
 \label{sec:discrete-kernel-proof}
+\begin{corollary}[Integer-position equivalence]\label{cor:discrete-kernel}
+Let $\Omega,\Lambda$ contain $K$ frequencies in $(0,\pi)$ and
+$\mathcal R_\Omega(d)=\operatorname{diag}_k R(\omega_kd)$.
+Position-independent invertible $A,B$ satisfy
+$A^\top\mathcal R_\Omega(d)B=\mathcal R_\Lambda(d)$ for $d=0,1$
+if and only if the frequency multisets agree.
+\end{corollary}
+
 Corollary~\ref{cor:discrete-kernel} applies to the full bilinear kernel on the
 $2K$-dimensional rotary space. Equality for all $q,k$ is matrix equality.
 At $d=0$, $A^\top B=I$; at $d=1$ the two rotation matrices are similar.
```


## sections/04_mature.tex

```diff
--- 
+++ 
@@ -7,9 +7,18 @@
 x'_{l+q}-x'_{l+q-1}=c+\epsilon_q\log s.
 \label{eq:increment-log-gap}
 \end{equation}
-Their unit sum allocates exactly $\log s$ of extra span across the band. Beyond it, constant displacement $m=1$ preserves native adjacent gaps: the extra gap returns to zero. The remaining choice is how to reach this junction.
+Their unit sum allocates exactly $\log s$ of extra span across the band.
+At pair $q$, a fixed exponent $m_q$ assigns wavelength growth $s^{m_q}$;
+the endpoints fix the total span, while the interior profile determines
+how this growth is distributed across channels.
+The outer bands preserve two distance references: high frequencies retain
+$R_{\omega^N}(d)$, while the low-frequency tail satisfies
+$R_{\omega^N/s}(sd)=R_{\omega^N}(d)$. At the tail, constant displacement
+$m=1$ preserves native adjacent gaps, so the extra gap returns to zero.
 
-MrRoPE-Pro uses increasing increments, $m_q=q(q+1)/[n(n+1)]$. We instead smooth the transition into the fully interpolated tail, whose subsequent increments are zero:
+MrRoPE-Pro uses increasing increments, $m_q=q(q+1)/[n(n+1)]$.
+We formulate a discrete tail-connection problem by penalizing variation
+in the extra log gaps and their mismatch with the fully interpolated tail:
 \begin{equation}
 \min_{\epsilon:\,\sum_{q=1}^n\epsilon_q=1}
 \sum_{q=1}^{n-1}(\epsilon_{q+1}-\epsilon_q)^2+\epsilon_n^2.
@@ -24,17 +33,19 @@
 \label{eq:tailspline}
 \end{equation}
 \end{theorem}
-Strict convexity and the tridiagonal stationarity system give Eq.~\eqref{eq:tailspline} (Appendix~\ref{sec:tailspline-details}). Adding the entry penalty $\epsilon_1^2$ yields the symmetric construction BM, with increments proportional to $q(n-q+1)$ (Appendix~\ref{sec:bm-construction}). Relative to MrPro, TailSpline accepts a larger entry jump and reduces the terminal jump by the exact factor $3/(2n+1)$: $0.086$ for Llama's $n=17$ and $0.081$ for OLMo's $n=18$ (Fig.~\ref{fig:construction-contrast}). These guarantees solve the declared objectives; task value is tested below.
+Strict convexity and the tridiagonal stationarity system give Eq.~\eqref{eq:tailspline} (Appendix~\ref{sec:tailspline-details}). Adding the entry penalty $\epsilon_1^2$ yields the symmetric construction BM, with increments proportional to $q(n-q+1)$ (Appendix~\ref{sec:bm-construction}). Relative to MrPro, TailSpline accepts a larger entry jump and reduces the terminal jump by the exact factor $3/(2n+1)$: $0.086$ for Llama's $n=17$ and $0.081$ for OLMo's $n=18$ (Fig.~\ref{fig:construction-contrast}). TailSpline also redistributes scaling across the entire transition:
+its wavelength multiplier is $s^{m_q^{\rm TS}}$, with
+$m_q^{\rm TS}\ge m_q^{\rm Pro}$ at every interior pair. On the Llama
+$s=4$ grid, the largest TailSpline/MrPro wavelength ratio is $1.766$.
+Appendix~\ref{sec:allocation-scale-response} derives this ordering and
+compares the response to scale with YaRN's frequency blend.
 
 \paragraph{What the comparison isolates.}
 TailSpline and MrPro retain the same outer bands and endpoints, so their
-comparison tests complete internal allocations. Total displacement is one
-statistic of that allocation. An additional control $C$ matches
-TailSpline's total displacement to test the remaining shape difference
-(Appendix~\ref{sec:tailspline-dose-control}). The completed classic diagnostic
-gives T--C $-0.41$ points in Full-13 AUC, with interval $[-2.63,1.82]$;
-T uses batch $1$ and C batch $2$, leaving this finer ordering unresolved
-(Appendix~\ref{sec:current-controls}).
+comparison tests complete internal allocations. At fixed support, total
+log-frequency displacement is a statistic of $z$. The control $C$ also
+matches this statistic to study the remaining shape difference
+(Appendices~\ref{sec:tailspline-dose-control} and~\ref{sec:current-controls}).
 
 \paragraph{Installation.}
 (1) Read the public native grid, reference length and scale $s$; (2) compute the $32/1$-turn band; (3) evaluate $m_k$, $\omega'_k=\omega_k^Ns^{-m_k}$ and $g=1+0.1\ln s$; (4) install before prefill, retaining one table across layers and input lengths. The baseline shares $g$. Construction costs $O(K)$ and leaves the rotary computation unchanged. Neither constructor reads weights, activations or calibration outputs.
```


## sections/02_related.tex

```diff
--- 
+++ 
@@ -7,7 +7,7 @@
 Round and Round studies positional and semantic frequency use; its $p$-RoPE gains include Gemma training from scratch \citep{barbero2025round}. HoPE studies high-frequency positional encoding \citep{chen2025hope}; Massive Values studies large Q/K components \citep{jin2025massive}. FoPE changes Fourier components, while LeRoPE learns frequencies and derives their attention/value gradients \citep{hua2025fope,karypis2026lerope}. We build on frequency design by isolating the value of interior placement and connecting positional structure to explicit constructions for learning and frozen extension.
 
 \paragraph{Group structure and changes of basis.}
-STRING characterizes differentiable matrix encodings satisfying an identity and a group-like relative-translation law, and relates them to RoPE through an orthogonal basis \citep{schenck2025string}. GRAPE connects group generators, learned frequencies and frequency--norm coupling \citep{zhang2026grape}. Our integer-position corollary uses rotation spectra to separate a change in frequencies from a fixed change of basis.
+STRING characterizes differentiable matrix encodings satisfying an identity and a group-like relative-translation law, and relates them to RoPE through an orthogonal basis \citep{schenck2025string}. GRAPE connects group generators, learned frequencies and frequency--norm coupling \citep{zhang2026grape}. Our integer-position criterion uses rotation spectra to separate a frequency change from a fixed change of basis (Appendix~\ref{sec:discrete-kernel-proof}).
 
 \paragraph{Task structure and positional dependence.}
 PINE changes inter-document masks and ordering to target document-level invariance \citep{wang2024pine}; PRoPE encodes relative camera geometry \citep{li2025cameras}. Selective RoPE and RePo learn input-dependent rotations or positions \citep{movahedi2026selectiverope,li2026repo}. These modify different objects from a static frequency allocation. HELMET separates application categories whose behavior synthetic retrieval alone does not predict \citep{yen2024helmet}, motivating distinct retrieval, natural-QA and native-window evidence rather than a single pooled score.
```


## appendix/a11_allocation_response.tex

完整新增内容见[附录草稿](proposed_appendix.tex)。


## main.tex

```diff
--- 
+++ 
@@ -129,5 +129,6 @@
 \input{sections/03_compatibility}
 \input{sections/05_threeband}
 \input{appendix/a8_profile_diagnostics}
+\input{appendix/a11_allocation_response}
 
 \end{document}
```


## appendix/a0_guide.tex

```diff
--- 
+++ 
@@ -1,6 +1,6 @@
 \section*{Guide to the Supplementary Material}
 The supplementary material follows the argument from the independent value of $z$ to positional structure, explicit constructions and task validation. Table~\ref{tab:argument-guide} maps claims to their supporting material; Table~\ref{tab:protocol-glossary} distinguishes the experimental contracts.
-The core reading path is fixed-range identification (Appendix~\ref{sec:exact-range-control}), complete-pair geometry and its ordering counterexample (Appendices~\ref{sec:geometry-proofs} and~\ref{sec:static-counterexamples}), then TailSpline construction and clean task gains (Appendices~\ref{sec:tailspline-details},~\ref{sec:clean-confirmation},~\ref{sec:clean16k-confirmation}). Cosh extrapolation, native-window and natural-QA evaluations provide complementary results under their stated protocols.
+The core reading path is fixed-range identification (Appendix~\ref{sec:exact-range-control}), complete-pair geometry and its ordering counterexample (Appendices~\ref{sec:geometry-proofs} and~\ref{sec:static-counterexamples}), then TailSpline construction and clean task gains (Appendices~\ref{sec:tailspline-details},~\ref{sec:clean-confirmation},~\ref{sec:clean16k-confirmation}). Allocation response and the rank--quality comparison are developed in Appendix~\ref{sec:allocation-response}. Cosh extrapolation, native-window and natural-QA evaluations provide complementary results under their stated protocols.
 \begin{table}[ht]
 \centering\small
 \caption{\textbf{Where each part of the argument is developed.}}
```
