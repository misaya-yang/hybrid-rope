<!-- FILE: main.tex -->
\documentclass{article}
\PassOptionsToPackage{table}{xcolor}
\usepackage{iclr2027_conference,times}

\usepackage[T1]{fontenc}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{placeins}
\usepackage{url}
\usepackage[colorlinks=true,linkcolor=black!70!blue,citecolor=black!70!blue,
            urlcolor=black!60!blue,filecolor=black!60!blue]{hyperref}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{amsthm}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{assumption}[theorem]{Assumption}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

\newcommand{\evq}{\textsc{EVQ-Cosh}}
\newcommand{\dhd}{d_{\mathrm{head}}}
\newcommand{\deff}{d_{\mathrm{eff}}}
\newcommand{\drot}{d_{\mathrm{rot}}}
\newcommand{\Ltr}{L_{\mathrm{train}}}
\newcommand{\Capp}{\mathcal{C}_{\mathrm{app}}}
\newcommand{\rs}{\textsc{YaRN-style}}

\title{RoPE Has a Spectral Budget}
\author{Anonymous authors\\Paper under double-blind review}

\begin{document}
\maketitle

\begin{abstract}
<!-- FILE: sections/00_abstract.tex -->
RoPE is usually controlled by a scalar base, but a finite table must also
decide how $K$ rotary pairs are allocated inside the sampled support. Writing
$x_k=-\log\omega_k=a+Rz_k$ separates support $(a,R)$ from normalised interior
allocation $z$. We show that $z$ is a separately identifiable training-time
variable. Across three paired $151.9$M runs, changing only the $30$ interior
frequencies consistently shifts modelling quality from the training window
toward longer contexts. Full sin/cos subspace geometry gives an exact,
phase-invariant accounting of finite-table redundancy, while an exact
transplant obstruction and a weight--table crossing show why the table must be
learned with the weights. \evq{} is one closed-form, zero-learned-parameter
intervention on this axis. Under complete-table protocols, the same
effective-context crossover recurs in three-seed $432$M scarce-channel MLA,
$750$M full-parameter continuation, and a $1.485$B
same-initialisation/same-scientific-recipe comparison, where Geo/\evq{} $16$K
PPL is $182.73/159.64$. Separately, matched $8$B adaptation supports
mature-model transfer and causal use of remote evidence; $1.485$B Q/K-only
adaptation carries real-document multi-hop QA beyond its physical training
window. Interior allocation is therefore a practical part of long-context
training, distinct from but interacting with support selection and
inference-time range transport.
\end{abstract}

<!-- FILE: sections/01_intro.tex -->
\section{Introduction}
\label{sec:intro}

RoPE has a spectral budget. Each attention head receives only
$K=\drot/2$ rotary pairs before training, but the familiar scalar base specifies
only their support. For an ordered table define
\begin{equation}
x_k=-\log\omega_k=a+Rz_k,
\qquad z_0=0,\quad z_{K-1}=1,
\label{eq:table-decomposition}
\end{equation}
where $a=x_0$, $R=x_{K-1}-x_0$, and $z$ is the normalised allocation inside the
sampled log-frequency span. Standard geometric RoPE fixes
$z_k=k/(K-1)$.

\paragraph{Our view: support says which frequencies are available; allocation
says how a finite head spends them.}
The scalar base sets the span, while $z$ determines how the finite channels use
it. This distinction matters only at finite $K$, exactly where every deployed
attention head operates.

This decomposition separates decisions that are often conflated. Position
interpolation, YaRN, and LongRoPE transport a trained spectrum toward a target
range \citep{chen2024position,peng2024yarn,ding2024longrope,shang2025longrope2};
FMRoPE changes scalar support while retaining a geometric exponent grid
\citep{oka2026fmrope}; LeRoPE and AdaRoPE learn broader frequency tables
\citep{karypis2026lerope,wang2026adarope}. These interventions can be composed
and need not act on only one coordinate. Our question is narrower and exactly
controlled: \emph{with $a$ and $R$ fixed, does changing only the $K-2$
interior entries of $z$ change what a model learns?}

It does. Across three paired $151.9$M training seeds, moving only those $30$
frequencies changes mean OOD NLL by $-0.281/-0.176/-0.146$ at
$2\times/4\times/8\times$, with every seed favouring Cosh at every OOD length,
at a $+0.026$ in-window cost (\S\ref{sec:exp-identify}). The anchored
intervention is the affine support normalisation of the deployed midpoint Cosh
quantiles and therefore has exactly the same $z$. A separate $50.9$M,
$12$-configuration, three-seed factorial finds the non-uniform direction for
both $1.25\times$ Cosh ($10/12$) and a deformation-matched exponential
($9/12$). Two distinct fixed analytic shapes therefore reproduce the
non-uniform direction and identify the allocation axis across analytic curves.

The same in-window/long-range crossover then appears under complementary
complete-table protocols (Fig.~\ref{fig:evidence-overview}). A three-seed
$432$M MLA stress test uses only $K{=}16$ rotary frequencies; a $750$M
full-parameter continuation starts both arms from the same $2$K Geo checkpoint;
and a $1.485$B comparison uses the same initialisation, scientific recipe,
data-order prefix, counted-token budget, and evaluation rows. The $1.485$B
Geo/\evq{} PPL is $161.19/167.45$ at the $4$K training cap but
$182.73/159.64$ at $16$K. Separately, matched $1.485$B Q/K-only adaptation
transfers real-document multi-hop QA beyond its physical window, while matched
$8$B LoRA provides mature-model probability and causal remote-source-use
evidence. The training protocols establish architecture and training-stage
persistence through $1.485$B; the separate adaptations establish capability at
$1.485$B and $8$B. The fixed-support study remains the causal owner of $z$.

We next ask what that axis means for a trained model
(\S\ref{sec:theory}). Each rotary frequency contributes a two-dimensional
sin/cos subspace. Canonical correlations give phase-invariant redundancy, and
block whitening yields the exact finite-basis identity
\[
r_2(R)=\frac{2K}{1+(K-1)\bar c}.
\]
In a standard $4$K table, $23$ slow pairs occupy $46$ nominal dimensions but
have Rényi-$2$ effective rank $2.00$. This is a static redundancy statement,
whose trained consequences must be measured. A weight--table crossing shows
that self-consistent systems remain usable while post-hoc swaps fail. We prove
the exact counterpart: unequal frequency multisets cannot be absorbed by fixed,
position-independent invertible Q/K maps. A RoPE table is therefore a
training-time coordinate system whose coefficients are learned with it.

Finally, a convex modelling surrogate yields a closed-form Cosh density and
inverse-CDF table with geometric RoPE as the $\tau\to0$ limit. \evq{} is thus
a reproducible, zero-learned-parameter point on the allocation axis.

\paragraph{Contributions.}
(i) We separate sampled support from interior allocation and identify the
latter with a three-seed fixed-support causal control.
(ii) We give a phase-invariant finite-basis geometry, an exact effective-rank
identity, and an exact frozen-transplant obstruction, paired with direct
co-adaptation evidence.
(iii) We derive a closed-form analytic allocation and show that the resulting
effective-context shift persists through $1.485$B from-initialisation training,
with separate matched $1.485$B and $8$B capability evidence.

<!-- FILE: sections/02_related.tex -->
\section{Related Work}
\label{sec:related}

Write $x_k=-\log\omega_k=a+Rz_k$ as in
Eq.~\eqref{eq:table-decomposition}. Prior work changes several, nonexclusive
parts of this parameterisation: it transports realised phases, changes support
$(a,R)$, or changes the normalised interior allocation $z$.
This decomposition separates when a method acts from which spectral degrees of
freedom it controls; the interventions are nonexclusive and composable.

\paragraph{Context-range extension.}
Relative position information can be injected additively, through learned or
biased attention terms \citep{shaw2018self,press2022alibi,chi2022kerple}, or
multiplicatively through rotation \citep{su2024roformer,sun2022xpos}; we take
the rotary operator as given and ask only where its frequencies are placed.
Position interpolation rescales positions to keep rotary phases within the
training range \citep{chen2024position}. YaRN, SelfExtend, PoSE, and CLEX refine
this extension through frequency-dependent scaling, remapped positions, or
extension-time training \citep{peng2024yarn,jin2024selfextend,zhu2024pose,chen2024clex}.
LongRoPE and LongRoPE2 search channel-wise rescaling factors for pretrained
models \citep{ding2024longrope,shang2025longrope2}. These methods may combine
several intervention levels; their common goal is to transport a learned model
to a declared longer context. Bounds based on the RoPE base characterize limits
of this regime \citep{xu2024base}.

\paragraph{Frequency use, scalar base, and operator geometry.}
Trained RoPE exhibits frequency-localized query/key norms; partial RoPE removes
the slowest rotations \citep{barbero2025round}. \mbox{Frequency Entropy} measures
rotational-pair utilization and identifies redundant extreme-entropy dimensions
\citep{oka2026frequencyentropy}; controlled access experiments causally link
available frequencies to positional versus symbolic behaviour
\citep{urrutia2026decoupling}. FMRoPE instead shifts the geometric spectrum by
setting the scalar base from training or evaluation length while retaining a
uniform exponent grid \citep{oka2026fmrope}. At a broader operator level, GRAPE
recovers RoPE as canonical planar group actions and generalizes their generators
and subspaces \citep{zhang2026grape}. Our identification keeps the rotary
operator, sampled endpoints, and log-span fixed, moves only finite interior
frequencies, and pairs trained comparisons with an exact frozen-transplant
obstruction (\S\ref{sec:exp-identify}).

CoPE softly clips slow rotations, while RoPE-ID applies high-frequency RoPE to
a subset of channels for long-input robustness
\citep{li2026copeclipped,wertheimer2026frayed}. In multimodal models,
MHRoPE and MRoPE-I treat frequency coverage across positional axes as a design
principle \citep{qwen2026mhrope}. These methods alter frequency behaviour,
channel participation, or axis assignment; our fixed-support control instead
moves only the interior entries of a fully rotary one-dimensional table.

\paragraph{Learning and constructing frequency tables.}
LeRoPE learns one log-scale per frequency band, shared across layers and heads,
and reports consistent non-geometric profiles on a 52M--2.5B model ladder
\citep{karypis2026lerope}. In its 217M ablation, retraining with a frequency
table frozen from an independent LeRoPE run retains $63.6\%$ of the full
validation-perplexity gain over RoPE, compared with $10.4\%$ for partial RoPE.
This isolates substantial value in the fixed table, although obtaining that
table still requires a separate learned run. AdaRoPE instead learns per-head
frequencies together with head-specific, length-dependent attention scaling
\citep{wang2026adarope}. Recent analysis further shows that the frequencies used
by a trained model depend on the relative-distance structure of its training
data, with the preferred frequency varying inversely with dependency width
\citep{wu2026datashapes}.

Resonance RoPE analytically snaps each wavelength to a nearby integer period to close
phase gaps during interpolation, producing a fixed non-geometric table that
can be combined with range scaling \citep{wang2024resonance}. \evq{} instead
derives an interior allocation density from a variational objective and tests
that variable with the sampled range held fixed.

\paragraph{EVQ-Cosh.}
\evq{} addresses a different fixed-table objective: it warps the exponent grid before
training using the closed-form solution of a stated variational surrogate. The
result is shared across heads, introduces no learned positional parameters, and
does not require a deployment target. It therefore complements range extension
and differs from LeRoPE and AdaRoPE in how the non-geometric table is obtained,
not in whether frequency allocation matters. Alternatives that learn a broader
positional function, including DAPE, FIRE, and FoPE
\citep{zheng2024dape,li2024fire,hua2025fope}, change a larger functional object
than a fixed RoPE table.

\paragraph{Evaluation.}
Long-context evaluation spans retrieval diagnostics and task suites
\citep{bai2024longbench,hsieh2024ruler}, and information can be lost despite
being present in the prompt \citep{liu2024lost,zhang2024found}. We use
PE-dominant diagnostics for mechanism and RULER for effective-context claims,
labelling the reported RULER endpoint as task-family-adapted length transfer.

<!-- FILE: sections/03_theory.tex -->
\section{Theory}
\label{sec:theory}

\subsection{A finite table is a set of two-dimensional subspaces}
\label{sec:subspaces}

Let $\phi \in [0,1]$ parameterise log-frequency, $\omega(\phi) = b^{-\phi}$, let
$\drot$ be the number of rotated dimensions and $K = \drot/2$ the number of
rotary pairs ($\drot{=}\dhd$ for standard MHA, smaller for compressed-RoPE
architectures such as MLA).

We reserve $z$ for the support-normalised coordinate in
Eq.~\eqref{eq:table-decomposition}. Standard implementations use raw geometric
exponents $u_k=k/K$, while deployed \evq{} uses midpoint quantiles
$u_k=(k+1/2)/K$; endpoint-inclusive grids used for static diagnostics are
labelled explicitly. Normalising the sampled geometric endpoints gives
$z_k=k/(K-1)$.

Equation~\eqref{eq:table-decomposition} is a coordinate identity for any
ordered positive table with nonzero span. In the geometric family
$z_k=k/(K-1)$, so fixing $(a,R)$ fixes every channel. A second table with the
same support and different interior $z$ therefore cannot be represented by a
different scalar base. Quantities such as median wavelength or the number of
channels completing a cycle are summaries of $z$, not omitted scalar controls.

A rotary pair contributes $f_\omega(\Delta) = C\cos(\omega\Delta) +
D\sin(\omega\Delta)$ to the relative-position attention logit, with $C,D$ set by
content. The object attached to $\omega$ is therefore the two-dimensional
subspace
\begin{equation}
V_\omega=\operatorname{span}\{\cos(\omega\Delta),\ \sin(\omega\Delta)\},
\label{eq:subspace}
\end{equation}
not a cosine feature: an analysis that fixes one content phase sees a slice of
$V_\omega$ and is not invariant to rotation inside the pair. Two channels are
redundant to the extent that their subspaces are. With
$x_\omega(\Delta)=[\cos(\omega\Delta)\ \ \sin(\omega\Delta)]$, Gram matrices
$S_\omega=\mathbb E[x_\omega^\top x_\omega]$ and
$H_{\omega\nu}=\mathbb E[x_\omega^\top x_\nu]$ over a separation prior, and the
whitened cross-Gram $Q_{\omega\nu}=S_\omega^{-1/2}H_{\omega\nu}S_\nu^{-1/2}$,
the singular values of $Q_{\omega\nu}$ are the canonical correlations between
$V_\omega$ and $V_\nu$, and
\begin{equation}
c_{\omega\nu}=\tfrac12\lVert Q_{\omega\nu}\rVert_F^2\in[0,1]
\label{eq:collision}
\end{equation}
is invariant to phase rotation inside a pair and to any change of basis of
either subspace. For $\Delta\sim\mathrm{Unif}[0,L]$, $H_{\omega\nu}$ has a closed
form in $\sin t/t$ and $(1{-}\cos t)/t$ at $t=(\omega{\mp}\nu)L$
(App.~\ref{sec:proofs}).

\subsection{The spectral budget is a dimension count}
\label{sec:budget}

Block-whitening every pair gives a global correlation Gram $R$ with diagonal
blocks $I_2$. Write $\bar c$ for the mean of the $c_{ij}$.

\begin{theorem}[Spectral budget identity]
\label{thm:budget}
$\operatorname{tr}R = 2K$ and $\operatorname{tr}(R^2)=2K[1+(K{-}1)\bar c]$, so the
Rényi-$2$ effective rank of the positional basis is exactly
\begin{equation}
r_2(R)=\frac{(\operatorname{tr}R)^2}{\operatorname{tr}(R^2)}
      =\frac{2K}{1+(K-1)\bar c}.
\label{eq:budget-identity}
\end{equation}
\end{theorem}

The denominator exactly characterises this static dimension summary: larger
mean pairwise redundancy lowers $r_2$. Allocation changes the entire Gram
spectrum, while block whitening removes per-channel energy, conditioning, and
the trained Q/K coefficients. We therefore use $r_2$ to account for positional
basis redundancy and the exact-range experiment of \S\ref{sec:exp-identify} to
measure trained behaviour.

The deflation is concentrated at the slow end.

\begin{proposition}[Low-frequency collapse]
\label{prop:collapse}
As $\omega L\to0$, $V_\omega\to\operatorname{span}\{1,\Delta\}$, and for slow
channels $x{=}\omega L$, $y{=}\nu L$,
$2-\lVert Q_{x,y}\rVert_F^2=\tfrac{19}{12600}(x^2{-}y^2)^2+O(\epsilon^6)$: slow
subspaces approach a common two-dimensional limit at fourth order. Under the
softmax metric $F=\operatorname{diag}(p)-pp^\top$, which annihilates the constant
direction, the centred limit is
$\operatorname{span}\{\Delta-\mathbb E_p\Delta,\ \Delta^2-\mathbb E_p\Delta^2\}$.
\end{proposition}

Concretely, at $L{=}4096$, $b{=}5{\times}10^5$, $K{=}64$ the $23$ pairs with
$\omega L\le1$ occupy $46$ nominal dimensions but have block-whitened
$r_2 = 2.00$: about $96\%$ of that sub-budget is redundant. Slow channels are
\emph{redundant}, not unused --- they still carry content, and
Prop.~\ref{prop:collapse} says only that many of them describe nearly the same
positional direction.
This common long-context failure mode motivates allocating a finite table; the
full subspace geometry, rather than the asymptotic threshold alone, covers the
$b{=}256$ exact-range control.

\subsection{The table is a training-time coordinate system}
\label{sec:coordinate}

Theorem~\ref{thm:budget} constrains what a table can express. It does not
predict what a trained model does with one, and the gap is not small. In a
frozen $50$M diagnostic ($L{=}512$, $b{=}5{\times}10^5$,
$\tau_{\mathrm{EVQ}}{=}2.83$; no training, no gradient) we cross two trained
weight sets with two runtime tables
(Table~\ref{tab:coadapt}). Substituting \evq{} for the geometric table at
inference \emph{raises} the block-whitened $r_2$ from $4.57$ to $12.54$
while perplexity degrades from $7.14$ to $76.20$.

<!-- FILE: tables/table_coadapt.tex -->
\begin{table}[tb]
\centering
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{0.95}
\caption{\textbf{Static geometry does not predict trained behaviour.} Two
seed-$42$ $50$M models trained with the geometric and \evq{} tables, each
evaluated under both runtime tables with all parameters frozen ($1{,}920$
head--query observations). The worst cell is also the one whose static
$r_2$ improves most.}
\label{tab:coadapt}
\begin{tabular}{@{}llcc@{}}
\toprule
Trained weights & Runtime table & PPL & Rényi-$2$ rank $r_2$ \\
\midrule
Geo   & Geo   & $\mathbf{7.14}$ & $4.57$ \\
Geo   & \evq{} & $76.20$ & $\mathbf{12.54}$ \\
\evq{} & Geo   & $23.05$ & $4.57$ \\
\evq{} & \evq{} & $\mathbf{7.16}$ & $12.54$ \\
\bottomrule
\end{tabular}
\end{table}

Two consequences discipline the rest of the paper. First, static criteria such
as collision, effective rank, and log-determinant do not by themselves predict
trained quality, so every task claim in \S\ref{sec:experiments} is made on
trained models. Second, the two \emph{self-consistent} systems are essentially
tied in-window ($7.14$ vs.\ $7.16$) while both mismatched cells collapse: on LM
loss the table and weight main effects are $+0.599$ and $-0.597$, and their
interaction contrast is $-3.537$ (configuration-level bootstrap
CI $[-5.17,-3.04]$). The table behaves as a coordinate system fixed before
training, with the weights learning coefficients in it.

\begin{theorem}[Post-hoc transplant obstruction]
\label{thm:obstruction}
Let $R_\Omega(\Delta)$ be the block-rotation operator of a frequency multiset
$\Omega$. If position-independent invertible maps $A,B$ satisfy
$A^\top R_{\Omega'}(\Delta)B = R_\Omega(\Delta)$ for all $\Delta$ in an open
interval containing zero, then $\Omega'$ and $\Omega$ have the same frequency
multiset, up to sign and permutation. Repeated frequencies may mix within their
full equal-frequency invariant subspace.
\end{theorem}

\subsection{A closed-form allocation}
\label{sec:construction}

A positive density $\rho\in C^2([0,1])$ with $\int_0^1\rho=1$ induces a discrete
allocation by inverse-CDF quantisation, $\phi_k=F_\rho^{-1}(u_k)$; geometric RoPE
is $\rho\equiv1$. The exact canonical-correlation objective is oscillatory. To
obtain a closed-form construction, we use the convex modelling surrogate
\begin{equation}
\Capp[\rho]=\frac{\alpha}{2}\int_0^1\!\rho^2\,d\phi
 +\frac{\beta}{2}\iint_{[0,1]^2}\!\!\rho(\phi)\rho(\psi)\min(\phi,\psi)\,d\phi\,d\psi,
\label{eq:Capp}
\end{equation}
with $\alpha,\beta>0$. The diagonal term penalises channel-load concentration;
$\min(\phi,\psi)$ is the Green kernel of $-\partial_\phi^2$ and encodes a
penalty for pairs of channels that are both slow, matching the direction of
Prop.~\ref{prop:collapse}. The full geometry and the surrogate play distinct
roles: the former measures redundancy, while the latter supplies an analytic
allocation family.

\begin{theorem}[Stationary allocation under the surrogate]
\label{thm:ode}
Under \eqref{eq:Capp} with $\alpha{>}0,\beta{\geq}0$ and constraints
$\rho\in C^2([0,1])$, $\rho{>}0$, $\int_0^1\rho{=}1$, the unique constrained
minimiser is
\begin{equation}
\rho_\tau(\phi)=\frac{\tau\cosh(\tau(1{-}\phi))}{\sinh\tau},
\qquad \tau=\sqrt{\beta/\alpha};
\label{eq:rho-tau}
\end{equation}
at $\beta{=}0$ it is the $\tau{\to}0$ limit $\rho_0\equiv1$.
\end{theorem}

\begin{corollary}[Closed-form quantiles]
\label{cor:quantiles}
$F_\tau(\phi)=1-\sinh(\tau(1{-}\phi))/\sinh\tau$, so
\begin{equation}
\phi_k(\tau)=1-\tfrac{1}{\tau}\operatorname{arcsinh}\!\left((1-u_k)\sinh\tau\right),
\qquad u_k=\tfrac{k+1/2}{K},
\label{eq:warp}
\end{equation}
which tends to $u_k$ as $\tau\to0$, recovering the uniform-log midpoint grid.
\end{corollary}

\paragraph{The recipe, and the status of $\tau$.} With $\deff$ the effective head
dimension, equal to $\dhd$ for the full-rotary architectures studied here, and
$\Pi$ denoting the training protocol,
\begin{equation}
\label{eq:evq-practical}
\tau = c(\Pi)\deff/\sqrt{\Ltr},\qquad
u_k = (k+\tfrac12)/K,\qquad
\omega_k^{\mathrm{EVQ}} = b^{-\phi_k(\tau)} .
\end{equation}

<!-- FILE: sections/04_experiments.tex -->
\section{Experiments}
\label{sec:experiments}

NLL/PPL, teacher-forced retrieval, causal source use, strict autoregressive
exact match, real-document QA, and RULER macro remain separate endpoints.
Every trained system is evaluated with its own table; protocol and seed scope
are reported with each result and in the appendices.

\subsection{Identifying allocation at fixed support}
\label{sec:exp-identify}

\paragraph{Protocol.}
Within each of three $151.9$M training seeds, the paired arms share
architecture, trainable initialisation, token order, optimiser, LR schedule,
global batch, a
$499{,}974{,}144$-token budget, and $32$ frozen evaluation anchors. The
baseline is our paper-faithful implementation of the uniform-in-log rule in
\citet[\S6.1]{oka2026fmrope}, using training base $256$; the intervention is
endpoint-normalised Cosh spacing at $\tau{=}4$. Their sampled extrema and
log-frequency span are identical, so only the $K{-}2=30$ interior frequencies
move. The endpoint-normalised and deployed midpoint Cosh grids have exactly the
same normalised allocation $z$ in Eq.~\eqref{eq:table-decomposition}.

\paragraph{Result.}
Cosh-minus-uniform NLL is $+0.026$ at the $256$-token training length and
$-0.281/-0.176/-0.146$ at $512$/$1$K/$2$K (Fig.~\ref{fig:evidence-overview}a).
All three training seeds favour Cosh at every OOD length. Since every sampled
scalar support quantity is pinned, no geometric base can reproduce the
intervention: \emph{fixed support $+$ different $z$ $\Rightarrow$ different
trained-model behaviour.}

\paragraph{Cross-configuration and shape breadth.}
A separate pre-specified $50.9$M factorial spans two bases, two training
lengths, three head dimensions, and three paired seeds. Over its $12$
structural configurations, $1.25\times$ Cosh and a deformation-matched
exponential favour non-uniform allocation in $10/12$ and $9/12$
configurations (App.~Table~\ref{tab:m4}). The factorial therefore reproduces
the non-uniform direction across configurations
and two fixed analytic shapes; the $151.9$M study supplies the replicated
direct effect size. The separately retargeted deployment condition is reported
in App.~\ref{sec:identification-details}.

\subsection{A finite budget survives architecture and scale}
\label{sec:exp-mature}

\paragraph{Scarce-channel MLA.}
A $432$M MLA transformer trains from scratch over three seeds with only
$d_{\mathrm{rope}}{=}32$, hence $K{=}16$ rotary frequencies. Geo/\evq{} PPL
is $35.4/35.8$ at the $8$K training length ($+1.1\%$) but $138.8/95.6$ at
$16$K ($-31.1\%$), with all three seeds favouring \evq{} at $16$K
(App.~Table~\ref{tab:mla}). This is the direct
scarce-budget stress test predicted by the finite-$K$ view.

\paragraph{Full-parameter continuation.}
A $750$M single-seed experiment starts both arms from the same $2$K geometric
checkpoint and continues all parameters for $500$M tokens at $4$K with either
the Geo or \evq{} table. PPL changes from $22.0/22.3$ at $4$K to $45.1/24.4$
at $16$K, while strict autoregressive exact retrieval at $8$K changes from
$0\%$ to $77.5\%$ (App.~Table~\ref{tab:750m}). This is full-parameter
continued pretraining from a shared checkpoint.

\paragraph{$1.485$B from initialisation.}
Under the same initialisation, architecture, scientific recipe, reconstructed
data-order prefix, $2.097$B-token budget, and evaluation rows, Geo/\evq{} PPL
is $177.99/191.36$, $161.19/167.45$, $163.88/156.87$, and $182.73/159.64$ at
$2$/$4$/$8$/$16$K (Fig.~\ref{fig:evidence-overview}b). Thus the comparison
crosses beyond the $4$K training cap, with $122/128$ and $126/128$ documents
favouring \evq{} at $8$K/$16$K.

\paragraph{Range composition.}
In a separate $454$M three-seed MHA study, the same fixed-scale \rs{} transform
raises $8$K teacher-forced retrieval from $41\%$ to $61\%$ on Geo but from
$53\%$ to $100\%$ on \evq{}; transformed $8$K PPL is $82.9$ versus $70.9$
(App.~Table~\ref{tab:evq-ramp}). A shared range operation therefore does not
erase the table learned during training.

\subsection{From effective context to remote-content use}
\label{sec:exp-capability}

\paragraph{$1.485$B real-document QA and task adaptation.}
Starting from matched OLMo-2 Stage-A parents, both arms freeze inherited V/O
LoRA and continue only Q/K for $300$ steps with the same data, order,
optimiser, trainable count, and $4$K/$8$K/$16$K phase exposure; physical
sequences remain at most $4$K. On held-out 2WikiMultiHopQA,
Native/\evq{} exact match is $22.0/21.5\%$ at $4$K, $0/17.5\%$ at $8$K, and
$0/4.0\%$ at $16$K (Fig.~\ref{fig:evidence-overview}c)
\citep{bai2024longbench}. An independent continuation over all $13$ RULER
families gives official macro $72.19/42.44\%$, $2.02/31.63\%$, and
$0.38/5.03\%$ at the same lengths \citep{hsieh2024ruler}. The first result is
real-document QA; the second is task-family-adapted length transfer.

\paragraph{Separate matched $8$B adaptation.}
On LLaMA-3-8B-Instruct \citep{grattafiori2024llama3}, matched rank-$64$
Q/K/V/O LoRA gives Native/\evq{} PPL $6.82/10.07$, $108.96/24.07$, and
$991.48/127.91$ at $8$/$16$/$32$K. The long-range direction holds on all $24$
frozen natural-text packs and all three source domains. At true $16$K, median
target-block hit@16 rises from $18.75\%$ to $64.06\%$, and deleting the remote
gold block changes NLL by $-0.0095/+1.5055$. These are mature-model probability
and causal source-use results. A separate
$516$-step task-family continuation gives Native/\evq{} RULER macro
$94.44/77.60\%$ at $8$K and $0.295/14.03\%$ at $16$K, measuring
task-family-adapted length transfer.

Finally, a two-seed $129.6$M video DiT with bidirectional attention and 3D RoPE
favours \evq{} on training, all-frame extrapolation, and far-frame MSE in both
runs (App.~\ref{sec:video-dit}), extending the scope check to bidirectional 3D
RoPE.

\subsection{What is schedule-specific}
\label{sec:exp-tau}

The finite-$\tau$ rule is a zero-search operating-basin selector. An independent
sweep selects $\tau{=}5$ against the rule value $5.657$, only $0.0119$ NLL
apart; across the factorial and boundary arms, the best tested multipliers stay
within $0.75\times$--$1.5\times$ of the rule. A deformation-matched exponential
also improves in $9/12$ configurations, so the fixed zero-parameter comparisons
attribute the direction to allocation rather than learned capacity. Cosh
contributes a closed-form solution of the stated surrogate. Full multiplier,
boundary, and matched-shape results are in App.~\ref{sec:identification-details}.

<!-- FILE: sections/05_discussion.tex -->
\section{Discussion}
\label{sec:discussion}

A finite RoPE table exposes support $(a,R)$ and normalised allocation $z$.
Scalar ``effective base'', median wavelength, and cycle counts are useful
summaries of $z$, but they do not reproduce a non-geometric table while its
endpoints remain fixed. The exact-range controls identify this remaining degree
of freedom directly. Its full sin/cos Gram determines a static positional basis,
for which Theorem~\ref{thm:budget} gives an exact Rényi-$2$ effective-rank
accounting; its trained value depends on how model weights use that basis. A
table with higher static rank can therefore still be catastrophically wrong for
weights trained in another coordinate system.

This distinction explains why allocation must be installed during training or
adaptation. The $50$M table-by-weights crossing shows strong co-adaptation,
while the mature-model comparisons show persistence after training or
adaptation; Theorem~\ref{thm:obstruction} rules
out exact preservation by fixed invertible Q/K maps when frequency multisets
differ. Its exact scope explains why post-hoc replacement and training under a
table are different interventions, while the adaptation results quantify how
much low-rank retraining can recover. The modest from-initialisation trade-off
and strong frozen table-by-weights interaction are therefore compatible.

Allocation also differs from range transport. The former chooses which finite
channels a model trains against; the latter changes how a realised spectrum is
deployed at a target length. The same \rs{} transform has substantially
different leverage on Geo and \evq{} substrates, so a shared range operation
does not erase the training table. The target-retargeted diagnostic in
App.~\ref{sec:identification-details} further shows that support and allocation
interact under deployment. Holding one fixed identifies the other; together
they form separate coordinates of the empirical design. LeRoPE
supplies the complementary learned
result: a table obtained in one run retains substantial value when frozen for a
new training run \citep{karypis2026lerope}. Our contribution is the explicit
support/allocation decomposition, phase-invariant finite-basis geometry,
fixed-support causal identification, exact frozen-retrofit obstruction, and a
closed-form point on the allocation axis.

Static geometry characterises positional-basis redundancy; trained comparisons
establish extrapolation and task behaviour. \evq{} is useful because it is
analytic, zero-learned-parameter, and reproducible, while the factorial extends
the direction to another fixed analytic allocation. The broader result is simpler:
at finite $K$, interior frequency placement is part of the training design.
Practically, support, allocation, and deployment-time range transport should be
chosen jointly but evaluated separately, with non-geometric allocation
installed during training or adaptation.

<!-- FILE: sections/06_ethics.tex -->
\subsection*{Ethics statement}
This work introduces no new dataset, deployed system, or human-subject
component. It changes only the initialisation of the RoPE inverse-frequency
table and adds no learned parameters, so it is inexpensive to audit, ablate,
or revert. All models, corpora, and evaluation suites used here are public
research artefacts used within their stated licences.

<!-- FILE: sections/07_reproducibility.tex -->
\subsection*{Reproducibility statement}
The \evq{} table is fully specified by Eq.~\eqref{eq:warp}: the schedule is a
pure function of $(\tau,K)$ that replaces the inverse-frequency buffer, with no
learned parameters or auxiliary loss. Equation~\eqref{eq:evq-practical} is the
zero-search full-RoPE text default; every protocol reports its realised
$\tau$ explicitly, including the empirical MLA and video settings and the
mature-model tables in App.~\ref{sec:mature-details}. Complete proofs of Theorems~\ref{thm:budget}, \ref{thm:obstruction} and
\ref{thm:ode}, Proposition~\ref{prop:collapse},
Corollary~\ref{cor:quantiles} and Lemma~\ref{lem:budget-crossing}, together
with every stated assumption, are in App.~\ref{sec:proofs}.

<!-- FILE: sections/08_ai_use.tex -->
\subsection*{AI use statement}
We used generative AI tools to help develop conceptual frameworks and
mathematical claims; propose and refine hypotheses; draft and check proofs; give
feedback on methodology and experiments; implement methods and deterministic
data-reformatting and analysis code; translate or rephrase research notes; and
interpret results.

<!-- FILE: refs/references.bib -->
@inproceedings{vaswani2017attention, title = {Attention Is All You Need}, author = {Vaswani, Ashish and others}, booktitle = {NeurIPS}, year = {2017} }
@article{su2024roformer, title = {{RoFormer}: Enhanced Transformer with Rotary Position Embedding}, author = {Su, Jianlin and others}, journal = {Neurocomputing}, year = {2024} }
@inproceedings{peng2024yarn, title = {{YaRN}: Efficient Context Window Extension of Large Language Models}, author = {Peng, Bowen and others}, booktitle = {ICLR}, year = {2024} }
@inproceedings{ding2024longrope, title = {{LongRoPE}: Extending {LLM} Context Window Beyond 2 Million Tokens}, author = {Ding, Yiran and others}, booktitle = {ICML}, year = {2024} }
@inproceedings{zheng2024dape, title = {{DAPE}: Data-Adaptive Positional Encoding for Length Extrapolation}, author = {Zheng, Chuanyang and others}, booktitle = {NeurIPS}, year = {2024} }
@inproceedings{hsieh2024ruler, title = {{RULER}: What's the Real Context Size of Your Long-Context Language Models?}, author = {Hsieh, Cheng-Ping and others}, journal = {arXiv}, year = {2024} }
@inproceedings{oka2026fmrope, title = {Frequency Bands in {RoPE}: Base Frequency and Context Length Shape the Interpolation--Extrapolation Trade-off}, author = {Oka, Yui and others}, booktitle = {ICLR}, year = {2026} }
@inproceedings{oka2026frequencyentropy, title = {Probing Rotary Position Embeddings through Frequency Entropy}, author = {Oka, Yui and others}, booktitle = {ICLR}, year = {2026} }
@inproceedings{urrutia2026decoupling, title = {Decoupling Positional and Symbolic Attention in Transformers}, author = {Urrutia, Felipe and others}, booktitle = {ICLR}, year = {2026} }
@inproceedings{zhang2026grape, title = {Group Representational Position Encoding}, author = {Zhang, Yifan and others}, booktitle = {ICLR}, year = {2026} }
@article{karypis2026lerope, title = {{LeRoPE}: Learnable {RoPE} Frequencies Improve Language Modeling}, author = {Karypis, Petros and others}, journal = {arXiv}, year = {2026} }
@inproceedings{wang2026adarope, title = {{AdaRoPE}: Not All Attention Heads Should Rotate and Scale Equally}, author = {Wang, Shaowen and others}, booktitle = {ICML}, year = {2026} }
@article{wu2026datashapes, title = {How Data Shapes {RoPE} Frequency Usage: From Positional Scale Matching to Length Generalization}, author = {Wu, Xinyi and others}, journal = {arXiv}, year = {2026} }

<!-- FILE: appendix/a1_proofs.tex -->
\section{Proof Details}
\label{sec:proofs}

\subsection{Subspace geometry: cross-Gram, budget identity, and collapse}
\label{sec:geometry-proofs}

\paragraph{Closed-form cross-Gram.} For $\Delta\sim\mathrm{Unif}[0,L]$ and
$x_\omega(\Delta)=[\cos(\omega\Delta)\ \ \sin(\omega\Delta)]$, write
$d=(\omega{-}\nu)L$, $s=(\omega{+}\nu)L$, $a(t)=\sin t/t$ and
$b_\star(t)=(1{-}\cos t)/t$. Expanding the products of sinusoids and integrating
termwise gives
\begin{equation}
H_{\omega\nu}=\mathbb E[x_\omega^\top x_\nu]=\frac12
\begin{bmatrix}
a(d)+a(s) & b_\star(s)-b_\star(d)\\
b_\star(s)+b_\star(d) & a(d)-a(s)
\end{bmatrix},
\label{eq:cross-gram}
\end{equation}
and $S_\omega=H_{\omega\omega}$. Since $Q_{\omega\nu}=S_\omega^{-1/2}H_{\omega\nu}S_\nu^{-1/2}$
is the whitened cross-Gram of two two-dimensional subspaces, its singular values
are the canonical correlations, and $c_{\omega\nu}=\frac12\|Q_{\omega\nu}\|_F^2$
is invariant under $x_\omega\mapsto x_\omega O$ for any invertible $O$ (in
particular under rotation of the phase inside a pair), because $S_\omega$
transforms congruently and the whitening cancels $O$.

\begin{proof}[Proof of Theorem~\ref{thm:budget}]
$R$ is the block matrix with blocks $R_{ij}=Q_{\omega_i\omega_j}$ and
$R_{ii}=I_2$. Hence $\operatorname{tr}R=\sum_{i=1}^K\operatorname{tr}I_2=2K$.
For the second moment,
$\operatorname{tr}(R^2)=\sum_{i,j}\operatorname{tr}(R_{ij}R_{ji})
=\sum_i\operatorname{tr}(I_2)+\sum_{i\neq j}\|Q_{\omega_i\omega_j}\|_F^2
=2K+2K(K-1)\bar c$, using $R_{ji}=R_{ij}^\top$ and the definition
$\bar c=\frac{1}{K(K-1)}\sum_{i\neq j}c_{\omega_i\omega_j}$ with
$\|Q_{ij}\|_F^2=2c_{ij}$. Dividing gives \eqref{eq:budget-identity}.
\end{proof}

\begin{proof}[Proof of Proposition~\ref{prop:collapse}]
With $t=\Delta/L$ and $x=\omega L$, $\cos(xt)=1-x^2t^2/2+O(x^4)$ and
$\sin(xt)/x=t-x^2t^3/6+O(x^4)$, so after rescaling the second coordinate the
basis of $V_\omega$ tends to $\{1,\Delta\}$. Substituting these expansions into
\eqref{eq:cross-gram} and whitening, the leading deficit between two slow
subspaces is $2-\|Q_{x,y}\|_F^2=\frac{19}{12600}(x^2-y^2)^2+O(\epsilon^6)$;
numerically at $x{=}0.05,y{=}0.10$ the exact-to-leading ratio is $1.00058$. For
the softmax metric, $F=\operatorname{diag}(p)-pp^\top$ satisfies $F\mathbf1=0$,
so constants are annihilated; centring by $\mathbb E_p$ and rescaling gives
$\overline{\sin(\omega\Delta)}/\omega\to\Delta-\mathbb E_p\Delta$ and
$-2\overline{\cos(\omega\Delta)}/\omega^2\to\Delta^2-\mathbb E_p\Delta^2$,
which span the centred limit whenever $p$ has nondegenerate support on at least
three distances.
\end{proof}

\subsection{Post-hoc transplant obstruction}
\label{sec:obstruction-proof}

\begin{proof}[Proof of Theorem~\ref{thm:obstruction}]
Both $R_\Omega$ and $R_{\Omega'}$ are block-diagonal with $2\times2$ rotation
blocks, hence $R_\Omega(0)=R_{\Omega'}(0)=I$. Evaluating the hypothesis at
$\Delta{=}0$ gives $A^\top B=I$, so $B=A^{-\top}$ and the hypothesis becomes the
similarity $A^\top R_{\Omega'}(\Delta)A^{-\top}=R_\Omega(\Delta)$ on an interval.
Each side is the exponential of a constant generator, $R_\Omega(\Delta)=\exp(\Delta G_\Omega)$
with $G_\Omega=\bigoplus_k \omega_k J$, $J=\left[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right]$.
Differentiating at $\Delta{=}0$ gives $A^\top G_{\Omega'}A^{-\top}=G_\Omega$, so
the generators are similar and share a spectrum. The spectrum of $G_\Omega$ is
$\{\pm i\omega_k\}_{k\le K}$ with multiplicity, so the frequency multisets agree
up to sign and permutation. When a frequency is repeated, similarity may mix
the entire equal-frequency invariant subspace; it need not preserve the original
two-dimensional block decomposition. For integer positions the same argument
is replaced by a one-step spectral argument: evaluating at $\Delta=0$ again
gives $B=A^{-\top}$, while $\Delta=1$ makes $R_{\Omega'}(1)$ and
$R_\Omega(1)$ similar. Their eigenvalue multisets
$\{e^{\pm i\omega_k}\}$ therefore agree, which identifies frequencies up to
sign, permutation, and the $2\pi$ alias.
\end{proof}

\subsection{Collision-only surrogate stationary density}
\begin{proof}[Proof of Theorem~\ref{thm:ode}]
Minimizing \eqref{eq:Capp} under $\int_0^1 \rho = 1$ leads to Euler-Lagrange ODE $\rho''(\phi) - \tau^2 \rho(\phi) = 0$ with boundary conditions $\rho'(0) = -\tau^2$ and $\rho'(1) = 0$. Solving gives $\rho_\tau(\phi) = \frac{\tau \cosh(\tau(1-\phi))}{\sinh\tau}$.
\end{proof}

\begin{proof}[Proof of Corollary~\ref{cor:quantiles}]
Integrating $\rho_\tau$ gives $F_\tau(\phi) = 1 - \frac{\sinh(\tau(1-\phi))}{\sinh\tau}$. Inverting $F_\tau(\phi_k) = u_k$ yields $\phi_k(\tau) = 1 - \frac{1}{\tau}\operatorname{arcsinh}((1-u_k)\sinh\tau)$.
\end{proof}

<!-- FILE: appendix/a2_experiment_details.tex -->
\section{Experimental Details}
\label{sec:experiment-details}

<!-- FILE: tables/table6_750m_continue_supporting.tex -->
\begin{table}[tb]
\caption{Single-seed $750$M continued-pretraining check ($2$K$\rightarrow4$K,
$500$M continuation tokens).}
\label{tab:750m}
\centering
\small
\begin{tabular}{@{}lcc@{}}
\toprule
Metric & Geo & \evq{} \\
\midrule
PPL@2K & $25.9$ & $26.2$ \\
PPL@4K & $22.0$ & $22.3$ \\
PPL@8K & $23.4$ & $\mathbf{19.6}$ \\
PPL@16K & $45.1$ & $\mathbf{24.4}$ \\
\midrule
Strict AR exact @8K & $0\%$ & $\mathbf{77.5\%}$ \\
Global strict AR exact & $66.67\%$ & $\mathbf{92.5\%}$ \\
\bottomrule
\end{tabular}
\end{table}

<!-- FILE: appendix/a5_identification.tex -->
\section{Identification protocol and provenance}
\label{sec:identification-details}

<!-- FILE: tables/table_m4.tex -->
\begin{table}[tb]
\caption{\textbf{Exact-range factorial} ($50.9$M; $12$ configurations, $3$ seeds).}
\label{tab:m4}
\begin{tabular}{@{}lcccc@{}}
\toprule
Contrast & $\Delta$NLL & 95\% CI & $\Delta{<}0$ configs & $p$ \\
\midrule
Cosh $0.75\times$ $-$ Geo & $-0.00912$ & $[-0.018,-0.001]$ & $8/12$ & $0.082$ \\
Cosh rule $-$ Geo & $-0.00988$ & $[-0.021,0.002]$ & $7/12$ & $0.125$ \\
Cosh $1.25\times$ $-$ Geo & $\mathbf{-0.01210}$ & $[-0.021,-0.003]$ & $\mathbf{10/12}$ & $0.027$ \\
Matched exponential $-$ Geo & $-0.01062$ & $[-0.021,-0.001]$ & $9/12$ & $0.071$ \\
\midrule
Cosh (rule) $-$ exponential & $+0.00074$ & $[-0.006,0.008]$ & $7/12$ & $0.836$ \\
\bottomrule
\end{tabular}
\end{table}

<!-- FILE: tables/table_pe_dominant.tex -->
\begin{table}[h]
\caption{Learned-frequency comparator ($125$M, $\Ltr{=}128$, FineWeb-Edu, $128\to8$K).}
\label{tab:pe-dominant}
\centering
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Method ($128\!\rightarrow\!8$K) & Extra params & PPL@$128$ & PPL@$8$K & $\Delta$ vs Geo \\
\midrule
Geo & 0 & 184.9 & 513.7 & --- \\
Learnable $\tau$ & 1 & $\mathbf{181.2{\scriptstyle\,\pm 1.3}}$ & $437.9{\scriptstyle\,\pm 12.2}$ & $-14.8\%$ \\
Learned inv-freq (layer-shared) & 32 & 183.6 & 455.3 & $-11.4\%$ \\
\evq{} & 0 & 182.0 & \textbf{333.7} & $\mathbf{-35.0\%}$ \\
\bottomrule
\end{tabular}
\end{table}

<!-- FILE: appendix/a6_mature_scale.tex -->
\section{Mature-scale protocols}
\label{sec:mature-details}

<!-- FILE: tables/table_ruler.tex -->
\begin{table}[tb]
\caption{\textbf{RULER $13$-family scores (\%).}}
\label{tab:ruler}
\begin{tabular}{@{}llccc@{}}
\toprule
Model / protocol & Arm / metric & $1\times$ & $2\times$ & $4\times$ \\
\midrule
OLMo-2 $1.485$B, Q/K-only & Native, official macro & $\mathbf{72.19}$ & $2.02$ & $0.38$ \\
                         & \evq{}, official macro & $42.44$ & $\mathbf{31.63}$ & $\mathbf{5.03}$ \\
LLaMA-3-8B, Q/K/V/O       & Native, official macro & $\mathbf{94.44}$ & $0.295$ & --- \\
                         & \evq{}, official macro & $77.60$ & $\mathbf{14.03}$ & --- \\
\bottomrule
\end{tabular}
\end{table}

<!-- FILE: appendix/a3_supporting_results.tex -->
\section{Supporting Results}
<!-- FILE: tables/table_evq_ramp.tex -->
\begin{table}[tb]
\caption{\textbf{Substrate $\times$ range composition} ($454$M, $\Ltr{=}2048$, $3$ seeds).}
\label{tab:evq-ramp}
\begin{tabular}{@{}lccccc@{}}
\toprule
Method & PK@$8$K & PK@$12$K & PK@$16$K & PPL@$8$K & PPL@$16$K \\
\midrule
Geo & $41\%$ & 57\% & 51\% & 161.9 & 253.2 \\
Geo$+\rs{}$ & $61\%$ & 59\% & 51\% & 82.9 & 157.7 \\
\evq{} & $53\%$ & 63\% & 50\% & 150.3 & 229.5 \\
\textbf{\evq{}$+\rs{}$} & $\mathbf{100\%}$ & \textbf{79\%} & \textbf{68\%} & \textbf{70.9} & \textbf{107.5} \\
\bottomrule
\end{tabular}
\end{table}

<!-- FILE: appendix/a4_supporting_experiments.tex -->
\section{Supporting Experiments}
\end{document}
