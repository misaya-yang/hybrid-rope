<!-- FILE: main.tex -->

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  EVQ-Cosh --- ICLR 2027 submission
%
%  Build:  ./compile.sh          (or: make -f build.mk)
%
%  VENUE
%  -----
%  ICLR 2027.  Abstract deadline 2026-09-18 AoE, full paper 2026-09-25 AoE.
%  Official style files (iclr2027_conference.sty/.bst, fancyhdr.sty,
%  natbib.sty, math_commands.tex) are the unmodified contents of
%  https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip
%
%  HARD FORMAT RULES (enforced by compile.sh)
%    * main text <= 9 pages (10 only at rebuttal / camera ready)
%    * references and appendix do not count toward the limit
%    * AI use statement is REQUIRED and does not count toward the limit
%    * Ethics and Reproducibility statements are recommended, do not count
%    * double blind: no author names, no de-anonymising links
%
%  The previous ICML-formatted template files are preserved unused in
%  venue_icml_fallback/ .
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass{article}
\PassOptionsToPackage{table}{xcolor}
\usepackage{iclr2027_conference,times}

\usepackage[T1]{fontenc}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{placeins}
\usepackage{url}
% plain hyperref draws coloured boxes around every link; use subtle colours instead
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

% float packing: single-column ICLR body is tight, allow denser float pages
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dbltopfraction}{0.95}

\graphicspath{{figs/}}

% ---- shorthands -------------------------------------------------------------
\newcommand{\evq}{\textsc{EVQ-Cosh}}
\newcommand{\dhd}{d_{\mathrm{head}}}
\newcommand{\drot}{d_{\mathrm{rot}}}
\newcommand{\Ltr}{L_{\mathrm{train}}}
\newcommand{\Capp}{\mathcal{C}_{\mathrm{app}}}

\title{Beyond the Base: Exponent Allocation in RoPE}

% Authors must not appear in the submitted version.
\author{Anonymous authors\\Paper under double-blind review}

%\iclrfinalcopy % Uncomment for camera-ready version, but NOT for submission.

\begin{document}

\maketitle

\begin{abstract}
<!-- FILE: sections/00_abstract.tex -->

RoPE assigns its rotary pairs frequencies through a base and an exponent
sequence. We study how the distribution of those exponents changes positional
representation and length generalization. Paired training with identical
frequency endpoints shows that changing only the interior exponents improves
all three tested extrapolation lengths in all three seeds. Retargeting the
range changes the ordering, and frozen weights prefer runtime tables derived
from the allocation they trained with. A full sine--cosine subspace analysis
characterizes how exponent allocation changes the positional basis, including
the shared two-dimensional limit of slow frequencies. We also derive an analytic Cosh family
from a specified convex allocation criterion and evaluate this explicit
construction in language-model training and adaptation. After matched
Llama-$3$-$8$B adaptation at $8$K, $32$K perplexity falls from $991.5$ to $127.9$.
For frozen models, we study exponent displacements relative to the native table.
A boundary-matched adjustment improves OLMo's mean token F1 over MrRoPE-Pro from $21.62\%$ to
$25.44\%$ across five natural-QA tasks. The results connect exponent shape,
sampled range, and learned compatibility as joint considerations in RoPE design.

\end{abstract}

<!-- FILE: sections/01_intro.tex -->

\section{Introduction}
\label{sec:intro}

Rotary position embedding (RoPE) assigns a different frequency to each pair
of query and key coordinates \citep{su2024roformer}. Its usual form,
$\omega_k=b^{-\phi_k}$, combines a base $b$ with equally spaced exponents.
Base-selection and context-scaling methods have established that the range of
positional scales matters \citep{liu2024scaling,xu2024base,oka2026fmrope}.
A finite table also makes a distributional choice: how its rotary pairs occupy
that range. We study this choice through the exponent sequence.

Three questions organize the study. What changes in the positional basis when
exponents are redistributed? Does this redistribution affect learned behavior
when the frequency range is fixed? And how does an already trained model
respond to adjustments of its native exponents? These questions connect
geometric analysis, controlled interventions, and practical table design.

Figure~\ref{fig:spectral-budget-overview} shows the basic intervention.
Two allocations share their endpoints and $32$ rotary pairs; only the $30$
interior exponents differ. Across three paired training seeds, this change
lowers NLL at each tested extrapolation length. A separate range-retargeted
evaluation reverses the ordering. The weights-by-table crossings add another
finding: the preferred runtime table depends strongly on which allocation
trained the weights. Together, these controls make exponent shape observable
while revealing its interaction with range and learned representation.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{fig_evidence_overview.pdf}
  \caption{\textbf{Interior exponents change learned behavior at fixed frequency
  range.} Left: two allocations of the same $32$ rotary pairs with identical
  endpoints. Right: paired $151.9$M models trained at length $256$; changing
  the $30$ interior exponents lowers extrapolation NLL in every seed.
  Thin lines show the three training seeds and the heavy line their mean.}
  \label{fig:spectral-budget-overview}
\end{figure}

The positional geometry provides a way to analyze these distributions.
Each rotary pair supplies an entire sine--cosine subspace. Canonical
correlations compare these subspaces without privileging a content phase,
and expose the common low-frequency limit
$\operatorname{span}\{1,\Delta\}$. Thus many slow pairs can supply few
distinct positional directions over a bounded interval. The learned content
coefficients determine how the model uses those directions.

We then study concrete exponent designs in two settings. A convex
cumulative-tail criterion yields an analytic Cosh distribution, whose
quantiles define a fixed table for training and adaptation. Its experiments
cover scarce rotary channels, continued training, and matched $8$B adaptation.
For frozen models, we instead construct displacements from the native
exponents and compare their placement and intermediate-band shape.
These comparisons include a static Qwen table--amplitude configuration that improves $64$K RULER over
YaRN while retaining the $32$K score, and a boundary-matched adjustment with
improvements on five OLMo natural-QA tasks. The tested Qwen checkpoints favor
a different intermediate profile, making learned compatibility part of the
exponent-design problem.

\paragraph{Contributions.}
We (i) define exponent allocation and analyze its full-subspace positional
geometry; (ii) identify its effects through fixed-range, cross-shape, and
weights-by-table controls, and evaluate a closed-form construction in trained
models; and (iii) develop and test native-relative exponent adjustments in
frozen models, including a boundary-matched intermediate-band rule.

<!-- FILE: sections/02_exponents.tex -->

\section{Exponent Allocation}
\label{sec:exponents}
\label{sec:identification}

Let $K=\drot/2\ge2$ be the number of rotary pairs and $b>1$ the base. Standard RoPE uses
\begin{equation}
\omega_k=b^{-\phi_k},\qquad \phi_k=\frac{k}{K},\qquad k=0,\ldots,K-1.
\label{eq:exponent-rope}
\end{equation}
The base controls the spacing of this geometric frequency family. Allowing
$\phi_k$ to follow a non-linear sequence also changes the density of samples
across positional scales. To describe that shape without conflating it with
an overall shift or stretch, take an ordered table $\omega_0>\cdots>\omega_{K-1}>0$ and write
\begin{equation}
x_k=-\log\omega_k=a+Rz_k,\qquad
z_k=\frac{x_k-x_0}{x_{K-1}-x_0},\quad z_0=0,\quad z_{K-1}=1.
\label{eq:table-decomposition}
\end{equation}
Here $a=x_0$ and $R=x_{K-1}-x_0$ specify the sampled log-frequency range.
Geometric tables have $z_k=k/(K-1)$ at every base. Holding $(a,R)$ fixed while
changing $z$ therefore changes only the normalized exponent allocation.
The standard exponent grid $k/K$ in~\eqref{eq:exponent-rope} and its
endpoint-normalized shape $k/(K-1)$ thus describe the same geometric table.

This representation gives both a construction and an experimental control.
A density over exponents generates a finite table through its quantiles. The
same quantiles can either be installed directly or affinely mapped onto fixed
endpoints to test the effect of their shape. We use the anchored version for
fixed-range comparisons and state the installed grid in each model protocol.
For a mature checkpoint, the corresponding variable is the displacement from
its original exponents; \S\ref{sec:mature-adjustments} develops that form.

The analysis below asks what positions these finitely many exponents can
represent. The experiments then ask how models learn to use the resulting
basis and how existing weights respond to a change in it.

<!-- FILE: sections/03_findings.tex -->

\section{Controlled Effects of Exponent Allocation}
\label{sec:controlled-findings}

We begin with interventions that distinguish exponent shape, sampled range,
and the weights learned under an allocation. The Cosh quantiles used here
are constructed in \S\ref{sec:construction}; endpoint anchoring keeps their
range fixed. An exponential deformation supplies a second nonuniform family.

<!-- FILE: sections/02_identification.tex -->

\subsection{Does exponent shape matter at fixed range?}
\label{sec:exp-identify}

We trained paired $151.9$M models at length $256$ with $K=32$ rotary pairs.
Within each of three seeds, architecture, initialization, token order, optimizer,
schedule, and the $499{,}974{,}144$-token budget are identical. The geometric
arm follows the FMRoPE training-base rule $b=L_{\mathrm{train}}$
\citep{oka2026fmrope}; the Cosh arm anchors its $\tau=4$ quantiles to the same
two endpoints. Only the $30$ interior exponents differ.

At the fixed training range, Cosh lowers tail NLL at $512$, $1$K, and $2$K in
all three seeds (Fig.~\ref{fig:spectral-budget-overview}). The mean
Cosh-minus-geometric differences at $256/512/1024/2048$ are
$+0.026/-0.281/-0.176/-0.146$. Exponent shape therefore changes learned
length generalization even when no scalar base change is available to explain
the result.

A separate evaluation retargets each table's range to the declared test length
while preserving its normalized exponent shape. This intervention favors the
geometric arm, with Cosh-minus-geometric NLL
$+0.060/+0.227/+0.460$ at $512/1$K/$2$K. Range selection and exponent
allocation consequently need to be evaluated together when composing a
deployment table. The complete paired curves and training details appear in
Appendix~\ref{sec:identification-details}.

\subsection{Different allocation shapes across configurations}

A $50.9$M factorial varies two bases, two training lengths, three head
dimensions, and three paired seeds, with endpoints pinned in every comparison.
It evaluates three Cosh strengths and a deformation-matched exponential.
At the reference Cosh strength, the weighted OOD objective improves in
$7/12$ configurations; the preassigned $1.25\times$ strength improves
$10/12$, and the matched exponential improves $9/12$.
The reference-Cosh-minus-exponential mean difference is $+0.00074$ NLL,
with interval $[-0.006,0.008]$ (App.~Table~\ref{tab:m4}).
The observed means favor several preassigned nonuniform shapes, with
configuration-dependent strengths and uncertainty in the cross-configuration
contrasts. Table~\ref{tab:m4} reports the unadjusted paired tests.


\subsection{Weights learn to use their exponent basis}
\label{sec:coadaptation-finding}

A model trained with one exponent distribution has learned query and key
coefficients in that basis. The weights-by-table crossing makes this dependence
visible: in two $50$M models, replacing the Geo-trained model's table by Cosh
changes PPL from $7.14$ to $76.20$; the reverse swap changes $7.16$ to $23.05$
(Fig.~\ref{fig:weight-table-crossing}). A $151.9$M crossing replicates the
preferred-table reversal in two training seeds. These crossings connect exponent design during learning with adjustments of
a fixed checkpoint: the installed table acts together with learned coefficients.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_weight_table_crossing.pdf}
\caption{\textbf{Weights co-adapt to their exponent allocation.}
Each row freezes one trained weight set and changes only its runtime table.
Left: the $50$M crossing reports PPL on eight windows.
Right: the $151.9$M crossing reports mean tail NLL on $32$ anchors in each
of two training seeds, using extensions derived from the two training tables.
FMR denotes the geometric table at the fixed training range.
Shading shows the NLL increase relative to the matched diagonal cell.}
\label{fig:weight-table-crossing}
\end{figure}


<!-- FILE: sections/03_theory.tex -->

\section{Positional Geometry of Exponent Distributions}
\label{sec:theory}

\subsection{What different exponents represent}
\label{sec:subspaces}
\label{sec:budget}

A rotary pair contributes
$C\cos(\omega\Delta)+D\sin(\omega\Delta)$ to an attention logit, where
$\Delta$ is the relative position and $C,D$ depend on query and key content.
The positional object associated with its exponent is therefore the subspace
\begin{equation}
V_\omega=\operatorname{span}\{\cos(\omega\Delta),\sin(\omega\Delta)\}.
\label{eq:subspace}
\end{equation}
Comparing these two-dimensional spaces accounts for every content phase within
a pair. Let $x_\omega(\Delta)=[\cos(\omega\Delta)\ \sin(\omega\Delta)]$,
$S_\omega=\mathbb E[x_\omega^\top x_\omega]$, and
$H_{\omega\nu}=\mathbb E[x_\omega^\top x_\nu]$ under a specified distribution
of separations. For nonsingular $S_\omega,S_\nu$, the singular values of
$Q_{\omega\nu}=S_\omega^{-1/2}H_{\omega\nu}S_\nu^{-1/2}$ are the canonical
correlations of the two spaces. Their average squared correlation,
\begin{equation}
c_{\omega\nu}=\tfrac12\|Q_{\omega\nu}\|_F^2\in[0,1],
\label{eq:collision}
\end{equation}
is unchanged by rotation of the phase inside either pair. It measures how
much of the positional basis one frequency shares with another.

Stacking all block-whitened pairs gives a Gram matrix $\Gamma$ with $I_2$
on its diagonal blocks. If $\bar c$ is the mean of $c_{ij}$ over ordered
distinct pairs, its R\'enyi-$2$ effective rank satisfies
\begin{equation}
r_2(\Gamma)
=\frac{(\operatorname{tr}\Gamma)^2}{\operatorname{tr}(\Gamma^2)}
=\frac{2K}{1+(K-1)\bar c}.
\label{eq:budget-identity}
\end{equation}
At fixed $K$ and range, exponent allocation changes this basis through the
off-diagonal correlations. For uniform separations on $[0,L]$, the cross-Gram
has an exact trigonometric form, so both the correlations and effective rank
can be computed directly (App.~\ref{sec:geometry-proofs}).

\subsection{How slow exponents share positional directions}

\begin{proposition}[Shared slow-frequency subspace]
\label{prop:collapse}
Under the uniform measure on $[0,L]$, as $\omega L\to0$ through $\omega>0$,
the two-dimensional subspace $V_\omega$ approaches $\operatorname{span}\{1,\Delta\}$. If
$\epsilon=\max\{|\omega L|,|\nu L|\}\to0$, then
$2-\|Q_{\omega\nu}\|_F^2=O(\epsilon^4)$.
\end{proposition}

The limit follows from $\cos(\omega\Delta)\to1$ and
$\sin(\omega\Delta)/\omega\to\Delta$; Appendix~\ref{sec:geometry-proofs}
derives the leading fourth-order coefficient. Several slow pairs can thus
supply many coordinate dimensions but very few distinct positional directions.
For $b=500{,}000$, $K=64$, and $L=4096$, the $23$ standard-grid pairs with
$\omega L\le1$ have $46$ nominal dimensions and $r_2=2.00013$ under this
measure. Block whitening normalizes each pair's energy and conditioning, so
$r_2$ measures directional overlap; the raw feature scales are analyzed in
Appendix~\ref{sec:geometry-proofs}.

Exponent redistribution changes how many pairs occupy these shared directions
and how many cover faster positional variations. Frequency-use studies also show that
slow pairs can carry semantic information
\citep{barbero2025round,urrutia2026decoupling}; their positional similarity
motivates redistributing the rotary pairs while retaining their content
coordinates. The learned weights determine how the resulting basis is used.

\subsection{An analytic allocation construction}
\label{sec:construction}

To construct a smooth exponent table, let a positive density
$\rho\in C^2([0,1])$, with $\int_0^1\rho=1$, allocate samples over
$[0,1]$. Uniform $\rho$ recovers geometric spacing. Define the cumulative
slow-tail mass $S_\rho(t)=\int_t^1\rho(\phi)\,d\phi$ and choose the
convex allocation criterion
\begin{equation}
\Capp[\rho]
=\frac{\alpha}{2}\int_0^1\rho(\phi)^2\,d\phi
+\frac{\beta}{2}\int_0^1 S_\rho(t)^2\,dt,
\qquad \alpha>0,\ \beta\ge0.
\label{eq:Capp}
\end{equation}
The first term discourages concentration in a narrow band; the second
penalizes the squared mass beyond each slow-end threshold. Equivalently,
$\int S_\rho^2=\iint\rho(\phi)\rho(\psi)\min(\phi,\psi)\,d\phi\,d\psi$.
These specified design preferences provide a tractable surrogate for
constructing a nonuniform allocation.

\begin{theorem}[Cosh allocation]
\label{thm:ode}
The unique minimizer of \eqref{eq:Capp} over positive
$\rho\in C^2([0,1])$ with unit integral is
\begin{equation}
\rho_\tau(\phi)=\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau},
\qquad \tau=\sqrt{\beta/\alpha},
\label{eq:rho-tau}
\end{equation}
with $\rho_0(\phi)=1$ obtained by continuity at $\beta=0$.
\end{theorem}

The Euler--Lagrange equation reduces to $\rho''=\tau^2\rho$ with
$\rho'(1)=0$; normalization fixes its scale. Integrating and inverting the CDF
gives the finite table
\begin{equation}
\phi_k(\tau)
=1-\frac{1}{\tau}\operatorname{arcsinh}\!\left((1-u_k)\sinh\tau\right),
\qquad u_k=\frac{k+1/2}{K}.
\label{eq:warp}
\end{equation}
We call this one-parameter inverse-CDF family \evq{}. For $\tau>0$, its density decreases from the
fast to the slow end, allocating more samples to faster positional variations.
The limit $\tau\to0$ gives uniform-log midpoint quantiles.

After choosing $\tau$ and a grid, installation evaluates \eqref{eq:warp}
once and sets
$\omega_k=b^{-\phi_k}$. The frequency table stays fixed during training or
adaptation, and the attention operator and trainable parameter count are unchanged.
For the fixed-range experiment we normalize these quantiles using
\eqref{eq:table-decomposition} and anchor them to the baseline endpoints.
Table~\ref{tab:allocation-protocols} records each experiment's $\tau$, grid,
and selection convention; the operating-rule study is in
Appendix~\ref{sec:tau-scaling}.

\FloatBarrier
<!-- FILE: sections/04_experiments.tex -->

\section{Model Effects of an Analytic Allocation}
\label{sec:experiments}
\label{sec:exp-mature}

We apply Cosh quantiles in three settings: an MLA architecture with few
rotary channels, continued full-parameter training, and adaptation of a
pretrained language model. Together they test how an explicit allocation
supports language modeling, retrieval, and source use.

\subsection{Language modeling and retrieval}
\label{sec:exp-capability}

\paragraph{Scarce rotary channels.}
The $432$M MLA model uses $16$ rotary pairs and trains on $500$M tokens at
length $8192$ \citep{deepseekv2,deepseekv3}. Across three seeds, \evq{} reduces
$16$K perplexity from $138.8$ to $95.6$; at $8$K the corresponding values are
$35.4$ and $35.8$ (Table~\ref{tab:training-models}). This tests exponent
allocation in an architecture where only a small set of dimensions carries
explicit rotation.

\paragraph{Continued full-parameter training.}
The matched $750$M continuation trains both arms at length $4096$.
At $16$K, \evq{} reduces perplexity from $45.1$ to $24.4$, while the $4$K
values remain close at $22.0$ and $22.3$. Across $40$ associated $8$K passkey
trials, greedy autoregressive exact match rises from $0$ to $77.5\%$.
The per-length results and generation protocol are in
Appendix~\ref{sec:experiment-details}.

\begin{table}[ht]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{\textbf{Model effects of the Cosh exponent table.}
Each row compares Geo and \evq{} within one matched protocol. Entries are
mean perplexities over the indicated training replications; lower is better.}
\label{tab:training-models}
\begin{tabular}{@{}lc@{\hspace{14pt}}ccrr@{}}
\toprule
Model & Seeds & Train length & Eval. length & Geo & \evq{} \\
\midrule
$432$M MLA & 3 & $8$K & $8$K & 35.4 & 35.8 \\
           &   &   & $16$K & 138.8 & \textbf{95.6} \\
$750$M continuation & 1 & $4$K & $4$K & 22.0 & 22.3 \\
           &   &   & $16$K & 45.1 & \textbf{24.4} \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Matched Llama-$3$-$8$B adaptation.}
We compare Native and \evq{} under the same $300$-step rank-$64$ Q/K/V/O LoRA
adaptation, using physical $8$K sequences \citep{hu2022lora,grattafiori2024llama3}.
The evaluation comprises $24$ frozen temporal text packs from three domains.
Figure~\ref{fig:8b-length-curve} shows the full curve:
Native-LoRA/\evq{}-LoRA PPL is $6.82/10.07$ at $8$K,
$108.96/24.07$ at $16$K, and $991.48/127.91$ at $32$K.
The long-context NLL improvement holds in every pack in both extended lengths.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.86\linewidth]{fig_8b_length_curve.pdf}
  \caption{\textbf{The Cosh construction improves extrapolation under matched adaptation.} Llama-$3$-$8$B arms share an $8$K, $300$-step LoRA protocol and
  are evaluated on the same $24$ text packs. The logarithmic PPL axis shows
  all three evaluation lengths. PPL is the exponential of the equal-domain
  mean NLL.}
  \label{fig:8b-length-curve}
\end{figure}

Source interventions help interpret this result. On ten true-$16$K passkey
cases, median target-block hit@16 rises from $18.75\%$ to $64.06\%$.
Removing decode attention to the remote gold block raises pooled answer-token NLL by
$1.5055$ in the \evq{} arm, compared with $-0.0095$ in Native.
Thus the adapted model's answer probability depends on the distant source.
Appendix~\ref{sec:llama8b} reports this intervention and the separate
RULER-family continuation.

That continuation provides an autoregressive comparison: both arms receive
the same $516$ additional steps on $13$ RULER families at physical length
$8$K, then generate answers on new rows. At $16$K, the official RULER macro
is $0.30\%$ for Native-LoRA and $14.03\%$ for \evq{}-LoRA; at $8$K it is
$94.44/77.60\%$. The official task metrics allow partial credit;
normalized whole-response exact match at $16$K is $0/1.54\%$.
These scores measure length transfer after matched task-family supervision.

\paragraph{Composition with range extension.}
In the $454$M three-seed experiment, applying the same fixed-index
smooth-ramp scaler to both trained tables gives $16$K PPL of $157.7$ for
Geo and $107.5$ for \evq{}. The training-time allocation thus remains useful
after this shared inference-time transformation.
Appendix~\ref{sec:range-composition} gives the operator, training protocol,
and complete comparison.

\paragraph{Further settings.}
The same-initialization OLMo scale check and video-DiT comparison are reported
with their full protocols in Apps.~\ref{sec:olmo2-1b} and~\ref{sec:video-dit}.

<!-- FILE: sections/04_mature.tex -->

\section{Adjusting Exponents in Frozen Models}
\label{sec:mature-adjustments}
\label{sec:exp-frozen-support}
\label{sec:exp-zero-training-system}
\label{sec:coordinate}

The frozen crossings in \S\ref{sec:coadaptation-finding} show that trained
weights use a particular exponent basis. This motivates adjustments defined
relative to the checkpoint's native table. We compare how these adjustments
place the available exponent shifts within the existing rotary pairs.

\subsection{A common coordinate for exponent adjustments}

For native frequencies $\omega_k^N$ and an installed table $\omega'_k$, define
\begin{equation}
d_k=\log\frac{\omega_k^N}{\omega'_k},\qquad
\omega'_k=\omega_k^N e^{-d_k},\qquad
\phi'_k-\phi_k^N=\frac{d_k}{\log b}.
\label{eq:movement-allocation}
\end{equation}
Several familiar constructions then have explicit displacement profiles:
\begin{equation}
d_k=
\begin{cases}
-\log(1-w_k+w_k/s), & \text{frequency blend},\\
m_k\log s, & \text{log-frequency shift},\\
\sum_{j<k}\log\lambda_j, & \text{mixed-radix conversion}.
\end{cases}
\label{eq:adjustment-families}
\end{equation}
Here $s$ is the extension factor, $w_k$ is a blend weight, and $m_k$ is the
normalized cumulative exponent shift. YaRN's NTK-by-parts frequency map uses the first construction;
MrRoPE uses the third \citep{peng2024yarn,tian2026mrrope}.
The coordinate lets us compare their realized exponent shapes directly.

\subsection{Fixed-range and single-table comparisons}

\paragraph{Changing only the normalized exponent allocation.}
On frozen OLMo-$2$-$1$B and Qwen-$2.5$-$1.5$B, we hold the endpoints, $64$
pairs, attention amplitude, inputs, and decoder fixed. A geometric allocation,
a residual-guided allocation, and its nearest coarse ramp give the results in
Table~\ref{tab:frozen-exponent-main}. The residual-guided construction moves
pairs according to their remaining energy after projection onto the other
sine--cosine pairs; the ramp approximates that profile without task labels.
Both structured profiles produce large gains over uniform exponents.
Their definitions and reference Native/YaRN rows are in
Appendix~\ref{sec:frozen-fixed-support}.

\begin{table}[t]
\centering
\small
\caption{\textbf{Frozen-model RULER scores (\%) at matched exponent range.}
OLMo uses a held-out nine-task confirmation at $16$K; Qwen uses a four-task
development panel at $64$K, with $20$ rows per task.
Within each model only the interior allocation changes.}
\label{tab:frozen-exponent-main}
\begin{tabular}{@{}lrr@{}}
\toprule
Exponent allocation & OLMo-$1$B & Qwen-$1.5$B \\
\midrule
Geometric & 0.56 & 57.75 \\
Coarse ramp & \textbf{61.04} & 64.00 \\
Residual-guided & 60.47 & \textbf{66.50} \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{One static table at both lengths.}
A frozen normalized-index profile transfers a reference displacement curve
onto the native slot grid and installs $\omega'_k=\omega_k^N s^{-m_k}$.
Its placement can be compared with direct evaluation from local frequency
gaps. On a new-seed Gemma-$1.1$-$2$B comparison with $128$ rotary pairs
\citep{gemmateam2024gemma},
normalized-index interpolation scores $79.00\%$ versus $72.81\%$ at $16$K.
Both arms share the frequency endpoints, amplitude, and $320$ inputs;
the paired improvement is $6.19$ points with a $95\%$ bootstrap interval $[2.81,9.63]$.
Appendix~\ref{sec:coordinate-confirmation} gives the two constructions and
the corresponding $32$-pair comparison.

On Qwen-$2.5$-$0.5$B, a new-seed $13$-task RULER evaluation uses $20$ examples
per task at $32$K and $64$K. Every request uses the same $s=2$ table throughout
prefill and decoding, including requests within the native window.
The installed table and amplitude together give the comparison in
Table~\ref{tab:static-index-main}: $6.09$ points above YaRN at $64$K,
with a $95\%$ paired task-stratified bootstrap interval $[2.76,9.58]$ points.
At $32$K the two methods are nearly equal. Appendix~\ref{sec:index-adjustment}
gives the frozen profile and the full task breakdown.

\begin{table}[t]
\centering
\small
\caption{\textbf{A static extension across the native and extended windows.}
Qwen-$0.5$B RULER-13 macro (\%); $g$ is the fixed cosine/sine amplitude used by
each method.}
\label{tab:static-index-main}
\begin{tabular}{@{}lrrr@{}}
\toprule
Installed table & $g$ & $32$K & $64$K \\
\midrule
Native & 1.0000 & 54.78 & 22.05 \\
YaRN-$2$ & 1.0693 & 55.94 & 45.37 \\
Normalized-index-$2$ & 1.0513 & 55.92 & \textbf{51.46} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Redistributing the intermediate-band shift}
\label{sec:bm-main}

MrRoPE-Pro progressively distributes the cumulative shift across an intermediate
frequency band. We compare it with a boundary-matched (BM) profile, which keeps
the same band endpoints and total scale while smoothing the per-slot radix
increments toward the unmodified outer bands. Both are explicit constructions
of $m_k$ in \eqref{eq:adjustment-families}; their exact discrete formulas are
given in Appendix~\ref{sec:bm-construction}.

In an OLMo-$1$B six-task confirmation, BM improves the $16$K macro from
$2.78\%$ for MrRoPE-Pro to $51.32\%$; MrRoPE-Uni scores $32.12\%$.
The comparison extends to natural inputs: across $631$ eligible long examples
from five QA tasks, the task-equal F1 rises from $21.62\%$ to $25.44\%$.
The $95\%$ paired bootstrap interval for the improvement is $[1.32,6.29]$ points.
Every task has a positive mean difference
(Fig.~\ref{fig:bm-natural-qa}). All methods use a fixed factor-four table;
the input and decoding details are in Appendix~\ref{sec:bm-natural}.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{fig_bm_natural_qa.pdf}
  \caption{\textbf{An intermediate-band redistribution improves OLMo natural QA.}
  Complete-output token F1 on untruncated long inputs, using the same $s=4$
  configuration and decoder in both arms. Sample counts are shown by task;
  the final row averages the five task means.}
  \label{fig:bm-natural-qa}
\end{figure}

The three-model comparison reveals distinct preferences for the intermediate
allocation (Table~\ref{tab:bm-models}). BM gives the higher OLMo scores and
Qwen-$3$B's higher $32$K score; MrRoPE-Pro gives the higher $128$K scores on
both Qwen checkpoints. With band boundaries and cumulative scale fixed,
these differences connect the preferred exponent shape to the checkpoint and
operating length. Appendix~\ref{sec:bm-model-comparison} reports every task.

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Matched six-task RULER comparisons (\%).}
Each length column averages six task scores. Both methods use static $s=4$;
the row counts refer to short/long prompts per arm.}
\label{tab:bm-models}
\begin{tabular}{@{}lclrrrr@{}}
\toprule
& & & \multicolumn{2}{c}{Short} & \multicolumn{2}{c}{Long}\\
Model & Lengths & Rows & MrPro & BM & MrPro & BM \\
\midrule
OLMo-$1$B & $4/16$K & $24/48$ & 37.85 & 81.81 & 2.78 & 51.32 \\
Qwen-$3$B & $32/128$K & $12/24$ & 87.22 & 91.67 & 78.13 & 70.83 \\
Qwen-$7$B & $32/128$K & $6/12$ & 83.33 & 80.00 & 84.44 & 71.11 \\
\bottomrule
\end{tabular}
\end{table}

<!-- FILE: sections/02_related.tex -->

\section{Related Work}
\label{sec:related}

\paragraph{Range, scaling, and exponent shape.}
RoPE's rotation operator \citep{su2024roformer} admits many frequency tables.
Scaling-law and base-bound analyses study how the geometric family's range
affects length generalization \citep{liu2024scaling,xu2024base}.
FMRoPE relates the learned frequency band to the base and training length,
and chooses a base accordingly \citep{oka2026fmrope}. In that family the
normalized exponents remain equally spaced; our fixed-range comparisons vary
their interior distribution.

Position interpolation scales positions uniformly, while YaRN blends
frequencies across bands \citep{chen2024position,peng2024yarn}.
LongRoPE combines dimension-wise scaling with a token-position threshold
\citep{ding2024longrope,shang2025longrope2}.
MrRoPE gives uniform and progressive intermediate-band rules through
mixed-radix conversion \citep{tian2026mrrope}.
Equation~\eqref{eq:adjustment-families} expresses these frequency factors as
exponent displacements. Our second stage compares their shapes; the first
derives an exponent density and tests it at fixed frequency endpoints.

\paragraph{Learning and using frequency bands.}
LeRoPE learns one log-space scale per pair, shared across layers and heads;
AdaRoPE learns head-specific frequencies and attention scaling
\citep{karypis2026lerope,wang2026adarope}. Both optimize entries that can
be expressed as exponent changes. Our contribution lies in their allocation
geometry, explicit density construction, and controlled separation from range.

Frequency-use studies connect fast and slow bands to positional and symbolic
attention, retrieval, and data-dependent scales
\citep{barbero2025round,urrutia2026decoupling,chiang2025rotary,
oka2026frequencyentropy,wu2026datashapes}.
DoPE uses truncated matrix entropy to identify low-rank structures in rotated
activations and selectively modifies positional encoding \citep{xiong2025dope}.
Our canonical correlations instead describe the supplied sine--cosine
subspaces as their exponents vary.

\paragraph{Changing the positional operator.}
FoPE introduces Fourier components, while HoPE and clipped encodings alter the
treatment of slow rotations \citep{hua2025fope,chen2025hope,li2026copeclipped}.
GRAPE generalizes positional transformations through group actions;
Selective RoPE learns input-dependent rotations
\citep{zhang2026grape,movahedi2026selectiverope}.
RePo learns contextual token positions before applying the encoding
\citep{li2026repo}. These works change the positional map or its dependence on
content. Our construction retains the standard rotation and designs its
exponent table. Recent analyses of long-context positional and token confusion
further motivate studying the interaction between frequency choice and the
representations it acts on \citep{du2026distinguishes}.

<!-- FILE: sections/05_discussion.tex -->

\section{Discussion}
\label{sec:discussion}

The exponent distribution determines how a finite RoPE table samples
positional scales. Fixed-range training establishes a behavioral effect of
its interior shape, while range retargeting and weights-by-table crossings
show that this effect interacts with the installed range and learned
coefficients. The full sine--cosine analysis makes the underlying positional
basis explicit and explains why slow exponents can supply strongly
correlated directions.

These findings support two forms of design. Before or during learning, a
specified allocation criterion can generate an explicit table, as illustrated
by the Cosh construction and its trained-model results. After learning,
adjustments relative to the native table provide a way to preserve its
coordinate structure while reallocating the extension across pairs.
The static-profile and boundary-matched experiments show practical gains,
as well as checkpoint-dependent preferences for the adjustment shape.

A natural next step is to study exponent allocation under sparse attention.
Selective token access and compressed representations change which positional
distinctions reach the reader. The framework developed here supplies a way
to examine which exponent distributions serve those architectures.


%% ---- main text ends here; everything below is exempt from the page limit ----
\label{page:bodyend}

\subsection*{Ethics statement}
<!-- FILE: sections/06_ethics.tex -->

This work introduces no new dataset, deployed system, or human-subject
component. The proposed training-time construction changes only the RoPE
inverse-frequency table and adds no learned parameters; the mature-checkpoint
studies use the frozen-table or LoRA interventions stated in their protocols.
The table construction remains inexpensive to audit, ablate, or revert. All
models, corpora, and evaluation suites used here are public research artefacts
used within their stated licences.

A better allocated frequency table may reduce the adaptation or inference
effort required to reach a target context range, and the same mechanism could
be used to extend the context capability of language or video models, with the
general risks associated with stronger generative models. Our evaluations
measure positional and long-context endpoints, not factual reliability or
safety; deployment claims require application-specific evaluation.


\subsection*{Reproducibility statement}
<!-- FILE: sections/07_reproducibility.tex -->

The appendices specify the installed frequency tables, exponent grids,
attention amplitudes, training and adaptation recipes, datasets, and evaluation
procedures. The fixed-range experiment reports three paired training seeds;
the larger-model and frozen-model tables identify their comparison units and
sample counts. The accompanying source archive contains the active TeX,
bibliography, plotted figures, numerical summaries, a SHA256 manifest, and a
standalone verification of the explicit Gram-matrix examples. The appendices
provide the derivations and exponent-adjustment formulas. Model checkpoints
and training-data streams are separate from this manuscript archive; its
build reproduces the paper, and its numerical records support inspection of
the reported comparisons.


\subsection*{AI use statement}
<!-- FILE: sections/08_ai_use.tex -->

We used generative AI tools to help develop conceptual frameworks and
mathematical claims; propose and refine hypotheses; draft and check proofs; give
feedback on methodology and experiments; implement methods and deterministic
data-reformatting and analysis code; translate or rephrase research notes; and
interpret results. We did not use generative AI to generate dataset content or
to perform qualitative or thematic data analysis; survey, interview, and
transcription tasks do not apply to this work.

For tasks for which disclosure is recommended, we also used these tools to
suggest experimental parameters; write, edit, and test software and artefacts;
create or modify figures; draft, edit, and structure parts of the manuscript;
brainstorm and identify research gaps; search for and summarise literature; and
format references. Every empirical quantity reported here comes from executed
code over recorded run artefacts rather than language-model generation.

All AI-assisted work was reviewed before inclusion. We take responsibility for
the final text, claims, proofs, code, figures, and experimental artefacts.


\bibliographystyle{iclr2027_conference}
\bibliography{refs/references}
<!-- FILE: refs/references.bib -->

@inproceedings{vaswani2017attention,
  title = {Attention Is All You Need},
  author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {30},
  year = {2017}
}

@inproceedings{shaw2018self,
  title = {Self-Attention with Relative Position Representations},
  author = {Shaw, Peter and Uszkoreit, Jakob and Vaswani, Ashish},
  booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year = {2018}
}

@article{raffel2020exploring,
  title = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  author = {Raffel, Colin and Shazeer, Noam and Roberts, Adam and Lee, Katherine and Narang, Sharan and Matena, Michael and Zhou, Yanqi and Li, Wei and Liu, Peter J.},
  journal = {Journal of Machine Learning Research},
  volume = {21},
  number = {140},
  pages = {1--67},
  year = {2020}
}

@inproceedings{press2022alibi,
  title = {Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation},
  author = {Press, Ofir and Smith, Noah A. and Lewis, Mike},
  booktitle = {International Conference on Learning Representations},
  year = {2022}
}

@article{su2024roformer,
  title = {{RoFormer}: Enhanced Transformer with Rotary Position Embedding},
  author = {Su, Jianlin and Ahmed, Murtadha and Lu, Yu and Pan, Shengfeng and Wen, Bo and Liu, Yunfeng},
  journal = {Neurocomputing},
  volume = {568},
  pages = {127063},
  year = {2024}
}

@article{chen2024position,
  title = {Extending Context Window of Large Language Models via Positional Interpolation},
  author = {Chen, Shouyuan and Wong, Sherman and Chen, Liangjian and Tian, Yuandong},
  journal = {arXiv preprint arXiv:2306.15595},
  year = {2023},
  doi = {10.48550/arXiv.2306.15595},
  url = {https://arxiv.org/abs/2306.15595}
}

@inproceedings{peng2024yarn,
  title = {{YaRN}: Efficient Context Window Extension of Large Language Models},
  author = {Peng, Bowen and Quesnelle, Jeffrey and Fan, Honglu and Shippole, Enrico},
  booktitle = {International Conference on Learning Representations},
  year = {2024},
  url = {https://openreview.net/forum?id=wHBfxhZu1u}
}

@inproceedings{ding2024longrope,
  title = {{LongRoPE}: Extending {LLM} Context Window Beyond 2 Million Tokens},
  author = {Ding, Yiran and Zhang, Li Lyna and Zhang, Chengruidong and Xu, Yuanyuan and Shang, Ning and Xu, Jiahang and Yang, Fan and Yang, Mao},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series = {PMLR},
  volume = {235},
  pages = {11091--11104},
  year = {2024},
  url = {https://proceedings.mlr.press/v235/ding24i.html}
}

@inproceedings{zheng2024dape,
  title = {{DAPE}: Data-Adaptive Positional Encoding for Length Extrapolation},
  author = {Zheng, Chuanyang and Gao, Yihang and Shi, Han and Huang, Minbin and Li, Jingyao and Xiong, Jing and Ren, Xiaozhe and Ng, Michael K. and Jiang, Xin and Li, Zhenguo and Li, Yu},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  year = {2024},
  doi = {10.52202/079017-0838},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f050fa9f0d898e3f265d515f50ae8f9-Abstract-Conference.html}
}

@inproceedings{xu2024base,
  title = {Base of {RoPE} Bounds Context Length},
  author = {Xu, Mingyu and Men, Xin and Wang, Bingning and Zhang, Qingyu and Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Chen, Weipeng},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  year = {2024},
  doi = {10.52202/079017-2773},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/hash/9f12dd32d552f3ad9eaa0e9dfec291be-Abstract-Conference.html}
}

@inproceedings{li2024fire,
  title = {Functional Interpolation for Relative Positions Improves Long Context Transformers},
  author = {Li, Shanda and You, Chong and Guruganesh, Guru and Ainslie, Joshua and Ontanon, Santiago and Zaheer, Manzil and Sanghai, Sumit and Yang, Yiming and Kumar, Sanjiv and Bhojanapalli, Srinadh},
  booktitle = {International Conference on Learning Representations},
  year = {2024},
  url = {https://openreview.net/forum?id=rR03qFesqk}
}

@inproceedings{chi2022kerple,
  title = {{KERPLE}: Kernelized Relative Positional Embedding for Length Extrapolation},
  author = {Chi, Ta-Chung and Fan, Ting-Han and Ramadge, Peter J. and Rudnicky, Alexander I.},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {35},
  year = {2022}
}

@inproceedings{bai2024longbench,
  title = {{LongBench}: A Bilingual, Multitask Benchmark for Long Context Understanding},
  author = {Bai, Yushi and Lv, Xin and Zhang, Jiajie and Lyu, Hongchang and Tang, Jiankai and Huang, Zhidian and Du, Zhengxiao and Liu, Xiao and Zeng, Aohan and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {3119--3137},
  year = {2024},
  doi = {10.18653/v1/2024.acl-long.172},
  url = {https://aclanthology.org/2024.acl-long.172/}
}

@article{penedo2024fineweb,
  title = {The {FineWeb} Datasets: Decanting the Web for the Finest Text Data at Scale},
  author = {Penedo, Guilherme and Kydl{\'i}{\v c}ek, Hynek and Ben Allal, Loubna and Lozhkov, Anton and Mitchell, Margaret and Raffel, Colin and von Werra, Leandro and Wolf, Thomas},
  journal = {arXiv preprint arXiv:2406.17557},
  year = {2024},
  doi = {10.48550/arXiv.2406.17557},
  url = {https://arxiv.org/abs/2406.17557}
}

@article{eldan2023tinystories,
  title = {{TinyStories}: How Small Can Language Models Be and Still Speak Coherent English?},
  author = {Eldan, Ronen and Li, Yuanzhi},
  journal = {arXiv preprint arXiv:2305.07759},
  year = {2023}
}

@inproceedings{pang2022quality,
  title = {{QuALITY}: Question Answering with Long Input Texts, Yes!},
  author = {Pang, Richard Yuanzhe and Parrish, Alicia and Joshi, Nitish and Nangia, Nikita and Phang, Jason and Chen, Angelica and Padmakumar, Vishakh and Ma, Johnny and Thompson, Jana and He, He and Bowman, Samuel},
  booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages = {5336--5358},
  year = {2022},
  doi = {10.18653/v1/2022.naacl-main.391}
}

@inproceedings{bai2024longalign,
  title = {{LongAlign}: A Recipe for Long Context Alignment of Large Language Models},
  author = {Bai, Yushi and Lv, Xin and Zhang, Jiajie and He, Yuze and Qi, Ji and Hou, Lei and Tang, Jie and Dong, Yuxiao and Li, Juanzi},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages = {1376--1395},
  year = {2024},
  doi = {10.18653/v1/2024.findings-emnlp.74}
}

@inproceedings{hsieh2024ruler,
  title = {{RULER}: What's the Real Context Size of Your Long-Context Language Models?},
  author = {Hsieh, Cheng-Ping and Sun, Simeng and Kriman, Samuel and Acharya, Shantanu and Rekesh, Dima and Jia, Fei and Zhang, Yang and Ginsburg, Boris},
  booktitle = {First Conference on Language Modeling},
  year = {2024},
  url = {https://openreview.net/forum?id=kIoBbc76Sy}
}

@article{liu2024lost,
  title = {Lost in the Middle: How Language Models Use Long Contexts},
  author = {Liu, Nelson F. and Lin, Kevin and Hewitt, John and Paranjape, Ashwin and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {12},
  pages = {157--173},
  year = {2024}
}

@inproceedings{zhu2024pose,
  title = {{PoSE}: Efficient Context Window Extension of {LLMs} via Positional Skip-wise Training},
  author = {Zhu, Dawei and Yang, Nan and Wang, Liang and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
  booktitle = {International Conference on Learning Representations},
  year = {2024}
}

@inproceedings{jin2024selfextend,
  title = {{LLM} Maybe {LongLM}: {SelfExtend} {LLM} Context Window Without Tuning},
  author = {Jin, Hongye and Han, Xiaotian and Yang, Jingfeng and Jiang, Zhimeng and Liu, Zirui and Chang, Chia-Yuan and Chen, Huiyuan and Hu, Xia},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series = {PMLR},
  volume = {235},
  pages = {22099--22114},
  year = {2024}
}

@inproceedings{videorope2025,
  title = {{VideoRoPE}: What Makes for Good Video Rotary Position Embedding?},
  author = {Wei, Xilin and Liu, Xiaoran and Zang, Yuhang and Dong, Xiaoyi and Zhang, Pan and Cao, Yuhang and Tong, Jian and Duan, Haodong and Guo, Qipeng and Wang, Jiaqi and Qiu, Xipeng and Lin, Dahua},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series = {PMLR},
  volume = {267},
  pages = {66118--66136},
  year = {2025},
  url = {https://proceedings.mlr.press/v267/wei25h.html}
}

@article{radford2019gpt2,
  title = {Language Models are Unsupervised Multitask Learners},
  author = {Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal = {OpenAI Technical Report},
  year = {2019}
}

@inproceedings{zhao2025riflex,
  title = {{RIFLEx}: A Free Lunch for Length Extrapolation in Video Diffusion Transformers},
  author = {Zhao, Min and He, Guande and Chen, Yixiao and Zhu, Hongzhou and Li, Chongxuan and Zhu, Jun},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series = {PMLR},
  volume = {267},
  pages = {77539--77557},
  year = {2025},
  url = {https://proceedings.mlr.press/v267/zhao25m.html}
}

@inproceedings{qiu2024freenoise,
  title = {{FreeNoise}: Tuning-Free Longer Video Diffusion via Noise Rescheduling},
  author = {Qiu, Haonan and Xia, Menghan and Zhang, Yong and He, Yingqing and Wang, Xintao and Shan, Ying and Liu, Ziwei},
  booktitle = {International Conference on Learning Representations},
  year = {2024}
}

@inproceedings{shang2025longrope2,
  title = {{LongRoPE2}: Near-Lossless {LLM} Context Window Scaling},
  author = {Shang, Ning and Zhang, Li Lyna and Wang, Siyuan and Zhang, Gaokai and Lopez, Gilsinia and Yang, Fan and Chen, Weizhu and Yang, Mao},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series = {PMLR},
  volume = {267},
  pages = {54203--54218},
  year = {2025},
  url = {https://proceedings.mlr.press/v267/shang25a.html}
}

@inproceedings{chen2024clex,
  title = {{CLEX}: Continuous Length Extrapolation for Large Language Models},
  author = {Chen, Guanzheng and Li, Xin and Meng, Zaiqiao and Liang, Shangsong and Bing, Lidong},
  booktitle = {International Conference on Learning Representations},
  year = {2024}
}

% Round and Round We Go — empirical analysis of RoPE frequency usage in Gemma;
% finds that high frequencies carry positional patterns and low frequencies carry semantic information.
% Independent empirical support for EVQ-Cosh's premise that the two frequency regimes are not equally important.
@inproceedings{barbero2025round,
  title = {Round and Round We Go! What makes Rotary Positional Encodings useful?},
  author = {Barbero, Federico and Vitvitskyi, Alex and Perivolaropoulos, Christos and Pascanu, Razvan and Veli{\v{c}}kovi{\'c}, Petar},
  booktitle = {International Conference on Learning Representations},
  year = {2025},
  url = {https://proceedings.iclr.cc/paper_files/paper/2025/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html}
}

@inproceedings{wang2024resonance,
  title = {Resonance {RoPE}: Improving Context Length Generalization of Large Language Models},
  author = {Wang, Suyuchen and Kobyzev, Ivan and Lu, Peng and Rezagholizadeh, Mehdi and Liu, Bang},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  pages = {586--598},
  year = {2024},
  doi = {10.18653/v1/2024.findings-acl.32},
  url = {https://aclanthology.org/2024.findings-acl.32/}
}

@inproceedings{zhang2024found,
  title = {Found in the Middle: How Language Models Use Long Contexts Better via Plug-and-Play Positional Encoding},
  author = {Zhang, Zhenyu and Chen, Runjin and Liu, Shiwei and Yao, Zhewei and Ruwase, Olatunji and Chen, Beidi and Wu, Xiaoxia and Wang, Zhangyang},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  year = {2024}
}

@article{deepseekv2,
  title = {{DeepSeek-V2}: A Strong, Economical, and Efficient Mixture-of-Experts Language Model},
  author = {DeepSeek-AI},
  journal = {arXiv preprint arXiv:2405.04434},
  year = {2024},
  doi = {10.48550/arXiv.2405.04434},
  url = {https://arxiv.org/abs/2405.04434}
}

@article{deepseekv3,
  title = {{DeepSeek-V3} Technical Report},
  author = {DeepSeek-AI},
  journal = {arXiv preprint arXiv:2412.19437},
  year = {2024}
}

@article{touvron2023llama2,
  title = {Llama 2: Open Foundation and Fine-Tuned Chat Models},
  author = {Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and Almahairi, Amjad and Babaei, Yasmine and Bashlykov, Nikolay and Batra, Soumya and Bhargava, Prajjwal and Bhosale, Shruti and others},
  journal = {arXiv preprint arXiv:2307.09288},
  year = {2023}
}

% Intentional large-author exception: retain the official leading authors and
% BibTeX's `and others` abbreviation rather than expanding hundreds of names.
@article{grattafiori2024llama3,
  title = {The {Llama} 3 Herd of Models},
  author = {Grattafiori, Aaron and Dubey, Abhimanyu and Jauhri, Abhinav and others},
  journal = {arXiv preprint arXiv:2407.21783},
  year = {2024},
  doi = {10.48550/arXiv.2407.21783},
  url = {https://arxiv.org/abs/2407.21783}
}

% Intentional large-author exception: retain the official leading authors and
% BibTeX's `and others` abbreviation rather than expanding the full team list.
@article{qwen2024qwen25,
  title = {{Qwen2.5} Technical Report},
  author = {Yang, An and Yang, Baosong and Zhang, Beichen and Hui, Binyuan and Zheng, Bo and Yu, Bowen and Li, Chengyuan and Liu, Dayiheng and Huang, Fei and others},
  journal = {arXiv preprint arXiv:2412.15115},
  year = {2024},
  doi = {10.48550/arXiv.2412.15115},
  url = {https://arxiv.org/abs/2412.15115}
}

@article{yang2024cogvideox,
  title = {{CogVideoX}: Text-to-Video Diffusion Models with An Expert Transformer},
  author = {Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal = {arXiv preprint arXiv:2408.06072},
  year = {2024}
}

@article{wan2025wan,
  title = {Wan: Open and Advanced Large-Scale Video Generative Models},
  author = {{Wan Team}},
  journal = {arXiv preprint arXiv:2503.20314},
  year = {2025}
}

@article{kong2024hunyuanvideo,
  title = {{HunyuanVideo}: A Systematic Framework For Large Video Generative Models},
  author = {Kong, Weijie and Tian, Qi and Zhang, Zijian and Min, Rox and Dai, Zuozhuo and Zhou, Jin and Xiong, Jiangfeng and Li, Xin and Wu, Bo and others},
  journal = {arXiv preprint arXiv:2412.03603},
  year = {2024}
}

@article{opensora2024,
  title = {Open-{Sora}: Democratizing Efficient Video Production for All},
  author = {Zheng, Zangwei and Peng, Xiangyu and Yang, Tianji and Shen, Chenhui and Li, Shenggui and Liu, Hongxin and Zhou, Yukun and Li, Tianyi and You, Yang},
  journal = {arXiv preprint arXiv:2412.20404},
  year = {2024}
}

@article{ma2024latte,
  title = {Latte: Latent Diffusion Transformer for Video Generation},
  author = {Ma, Xin and Wang, Yaohui and Jia, Gengyun and Chen, Xinyuan and Liu, Ziwei and Li, Yuan-Fang and Chen, Cunjian and Qiao, Yu},
  journal = {arXiv preprint arXiv:2401.03048},
  year = {2024}
}

@inproceedings{sun2022xpos,
  title = {A Length-Extrapolatable Transformer},
  author = {Sun, Yutao and Dong, Li and Patra, Barun and Ma, Shuming and Huang, Shaohan and Benhaim, Alon and Chaudhary, Vishrav and Song, Xia and Wei, Furu},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
  year = {2023}
}

@article{roziere2023codellama,
  title = {Code Llama: Open Foundation Models for Code},
  author = {Rozi{\`e}re, Baptiste and Gehring, Jonas and Gloeckle, Fabian and Sootla, Sten and Gat, Itai and Tan, Xiaoqing Ellen and Adi, Yossi and Liu, Jingyu and Sauvestre, Romain and Remez, Tal and others},
  journal = {arXiv preprint arXiv:2308.12950},
  year = {2023}
}

@inproceedings{srivastava2015unsupervised,
  title = {Unsupervised Learning of Video Representations using {LSTMs}},
  author = {Srivastava, Nitish and Mansimov, Elman and Salakhutdinov, Ruslan},
  booktitle = {Proceedings of the 32nd International Conference on Machine Learning},
  series = {PMLR},
  volume = {37},
  pages = {843--852},
  year = {2015},
  url = {https://proceedings.mlr.press/v37/srivastava15.html}
}

@inproceedings{qwen2026mhrope,
  title = {Revisiting Multimodal Positional Encoding in Vision--Language Models},
  author = {Huang, Jie and Liu, Xuejing and Song, Sibo and Hou, Ruibing and Chang, Hong and Lin, Junyang and Bai, Shuai},
  booktitle = {International Conference on Learning Representations},
  year = {2026},
  note = {arXiv:2510.23095}
}

@inproceedings{li2025hope,
  title = {{HoPE}: Hybrid of Position Embedding for Long Context Vision--Language Models},
  author = {Li, Haoran and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2025},
  note = {arXiv:2505.20444}
}

@inproceedings{chen2025hope,
  title = {{HoPE}: A Novel Positional Encoding Without Long-Term Decay for Enhanced Context Awareness and Extrapolation},
  author = {Chen, Yuhan and Lv, Ang and Luan, Jian and Wang, Bin and Liu, Wei},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {23044--23056},
  year = {2025},
  doi = {10.18653/v1/2025.acl-long.1123},
  url = {https://aclanthology.org/2025.acl-long.1123/}
}

@inproceedings{yang2025pathattention,
  title = {{PaTH} Attention: Position Encoding via Accumulating Householder Transformations},
  author = {Yang, Songlin and Shen, Yikang and Wen, Kaiyue and Tan, Shawn and Mishra, Mayank and Ren, Liliang and Panda, Rameswar and Kim, Yoon},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2025},
  note = {NeurIPS 2025 poster}
}

@article{dai2025hyperbolicrope,
  title = {{HoPE}: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models},
  author = {Dai, Chang and Shan, Hongyu and Song, Mingyang and Liang, Di},
  journal = {arXiv preprint arXiv:2509.05218},
  year = {2025}
}

@inproceedings{hua2025fope,
  title = {Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization},
  author = {Hua, Ermo and Jiang, Che and Lv, Xingtai and Zhang, Kaiyan and Sun, Youbang and Fan, Yuchen and Zhu, Xuekai and Qi, Biqing and Ding, Ning and Zhou, Bowen},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series = {PMLR},
  volume = {267},
  pages = {24932--24949},
  year = {2025},
  url = {https://proceedings.mlr.press/v267/hua25b.html}
}

@article{veisi2025carope,
  title = {Context-aware Rotary Position Embedding},
  author = {Veisi, Ali and Fartoot, Delaram and Amirzadeh, Hamidreza},
  journal = {arXiv preprint arXiv:2507.23083},
  year = {2025}
}

@article{li2026copeclipped,
  title = {{CoPE}: Clipped {RoPE} as a Scalable Free Lunch for Long Context {LLMs}},
  author = {Li, Haoran and Ren, Sucheng and Yuille, Alan and Wang, Feng},
  journal = {arXiv preprint arXiv:2602.05258},
  year = {2026},
  doi = {10.48550/arXiv.2602.05258},
  url = {https://arxiv.org/abs/2602.05258}
}

@inproceedings{wertheimer2026frayed,
  title = {Frayed {RoPE} and Long Inputs: A Geometric Perspective},
  author = {Wertheimer, Davis and Zhang, Aozhong and Liu, Derrick and Yin, Penghang and Wang, Naigang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2026},
  url = {https://openreview.net/forum?id=W8ZXfNaqku},
  note = {arXiv:2603.18017}
}

@inproceedings{oka2026fmrope,
  title     = {Frequency Bands in {RoPE}: Base Frequency and Context Length Shape the Interpolation--Extrapolation Trade-off},
  author    = {Oka, Yui and Saito, Itsumi and Nishida, Kyosuke and Saito, Kuniko},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=PR1PPxvG9Q}
}

@inproceedings{oka2026frequencyentropy,
  title     = {Probing Rotary Position Embeddings through Frequency Entropy},
  author    = {Oka, Yui and Hanafusa, Kentaro and Hasegawa, Taku and Nishida, Kyosuke and Saito, Kuniko},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=1JZuEDq62N}
}

@inproceedings{urrutia2026decoupling,
  title     = {Decoupling Positional and Symbolic Attention Behavior in Transformers},
  author    = {Urrutia, Felipe and Salas, Jorge and Kozachinskiy, Alexander and Buc Calderon, Cristian and Pasten, Hector and Rojas, Cristobal},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=V38yAoqddQ}
}

@inproceedings{zhang2026grape,
  title     = {Group Representational Position Encoding},
  author    = {Zhang, Yifan and Chen, Zixiang and Liu, Yifeng and Qin, Zhen and Yuan, Huizhuo and Xu, Kangping and Yuan, Yang and Gu, Quanquan and Yao, Andrew Chi-Chih},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2026/file/5cb58625f49ddf70fe2d527e9e4bbae5-Paper-Conference.pdf}
}

@inproceedings{tian2026mrrope,
  title     = {{MrRoPE}: Mixed-radix Rotary Position Embedding},
  author    = {Tian, Qingyuan and Zhu, Wenhong and Liu, Xiaoran and Wang, Xiaofeng and Wang, Rui},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2026/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html}
}

@inproceedings{movahedi2026selectiverope,
  title     = {Selective Rotary Position Embedding},
  author    = {Movahedi, Sajad and Carstensen, Timur and Afzal, Arshia and Hutter, Frank and Orvieto, Antonio and Cevher, Volkan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=AQo1SEElNb}
}

@inproceedings{gu2026deconstructing,
  title     = {Deconstructing Positional Information: From Attention Logits to Training Biases},
  author    = {Gu, Zihan and Chen, Ruoyu and Zhang, Han and Zhang, Hua and Hu, Yue},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=D0u0glT060}
}

@inproceedings{kazemnejad2023impact,
  title     = {The Impact of Positional Encoding on Length Generalization in Transformers},
  author    = {Kazemnejad, Amirhossein and Padhi, Inkit and Ramamurthy, Karthikeyan Natesan and Das, Payel and Reddy, Siva},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  year      = {2023},
  doi       = {10.52202/075280-1082},
  url       = {https://openreview.net/forum?id=Drrl2gcjzl}
}

@inproceedings{li2026repo,
  title     = {{RePo}: Language Models with Context Re-Positioning},
  author    = {Li, Huayang and Zhao, Tianyu and Cai, Deng and Sproat, Richard},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  note      = {arXiv:2512.14391},
  url       = {https://icml.cc/virtual/2026/poster/64002}
}

@article{karypis2026lerope,
  title   = {{LeRoPE}: Learnable {RoPE} Frequencies Improve Language Modeling},
  author  = {Karypis, Petros and O'Brien, Sean and Kadekodi, Shreyas and Zhu, Rui and McAuley, Julian},
  journal = {arXiv preprint arXiv:2607.10134},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.10134},
  url     = {https://arxiv.org/abs/2607.10134}
}

@article{xiong2025dope,
  title   = {{DoPE}: Denoising Rotary Position Embedding},
  author  = {Xiong, Jing and Fan, Liyang and Shen, Hui and Su, Zunhai and Yang, Min and Kong, Lingpeng and Wong, Ngai},
  journal = {arXiv preprint arXiv:2511.09146},
  year    = {2025},
  doi     = {10.48550/arXiv.2511.09146},
  url     = {https://arxiv.org/abs/2511.09146}
}

@article{du2026distinguishes,
  title   = {{RoPE} Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably},
  author  = {Du, Yufeng and Harris, Phillip and Tian, Minyang and Huerta, Eliu A. and Ronanki, Srikanth and Rongali, Subendhu and Galstyan, Aram and Peng, Hao},
  journal = {arXiv preprint arXiv:2605.15514},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.15514},
  url     = {https://arxiv.org/abs/2605.15514}
}

@inproceedings{wang2026adarope,
  title     = {{AdaRoPE}: Not All Attention Heads Should Rotate and Scale Equally},
  author    = {Wang, Shaowen and Zheng, Yuke and Zhu, Tansheng and Chen, Shuang and Liu, Shaofan and Zheng, Suncong and Li, Jian},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  note      = {arXiv:2607.19363},
  url       = {https://icml.cc/virtual/2026/poster/60704}
}

@article{wu2026datashapes,
  title   = {How Data Shapes {RoPE} Frequency Usage: From Positional Scale Matching to Length Generalization},
  author  = {Wu, Xinyi and Liu, Siyuan and Jadbabaie, Ali},
  journal = {arXiv preprint arXiv:2607.07678},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.07678},
  url     = {https://arxiv.org/abs/2607.07678}
}

@article{tang2026jetlong,
  title   = {{Jet-Long}: Efficient Long-Context Extension with Dynamic Bifocal {RoPE}},
  author  = {Tang, Haozhan and Wang, Zerui and Gu, Yuxian and Han, Song and Cai, Han},
  journal = {arXiv preprint arXiv:2607.07740},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.07740},
  url     = {https://arxiv.org/abs/2607.07740}
}

@article{olmo2furious,
  title   = {2 {OLMo} 2 Furious},
  author  = {{Team OLMo} and Walsh, Pete and Soldaini, Luca and Groeneveld, Dirk and Lo, Kyle and Arora, Shane and Bhagia, Akshita and Gu, Yuling and Huang, Shengyi and Jordan, Matt and Lambert, Nathan and Schwenk, Dustin and Tafjord, Oyvind and Anderson, Taira and Atkinson, David and Brahman, Faeze and Clark, Christopher and Dasigi, Pradeep and Dziri, Nouha and Ettinger, Allyson and Guerquin, Michal and Heineman, David and Ivison, Hamish and Koh, Pang Wei and Liu, Jiacheng and Malik, Saumya and Merrill, William and Miranda, Lester James V. and Morrison, Jacob and Murray, Tyler and Nam, Crystal and Poznanski, Jake and Pyatkin, Valentina and Rangapur, Aman and Schmitz, Michael and Skjonsberg, Sam and Wadden, David and Wilhelm, Christopher and Wilson, Michael and Zettlemoyer, Luke and Farhadi, Ali and Smith, Noah A. and Hajishirzi, Hannaneh},
  journal = {arXiv preprint arXiv:2501.00656},
  year    = {2025},
  doi     = {10.48550/arXiv.2501.00656},
  url     = {https://arxiv.org/abs/2501.00656}
}

@inproceedings{hu2022lora,
  title     = {{LoRA}: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  url       = {https://openreview.net/forum?id=nZeVKeeFYf9}
}

@inproceedings{liu2024scaling,
  title     = {Scaling Laws of {RoPE}-based Extrapolation},
  author    = {Liu, Xiaoran and Yan, Hang and An, Chenxin and Qiu, Xipeng and Lin, Dahua},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  url       = {https://openreview.net/forum?id=JO7k0SJ5V6},
  note      = {arXiv:2310.05209}
}

@inproceedings{chiang2025rotary,
  title     = {The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval},
  author    = {Chiang, Ting-Rui and Yogatama, Dani},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {13552--13562},
  year      = {2025},
  doi       = {10.18653/v1/2025.findings-acl.697},
  url       = {https://aclanthology.org/2025.findings-acl.697/}
}

@inproceedings{black2022gptneox,
  title     = {{GPT}-{N}eo{X}-20{B}: An Open-Source Autoregressive Language Model},
  author    = {Black, Sidney and Biderman, Stella and Hallahan, Eric and Anthony, Quentin and Gao, Leo and Golding, Laurence and He, Horace and Leahy, Connor and McDonell, Kyle and Phang, Jason and Pieler, Michael and Prashanth, Usvsn Sai and Purohit, Shivanshu and Reynolds, Laria and Tow, Jonathan and Wang, Ben and Weinbach, Samuel},
  booktitle = {Proceedings of BigScience Episode \#5 -- Workshop on Challenges \& Perspectives in Creating Large Language Models},
  pages     = {95--136},
  year      = {2022},
  doi       = {10.18653/v1/2022.bigscience-1.9},
  url       = {https://aclanthology.org/2022.bigscience-1.9/}
}

@inproceedings{merity2017wikitext,
  title     = {Pointer Sentinel Mixture Models},
  author    = {Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017},
  url       = {https://openreview.net/forum?id=Byj72udxe},
  note      = {arXiv:1609.07843}
}

@inproceedings{ho2020twowiki,
  title     = {Constructing A Multi-hop {QA} Dataset for Comprehensive Evaluation of Reasoning Steps},
  author    = {Ho, Xanh and Duong Nguyen, Anh-Khoa and Sugawara, Saku and Aizawa, Akiko},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
  pages     = {6609--6625},
  year      = {2020},
  doi       = {10.18653/v1/2020.coling-main.580},
  url       = {https://aclanthology.org/2020.coling-main.580/}
}

@inproceedings{dasigi2021qasper,
  title     = {A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers},
  author    = {Dasigi, Pradeep and Lo, Kyle and Beltagy, Iz and Cohan, Arman and Smith, Noah A. and Gardner, Matt},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages     = {4599--4610},
  year      = {2021},
  doi       = {10.18653/v1/2021.naacl-main.365},
  url       = {https://aclanthology.org/2021.naacl-main.365/}
}

@article{gray1998quantization,
  title   = {Quantization},
  author  = {Gray, Robert M. and Neuhoff, David L.},
  journal = {IEEE Transactions on Information Theory},
  volume  = {44},
  number  = {6},
  pages   = {2325--2383},
  year    = {1998},
  doi     = {10.1109/18.720541}
}

@inproceedings{ji2025mha2mla,
  title={Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs},
  author={Ji, Tao and Guo, Bin and Wu, Yuanbin and Guo, Qipeng and Shen, Lixing and Chen, Zhan and Qiu, Xipeng and Zhang, Qi and Gui, Tao},
  booktitle={Proceedings of ACL},
  year={2025},
  url={https://arxiv.org/abs/2502.14837}
}
@article{gemmateam2024gemma,
  title={Gemma: Open Models Based on Gemini Research and Technology},
  author={{Gemma Team} and others},
  journal={arXiv preprint arXiv:2403.08295},
  year={2024},
  url={https://arxiv.org/abs/2403.08295}
}

@article{chen2023longlora,
  title={LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models},
  author={Chen, Yukang and Qian, Shengju and Tang, Haotian and Lai, Xin and Liu, Zhijian and Han, Song and Jia, Jiaya},
  journal={arXiv preprint arXiv:2309.12307},
  year={2023},
  doi={10.48550/arXiv.2309.12307},
  url={https://arxiv.org/abs/2309.12307}
}


\newpage
\appendix

<!-- FILE: appendix/a1_proofs.tex -->

\section{Proof Details}
\label{sec:proofs}

\FloatBarrier
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

\paragraph{Basis invariance.}
The cancellation can be stated exactly. Under independent invertible basis
changes $x_\omega\mapsto x_\omega A_\omega$ and
$x_\nu\mapsto x_\nu A_\nu$, let
$U_j=S_j^{1/2}A_j(A_j^\top S_j A_j)^{-1/2}$ for $j\in\{\omega,\nu\}$. Then
$U_j^\top U_j=I$, and the transformed whitened cross-Gram is
$\widetilde Q_{\omega\nu}=U_\omega^\top Q_{\omega\nu}U_\nu$. Left and right
orthogonal factors preserve the two canonical correlations and therefore
$c_{\omega\nu}$. Thus $c_{\omega\nu}$ is an intrinsic measure of overlap
between the two positional subspaces.

\begin{theorem}[Spectral budget identity]
\label{thm:budget}
For the block-whitened Gram matrix $\Gamma$ with diagonal blocks $I_2$,
$\operatorname{tr}\Gamma=2K$ and
$\operatorname{tr}(\Gamma^2)=2K[1+(K-1)\bar c]$. Hence its
R\'enyi-$2$ effective rank is given by \eqref{eq:budget-identity}.
\end{theorem}

\begin{proof}[Proof of Theorem~\ref{thm:budget}]
$\Gamma$ is the block matrix with blocks $\Gamma_{ij}=Q_{\omega_i\omega_j}$ and
$\Gamma_{ii}=I_2$. Hence $\operatorname{tr}\Gamma=\sum_{i=1}^K\operatorname{tr}I_2=2K$.
For the second moment,
$\operatorname{tr}(\Gamma^2)=\sum_{i,j}\operatorname{tr}(\Gamma_{ij}\Gamma_{ji})
=\sum_i\operatorname{tr}(I_2)+\sum_{i\neq j}\|Q_{\omega_i\omega_j}\|_F^2
=2K+2K(K-1)\bar c$, using $\Gamma_{ji}=\Gamma_{ij}^\top$ and the definition
$\bar c=\frac{1}{K(K-1)}\sum_{i\neq j}c_{\omega_i\omega_j}$ with
$\|Q_{ij}\|_F^2=2c_{ij}$. Dividing gives \eqref{eq:budget-identity}.
\end{proof}

The identity determines R\'enyi-$2$ rank exactly from the mean pairwise
redundancy. Shannon effective rank and log-determinant retain higher-order
multi-subspace dependence on the full spectrum.

\begin{proof}[Proof of Proposition~\ref{prop:collapse}]
Put $t=\Delta/L$, $x=\omega L$, and use $\theta=x^2$ as the local parameter.
The rescaled pair has the analytic basis
\begin{align}
u_1(\theta,t)&=\cos(\sqrt\theta\,t)
 =1-\tfrac12\theta t^2+O(\theta^2),\\
u_2(\theta,t)&=\frac{\sin(\sqrt\theta\,t)}{\sqrt\theta}
 =t-\tfrac16\theta t^3+O(\theta^2).
\label{eq:slow-basis-expansion}
\end{align}
Thus $V_\omega\to V_0=\operatorname{span}\{1,\Delta\}$.  The fourth-order
coefficient follows by retaining the part of the first perturbation transverse
to $V_0$. Under the
uniform measure on $t\in[0,1]$,
\begin{align}
\Pi_0^\perp t^2&=p_2(t):=t^2-t+\tfrac16,\\
\Pi_0^\perp t^3&=p_3(t):=t^3-\tfrac9{10}t+\tfrac15,
\label{eq:slow-residual-polynomials}
\end{align}
where $\Pi_0^\perp$ is the $L_2[0,1]$ projection off
$\operatorname{span}\{1,t\}$.  In the base $[1,t]$, its Gram and the residual
perturbation Gram are
\begin{equation}
G_0=
\begin{bmatrix}1&1/2\\[1pt]1/2&1/3\end{bmatrix},\qquad
M=
\begin{bmatrix}
\langle-p_2/2,-p_2/2\rangle&\langle-p_2/2,-p_3/6\rangle\\
\langle-p_3/6,-p_2/2\rangle&\langle-p_3/6,-p_3/6\rangle
\end{bmatrix}
=\begin{bmatrix}1/720&1/1440\\[1pt]1/1440&1/2800\end{bmatrix}.
\label{eq:slow-residual-gram}
\end{equation}
The squared chordal distance between the two nearby planes is therefore
\begin{equation}
2-\lVert Q_{x,y}\rVert_F^2
=(x^2-y^2)^2\operatorname{tr}(G_0^{-1}M)+O(\epsilon^6)
=\frac{19}{12600}(x^2-y^2)^2+O(\epsilon^6),
\label{eq:slow-collapse-expanded}
\end{equation}
with $\epsilon=\max\{|x|,|y|\}\to0$.  The exact coefficient is reproduced
with rational polynomial inner products by
\texttt{scripts/analysis/full\_rope\_collision\_audit.py}; at
$x{=}0.05,y{=}0.10$ its analytic Gram gives exact/leading ratio $1.00058$.

For the softmax metric with fixed $p$,
$F=\operatorname{diag}(p)-pp^\top$ satisfies $F\mathbf1=0$, so the constant
direction is annihilated before whitening.  Writing
$\overline h=h-\mathbb E_p h$, the same expansion gives
\begin{equation}
\frac{\overline{\sin(\omega\Delta)}}{\omega}
 \longrightarrow \Delta-\mathbb E_p\Delta,
\qquad
-\frac{2\overline{\cos(\omega\Delta)}}{\omega^2}
 \longrightarrow \Delta^2-\mathbb E_p\Delta^2.
\end{equation}
These functions span the centred limit whenever $p$ has nondegenerate support
on at least three distances. The separation weights $p$ therefore determine
which polynomial directions survive centering.
\end{proof}

\paragraph{Directional overlap and feature scale.}
For the original, unwhitened pair on $[0,L]$, put $x=\omega L$ and
$\operatorname{sinc}x=\sin x/x$. The self-Gram eigenvalues are
\[
\lambda_\pm(S_\omega)=\frac{1\pm|\operatorname{sinc}x|}{2},
\qquad \lambda_-(S_\omega)\sim\frac{x^2}{12},\qquad
\kappa(S_\omega)\sim\frac{12}{x^2}\quad(x\to0).
\]
Thus the slow pair has a poorly scaled second direction before normalization.
Block whitening removes this within-pair scaling and measures overlap between
positional directions. Raw feature energy, coefficient magnitude, and numerical
precision determine how strongly those directions can contribute to a model.

The standard $u_k=k/K$ grid gives $23$ slow pairs in
Figure~\ref{fig:frequency-geometry}; the endpoint-inclusive grid used in
Figure~\ref{fig:spectral-budget-scaling} gives $24$ at $K=64$.
Both figures use the uniform separation measure.

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.94\linewidth]{fig_frequency_geometry.pdf}
  \caption{\textbf{Where a finite frequency table spends its budget.}
  (a)~At $K{=}32$, Cosh concentrates interior channels toward the fast end
  while endpoint normalisation preserves sampled support; the deployed
  midpoint table is shown separately.
  (b)~For standard RoPE at $L{=}4096$, $b{=}5{\times}10^5$, and $K{=}64$,
  phase-invariant full-subspace redundancy concentrates in the $23$ pairs with
  $\omega L\le1$.}
  \label{fig:frequency-geometry}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.96\linewidth]{fig_spectral_budget_scaling.pdf}
  \caption{\textbf{Spectral-budget anatomy across channel counts.}
  Panel (a) uses the endpoint-inclusive diagnostic grid at $L{=}4096$ and
  $b{=}500$K: as $K$ grows from $16$ to $64$, the nominal dimension assigned to
  pairs with $\omega L\le1$ grows from $12$ to $48$, while their phase-invariant
  effective rank remains approximately two. Panel (b) evaluates the complete
  fixed-endpoint table under the same uniform separation prior, using the
  zero-search reference $\tau{=}2K/\sqrt{L}$; anchored \evq{} spends
  progressively more of the available full-pair dimension than Geo as the
  budget grows.}
  \label{fig:spectral-budget-scaling}
\end{figure}

\FloatBarrier
\subsection{Effective dimension and multiscale coverage}
\label{sec:rank-coverage}

The budget identity is an exact account of positional dimension under its
separation prior. A second requirement is where that dimension is placed across
scales.  The distinction is already visible in an exact construction that also
identifies the complete parity class omitted by the usual $2\pi/L$ Fourier
description.

\begin{proposition}[Parity-lattice orthogonality and recurrence]
\label{prop:parity-lattice}
Let $a_1,\ldots,a_K$ be distinct positive integers of one parity and set
\begin{equation}
\omega_k=\frac{\pi a_k}{L}.
\label{eq:parity-lattice}
\end{equation}
Under $\Delta\sim\mathrm{Unif}[0,L]$, the $K$ full sin/cos subspaces are
exactly mutually orthogonal: their block-whitened Gram is
$\Gamma=I_{2K}$ and $r_2(\Gamma)=2K$.  If support is free but restricted to
$0<\omega\le1$, the largest same-parity class in this construction has
\begin{equation}
K_{\mathrm{par}}(L)
=\left\lceil\frac{\lfloor L/\pi\rfloor}{2}\right\rceil
\label{eq:parity-capacity}
\end{equation}
pairs.  The resulting feature map is recurrent: for even $a_k$,
$\Phi_\Omega(\Delta+L)=\Phi_\Omega(\Delta)$; for odd $a_k$,
$\Phi_\Omega(\Delta+L)=-\Phi_\Omega(\Delta)$ and hence
$\Phi_\Omega(\Delta+2L)=\Phi_\Omega(\Delta)$.
\end{proposition}

\begin{proof}
For $i\ne j$, same parity makes both $a_i-a_j$ and $a_i+a_j$ nonzero even
integers.  Consequently
\begin{equation}
(\omega_i\!\mp\!\omega_j)L=\pi(a_i\!\mp\!a_j)\in2\pi\mathbb Z,
\end{equation}
so every entry of the cross-Gram in~\eqref{eq:cross-gram} vanishes:
$a((\omega_i\mp\omega_j)L)=0$ and
$b_\star((\omega_i\mp\omega_j)L)=0$.  Likewise
$2\omega_iL=2\pi a_i$ gives $S_{\omega_i}=\tfrac12I_2$.
Thus the unwhitened Gram is $\tfrac12I_{2K}$ and block whitening gives
$\Gamma=I_{2K}$.  Among the integers
$1,\ldots,\lfloor L/\pi\rfloor$, the larger parity class has
$\lceil\lfloor L/\pi\rfloor/2\rceil$ elements, proving
\eqref{eq:parity-capacity} for this construction.  Finally,
$e^{i\omega_k(\Delta+L)}=(-1)^{a_k}e^{i\omega_k\Delta}$; the common parity
gives the stated period or antiperiod, and every antiperiodic table repeats
after $2L$.
\end{proof}

Proposition~\ref{prop:parity-lattice} couples maximal rank on $[0,L]$ with
recurrence beyond that interval. Its orthogonality also depends on the
separation measure: weighting the same lattice by the triangular causal
histogram introduces off-diagonal Gram terms. Thus interval coverage and
pairwise rank capture different features of a positional basis.

\FloatBarrier
\subsection{Two static ordering counterexamples}
\label{sec:static-counterexamples}

Two explicit examples show how phase coupling and interval length change
the ordering of frequency tables.

\paragraph{Cosine-only collision can choose the lower-rank table.}
Take $K=4$, $L=8\pi$, and the two descending frequency tables
\[
\Omega_A=(1,5/8,3/8,1/4),\qquad
\Omega_B=(1,6.01/8,4.01/8,1/4).
\]
They share both endpoints. Define $C_{\cos}$ as the mean squared normalized
cosine--cosine overlap over distinct pairs. For $\Omega_A$, every cosine is
an integer harmonic on $[0,L]$, so $C_{\cos}(A)=0$ exactly. Mixed harmonic
parities nevertheless produce nonzero sine--cosine cross terms. For
$\Omega_B$, the small perturbation of an even-harmonic table gives
\begin{align}
C_{\cos}(A)&=0 < 1.38725\times10^{-5}=C_{\cos}(B),\\
r_2(A)&=6.29478 < 7.99971=r_2(B).
\label{eq:cosine-order-counterexample}
\end{align}
Thus a strict cosine-only preference can have the opposite full-subspace
rank ordering. Both values follow directly from Eq.~\eqref{eq:cross-gram}.

\paragraph{Full-subspace collision can reverse across lengths.}
Now use $K=4$, $L=2\pi$, and the fixed tables
\[
\Omega_A=(1,0.99,0.02,0.01),\qquad
\Omega_B=(1,2/3,1/3,0.01).
\]
Again the endpoints agree. Writing $C_M$ for mean phase-invariant collision
under the uniform measure on $[0,M]$, Eq.~\eqref{eq:cross-gram} gives
\begin{align}
C_L(A)&=0.538996 < 0.631721=C_L(B),\\
C_{2L}(A)&=0.384445 > 0.204995=C_{2L}(B),\\
C_{4L}(A)&=0.345188 > 0.033506=C_{4L}(B).
\label{eq:length-order-counterexample}
\end{align}
The close pairs in $A$ retain high overlap as the interval grows, while the
more evenly spread frequencies in $B$ become easier to distinguish. The
example makes the interval dependence of the geometry explicit.

All frequency entries and measures are specified above. The accompanying
\path{figs/verify_explicit_geometry.py} evaluates the closed-form Gram and
checks it against independent $128$-point Gauss--Legendre quadrature;
the largest discrepancy in the displayed metrics is below $4\times10^{-14}$.
These finite basis examples complement the trained allocation comparisons and
the weight--table crossings.

\FloatBarrier
\subsection{Post-hoc transplant obstruction}
\label{sec:obstruction-proof}

<!-- FILE: tables/table_coadapt.tex -->

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{0.95}
\caption{\textbf{Weights co-adapt to their training-time frequency table.} Two
seed-$42$ $50$M models trained with the geometric and \evq{} tables, each
evaluated under both runtime tables with all parameters frozen ($1{,}920$
head--query observations). Both runtime tables use the shared midpoint grid
$u_k=(k+1/2)/K$ for this diagnostic. The matched pairs remain near PPL $7$, whereas both
cross-swaps fail; the worst cell is also the one whose static $r_2$ improves
most.}
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


\begin{theorem}[Post-hoc transplant obstruction]
\label{thm:obstruction}
Let $\mathcal R_\Omega(\Delta)$ be the block-rotation operator of a frequency
multiset $\Omega$. If position-independent invertible maps $A,B$ satisfy
$A^\top\mathcal R_{\Omega'}(\Delta)B=\mathcal R_\Omega(\Delta)$ on an open
interval containing zero, then $\Omega'$ and $\Omega$ have the same frequency
multiset up to sign and permutation. Repeated frequencies may mix within their
full equal-frequency invariant subspace.
\end{theorem}

\begin{proof}[Proof of Theorem~\ref{thm:obstruction}]
Both $\mathcal R_\Omega$ and $\mathcal R_{\Omega'}$ are block-diagonal with $2\times2$ rotation
blocks, hence $\mathcal R_\Omega(0)=\mathcal R_{\Omega'}(0)=I$. Evaluating the hypothesis at
$\Delta{=}0$ gives $A^\top B=I$, so $B=A^{-\top}$ and the hypothesis becomes the
similarity $A^\top \mathcal R_{\Omega'}(\Delta)A^{-\top}=\mathcal R_\Omega(\Delta)$ on an interval.
Each side is the exponential of a constant generator, $\mathcal R_\Omega(\Delta)=\exp(\Delta G_\Omega)$
with $G_\Omega=\bigoplus_k \omega_k J$, $J=\left[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right]$.
Differentiating at $\Delta{=}0$ gives $A^\top G_{\Omega'}A^{-\top}=G_\Omega$, so
the generators are similar and share a spectrum. The spectrum of $G_\Omega$ is
$\{\pm i\omega_k\}_{k\le K}$ with multiplicity, so the frequency multisets agree
up to sign and permutation. When a frequency is repeated, similarity may mix
the entire equal-frequency invariant subspace, including across its original
two-dimensional blocks. For integer positions the same argument
is replaced by a one-step spectral argument: evaluating at $\Delta=0$ again
gives $B=A^{-\top}$, while $\Delta=1$ makes $\mathcal R_{\Omega'}(1)$ and
$\mathcal R_\Omega(1)$ similar. Their eigenvalue multisets
$\{e^{\pm i\omega_k}\}$ therefore agree, which identifies frequencies up to
sign, permutation, and the $2\pi$ alias.
\end{proof}

\FloatBarrier
\subsection{Proof of the closed-form variational optimum}

This subsection proves Theorem~\ref{thm:ode} directly from the stated
variational objective. Starting from the kernel
\begin{equation}
K_{\mathrm{app}}(\phi,\psi)=\alpha\delta(\phi-\psi)+\beta\min(\phi,\psi),
\end{equation}
the allocation functional is~\eqref{eq:Capp}, which under the mass normalization $\int_0^1\rho=1$ with Lagrange multiplier $\nu$ has constrained first variation
\begin{equation}
\alpha\rho(\phi) + \beta\,g(\phi) + \nu = 0,\qquad
g(\phi) = \int_0^1 \rho(\psi)\min(\phi,\psi)\,d\psi.
\end{equation}
\paragraph{Existence.}
For rigor, view $\mathcal{C}_{\mathrm{app}}$ first as a functional on $L^2([0,1])$ and minimize over $\mathcal{A}=\{\rho\in L^2([0,1]):\rho\ge0\ {\rm a.e.},\int_0^1\rho=1\}$. The $\alpha\|\rho\|_2^2$ term is coercive and weakly lower semicontinuous, the Green-kernel term is positive semidefinite and continuous, and the mass/nonnegativity constraints are weakly closed. A minimizer therefore exists by the direct method. Strict convexity gives uniqueness. The Euler solution derived below is strictly positive, so the nonnegativity constraint is inactive a posteriori and the classical $C^2$ derivation is justified.

Direct differentiation of $g$ gives
\begin{align}
g'(\phi)
&=
\int_\phi^1 \rho(\psi)\,d\psi,
&
g''(\phi)
&=
-\rho(\phi).
\end{align}
Differentiating the first variation twice in $\phi$ eliminates $\nu$ (a constant) and yields the homogeneous ODE
\begin{equation}
\rho''(\phi) - \tau^2\rho(\phi) = 0,\qquad \tau = \sqrt{\beta/\alpha}.
\label{eq:homogeneous-ode}
\end{equation}
Evaluating the pre-differentiated stationarity together with $g(0){=}0$, $g'(1){=}0$, and $\int_0^1\rho = 1$ gives the derivative boundary conditions
\begin{equation}
\rho'(0) = -\tau^2,\qquad \rho'(1) = 0,
\label{eq:bcs}
\end{equation}
since differentiating $\alpha\rho(\phi)+\beta g(\phi)+\nu = 0$ once and evaluating at $\phi{=}1$ gives $\alpha\rho'(1)+\beta g'(1) = 0$ hence $\rho'(1){=}0$, and at $\phi{=}0$ gives $\alpha\rho'(0)+\beta g'(0) = 0$ with $g'(0) = \int_0^1\rho = 1$, so $\rho'(0) = -\beta/\alpha = -\tau^2$. Substituting $\rho(\phi) = C_1\cosh(\tau\phi)+C_2\sinh(\tau\phi)$ into~\eqref{eq:homogeneous-ode} and~\eqref{eq:bcs} together with $\int_0^1\rho = 1$ gives the unique positive interior stationary density
\begin{equation}
\rho_\tau(\phi) \;=\; \frac{\tau\,\cosh(\tau(1-\phi))}{\sinh\tau},
\label{eq:rho-tau-closed}
\end{equation}
which is main-text Theorem~\ref{thm:ode}. Mass normalization enters the
boundary condition through $g'(0)=\int_0^1\rho=1$. Positivity holds on the closed interval since $\rho_\tau(\phi){\geq}\rho_\tau(1){=}\tau/\sinh\tau{>}0$ for $\tau{>}0$ (with $\rho_\tau(0){=}\tau\coth\tau{>}0$); the inequality constraint $\rho{>}0$ is therefore inactive throughout $[0,1]$ and the unconstrained Euler--Lagrange solution coincides with the KKT-constrained solution. For the degenerate case $\beta{=}0$ the ODE reduces to $\rho''{=}0$ with $\rho'(0){=}\rho'(1){=}0$ and $\int\rho{=}1$, forcing $\rho\equiv 1$, which is also the $\tau{\to}0$ limit of~\eqref{eq:rho-tau-closed}.

\paragraph{Convexity / PSD of $\mathcal{C}_{\mathrm{app}}$.} For $\alpha{>}0$ and $\beta{\geq}0$, the surrogate is convex because both quadratic terms are positive semidefinite. The diagonal $\alpha{\int}\rho^2$ is manifestly PSD on $L^2$, and the Green-kernel cross term satisfies the identity
\begin{equation}
\iint_{[0,1]^2} f(\phi)\,f(\psi)\,\min(\phi,\psi)\,d\phi\,d\psi \;=\; \int_0^1 \!\left(\int_s^1 f(u)\,du\right)^{\!2}\!ds \;\geq\; 0.
\label{eq:min-kernel-psd}
\end{equation}
The strictly convex functional on $L^2([0,1])$ therefore has the displayed
stationary density as its unique constrained minimizer.

The Cosh solution has an analytically invertible CDF, so its quantiles turn
the continuous allocation directly into a finite frequency table.

\FloatBarrier
\subsection{Single-crossing budget shift}
\label{sec:budget-crossing-proof}

\begin{lemma}[Single-crossing budget shift]
\label{lem:budget-crossing}
For every $\tau>0$, $\rho_\tau$ crosses the uniform density exactly once, at
\begin{equation}
\phi_c(\tau)=1-\tau^{-1}\operatorname{arcosh}(\sinh\tau/\tau)
\le 1-1/\sqrt3.
\label{eq:budget-crossing}
\end{equation}
\end{lemma}

\begin{proof}[Proof of Lemma~\ref{lem:budget-crossing}]
For $\tau>0$, $\rho_\tau(\phi)=\tau\cosh(\tau(1-\phi))/\sinh\tau$ is
strictly decreasing on $[0,1)$, with
$\rho_\tau(0)=\tau\coth\tau>1$ and
$\rho_\tau(1)=\tau/\sinh\tau<1$.  Hence the crossing is unique, and solving
$\rho_\tau(\phi_c)=1$ gives~\eqref{eq:budget-crossing}.  To bound it, expand
\[
\frac{\sinh\tau}{\tau}
=\sum_{n\ge0}\frac{\tau^{2n}}{(2n+1)!}
\ge \sum_{n\ge0}\frac{\tau^{2n}}{3^n(2n)!}
=\cosh\!\left(\frac{\tau}{\sqrt3}\right),
\]
where the coefficient-wise inequality is $3^n\ge2n+1$.  Monotonicity of
$\operatorname{arcosh}$ yields
$\operatorname{arcosh}(\sinh\tau/\tau)\ge\tau/\sqrt3$, proving the bound.
\end{proof}

\FloatBarrier
\subsection{Surrogate self-consistency theorem}
\label{sec:self-consistency}

The surrogate $\mathcal{C}_{\mathrm{app}}$ contains two integral terms $T_1(\tau) = \int_0^1 \rho^2\,d\phi$ and $T_2(\tau) = \int_0^1\!\!\int_0^1 \rho(\phi)\rho(\psi)\min(\phi,\psi)\,d\phi\,d\psi$, where $\rho(\phi) = \tau\cosh(\tau(1{-}\phi))/\sinh\tau$ is the normalized cosh density.

\begin{theorem}[Surrogate self-consistency]
\label{thm:self-consistency}
For all $\tau > 0$:
\begin{equation}
\tau^2 T_2(\tau) + T_1(\tau) = \tau\coth\tau.
\label{eq:self-consistency}
\end{equation}
\end{theorem}

\begin{proof}
The Green's function $g(\phi) = \int_0^1 \rho(\psi)\min(\phi,\psi)\,d\psi$ satisfies $g''(\phi) = -\rho(\phi)$ with $g(0) = 0$, $g'(1) = 0$ (from the $\min$ kernel structure; see~\S\ref{sec:proofs}). Define $h(\phi) = \rho(0) - \rho(\phi)$. Then $h'' = -\rho''$, and under the pure-tether ODE $\rho'' = \tau^2\rho$ this gives $h'' = -\tau^2\rho$, with $h(0) = 0$ and $h'(0) = -\rho'(0)$.

Define $f(\phi) = \tau^2 g(\phi) - h(\phi)$. Then $f'' = \tau^2 g'' - h'' = -\tau^2\rho - (-\tau^2\rho) = 0$, so $f$ is linear in $\phi$. The boundary values give $f(0) = \tau^2 g(0) - h(0) = 0$ and $f'(0) = \tau^2 g'(0) + \rho'(0) = \tau^2 - \tau^2 = 0$ (using $g'(0) = \int_0^1\rho = 1$ and $\rho'(0) = -\tau^2$ from the pure-tether density). Hence $f \equiv 0$, yielding $\tau^2 g(\phi) = h(\phi) = \rho(0) - \rho(\phi)$, i.e.,
\begin{equation}
g(\phi) = \frac{\rho(0) - \rho(\phi)}{\tau^2}.
\label{eq:green-identity}
\end{equation}
Integrating against $\rho$:
\begin{equation}
T_2 = \int_0^1 \rho(\phi)\,g(\phi)\,d\phi = \frac{1}{\tau^2}\bigl[\rho(0)\underbrace{\int_0^1\!\rho}_{=1} - T_1\bigr]
= \frac{\rho(0) - T_1}{\tau^2}.
\end{equation}
Since $\rho(0) = \tau\cosh\tau/\sinh\tau = \tau\coth\tau$, rearranging gives~\eqref{eq:self-consistency}.
\end{proof}

An immediate corollary follows.

\begin{corollary}[Closed-form $T_2$]
\begin{equation}
T_2(\tau) = \frac{\sinh 2\tau - 2\tau}{4\tau\sinh^2\tau}.
% \label{eq:T2-closed} % unused
\end{equation}
\end{corollary}

This follows by substituting
$T_1(\tau)=\tau^2(1+\sinh(2\tau)/(2\tau))/(2\sinh^2\tau)$ and
$\rho(0)=\tau\coth\tau$ into the theorem.

The identity provides a closed-form evaluation of the allocation energy.

\FloatBarrier
\subsection{Allocation strength and a reference rule}
\label{sec:lambda-cv}
\label{sec:tau-scaling}

The Cosh family varies with the strength $\tau$. A reference used in the
full-RoPE text experiments relates this strength to head width and training
length:
\begin{equation}
\tau = c\,\dhd/\sqrt{\Ltr},\qquad
u_k = (k+\tfrac12)/K,\qquad
\omega_k^{\mathrm{EVQ}} = b^{-\phi_k(\tau)} .
\label{eq:evq-practical}
\end{equation}
The reference sets $c=1$ before training. Table~\ref{tab:allocation-protocols}
records the strength, grid, and setting procedure used in each experiment.

\paragraph{Local scaling calculation.}
The scaling argument uses four explicit modelling assumptions: (i) the
pure-tether Cosh family is varied only through $\tau$; (ii) full-RoPE MHA has
$d_{\mathrm{rot}}=\dhd$; (iii) the reference post-softmax distribution is
diffuse, $p_0=1/L$; and (iv) a fixed trade-off $\lambda>0$ compares a normalised
channel-load stiffness with a specified phase-variance utility, with
$Q_1(L,b)>0$.  Under those assumptions and at small $\tau$,
\begin{align}
S_{\chi^2}(\tau)
&:=\frac1{\dhd}\int_0^1\frac{(1-\rho_\tau(\phi))^2}{\rho_\tau(\phi)}\,d\phi
=\frac{\tau^4}{45\dhd}+O(\tau^6),\\
U(\tau,L)
&=\frac{\dhd}{L}\left[Q_0(L,b)+\tau^2Q_1(L,b)+O(\tau^4)\right],
\end{align}
where
\begin{equation}
\begin{aligned}
Q_0(L,b)&=\int_0^1 q(Lb^{-\phi})\,d\phi,\\
Q_1(L,b)&=\int_0^1\!\left(\frac{(1-\phi)^2}{2}-\frac16\right)
q(Lb^{-\phi})\,d\phi,\\
q(x)&=\frac12+\frac{\sin(2x)}{4x}
      -\left(\frac{\sin x}{x}\right)^2.
\end{aligned}
\end{equation}
Stationarity of
$\mathcal F(\tau)=\tfrac12S_{\chi^2}(\tau)-\lambda U(\tau,L)$ gives the
leading small-$\tau$ balance
\begin{equation}
\tau_*^2=45\lambda Q_1(L,b)\frac{\dhd^2}{L},
\qquad
c_{\mathrm{loc}}(L,b,\lambda)=\sqrt{45\lambda Q_1(L,b)}.
\label{eq:tau-leading-balance}
\end{equation}
This calculation yields the scaling factor $\dhd/\sqrt L$ with a local
coefficient determined by $Q_1(L,b)$ and $\lambda$. The experiments evaluate
the exact Cosh quantiles at the protocol's chosen strength.

\paragraph{Reference and neighboring strengths.}
The staged study contains $99$ completed runs: the fixed $c{=}1$ reference
beats the midpoint-discretised Geo baseline under runner-defined weighted
extrapolation NLL in $7/9$ configuration means and $18/27$ paired seeds.  Its
registered neighbours are finite multiplier arms
at $0.75\times$, $1.25\times$, or $1.5\times$; the independent exact-range
factorial tests $0.75\times$, $1.00\times$, and $1.25\times$, and their order
varies by configuration.  The numeric owner is
\texttt{data/curated/phase16\_99run\_manifest.csv}; the tracked reproduction
entrypoint is
\texttt{scripts/core\_text\_phases/phase16\_formula\_optimality\_sweep.py}.
The reference and its neighboring strengths give a direct comparison of
allocation intensity across the tested configurations.

\FloatBarrier
\subsection{Discrete-channel transport gap under finite channel counts}
\label{sec:discrete-continuous-gap}

\paragraph{Setup.} Inverse-CDF quantization replaces the continuous density $\rho$ by an atomic measure. Let $F(\phi){=}\int_0^\phi\rho$, $Q(u){=}F^{-1}(u)$, midpoint grid $u_k{=}(k{-}\tfrac{1}{2})/K$, and $\phi_k{=}Q(u_k)$. The empirical channel measure is $\mu_K{=}K^{-1}\sum_{k=1}^{K}\delta_{\phi_k}$. We compare the measures in Wasserstein distance and approximate the density
with the associated quantile-cell histogram $\rho_K$.

\paragraph{Transport bounds.} Assume $0<m\le\rho\le M$ and $\|\rho'\|_\infty\le B$; then $Q$ is $(1/m)$-Lipschitz. The midpoint quantization errors satisfy
\begin{equation}
W_\infty(\mu_K,\mu_\rho) \;\le\; \frac{1}{2Km}, \qquad W_1(\mu_K,\mu_\rho) \;\le\; \frac{1}{4Km}.
\label{eq:wasserstein-bound}
\end{equation}
Defining the quantile-cell histogram $\rho_K(\phi){=}1/(K|I_k|)$ on $I_k{=}[Q((k{-}1)/K),\,Q(k/K)]$, the histogram density satisfies
\begin{equation}
\|\rho_K-\rho\|_1 \;\le\; \frac{B}{Km}, \qquad \|\rho_K-\rho\|_\infty \;\le\; \frac{B}{Km}.
\label{eq:density-bound}
\end{equation}
The transport bounds couple each quantile cell to its midpoint; the histogram
bound follows from the mean-value theorem and $|I_k|\le 1/(Km)$.

\paragraph{Application to EVQ-Cosh.} For $\rho_\tau(\phi)=\tau\cosh(\tau(1{-}\phi))/\sinh\tau$, $m_\tau=\tau/\sinh\tau$ and $B_\tau=\tau^2$. Substituting,
\begin{equation}
W_1(\mu_{K,\tau},\mu_{\rho_\tau}) \;\le\; \frac{\sinh\tau}{4K\tau}, \qquad \|\rho_{K,\tau}-\rho_\tau\|_1 \;\le\; \frac{\tau\sinh\tau}{K} \;=\; \frac{\tau^2}{K}+O\!\left(\tfrac{\tau^4}{K}\right).
\label{eq:evq-discrete-bound}
\end{equation}

\paragraph{Kernel-integrated errors.} For a smooth two-variable kernel $K_{\mathrm{sm}}$ with Lipschitz constant $L_K$ in each argument,
\begin{equation}
\left|\iint K_{\mathrm{sm}}\,d\mu_K\,d\mu_K - \iint K_{\mathrm{sm}}\,d\mu_\rho\,d\mu_\rho\right| \;\le\; \frac{L_K}{2Km}.
\label{eq:kernel-bound}
\end{equation}
The kernel bound applies the $W_1$ estimate to each argument. The singular
$\delta$-component of $K_{\mathrm{app}}$ is represented by its cell average.

\paragraph{High-resolution distortion.} For midpoint scalar quantisation with
a smooth weighted distortion $w$, the standard high-resolution (Bennett-integral)
expansion \citep{gray1998quantization} is
\begin{equation}
\mathcal{D}_K[\rho] \;=\; \frac{1}{12K^2}\int_0^1 \frac{w(\phi)}{\rho(\phi)^2}\,d\phi + O(K^{-3}).
\label{eq:high-res-distortion}
\end{equation}
This expansion quantifies how finite exponent samples approximate a continuous
allocation under a smooth distortion. For endpoint-anchored comparisons,
$K=2$ leaves no interior entries and the geometric and Cosh grids coincide.

\paragraph{Large-$\tau$ regime.} At large $\tau$,
$m_\tau{=}\tau/\sinh\tau$ becomes small and the $1/(Km_\tau)$ factor grows.
The transport bounds are informative while this factor remains controlled.

<!-- FILE: appendix/a2_experiment_details.tex -->

\section{Experimental Details}
\label{sec:experiment-details}

\FloatBarrier
\subsection{Experiment organization}

This section gives the training, continuation, and video protocols.
Apps.~\ref{sec:identification-details} and~\ref{sec:mature-details} detail
the fixed-range and mature-model experiments.

\FloatBarrier
\subsection{Models and training settings}

Table~\ref{tab:evidence-map} collects the model settings and evaluation units.
Scratch text runs use FineWeb-Edu \citep{penedo2024fineweb}.

\begin{table}[ht]
\caption{Evidence and protocol map. Every row retains its own intervention,
experimental unit, and endpoint.}
\label{tab:evidence-map}
\centering
\small
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}
  >{\raggedright\arraybackslash}p{0.19\linewidth}
  >{\raggedright\arraybackslash}p{0.41\linewidth}
  >{\raggedright\arraybackslash}p{0.30\linewidth}@{}}
\toprule
Role & Model/regime and intervention & Unit, endpoint, and appendix \\
\midrule
Allocation identification & 151.9M scratch; fixed support, $30$ interiors & 3 training seeds; tail NLL; App.~\ref{sec:exact-range-control} \\
Co-adaptation replication & 151.9M frozen crossing; weights $\times$ runtime table & 2 training seeds; tail NLL; App.~\ref{sec:crossing-151m} \\
Shape/configuration breadth & 50.9M factorial; fixed support, 5 allocations & 12 configs $\times$ 3 seeds; weighted OOD NLL; App.~\ref{sec:exact-range-factorial} \\
Scarce rotary budget & 432M MLA-style; Geo/\evq{}, $K{=}16$ & 3 training seeds; PPL; App.~\ref{sec:mla-results} \\
Full-parameter scale & 750M/1.485B; continued/from-init Geo/\evq{} & one comparison each; PPL/retrieval; Apps.~\ref{sec:larger-scale-supporting}/\ref{sec:olmo2-1b} \\
Frozen allocation & OLMo/Qwen; same support and amplitude & 20 rows/task; RULER; App.~\ref{sec:frozen-fixed-support} \\
Zero-training deployment & OLMo; frozen long table, gain, session route & fixed checkpoint/rows; NLL/RULER/F1; App.~\ref{sec:frozen-fixed-support} \\
Adaptation/source use & 1.485B/8B; matched LoRA/source block & one trained pair; QA/RULER/NLL; Apps.~\ref{sec:olmo2-1b}/\ref{sec:llama8b} \\
Static index profile & Qwen-$0.5$B; one table across lengths & 13 tasks, 20 rows/task/length; App.~\ref{sec:index-adjustment} \\
Profile placement & Qwen/Gemma; direct gap versus normalized index & 4 tasks, 80 rows/task; App.~\ref{sec:coordinate-confirmation} \\
Intermediate-band shape & OLMo/Qwen; matched BM/MrPro profiles & 6 tasks; Table~\ref{tab:bm-models}; App.~\ref{sec:bm-model-comparison} \\
Natural-QA transfer & OLMo; matched static BM/MrPro profiles & 778 inputs, 631 long; token F1; App.~\ref{sec:bm-natural} \\
Video modeling & 129.6M video DiT; temporal Geo/\evq{} & 1 training seed; denoising MSE; App.~\ref{sec:video-dit} \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:hyperparams} lists the training hyperparameters shared across
the supporting text experiments.
<!-- FILE: tables/table_allocation_protocols.tex -->

\begin{table}[ht]
\centering\small
\caption{Cosh-family settings used in the reported experiments.
Choosing $\tau$ and a grid specifies a fixed table. Midpoint means
$u_k=(k+1/2)/K$; standard means $u_k=k/K$; anchoring fixes the sampled
endpoints after evaluating the quantiles.}
\label{tab:allocation-protocols}
\begin{tabular}{@{}p{.25\linewidth}p{.14\linewidth}p{.20\linewidth}p{.32\linewidth}@{}}
\toprule
Protocol & $\tau$ & Grid & Setting convention \\
\midrule
151.9M fixed-range & $4$ & Anchored midpoint & Fixed $64/\sqrt{256}$ reference \\
50.9M factorial & Varies & Anchored midpoint & Three preassigned multiples of $d_{\rm head}/\sqrt{L_{\rm train}}$ \\
50M crossing & $2.83$ & Midpoint & Reported training-table setting \\
432M MLA & $1.414$ & Midpoint & Pre-specified architecture-specific setting \\
454M composition & $1.5$ & Midpoint & Reported training recipe \\
750M continuation & $1.5$ & Anchored inclusive & Fixed continuation recipe; $u_k=k/(K-1)$ \\
Llama-3-8B & $1.414$ & Midpoint & Fixed adaptation recipe; approximately $128/\sqrt{8192}$ \\
OLMo-2 & $2$ & Standard & Fixed $128/\sqrt{4096}$ reference \\
129.6M Video DiT & $1.5$ & Midpoint & Best tested value in the video sweep \\
\bottomrule
\end{tabular}
\end{table}


\begin{table}[ht]
\caption{Training hyperparameters for the supporting scratch text experiments.}
\label{tab:hyperparams}
\centering\small
\begin{tabular}{@{}ll@{}}
\toprule
Parameter & Value \\
\midrule
Optimizer & AdamW ($\beta_1{=}0.9$, $\beta_2{=}0.95$, $\epsilon{=}10^{-8}$) \\
Learning rate & $6{\times}10^{-4}$ (50M), $2{\times}10^{-4}$ (MLA/750M) \\
LR schedule & Cosine decay to $0.1\times$ peak LR \\
Warmup & $2\%$ of total steps (linear) \\
Batch size (seqs) & 32 (50M), 14 (750M: micro 7, accumulation 2), 6 (MLA) \\
Weight decay & $0.1$ \\
Gradient clipping & max-norm $1.0$ \\
Dropout & $0.0$ \\
Sequence packing & Contiguous, no cross-document masking \\
\midrule
\multicolumn{2}{@{}l}{\emph{Video DiT (129.6M)}} \\
Optimizer & AdamW ($\beta_1{=}0.9$, $\beta_2{=}0.95$) \\
RoPE base & $10{,}000$ (temporal axis) \\
Activation & GELU (tanh approx.) \\
\midrule
\multicolumn{2}{@{}l}{\emph{MLA (432M)}} \\
LR / warmup & $2{\times}10^{-4}$ / $2\%$ of steps \\
Training tokens & 500M \\
\bottomrule
\end{tabular}
\end{table}

Within each text comparison, the architecture, optimizer, data order, and
training loop are fixed, and the only intervention is the initialized
inverse-frequency table. The scratch-text experiments use base
$b{=}500\mathrm{K}$ unless stated otherwise.

\FloatBarrier
\subsection{Composition with a fixed-index range operator}
\label{sec:range-composition}

The $454.2$M decoder has $24$ layers, width $1024$, $16$ heads,
$d_{\rm head}=64$, and base $b=500{,}000$. Geo ($\tau=0$) and \evq{}
($\tau=1.5$) use midpoint frequency grids and train from initialization
for $100$M tokens per arm at length $2048$, using FineWeb-Edu with a
$10\%$ synthetic-passkey mixture and seeds $42,123,7$.
The same inference-time operator is then applied to both trained tables.

For $k=0,\ldots,K-1$, define
\[
\ell=\lfloor0.20K\rfloor,\qquad h=\lfloor0.90K\rfloor,\qquad
t_k=\operatorname{clip}\!\left(\frac{k-\ell}{h-\ell},0,1\right),\qquad
r_k=t_k^2(3-2t_k).
\]
The repository's fixed-index, YaRN-style smooth-ramp operator installs
\begin{equation}
\mathcal R_s(\omega_k)=
\frac{\omega_k}{s^{r_k}T(s)^{r_k/2}},
\qquad T(s)=1+0.07\log_2s.
\label{eq:repo-range-scaler}
\end{equation}
Here $K=32$ and $s=8$, giving $(\ell,h,T)=(6,28,1.21)$.
The temperature factor is folded into the frequency divisor; cosine/sine
amplitude stays at $g=1$. Equation~\eqref{eq:repo-range-scaler} specifies
the operator used in this experiment, separately from the official-equation
YaRN comparisons in \S\ref{sec:mature-adjustments}.
Table~\ref{tab:evq-repo-ramp} reports the three-seed comparison.

<!-- FILE: tables/table_evq_ramp.tex -->

\begin{table}[ht]
\centering\small
\caption{Matched fixed-ramp composition on the $454$M three-seed experiment.
$\mathcal R_8$ is defined in Eq.~\eqref{eq:repo-range-scaler}.
PPL uses full-sequence scoring; PK is teacher-forced NLL-gap retrieval.
Entries are three-seed means; PK@$8$K also gives the standard deviation.}
\label{tab:evq-repo-ramp}
\begin{tabular}{@{}lccccc@{}}
\toprule
Table & PK@$8$K & PK@$12$K & PK@$16$K & PPL@$8$K & PPL@$16$K \\
\midrule
Geo & $41{\scriptstyle\pm5}\%$ & $57\%$ & $51\%$ & 161.9 & 253.2 \\
Geo+$\mathcal R_8$ & $61{\scriptstyle\pm3}\%$ & $59\%$ & $51\%$ & 82.9 & 157.7 \\
\evq{} & $53{\scriptstyle\pm8}\%$ & $63\%$ & $50\%$ & 150.3 & 229.5 \\
\evq{}+$\mathcal R_8$ & $\mathbf{100}{\scriptstyle\pm0}\%$ & $\mathbf{79\%}$ & $\mathbf{68\%}$ & \textbf{70.9} & \textbf{107.5} \\
\bottomrule
\end{tabular}
\end{table}


\FloatBarrier
\subsection{Larger-scale supporting evidence}
\label{sec:larger-scale-supporting}

The \(750\mathrm{M}\) `continue@4K' experiment starts both arms from the same
2K Geo checkpoint and applies a 500M-token continuation with either the Geo or
\evq{} table (seed 42). At 16K, PPL is \(45.1\) versus \(24.4\); on
40 passkey trials at 8K, strict autoregressive exact match is \(0\%\) versus
\(77.5\%\). The \evq{} continuation uses $\tau=1.5$, $K=32$, and the
endpoint-inclusive quantiles $u_k=k/(K-1)$. Writing $q_\tau$ for the
inverse CDF in Eq.~\eqref{eq:warp}, the installed frequencies are
\[
\omega_k=\exp\!\left[-\frac{K-1}{K}\log b\,
q_{1.5}\!\left(\frac{k}{K-1}\right)\right],\qquad b=500{,}000.
\]
This retains the Geo endpoints. Both arms are full-parameter continuations
from the shared checkpoint.

<!-- FILE: tables/table6_750m_continue_supporting.tex -->

\begin{table}[ht]
\caption{$750$M continued-pretraining check (seed 42; $2$K$\rightarrow4$K;
$500$M continuation tokens). Passkey retrieval is teacher-forced NLL-gap and
is saturated at $8$K; AR exact is greedy autoregressive exact match over $40$
trials per length. The pooled AR row combines $2$K, $4$K and $8$K
($120$ trials: Geo $80/120$, \evq{} $111/120$).}
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
NLL-gap retrieval @8K & $100\%$ & $100\%$ \\
Strict AR exact @8K & $0\%$ & $\mathbf{77.5\%}$ \\
Pooled strict AR exact (2K/4K/8K) & $66.67\%$ & $\mathbf{92.5\%}$ \\
\bottomrule
\end{tabular}
\end{table}


\FloatBarrier
\subsection{Modality-dependent \texorpdfstring{$\tau$}{tau} correction for video DiT}
\label{sec:tau-correction}

The text-style reference used for this sweep gives
$\tau{=}K_t/\sqrt{T}=2.83$ for $K_t{=}16$ and $T{=}32$; the best tested
Video DiT point is $\tau{=}1.5$ ($0.53\times$), which selects the supporting
video configuration.

\FloatBarrier
\subsection{Cross-modal evidence: video DiT with bidirectional attention}
\label{sec:video-dit}

VideoRoPE and RIFLEx make temporal-frequency allocation explicit in video
models \citep{videorope2025,zhao2025riflex}. We test the same allocation
principle in a $129.6$M video diffusion transformer with bidirectional attention
and 3D RoPE on Oscillating Moving MNIST
\citep{srivastava2015unsupervised}. Training uses 32 frames; evaluation uses
128 frames ($4\times$) under the same fixed temporal range transform in both
arms. The seed-42 Geo and \evq{} ($\tau{=}1.5$) branches are trained for
$15{,}000$ steps in the same head-to-head run. The denoising endpoint uses
noise level $t{=}0.5$ and the same $256$ evaluation videos for both arms.

Table~\ref{tab:dit-h2h} reports the matched comparison. \evq{} reduces
denoising MSE by $21\%$ on training frames, $16\%$ over all extrapolated
frames, and $35\%$ on far extrapolated frames under bidirectional 3D RoPE.

\begin{table}[ht]
\caption{DiT head-to-head evaluation (129.6M, seed 42, $15{,}000$ steps,
256 evaluation videos, bidirectional attention, 3D RoPE). Values are denoising
MSE at $t{=}0.5$.}
\label{tab:dit-h2h}
\centering
\small
\begin{tabular}{@{}l l r r r@{}}
\toprule
Seed & Metric & Geo ($\tau{=}0$) & \evq{} ($\tau{=}1.5$) & $\Delta$ \\
\midrule
42 & Train MSE & 0.00911 & \textbf{0.00720} & $-21\%$ \\
42 & All extrap & 0.00724 & \textbf{0.00606} & $-16\%$ \\
42 & Far extrap & 0.00989 & \textbf{0.00639} & $-35\%$ \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Low-phase-channel diagnostic.}
At base${}=10{,}000$, six of the 16 temporal frequencies accumulate less than
$0.1$ rad over the 32 training frames. This analytic count identifies the
low-phase regime tested by the matched behavioural comparison above.

\FloatBarrier
\subsection{What the schedule concretely does}
\label{sec:schedule-detail}

At $\tau{=}4$, $b{=}5\times10^5$, the warp moves budget toward short
inverse-frequency scales: the continuous quantile $u{=}0.5$ moves from
$1/\omega{=}707$ to $9.69$, and $u{=}0.9$ from $134{,}609$ to $1{,}720$.
The deployed midpoint grid therefore changes sampled extrema and span at finite
$K$; the exact-range experiments anchor those quantities explicitly. At
$\tau{=}2$, $\rho_\tau(0.1){=}1.71$ and $\rho_\tau(1){=}0.55$, while the band
$1/\omega{=}2048$ has relative density $0.76$. Thus the finite grid thins most
strongly at scales much slower than the training window while retaining more of
the training-scale band. The exact-range experiments measure the corresponding
trained allocation effect, and
Table~\ref{tab:coadapt} separately measures incompatibility after a frozen
post-hoc swap.

<!-- FILE: appendix/a5_identification.tex -->

\section{Controlled Allocation Experiments}
\label{sec:identification-details}

\FloatBarrier
\subsection{Exact-range control at \texorpdfstring{$151.9$M}{151.9M}}
\label{sec:exact-range-control}

\paragraph{Paired controls.} The two trained tables have identical frequency
endpoints. Within each seed pair the arms share: model architecture ($151{,}898{,}880$
parameters); trainable initialisation; training length $256$; token order;
optimiser, LR schedule, global batch; within-pair micro batch and accumulation;
training budget
($499{,}974{,}144$ tokens, $7{,}629$ optimiser steps); the $32$ frozen
evaluation anchors; the sampled highest frequency; the sampled lowest
frequency; and the log-frequency span. The single free variable is the location
of the $K{-}2 = 30$ interior frequencies.

The optimiser is fused AdamW with $\beta_1{=}0.9$, $\beta_2{=}0.95$, weight
decay $0.01$, and gradient clipping at $1.0$. Peak/minimum learning rates are
$6\times10^{-4}/6\times10^{-5}$ with $762$ warmup steps and cosine decay.
The global batch is $256$ sequences of length $256$ for every arm. Seed $42$
uses micro batch $64$ with four accumulation steps; seeds $137$ and $256$ use
micro batch $128$ with two accumulation steps. Each paired comparison therefore
has identical execution geometry, optimiser steps, and counted-token budget;
the aggregate uses training seed as its statistical unit under this shared
scientific protocol.

\paragraph{Model and data.} The model has $12$ layers, hidden size $768$,
$12$ heads of dimension $64$, MLP width $3{,}072$, and vocabulary size
$50{,}304$. Training uses the unshuffled FineWeb-Edu
\citep{penedo2024fineweb}
\texttt{sample/10BT/000\_00000.parquet} shard at pinned revision
\texttt{87f09149ef4734204d70ed1d046ddc9ca3f2b8f9}; validation uses the disjoint
\texttt{sample/10BT/004\_00000.parquet} shard. Both are tokenised with the
GPT-NeoX-20B tokenizer \citep{black2022gptneox} at revision
\texttt{c292233c833e336628618a88a648727eb3dff0a7}.

\paragraph{Arms.} \emph{Baseline:} FMRoPE at fixed training support,
implementing the uniform-in-log exponent grid specified in \S6.1 of
\citet{oka2026fmrope} at training base $256$.
\emph{Intervention:} anchored \evq{} at $\tau{=}4$. The anchored and
deployed midpoint \evq{} tables are different support
embeddings of the \emph{same normalised shape}: both use midpoint Cosh
quantiles $q_k$, and the diagnostic removes only their affine endpoint degrees
of freedom. This normalisation is what makes the exact-range control possible.

Writing $\Phi_\tau$ for the inverse CDF in \eqref{eq:warp}, let
$q_k{=}\Phi_\tau((k+\tfrac12)/K)$,
$s_k{=}(q_k-q_0)/(q_{K-1}-q_0)$, and
$R{=}((K-1)/K)\log B$.  The intervention uses
\begin{equation}
\omega_k^{\mathrm{anchor}}=\exp(-R s_k),
\qquad
\omega_k^{\mathrm{FMR}}=\exp\!\left(-R\frac{k}{K-1}\right).
\label{eq:exact-range-grid}
\end{equation}
Hence $k{=}0,K{-}1$ match exactly and only the $K{-}2$ interior entries differ.

\paragraph{Evaluation.} Two conditions. \texttt{fixed}: each checkpoint retains
its training range. \texttt{target-matched}: the range is set to the declared
evaluation length while preserving each arm's normalised interior spacing.
Metric is paired final-$128$-token teacher-forced NLL over the $32$ frozen
anchors; reported PPL is $\exp(\mathrm{mean\ NLL})$.
The anchors are common end positions selected with Python's
\texttt{random.Random(20260723)}: the eligible validation stream is divided
into $32$ equal-width bins, with one end position drawn per bin and a
$2{,}048$-token guard before the next bin. At anchor $e$, length $L$ uses
validation tokens $[e-L,e)$ and scores the final $128$ targets $[e-128,e)$.
Longer contexts therefore add earlier text around the same target tokens;
the longest windows at distinct anchors are disjoint.

\paragraph{Exact range transformation.}
\[
R(L)=\frac{K-1}{K}\log L,\qquad
\omega_k^{\rm target}(L)=\exp[-R(L)z_k].
\]
Here $z_k=k/(K-1)$ for FMRoPE and the anchored Cosh shape for its paired
arm. The fixed condition uses $L=256$ at every evaluation length;
the target-matched condition substitutes the declared evaluation length.
Both conditions use amplitude one.

\paragraph{Three-seed contrasts.}
Negative values favour anchored \evq{}. The aggregation unit is the training seed;
evaluation anchors remain paired observations within a seed.

\begin{table}[ht]
\caption{Exact-range anchored \evq{} minus FMRoPE tail-NLL contrasts.
Mean$\pm$SD is across three independent training seeds.}
\label{tab:exact-range-3seed}
\centering
\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}r rrr c rrr c@{}}
\toprule
& \multicolumn{4}{c}{Fixed training range}
& \multicolumn{4}{c}{Target-matched range} \\
Length & Seed 42 & Seed 137 & Seed 256 & Mean$\pm$SD
       & Seed 42 & Seed 137 & Seed 256 & Mean$\pm$SD \\
\midrule
$256$  & $+0.033$ & $+0.029$ & $+0.017$ & $+0.026\pm0.008$
       & $+0.033$ & $+0.029$ & $+0.017$ & $+0.026\pm0.008$ \\
$512$  & $-0.478$ & $-0.271$ & $-0.094$ & $-0.281\pm0.192$
       & $+0.061$ & $+0.053$ & $+0.067$ & $+0.060\pm0.007$ \\
$1{,}024$ & $-0.205$ & $-0.191$ & $-0.132$ & $-0.176\pm0.039$
       & $+0.182$ & $+0.156$ & $+0.344$ & $+0.227\pm0.102$ \\
$2{,}048$ & $-0.113$ & $-0.181$ & $-0.143$ & $-0.146\pm0.034$
       & $+0.279$ & $+0.399$ & $+0.701$ & $+0.460\pm0.218$ \\
\bottomrule
\end{tabular}}
\end{table}

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.96\linewidth]{fig_exact_range_control.pdf}
  \caption{\textbf{Support and interior allocation are distinct design
  coordinates.} Each thin line is one training seed and the heavy line is the
  mean. At pinned training support, all three seeds favour anchored \evq{} at
  every OOD length (a). Retargeting each arm's support reverses all three OOD
  comparisons (b). The fixed-support panel isolates the allocation effect; the
  tested target-matched policy reverses the ordering of the two tested policies.}
  \label{fig:exact-range-control}
\end{figure}

Across all three seeds, fixed training range favors Cosh at every extended
length, whereas target-matched range favors FMRoPE. The paired design thus
shows both an interior-allocation effect and a change in preferred allocation
when the same trained weights are evaluated under a retargeted range.

\FloatBarrier
\subsection{Weights-by-table co-adaptation at \texorpdfstring{$151.9$M}{151.9M}}
\label{sec:crossing-151m}

The three-seed experiment above identifies the training-time effect of $z$.
A separate frozen crossing asks whether the learned weights remain compatible
with the coordinate system on which they were trained. Seeds $137$ and $256$
provide paired FMRoPE and anchored-\evq{} checkpoints whose training tables
share exact endpoints. For each frozen checkpoint, we install the factor-four
runtime table derived from either training coordinate, keep the checkpoint,
amplitude, $32$ FineWeb-Edu validation anchors, and final-$128$-token NLL
endpoint fixed, and evaluate at $512$ and $1{,}024$ tokens.

For each training table $\Omega$, the derived table uses the following
label-free residual construction. Set $L_0=256$, $\Delta_j=2j$ for
$j=0,\ldots,127$, and
$p_j=(2L_0-4j-1)/(L_0(L_0+1)/2)$, the grouped causal-separation weights.
Let $X_k$ have columns $\sqrt{p_j}\cos(\omega_k\Delta_j)$ and
$\sqrt{p_j}\sin(\omega_k\Delta_j)$, and let $X_{-k}$ collect all other pairs.
Define
\[
U_k=\frac{\|X_k-X_{-k}X_{-k}^{\dagger}X_k\|_F^2}{\|X_k\|_F^2},
\qquad
\bar U_k=\frac{U_k-\min_i U_i}{\max_i U_i-\min_i U_i},
\qquad w_k=(1-\bar U_k)^2.
\]
The pseudoinverse uses relative singular-value cutoff $10^{-10}$; the
zero-span convention is $\bar U_k=0$. The frequency blend is
$\omega'_k=\omega_k(1-3w_k/4)$, cast to FP32 with its endpoints set exactly
to $\omega'_0=\omega_0$ and $\omega'_{K-1}=\omega_{K-1}/4$.
Both derived tables use $g=1+0.1\log4$ and remain unchanged across the
$512$- and $1{,}024$-token evaluations.

\begin{table}[ht]
\caption{Two-training-seed mean tail NLL at $1{,}024$ tokens for the
$151.9$M weights-by-table crossing (lower is better). Each row freezes one
trained weight set; each column installs one separately derived runtime table.}
\label{tab:crossing-151m}
\centering
\small
\begin{tabular}{@{}lrr@{}}
\toprule
Frozen weights & FMRoPE-derived table & Anchored-\evq{}-derived table \\
\midrule
FMRoPE-trained & \textbf{3.426} & 5.776 \\
Anchored-\evq{}-trained & 4.455 & \textbf{3.479} \\
\bottomrule
\end{tabular}
\end{table}

With
$I=[L(W_F,T_C)-L(W_F,T_F)]-[L(W_C,T_C)-L(W_C,T_F)]$,
the interaction is $3.400/3.251$ NLL for seeds $137/256$ at $1{,}024$
tokens and $2.663/2.564$ at $512$. Training seed is the replication unit;
the $32$ anchors are paired observations within a seed. Both seeds exhibit
the same preference for the runtime table derived from their training allocation.

\FloatBarrier
\subsection{Exact-range factorial at \texorpdfstring{$50.9$M}{50.9M}}
\label{sec:exact-range-factorial}

\paragraph{Design.} The $50.9$M decoder has six layers, width $512$, MLP width
$2{,}048$, and vocabulary size $50{,}304$. The structural grid is RoPE base
$500$K/$1$M $\times$ training length $256/1{,}024$ $\times$ $4/8/16$ heads
($d_{\rm head}{=}128/64/32$), giving $12$ configurations. Seeds are
$42/137/256$. Every arm trains for $8{,}388{,}608$ tokens in $128$ optimizer
steps ($65{,}536$ tokens per step) on the same deterministically repeated
WikiText-2 raw stream with the same $1\%$ supervised passkey mix; execution is
float32 on Apple M4 Max MPS. The schedule arms, including the $1.25\times$
multiplier, were specified before any result was inspected: Geo, anchored
\evq{} at $0.75\times$, $1.00\times$ (the formula point), $1.25\times$, and a
deformation-matched exponential. Main factorial $180/180$ completed and the
boundary-$\tau$ follow-up $12/12$ completed, covering the pre-specified matrix.

The exponential control uses
$z_\gamma(t)=(e^{\gamma t}-1)/(e^\gamma-1)$ on $t\in[0,1]$. Its $\gamma$ is
chosen without task results so that the RMS node displacement from the uniform
grid matches the formula-Cosh displacement, while the sampled endpoints and
log-span remain identical across arms.

\paragraph{Metric.} Weighted OOD NLL over $2L/4L/8L$,
\[
\bar N_{\mathrm{OOD}}
= \frac{\sum_{r\in\{2,4,8\}}\log_2(r+1)\,N_{rL}}
       {\sum_{r\in\{2,4,8\}}\log_2(r+1)},
\]
paired within structural configuration and seed, averaged within seed, then
equally weighted over the $12$ configurations. The statistical unit is the
configuration. Equivalent-PPL figures quoted anywhere are
$\exp(\mathrm{mean\ NLL})$.

<!-- FILE: tables/table_m4.tex -->

\begin{table}[tb]
\centering
\small
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{0.92}
\caption{\textbf{Exact-range factorial} ($50.9$M; $2$ bases $\times$ $2$
training lengths $\times$ $3$ head dimensions $={}12$ configurations, $3$ seeds).
All arms share sampled extrema and log-span with Geo. Metric: weighted OOD NLL
over $2\times/4\times/8\times$, paired within configuration and seed. All arms
were pre-specified. CIs are $10{,}000$
configuration-level bootstraps; $p$ is a two-sided exact sign-flip over all
$2^{12}$ assignments, unadjusted across contrasts, so the claim rests on the
cross-configuration direction
(App.~\ref{sec:identification-details}).}
\label{tab:m4}
\begin{tabular}{@{}lcccc@{}}
\toprule
Contrast & $\Delta$NLL & 95\% CI & $\Delta{<}0$ configs & $p$ \\
\midrule
\evq{} $0.75\times$ $-$ Geo & $-0.00912$ & $[-0.018,-0.001]$ & $8/12$ & $0.082$ \\
\evq{} rule $-$ Geo & $-0.00988$ & $[-0.021,0.002]$ & $7/12$ & $0.125$ \\
\evq{} $1.25\times$ $-$ Geo & $\mathbf{-0.01210}$ & $[-0.021,-0.003]$ & $\mathbf{10/12}$ & $0.027$ \\
Matched exponential $-$ Geo & $-0.01062$ & $[-0.021,-0.001]$ & $9/12$ & $0.071$ \\
\midrule
\multicolumn{5}{@{}l@{}}{\textit{The matched exponential reproduces the non-uniform direction:}}\\
\evq{} rule $-$ exponential & $+0.00074$ & $[-0.006,0.008]$ & $7/12$ & $0.836$ \\
\bottomrule
\end{tabular}
\end{table}


\paragraph{Absolute values.} Mean weighted OOD NLL (equivalent PPL): Geo
$6.211973$ ($498.68$); \evq{} $0.75\times$ $6.202858$ ($494.16$); \evq{}
$1.00\times$ $6.202094$ ($493.78$); \evq{} $1.25\times$ $6.199873$ ($492.69$);
matched exponential $6.201354$ ($493.42$).

\paragraph{Boundary-$\tau$ arms.} At formula $\tau{=}1$ the $1.5\times$ arm is
better; at formula $\tau{=}8$ the $0.75\times$ arm is. The improving direction
points toward the intermediate strengths at both tested extremes.

The factorial varies architecture, base, and training length on a repeated
WikiText-2 stream \citep{merity2017wikitext}. Its configuration-level contrasts
complement the longer FineWeb-Edu training comparison in
\S\ref{sec:exact-range-control}.

\section{Learned-frequency comparator}
\label{sec:pe-dominant}

<!-- FILE: tables/table_pe_dominant.tex -->

\begin{table}[h]
\caption{Learned-frequency comparator ($125$M, $\Ltr{=}128$, FineWeb-Edu,
$128\to8$K). Geo, learned-inv-freq, and \evq{} are seed $42$; learnable-$\tau$
is $3$-seed (42/137/256). The learned-inverse-frequency arm has one parameter
per pair, shared across layers and heads.}
\label{tab:pe-dominant}
\centering
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{0.92}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method ($128\!\rightarrow\!8$K) & Extra params & PPL@$128$ & PPL@$8$K & $\Delta$ vs Geo \\
\midrule
Geo & 0 & 184.9 & 513.7 & --- \\
Learnable $\tau$ & 1 & $\mathbf{181.2{\scriptstyle\,\pm 1.3}}$ & $437.9{\scriptstyle\,\pm 12.2}$ & $-14.8\%$ \\
Learned inv-freq ($100\times$ positional LR) & 32 & 183.6 & 455.3 & $-11.4\%$ \\
\quad same, $10\times$ positional LR & 32 & --- & 477.7 & $-7.0\%$ \\
\evq{} & 0 & 182.0 & \textbf{333.7} & $\mathbf{-35.0\%}$ \\
\bottomrule
\end{tabular}
\end{table}


The $32$-parameter arm learns one inverse frequency per rotary pair, shared
across layers and heads. This is the learned-table parameterization also
studied by \citet{karypis2026lerope}. Its positional-parameter learning-rate
sweep uses $10\times$ and $100\times$ the base rate; the stronger $100\times$
setting gives $455.3$ PPL. The learned table and the fixed Cosh allocation
both improve on Geo in this protocol.

<!-- FILE: appendix/a6_mature_scale.tex -->

\section{Mature-scale protocols}
\label{sec:mature-details}

<!-- FILE: tables/table_ruler.tex -->

\begin{table}[tb]
\centering
\scriptsize
\setlength{\tabcolsep}{2.8pt}
\renewcommand{\arraystretch}{0.91}
\caption{\textbf{RULER $13$-family scores (\%).}
Within each protocol, continuation data and evaluation are matched and one seed
is used per arm; evaluation uses $20$ examples per family and length. OLMo
Q/K-only freezes inherited V/O LoRA and trains only the
tensors RoPE acts on; LLaMA uses matched Q/K/V/O LoRA. Training and evaluation
rows are disjoint and share the $13$ generator families, measuring
task-family-adapted length transfer.}
\label{tab:ruler}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}llccc@{}}
\toprule
Model / protocol & Arm / metric & $1\times$ & $2\times$ & $4\times$ \\
\midrule
\multirow{2}{*}{OLMo-2 $1.485$B, full Q/K/V/O ($4$K cap)}
 & Native, official-scorer macro & $\mathbf{82.16}$ & $0.08$ & $0$ \\
 & \evq{}, official-scorer macro & $37.51$ & $\mathbf{21.29}$ & $\mathbf{6.13}$ \\
\addlinespace[2pt]
\multirow{2}{*}{OLMo-2 $1.485$B, Q/K-only ($4$K cap)}
 & Native, official-scorer macro & $\mathbf{72.19}$ & $2.02$ & $0.38$ \\
 & \evq{}, official-scorer macro & $42.44$ & $\mathbf{31.63}$ & $\mathbf{5.03}$ \\
\addlinespace[2pt]
\multirow{4}{*}{LLaMA-3-8B, Q/K/V/O ($8$K cap)}
 & Native, official-scorer macro & $\mathbf{94.44}$ & $0.295$ & --- \\
 & \evq{}, official-scorer macro & $77.60$ & $\mathbf{14.03}$ & --- \\
 & Native, normalized exact & $17.69$ & $0$ & --- \\
 & \evq{}, normalized exact & $\mathbf{21.54}$ & $\mathbf{1.54}$ & --- \\
\bottomrule
\end{tabular}%
}
\end{table}


\FloatBarrier
\subsection{OLMo-2 \texorpdfstring{$1.485$B}{1.485B}}
\label{sec:olmo2-1b}

All OLMo-2 \evq{} arms use the standard $u_k{=}k/K$ grid with $\tau{=}2$ and
base $500{,}000$; the frequency tensor is fixed throughout each continuation.

\paragraph{Matched routing conversion.}
We use OLMo-2-0425-1B-Instruct \citep{olmo2furious} ($1.485$B actual
parameters) with rank-$64$, alpha-$128$ Q/K/V/O LoRA \citep{hu2022lora}.
Native and \evq{} share the checkpoint, training rows and order, token budget,
optimizer, and evaluation rows; the fixed frequency substrate is the intended
intervention. Every backward pass contains at most $4$K physical tokens,
with explicit position IDs providing target-range relative phases. On $8$K
\texttt{niah\_single\_1}, strict first-generated-number exact is $0/100$
for Native and $69/100$ for the matched \evq{} seed; an independently trained
\evq{} seed scores $67/100$ on the same rows. The two \evq{} results are
training-seed replications on a shared evaluation set.

\paragraph{Complete numeric answers after matched routing continuation.}
A separate follow-up to the routing parents applies the same $100$-step
query-gap continuation and $32$-step answer-plus-EOS continuation to Native
and \evq{} (training seed $20260728$). Every backward pass uses at most
$4096$ physical tokens; explicit position IDs expose target-range phases,
with maximum observed position ID $16{,}257$. Evaluation uses real contiguous
prompts from the same numeric NIAH family and greedy decoding.
The endpoint requires the complete expected answer string followed by
observed terminal EOS, measured from generated token IDs.

\begin{table}[ht]
\centering\small
\caption{Complete-answer-string plus terminal-EOS exact match after the
matched routing follow-up, $100$ inputs per length and one trained pair.}
\label{tab:complete-answer-eos}
\begin{tabular}{@{}lrr@{}}
\toprule
Length & Native & \evq{} \\
\midrule
$4$K & $95/100$ & $100/100$ \\
$8$K & $18/100$ & $98/100$ \\
$16$K & $0/100$ & $60/100$ \\
\bottomrule
\end{tabular}
\end{table}

All final generations in both arms terminate with EOS. On the \evq{} subsets
whose source--query gap exceeds the original training range, exact match is
$48/50$ at $8$K and $31/66$ at $16$K. These outputs belong to the
additional continuation above, separately from the earlier first-number
retrieval comparison.

\paragraph{Selective Q/K phase adaptation.}
To test a less intrusive mature-checkpoint adaptation, we start from the
protocol-matched Native and \evq{} Stage-A parents, freeze every inherited V/O
LoRA tensor within each arm (verified bitwise), and continue only the existing Q/K tensors for
$300$ steps. Each arm trains $8{,}388{,}608$ Q/K parameters while freezing the
$8{,}388{,}608$ inherited V/O parameters, with global batch $8$, learning rate
$5{\times}10^{-5}$, and $20$ warmup steps. The QA and RULER generators use
seeds $20260728$ and $20260729$, respectively; the bundled Q/K
\texttt{metrics.json} records the full protocol. Both arms use the same
$1{:}1{:}2$ mixture of continuous-$4$K,
target-$8$K, and target-$16$K phase exposure, complete answer-plus-EOS
supervision, and physical sequences no longer than $4$K; long phases again use
explicit position IDs. On $200$ held-out 2WikiMultiHopQA rows per length
\citep{ho2020twowiki},
Native/\evq{} token-F1 is $25.99/24.84\%$ at $4$K, $0.07/21.48\%$ at $8$K,
and $0/8.57\%$ at $16$K; exact match is $22.0/21.5\%$, $0/17.5\%$, and
$0/4.0\%$, respectively. Prompts retain the official question and context and are
deterministically filled with answer-filtered 2Wiki distractors to reach the
target length, forming a held-out 2Wiki length-transfer evaluation. The RULER adapters are independent of
the QA adapters: Table~\ref{tab:ruler} reports both the earlier $276$-step
full-Q/K/V/O continuation and the $300$-step selective-Q/K continuation, each
trained over all $13$ families. Evaluation uses $20$ examples per family and
length. A paired family-stratified bootstrap over the tracked Q/K-only
per-example predictions gives \evq{}-minus-Native official-scorer macro differences
of $-29.75$ points at $4$K (95\% CI $[-34.42,-24.85]$), $+29.61$ at $8$K
($[26.54,32.74]$), and $+4.65$ at $16$K ($[3.17,6.38]$). These intervals
quantify evaluation-row variation for the trained pair.

\paragraph{From initialisation.}
Starting from the public step-$0$ initialisation, the same-recipe Geo/\evq{}
comparison gives PPL $177.99/191.36$, $161.19/167.45$,
$163.88/156.87$, and $182.73/159.64$ at $2$/$4$/$8$/$16$K under a matched
$2.097$B-token budget.
The \evq{}-minus-Geo NLL is
$+0.0724/+0.0381/-0.0437/-0.1351$ at $2$/$4$/$8$/$16$K, with
$122/128$ and $126/128$ held-out documents favouring \evq{} at $8$K and
$16$K.

The \evq{} branch uses a Hugging Face single-GPU loop; the released Geo
checkpoint was produced by AI2's distributed OLMo trainer. Both start from the
same public step-$0$ initialisation and match the pinned recipe, reconstructed
seed-$6198$ data-order prefix, $512$-sequence global batch, $4$K context,
$1{,}000$ optimizer steps, and $2{,}097{,}152{,}000$ counted input tokens.
This comparison includes the frequency-table change and the two trainer
implementations.

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.88\linewidth]{fig_olmo_scale_crossover.pdf}
  \caption{\textbf{Full-parameter scale evidence crosses beyond the training
  window.} The $1.485$B Geo and \evq{} arms start from the same public
  initialisation and follow the same scientific recipe for $2.097$B tokens.
  Error bars are paired bootstrap intervals over the same $128$ held-out
  documents, conditioned on this trained pair; the lower panel gives the
  document count favouring \evq{} at each length.}
  \label{fig:olmo-scale-crossover}
\end{figure}
\FloatBarrier

\FloatBarrier
\subsection{Frozen fixed-support allocation}
\label{sec:frozen-fixed-support}

The mature-checkpoint intervention in Table~\ref{tab:frozen-exponent-main}
holds both models frozen. The three matched-support arms contain $64$ rotary
pairs and share the Native fast endpoint and the Native slow endpoint divided
by four. The reference rows use the original Native table and the
Transformers YaRN frequency map.
The three matched-support arms also share the factor-four attention amplitude
$1+0.1\ln4$, evaluation rows, greedy decoder, precision, and hardware. The
uniform-allocation arm is log-linear between the endpoints. The coarse control
is a label-free, model-relative projection of the frozen derived table: among
the registered discrete linear ramps, it minimises displacement-space MSE,
using transition pairs $20$--$22$ for OLMo and $28$--$31$ for Qwen. The
remaining arm installs the frozen derived interior allocation. For that arm,
$\widetilde U_k$ is the normalised residual-energy fraction left after the full
sin/cos pair is projected onto all other pairs over Native separations weighted
by their causal pair count, $p(\Delta)\propto L-\Delta$; it sets
$w_k=(1-\widetilde U_k)^2$ in the frequency blend of Eq.~\eqref{eq:adjustment-families}, so less unique
pairs move farther.

\begin{table}[ht]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{\textbf{Exact frozen mature-checkpoint RULER macro (\%).} These are
the values plotted in Table~\ref{tab:frozen-exponent-main}. Reference rows are
separated from the fixed-support block; only the lower block changes interior
$z$ while holding the stated checkpoint protocol fixed.}
\label{tab:frozen-fixed-support}
\begin{tabular}{@{}lrr@{}}
\toprule
Installed table & OLMo unseen-nine, 16K & Qwen core-four, 64K \\
\midrule
Native & 0.00 & 54.50 \\
Official Transformers YaRN ($s{=}4$) & 7.94 & 60.25 \\
\midrule
Uniform allocation $z$ & 0.56 & 57.75 \\
Coarse label-free $z$ & 61.04 & 64.00 \\
Derived allocation $z$ & 60.47 & 66.50 \\
\bottomrule
\end{tabular}
\end{table}

Method selection used the core-four tasks NIAH single-1, NIAH multikey-2/3,
and variable tracking. After the table, amplitude, split rule, checkpoint, and
metric were frozen, the remaining nine RULER families---NIAH single-2/3,
multikey-1, multivalue, multiquery, CWE, FWE, and QA-1/2---formed the OLMo
confirmation set at $16$K, with $20$ rows per task. Qwen2.5-1.5B-Instruct
\citep{qwen2024qwen25} uses the four development RULER tasks
at $64$K, also with $20$ rows per task; it evaluates a factor-four table at
twice the Native context length. Row resampling conditions on each fixed
checkpoint and task set. Both reallocated tables produce the large OLMo
recovery. The derived-minus-uniform-allocation intervals are $[54.88,64.80]$ points for
OLMo and $[0.25,17.50]$ points for Qwen; the paired
derived-minus-coarse-control intervals contain zero on both models. The
control therefore attributes the effect to
fixed-support reallocation while leaving profile-specific ordering unresolved.

\paragraph{Fresh-distribution allocation and routing controls.}
On a disjoint $512$-document FineWeb-Edu holdout from a shard absent from the
calibration data, the session policy is exactly Native on all $512/512$ $4$K
rows. Table~\ref{tab:fresh-fineweb-controls} separates its routing contrast from
the fixed-support allocation contrasts. For the latter, support, attention gain,
Native/long route, checkpoint, rows, and runtime are fixed; only interior $z$
changes. The coarse ramp again matches the detailed derived profile, whereas
the same-support geometric allocation is slightly better at $8$K and fails on
all $512$ rows at $16$K.

\begin{table}[ht]
\caption{Fresh FineWeb-Edu holdout-$512$ final-$1{,}024$-token NLL contrasts.
The first row evaluates the complete session policy; the lower rows compare
long allocations at fixed support, gain, and route. Negative favours the first
named method.}
\label{tab:fresh-fineweb-controls}
\centering
\small
\begin{tabular}{@{}lrr@{}}
\toprule
Contrast & 8K & 16K \\
\midrule
Session-s4 $-$ Native & $-4.0953$ & $-4.4393$ \\
Geometric $-$ derived $z$ & $-0.0288$ & $+4.4560$ \\
Coarse ramp $-$ derived $z$ & $+0.0005$ & $+0.0006$ \\
\bottomrule
\end{tabular}
\end{table}

The OLMo deployment policy selects the exact Native table when observed prefill plus generation
reserve fits the Native window; otherwise one frozen factor-four table and
amplitude are selected before prefill and retained for the KV-cache lifetime.
It adds no learned parameters or training tokens. Statically forcing the same
long profile at $4$K regresses NLL from $2.7538$ to $2.8774$; routing, rather
than the long allocation, is therefore the identified in-window retention
mechanism.

\begin{table}[ht]
\caption{Compact zero-training deployment endpoints on the frozen OLMo
checkpoint, using the specified table, amplitude, and session routing.}
\label{tab:zero-training-policy-endpoints}
\centering
\small
\begin{tabular}{@{}lrr@{}}
\toprule
Endpoint & Official YaRN ($s{=}4$) & Zero-training policy \\
\midrule
PG-19 tail NLL, 8K/16K & $3.4404/3.7955$ & $\mathbf{3.1060/3.0977}$ \\
RULER-13 macro, 8K/16K (\%) & $23.82/5.88$ & $\mathbf{67.72/54.40}$ \\
Qasper token F1, 16K ($n{=}200$, \%) & $18.03$ & $\mathbf{24.57}$ \\
2Wiki token F1, 16K ($n{=}200$, \%) & $25.69$ & $\mathbf{26.66}$ \\
\bottomrule
\end{tabular}
\end{table}

Applying the long profile to every 2Wiki row scores $27.74\%$; the routed policy
pays that small difference to preserve the exact Native short path. Across the
six-task natural-context matrix, routed/YaRN macro is $28.58/21.27\%$ at
$2\times$ and $24.97/25.59\%$ at $4\times$. The QA arms use identical rows,
and every post-chat input plus generation reserve fits within $16$K.
Table~\ref{tab:frozen-exponent-main} gives the corresponding
allocation-only comparison.

\FloatBarrier
\subsection{LLaMA-3-8B-Instruct}
\label{sec:llama8b}

The LLaMA-3-8B-Instruct checkpoint \citep{grattafiori2024llama3} is frozen
apart from the stated adapters. The \evq{} arm in both LLaMA protocols uses midpoint quantisation with
$\tau{=}1.414$; Native retains the pretrained model's original frequency table.

\paragraph{Probability and causal source use.}
The matched seed-$42$ protocol uses $300$ optimizer steps of BF16 rank-$64$,
alpha-$128$ LoRA on Q/K/V/O, physical $8{,}192$-token sequences, and learning
rate $10^{-4}$. Arms share model and tokenizer bytes, frozen training rows and
order, seed, objective, optimizer, scheduler, batch/accumulation, LoRA capacity,
checkpointing, compile mode, and evaluator. Native uses the pretrained
endpoint grid, and Cosh uses midpoint quantiles; the same adaptation recipe is
applied to these two frequency tables.
The frozen adaptation tensor is drawn from LongAlpaca-12k
\citep{chen2023longlora}: $7{,}476$ training and $152$ validation rows after
length filtering, with effective batch $8$ (microbatch $2$, accumulation $4$).

\paragraph{Temporal holdout and aggregation.}
Evaluation uses arXiv titles/abstracts, Federal Register titles/abstracts, and
Stack Overflow question bodies selected by publication or creation timestamps
between January 1 and July 12, 2026. These are separate from the LongAlpaca
adaptation corpus. Collection was frozen before evaluation; each domain has
eight disjoint-document $32{,}768$-token packs formed by concatenation with EOS
separators. Positions and causal attention continue across document boundaries.
The $8$K and $16$K conditions are exact prefixes of the same $32$K packs.
All ordinary text targets in each prefix are scored, excluding document-first
tokens and inserted EOS separators. For domain $d$, let $\ell_d(L)$ and
$n_d(L)$ be the summed token loss and valid-token count over its eight packs.
The reported aggregate is
\[
\operatorname{NLL}(L)=\frac13\sum_{d=1}^{3}\frac{\ell_d(L)}{n_d(L)},
\qquad \operatorname{PPL}(L)=\exp(\operatorname{NLL}(L)).
\]
Table~\ref{tab:llama-temporal-protocol} gives token counts and the distribution
of the $24$ paired pack-level NLL differences. The accompanying
\texttt{figs/llama\_temporal\_summary.json} retains each pack's loss sum and count,
its domain identity, and the frozen collection hash.

\begin{table}[ht]
\centering\small
\caption{Llama-$3$-$8$B temporal holdout: scored tokens and the $24$ paired
pack NLL differences (\evq{} minus Native). Each length uses the same packs.}
\label{tab:llama-temporal-protocol}
\begin{tabular}{@{}lrrr@{}}
\toprule
Prefix & Valid tokens/arm & Valid tokens/pack & Pack $\Delta$NLL: min / median / max \\
\midrule
$8$K & 194,794 & 8,051--8,183 & $+0.193 / +0.432 / +0.613$ \\
$16$K & 389,672 & 16,111--16,353 & $-2.069 / -1.436 / -1.278$ \\
$32$K & 779,400 & 32,243--32,679 & $-2.319 / -2.074 / -1.754$ \\
\bottomrule
\end{tabular}
\end{table}
\FloatBarrier

The domain-macro \evq{}-minus-Native NLL is
$+0.390/-1.510/-2.048$ at $8$/$16$/$32$K.

\paragraph{Position-preserving source intervention.}
On ten frozen true-$16$K
passkey cases, median target-block hit@16 over $32$ retrieval heads rises from
$18.75\%$ to $64.06\%$. Blocking answer-side decode attention to the remote
gold block in every head preserves all prompt tokens, original rotary KV
indices, and the $128$-token block geometry. NLL pools the negative log
probabilities of the $30$ answer tokens ($3$ per case); deletion changes this
mean by $-0.0095$ for Native and $+1.5055$ for \evq{}. The \evq{} change is
positive in all ten cases. Case-level hit@16 ranges are $3.12$--$34.38\%$
for Native and $62.50$--$68.75\%$ for \evq{}.
These are matched teacher-forced probability
and position-preserving causal-routing measurements; autoregressive task
accuracy is evaluated separately. Descriptive Native/\evq{} PPL is
$6.817/10.068$, $108.958/24.068$, and $991.475/127.911$ at those lengths.

\paragraph{RULER-family continuation.}
A separate seed-$20420726$ continuation uses $516$ steps, rank-$64$/alpha-$128$
Q/K/V/O LoRA, global batch $8$ (micro batch $2$, accumulation $4$), BF16 and
Flash-only attention. Its $1{,}376$ physical-$8$K training rows comprise $96$
rows for each of $13$ RULER families plus $128$ natural-replay rows; it processes
$33{,}816{,}576$ input tokens. Training and evaluation rows have zero exact
overlap, and evaluation uses $20$ examples per family and length.
At $16$K, Native/\evq{} official-scorer macro is
$0.295/14.03\%$ and normalized exact is $0/1.54\%$. At $8$K,
Table~\ref{tab:ruler} reports official scorer macro ($94.44/77.60\%$) and
normalized exact ($17.69/21.54\%$).


<!-- FILE: appendix/a7_exponent_adjustments.tex -->

\section{Frozen Exponent-Adjustment Protocols}
\label{sec:adjustment-details}

This section specifies the static profiles and evaluation panels used in
\S\ref{sec:mature-adjustments}. An amplitude $g$ multiplies both the cosine
and sine tables, so its contribution to a rotated Q/K logit is $g^2$.
Every table is installed before prefill and retained for the entire request.
The accompanying \texttt{figs/recorded\_runtime\_identities.json} records
checkpoint, tokenizer, and evaluator identities. The index/placement studies
use RULER commit \texttt{c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a}.

\FloatBarrier
\subsection{Normalized-index profile}
\label{sec:index-adjustment}

For a native table with $K$ pairs, base $b$, and reference length
$L_{\rm ref}$, the normalized-index construction evaluates the profile on
a $64$-pair reference grid with the same $b,L_{\rm ref}$:
\begin{align}
\bar\omega_j&=b^{-j/64},&
c_{64}&=(1-b^{-1/64})^{-1},&
\xi_j&=\log\frac{L_{\rm ref}\bar\omega_j}{2\pi c_{64}},\\
\bar m_j&=\operatorname{clip}\left(
\frac{\xi_H-\xi_j}{\xi_H-\xi_L},0,1\right),&
\xi_H&=0.7382780681078285,&
\xi_L&=0.366403835112904 .
\end{align}
These two constants were fitted to the existing OLMo displacement profile and
then frozen. Linear interpolation of $(j/63,\bar m_j)$ at $k/(K-1)$ gives
$m_k$ on the target grid. The installed table is
$\omega'_k=\omega_k^N s^{-m_k}$.

The Qwen-$2.5$-$0.5$B-Instruct experiment uses its native $K=32$ pairs,
$b=10^6$, and original context length $L_{\rm ref}=32768$, with $s=2$ and fixed amplitude
$g=1+0.074\log2=1.0512928914$. The YaRN-$2$ comparison uses the
Transformers official-equation table and $g=1+0.1\log2=1.0693147181$;
Native uses $g=1$.

The full RULER-13 confirmation uses seed $202609027$, $20$ examples per
task and length, the same tokenizer and native chat template, and greedy
task-budgeted generation. It contains $520$ generations per arm across
$32768$ and $65536$ tokens. Per-task scores are shown in
Table~\ref{tab:index-full13}. The macro averages the thirteen task means
equally. Paired intervals use $10{,}000$ bootstrap samples within task strata
with seed $202609028$.
The $64$K index-minus-YaRN difference is $6.0897$ percentage points,
with interval $[2.7627,9.5835]$; at $32$K it is $-0.0256$ points,
with interval $[-3.2630,3.2179]$.

<!-- FILE: tables/table_index_full13.tex -->

\begin{table}[ht]
\centering\small
\caption{Qwen-$0.5$B RULER-13 per-task score (\%), $20$ examples per cell.}
\label{tab:index-full13}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{$32$K} & \multicolumn{3}{c}{$64$K}\\
Task & Native & Index & YaRN & Native & Index & YaRN\\
\midrule
Single-1 & 100.00 & 100.00 & 100.00 & 75.00 & 100.00 & 100.00 \\
Single-2 & 100.00 & 100.00 & 100.00 & 15.00 & 100.00 & 75.00 \\
Single-3 & 100.00 & 100.00 & 100.00 & 30.00 & 100.00 & 95.00 \\
MultiKey-1 & 90.00 & 95.00 & 90.00 & 50.00 & 90.00 & 65.00 \\
MultiKey-2 & 50.00 & 60.00 & 35.00 & 0.00 & 40.00 & 25.00 \\
MultiKey-3 & 5.00 & 0.00 & 10.00 & 0.00 & 5.00 & 5.00 \\
MultiValue & 62.50 & 81.25 & 81.25 & 22.50 & 63.75 & 58.75 \\
MultiQuery & 62.50 & 80.00 & 77.50 & 22.50 & 65.00 & 50.00 \\
Variable tracking & 48.00 & 42.00 & 46.00 & 3.00 & 28.00 & 36.00 \\
Common words & 2.50 & 2.00 & 2.50 & 7.00 & 0.50 & 0.00 \\
Frequent words & 46.67 & 31.67 & 35.00 & 31.67 & 46.67 & 40.00 \\
QA-1 & 20.00 & 10.00 & 30.00 & 5.00 & 15.00 & 20.00 \\
QA-2 & 25.00 & 25.00 & 20.00 & 25.00 & 15.00 & 20.00 \\
\bottomrule
\end{tabular}
\end{table}


\FloatBarrier
\subsection{Placement on another finite pair grid}
\label{sec:coordinate-confirmation}

The alternative direct-gap construction evaluates the same clipped profile
using the target table's local frequency spacing:
\[
c_K=(1-b^{-1/K})^{-1},\qquad
\xi_k^{(K)}=\log\frac{L_{\rm ref}\omega_k^N}{2\pi c_K},\qquad
m_k^{\rm gap}=\operatorname{clip}
\left(\frac{\xi_H-\xi_k^{(K)}}{\xi_H-\xi_L},0,1\right).
\]
Both placements share the frozen $\xi_H,\xi_L$ and amplitude coefficient
$0.074$. Within each comparison, endpoints, scale, amplitude, checkpoint,
inputs, decoder, and scorer are matched. Table~\ref{tab:coordinate-confirmation}
reports two independent-input confirmations with $80$ rows per task.

Qwen-$0.5$B uses the $32$-pair, $s=2$ setting above and data seed
$202609026$, with four tasks at each of $32$K and $64$K. These are separate
inputs from the subsequent RULER-13 confirmation.
Gemma-$1.1$-$2$B-Instruct uses $K=128$, $b=10{,}000$, a fixed operating
reference of $4096$, $s=4$, and $g=1+0.074\log4=1.1025857827$.
Its confirmation uses data seed $202609028$ at $16$K.
The task vector is NIAH single-1, multikey-2, multikey-3, and variable tracking.
Gemma's direct-gap scores are $100/88.75/20/82.50\%$;
normalized-index scores are $100/88.75/40/87.25\%$.
Both tables were frozen before these inputs were generated, and the earlier
selection rows are excluded. This confirmation compares the two placements;
the Native/YaRN baseline panel uses a different, earlier input set.

<!-- FILE: tables/table_coordinate_confirmation.tex -->

\begin{table}[ht]
\centering\small
\caption{Independent-input placement comparisons, $80$ examples per task and four tasks per panel. Intervals are for index minus direct-gap, in percentage points: $97.5\%$ for each Qwen length (Bonferroni over two lengths), $95\%$ for Gemma.}
\label{tab:coordinate-confirmation}
\begin{tabular}{@{}llrrl@{}}
\toprule
Model & Length & Direct-gap (\%) & Index (\%) & Difference interval\\
\midrule
Qwen-$0.5$B & $32$K & 49.69 & 53.37 & $[-0.25,+7.75]$ \\
Qwen-$0.5$B & $64$K & 46.62 & 46.13 & $[-4.87,+3.81]$ \\
Gemma-$2$B & $16$K & 72.81 & 79.00 & $[+2.81,+9.62]$ \\
\bottomrule
\end{tabular}
\end{table}


\FloatBarrier
\subsection{Boundary-matched intermediate exponents}
\label{sec:bm-construction}

Let $l$ be the last native frequency pair making more than $32$ turns over
the reference window and $h$ the first making less than one turn. Write
$N=h-l$ and $q=\operatorname{clip}(k-l,0,N)$. All profiles leave the fast
band unchanged and divide the slow-band frequencies by $s$:
\begin{equation}
\omega'_k=\omega_k^N s^{-m_q},\qquad m_0=0,\quad m_N=1.
\end{equation}
With the same band boundaries, the three intermediate allocations are
\begin{align}
m_q^{\rm Uni}&=\frac{q}{N},\\
m_q^{\rm Pro}&=\frac{q(q+1)}{N(N+1)},\\
m_q^{\rm BM}&=\frac{q(q+1)(3N+2-2q)}{N(N+1)(N+2)}.
\label{eq:bm-exponents}
\end{align}
The Uni and Pro rules follow the uniform and progressive mixed-radix
constructions of \citet{tian2026mrrope}. BM smooths the radix increments
$\epsilon_q=m_q-m_{q-1}$ at both band boundaries.

To obtain BM, minimize
$\sum_{q=0}^{N}(\epsilon_{q+1}-\epsilon_q)^2$ with
$\epsilon_0=\epsilon_{N+1}=0$ and $\sum_{q=1}^{N}\epsilon_q=1$.
The discrete Euler equation gives
\[
\epsilon_q=\frac{6q(N+1-q)}{N(N+1)(N+2)};
\]
summing gives \eqref{eq:bm-exponents}. The quadratic form is strictly convex
with the endpoints fixed, so this solution is unique. Figure~\ref{fig:bm-profiles}
shows the actual profile shapes for the tested transition widths.

\begin{figure}[ht]
\centering
\includegraphics[width=\linewidth]{fig_bm_exponent_profiles.pdf}
\caption{\textbf{Intermediate-band exponent displacements.}
All three profiles have the same cumulative scale at the endpoints. BM
redistributes the increments toward the middle of the transition band.}
\label{fig:bm-profiles}
\end{figure}

OLMo-$2$-$0425$-$1$B-Instruct uses $b=500{,}000$, reference length $4096$,
$K=64$, and $(l,h,N)=(14,32,18)$. Qwen-$2.5$-$3$B/$7$B-Instruct use
$b=10^6$, reference length $32768$, $K=64$, and $(l,h,N)=(23,40,17)$.
The factor-four comparisons use $g=1+0.1\log4=1.1386294361$ for both
MrRoPE-Pro and BM, with FP32 frequency tables and BF16 model execution.

\FloatBarrier
\subsection{Six-task model comparisons}
\label{sec:bm-model-comparison}

The shared task types are NIAH single-2, multikey-2, multiquery, variable
tracking, frequent words, and QA-1. OLMo's confirmation uses seed $20260910$
with four short and eight long examples per task, after a separate development
panel with seed $20260909$. Its $4$K/$16$K scores are listed in
Table~\ref{tab:bm-models}; MrRoPE-Uni on the same confirmation inputs scores
$76.88/32.12\%$, and official YaRN scores $54.38/6.94\%$.
The Qwen-$3$B panel has two $32$K and four $128$K examples per task.
The $7$B screen reuses the first one short and two long examples per task;
this selection was fixed before its outputs were inspected.
Table~\ref{tab:bm-full-tasks} reports every task mean.

All arms use the corresponding native chat template and the same task-specific
generation caps, score the entire saved response with the RULER task metric,
and retain generated token IDs and EOS flags. On Qwen-$7$B, a tokenwise MLP
chunk size of $4096$ makes $128$K evaluation fit in memory; both arms share
that path while retaining full attention.

<!-- FILE: tables/table_bm_tasks.tex -->

\begin{table}[ht]
\centering\small
\caption{All six-task BM/MrRoPE-Pro comparisons. Values are official RULER task scores (\%); model-specific sample counts are given in Table~\ref{tab:bm-models}.}
\label{tab:bm-full-tasks}
\begin{tabular}{@{}llrrrr@{}}
\toprule
& & \multicolumn{2}{c}{Short} & \multicolumn{2}{c}{Long}\\
Model & Task & MrPro & BM & MrPro & BM\\
\midrule
OLMo-$1$B & Single-2 & 75.00 & 100.00 & 0.00 & 100.00 \\
 & MultiKey-2 & 25.00 & 100.00 & 0.00 & 37.50 \\
 & MultiQuery & 18.75 & 87.50 & 0.00 & 43.75 \\
 & Variable tracking & 25.00 & 70.00 & 0.00 & 10.00 \\
 & Frequent words & 58.33 & 58.33 & 16.67 & 41.67 \\
 & QA-1 & 25.00 & 75.00 & 0.00 & 75.00 \\
\midrule
Qwen-$3$B & Single-2 & 100.00 & 100.00 & 100.00 & 100.00 \\
 & MultiKey-2 & 100.00 & 100.00 & 75.00 & 50.00 \\
 & MultiQuery & 100.00 & 100.00 & 93.75 & 100.00 \\
 & Variable tracking & 90.00 & 100.00 & 75.00 & 75.00 \\
 & Frequent words & 83.33 & 100.00 & 75.00 & 75.00 \\
 & QA-1 & 50.00 & 50.00 & 50.00 & 25.00 \\
\midrule
Qwen-$7$B & Single-2 & 100.00 & 100.00 & 100.00 & 100.00 \\
 & MultiKey-2 & 100.00 & 100.00 & 100.00 & 100.00 \\
 & MultiQuery & 100.00 & 100.00 & 100.00 & 100.00 \\
 & Variable tracking & 100.00 & 80.00 & 90.00 & 60.00 \\
 & Frequent words & 100.00 & 100.00 & 66.67 & 66.67 \\
 & QA-1 & 0.00 & 0.00 & 50.00 & 0.00 \\
\bottomrule
\end{tabular}
\end{table}


\FloatBarrier
\subsection{Natural question answering}
\label{sec:bm-natural}

The natural comparison reuses frozen, source-verified LongBench prompts
\citep{bai2024longbench}. Inputs are untruncated, retain their source questions
and references, and fit the $16$K input-plus-generation budget.
Selection follows the pre-existing source-hash order; no generated answer
participates in selection. The complete eligible pool contains $778$ questions,
of which $631$ have more than $4096$ input tokens.
Both methods retain the same factor-four table at every length.

Greedy generation uses a $32$-token cap for HotpotQA and 2WikiMQA,
$128$ for Qasper and NarrativeQA, and $64$ for MultiFieldQA-en.
The metric is token F1 on the complete saved generated response against
the task references. Table~\ref{tab:bm-qa-complete} gives the long-input
and original-window-length strata, including EOS counts.
The long-input macro averages the five task means; the shorter-input macro
averages its four available tasks. The respective BM-minus-MrPro changes are
$+3.8194$ and $+2.0028$ points, with paired within-task bootstrap intervals
$[1.3161,6.2920]$ and $[-3.5618,7.4742]$.
The MultiFieldQA long-input row includes $40$ capped responses in each arm;
these remain in the F1 calculation.

<!-- FILE: tables/table_bm_qa_all.tex -->

\begin{table}[ht]
\centering\small
\caption{Natural QA under static $s=4$. Token F1 uses the whole generated response; EOS counts are recorded separately.}
\label{tab:bm-qa-complete}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Input length & Task & $n$ & MrPro F1 (\%) & BM F1 (\%) & MrPro EOS & BM EOS\\
\midrule
$4$K--$16$K & HotpotQA & 166 & 28.63 & 33.54 & 160 & 163 \\
 & 2WikiMQA & 173 & 21.55 & 25.43 & 169 & 171 \\
 & Qasper & 119 & 13.94 & 19.06 & 119 & 119 \\
 & NarrativeQA & 61 & 9.51 & 11.48 & 61 & 61 \\
 & MultiFieldQA & 112 & 34.49 & 37.72 & 72 & 72 \\
\midrule
$\le4$K & HotpotQA & 5 & 58.33 & 51.67 & 5 & 5 \\
 & 2WikiMQA & 26 & 34.62 & 44.98 & 26 & 26 \\
 & Qasper & 78 & 26.16 & 28.14 & 78 & 78 \\
 & MultiFieldQA & 38 & 50.94 & 53.27 & 35 & 33 \\
\bottomrule
\end{tabular}
\end{table}


<!-- FILE: appendix/a3_supporting_results.tex -->

\section{Supporting Results}

\subsection{Multi-head Latent Attention}
\label{sec:mla-results}

We report allocation at one fixed rotary budget in a 432M-parameter transformer following
the MLA architectural pattern \citep{deepseekv2}, with 24 layers, width 1024,
16 heads, and per-head width 64.
Its decoupled rotary
subspace has $d_{\mathrm{rope}}{=}32$ ($K{=}16$ frequencies); the remaining
$d_{\mathrm{nope}}{=}32$ dimensions are non-rotary, and the shared KV latent
rank is 256. The model trains from scratch on 500M FineWeb-Edu tokens
\citep{penedo2024fineweb} at
$L_{\mathrm{train}}{=}8192$ and base $500\mathrm{K}$ over seeds 42, 43, and
88. \evq{} uses the pre-specified $\tau=1.414$ setting.
At $16$K, Geo PPL is
$141.1/132.5/142.8$ for seeds $42/43/88$, whereas \evq{} is
$93.7/92.7/100.3$; all three paired directions agree.

For the supporting overlay, let $\lambda_k=2\pi/\omega_k$ and
$t_k=\operatorname{clip}((\lambda_k-L_{\mathrm{train}})/(31L_{\mathrm{train}}),0,1)$.
The MLA wavelength-blend operator sets
$\omega'_k=\omega_k/[1+t_k(s-1)]$: it preserves wavelengths below the training
length, applies the full factor $s$ above $32L_{\mathrm{train}}$, and interpolates
between them.

\begin{table}[ht]
\caption{MLA validation (432M, $d_{\mathrm{rope}}{=}32$, 3-seed mean
$\pm$ std). Rows marked ``MLA wavelength-blend'' use this run's wavelength-blend
operator defined above.}
\label{tab:mla}
\centering
\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}l r r r r@{}}
\toprule
Method & PPL@8K & PPL@16K & PPL@24K & PPL@32K \\
\midrule
Geo & $35.4{\pm}0.9$ & $138.8{\pm}5.5$ & $241.5{\pm}2.6$ & $323.7{\pm}5.2$ \\
Geo + MLA wavelength-blend ($s=4$) & $35.5{\pm}0.9$ & $117.9{\pm}6.5$ & $204.9{\pm}6.9$ & $278.5{\pm}6.3$ \\
\textbf{\evq{}} & $35.8{\pm}0.8$ & $\mathbf{95.6{\pm}4.1}$ & $204.9{\pm}14.7$ & $291.6{\pm}20.5$ \\
\textbf{\evq{} + MLA wavelength-blend ($s=4$)} & $35.8{\pm}0.8$ & $\mathbf{71.1{\pm}4.1}$ & $\mathbf{153.2{\pm}12.5}$ & $\mathbf{236.6{\pm}15.9}$ \\
\bottomrule
\end{tabular}
}
\end{table}

\evq{} reduces 16K PPL by $31.1\%$ ($138.8\to95.6$), while 8K PPL changes by
$+0.9\%$. All three training seeds favour \evq{} at 16K. The \evq{} plus MLA wavelength-blend
row has the lowest mean PPL at 16K and beyond, showing that the allocation
also composes with this wavelength-dependent extension.

\FloatBarrier


\end{document}
