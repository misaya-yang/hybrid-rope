# LaTeX Patterns for NeurIPS Papers

## NeurIPS 2026 Template Setup

```latex
\documentclass{article}
\usepackage[nonatbib]{neurips_2025}  % or neurips_2026 when available
\usepackage{natbib}

% Essential packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{microtype}           % Better typography
\usepackage{url}
\usepackage{booktabs}            % Professional tables
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsthm}
\usepackage[table]{xcolor}
\usepackage{hyperref}

% Hyperlink styling
\hypersetup{
  colorlinks=true,
  linkcolor=blue,
  citecolor=blue,
  urlcolor=blue,
}

% Theorem environments
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\newtheorem{remark}{Remark}
\newtheorem{definition}{Definition}
```

### Page Limits
- **Main body**: 9 pages (NeurIPS 2026)
- **References**: Unlimited (do not count toward page limit)
- **Appendix**: Unlimited (after `\appendix`, does not count)
- **NeurIPS Checklist**: Required, after main body before appendix

### Submission Modes
```latex
% During review (anonymous):
\usepackage[nonatbib]{neurips_2025}
\author{Anonymous Authors}

% Camera-ready:
\usepackage[final,nonatbib]{neurips_2025}
\author{Real Names \\ Affiliation \\ \texttt{email}}

% Preprint (e.g., arXiv):
\usepackage[preprint,nonatbib]{neurips_2025}
```

---

## Typography Conventions

### Math Mode
- Scalars: italic lowercase `$x$`, `$\alpha$`
- Vectors: bold lowercase `$\mathbf{x}$` or `$\boldsymbol{\theta}$`
- Matrices: bold uppercase `$\mathbf{W}$`
- Sets: calligraphic `$\mathcal{D}$`
- Operators: upright `$\operatorname{softmax}$`, `$\operatorname{arcsinh}$`
- Constants: upright when standard (`$\mathrm{e}$` for Euler's number)

### Text Formatting
- Method names: typically formatted as-is or with `\textsc{}` for acronyms
- First mention of an acronym: spell out + define: "Rotary Position Embedding (RoPE)"
- Use `\emph{}` sparingly — for genuine emphasis or first-time definitions
- Never use `\textbf{}` for emphasis in running text (save for table highlights)

### Numbers and Units
- Use `$125\mathrm{M}$` for model sizes (math mode + upright unit)
- Use `$8\mathrm{K}$` for context lengths
- Separate thousands with `\,`: `$500\mathrm{K}$` or `$1{,}000$`
- Percentage: `$31.1\%$` (math mode)
- Use `\times` for multiplication: `$24\times$` training length
- Ranges: en-dash in text ("pages 1--5"), `$16\mathrm{K}$--$24\mathrm{K}$` in math

### Spacing
- Thin space before units: `$128\,\text{tokens}$`
- Non-breaking space before citations: `shown by prior work~\citep{...}`
- Use `\,` for grouping in large numbers: `$100{,}000$`
- Tie (`~`) between figure/table and number: `Figure~\ref{fig:main}`

---

## Tables

### Standard Table Template
```latex
\begin{table}[t]
  \caption{Descriptive caption that tells the main takeaway. Include what's
  compared, the metric, and the key finding. Bold marks best results.}
  \label{tab:my-table}
  \centering
  \small  % or \footnotesize for wide tables
  \begin{tabular}{lcccc}
    \toprule
    Method & Metric 1 & Metric 2 & Metric 3 \\
    \midrule
    Baseline A & 82.3 & 45.1 & 91.2 \\
    Baseline B & 84.7 & 47.3 & 92.1 \\
    \midrule
    Ours       & \textbf{89.2} & \textbf{52.8} & \textbf{95.4} \\
    \bottomrule
  \end{tabular}
\end{table}
```

### Table Rules
- Always use `booktabs`: `\toprule`, `\midrule`, `\bottomrule`
- Never use `\hline` or vertical rules (`|`)
- Use `\midrule` to separate logical groups
- Bold best results with `\textbf{}`
- Right-align numbers, left-align text
- Use `\small` or `\footnotesize` for wide tables
- Include error bars for multi-seed: `$82.3_{\pm 1.2}$`

### Multi-row Tables
```latex
\multirow{2}{*}{Method} & \multicolumn{2}{c}{Setting A} & \multicolumn{2}{c}{Setting B} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
& PPL & Retrieval & PPL & Retrieval \\
```

---

## Figures

### Standard Figure Template
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{fig_name.pdf}
  \caption{Self-contained caption. Describe what the figure shows, what the
  axes represent, and the key takeaway. The caption should be understandable
  without reading the main text.}
  \label{fig:my-figure}
\end{figure}
```

### Side-by-Side Figures
```latex
\begin{figure}[t]
  \centering
  \begin{minipage}[t]{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{fig_left.pdf}
    \caption{Left panel description.}
    \label{fig:left}
  \end{minipage}
  \hfill
  \begin{minipage}[t]{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{fig_right.pdf}
    \caption{Right panel description.}
    \label{fig:right}
  \end{minipage}
\end{figure}
```

### Figure Guidelines
- Use PDF format for vector graphics (plots, diagrams)
- Use PNG only for raster images (attention maps, photos)
- Captions go below figures, above tables
- Make figures readable at print size — test at actual column width
- Use consistent color schemes across all figures
- Include legends within the figure, not just in the caption

---

## Equations

### Numbered vs. Unnumbered
- Number equations you reference: use `equation` environment
- Don't number one-off definitions: use `equation*` or inline `$...$`
- For multi-line: use `align` (numbered) or `align*` (unnumbered)

### Alignment
```latex
\begin{align}
  \mathcal{J}[\rho]
  &= \frac{\alpha}{2}\int \rho(\phi)^2\,d\phi \label{eq:objective} \\
  &\quad + \frac{\beta}{2}\iint \rho(\phi)\rho(\psi)\min(\phi,\psi)\,d\phi\,d\psi \nonumber \\
  &\quad - \mu\int \rho(\phi)b^{-2\phi}\,d\phi. \nonumber
\end{align}
```

### Display Math Punctuation
Displayed equations are part of the sentence — punctuate accordingly:
```latex
The stationary density satisfies
\begin{equation}
  \rho''(\phi) - \tau^2 \rho(\phi) = \gamma b^{-2\phi},
\end{equation}
with boundary conditions...
```

---

## Citations

### Citation Commands
- `\citep{key}` → (Author, Year) — use in parenthetical references
- `\citet{key}` → Author (Year) — use when author is grammatical subject
- `\citep{key1, key2, key3}` → (A, 2020; B, 2021; C, 2022) — multiple refs

### BibTeX Tips
- Use full first names for consistency
- Check for published versions of arXiv papers — use the conference version
- Use the `@inproceedings` type for conference papers, `@article` for journals
- Include DOI or URL for accessibility
- Sort `.bib` file alphabetically by cite key

### Common Citation Patterns
```latex
% Subject of sentence:
\citet{vaswani2017attention} introduced the transformer architecture.

% Parenthetical:
Recent work has explored context extension~\citep{chen2023position, peng2024yarn}.

% Multiple with context:
Several methods address this~\citep[see][for a survey]{survey2024lengthextrapolation}.
```

---

## Algorithm Boxes

```latex
\usepackage[ruled,vlined]{algorithm2e}

\begin{algorithm}[t]
  \caption{EVQ-Cosh Initialization}
  \label{alg:evq}
  \KwIn{Head dimension $d_{\mathrm{head}}$, training length $L$, base $b$}
  \KwOut{Inverse frequencies $\omega_0, \ldots, \omega_{K-1}$}
  $\tau \leftarrow d_{\mathrm{head}} / \sqrt{L}$ \;
  $K \leftarrow d_{\mathrm{head}} / 2$ \;
  \For{$k = 0$ \KwTo $K-1$}{
    $u_k \leftarrow k / K$ \;
    $\phi_k \leftarrow 1 - \frac{1}{\tau}\operatorname{arcsinh}((1 - u_k)\sinh\tau)$ \;
    $\omega_k \leftarrow b^{-\phi_k}$ \;
  }
  \Return $\{\omega_k\}_{k=0}^{K-1}$
\end{algorithm}
```

---

## Common Pitfalls

### Avoid These
- `\textwidth` in two-column mode when you mean `\linewidth`
- Missing `~` before `\citep` causing line break before citation
- Orphaned figures (figures far from their text reference)
- Using `\textbf` inside math mode — use `\mathbf` instead
- Forgetting `\label` after `\caption` (must come after, not before)
- Using `[h]` placement — prefer `[t]` or `[tb]` for NeurIPS

### Compilation Order
```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main  # resolve all cross-references
```
Run at least 3 times after any reference/citation changes.
