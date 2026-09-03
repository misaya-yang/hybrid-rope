# ICLR 2027 citation + novelty patch (Codex)

Verified 2026-08-26 against ACL Anthology, NeurIPS/ICLR proceedings, iclr.cc virtual 2026, arXiv abs pages, and live `paper-2027/` TeX. Not a numerical owner. Do not edit `paper/`. Do not flatten related work into a zoo. Compile with `paper-2027/compile.sh`.

> **Status (2026-08-28):** historical Codex execution record — the revision cycle this served closed 2026-08-28, outcome committed at 93d7eac; the narrative plan it sequences after is itself superseded (see its banner). Do not execute from this file; the citation/novelty verifications below stand as of 2026-08-26.

Companion: execute after
[`ICLR2027_NARRATIVE_OPTIMIZATION_PLAN_20260826.md`](../archive/2026-08/ICLR2027_NARRATIVE_OPTIMIZATION_PLAN_20260826.md).
If that pass deletes a MUST cite below, put it back.

---

## 1. Already closed (do not reopen)

NeurIPS `zWsa` / AC `AC.1`: FMRoPE uncited + no matched control.

Live TeX already: cites Oka ICLR 2026; implements `\citet[\S6.1]{oka2026fmrope}` (\(\theta=L_{\mathrm{train}}\), uniform grid); 3/3 OOD at pinned support; target-matched FMRoPE **wins** `+0.060/+0.227/+0.460`. Do not name FMRoPE in the abstract. Do not write “we beat FMRoPE.”

---

## 2. Verified records (use these strings only)

Venue rule: peer-reviewed venue if independently confirmed; else arXiv. Do not invent ICML/ICLR for CoPE, Wu, Jet-Long, LeRoPE.

### MUST cite (missing from PDF; novelty-attack)

| Key | Title | Authors (official order) | Source | Live status |
| --- | --- | --- | --- | --- |
| `xu2024base` | Base of RoPE Bounds Context Length | Mingyu Xu, Xin Men, Bingning Wang, Qingyu Zhang, Hongyu Lin, Yaojie Lu, Xianpei Han, Weipeng Chen | NeurIPS 2024, DOI `10.52202/079017-2773`, arXiv:2405.14591 | **in bib, uncited** (NeurIPS related work cited it) |
| `liu2024scaling` | Scaling Laws of RoPE-based Extrapolation | Xiaoran Liu, Hang Yan, Chenxin An, Xipeng Qiu, Dahua Lin | ICLR 2024 proceedings (no Shuo Zhang). OpenReview `JO7k0SJ5V6`. arXiv:2310.05209 | **not in bib** |
| `wu2026datashapes` | How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization | Xinyi Wu, Siyuan Liu, Ali Jadbabaie | arXiv:2607.07678, submitted 2026-07-08. **No venue.** | **in bib, uncited** |
| `chen2025hope` | HoPE: A Novel Positional Encoding Without Long-Term Decay for Enhanced Context Awareness and Extrapolation | Yuhan Chen, Ang Lv, Jian Luan, Bin Wang, Wei Liu | ACL 2025 long, anthology `2025.acl-long.1123`, pages 23044–23056, DOI `10.18653/v1/2025.acl-long.1123`. Method name in abstract: **High-frequency rotary Position Encoding (HoPE)**. | **in bib, uncited** (NeurIPS related work cited it). Bib authors already correct. |
| `li2026copeclipped` | CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs | Haoran Li, Sucheng Ren, Alan Yuille, Feng Wang | arXiv:2602.05258, submitted 2026-02-05. **No conference acceptance.** Not the ICML 2026 VLM poster also named CoPE. | **in bib, uncited** (NeurIPS related work cited it) |
| `wertheimer2026frayed` | Frayed RoPE and Long Inputs: A Geometric Perspective | Davis Wertheimer, Aozhong Zhang, Derrick Liu, Penghang Yin, Naigang Wang | ICLR 2026 poster (2026-04-24). arXiv:2603.18017 comments “Accepted by ICLR 2026”. OpenReview `W8ZXfNaqku`. Method: **RoPE-ID**. | **in bib as `@article` arXiv — upgrade to `@inproceedings` ICLR 2026; uncited** |
| `chiang2025rotary` | The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval | Ting-Rui Chiang, Dani Yogatama | Findings of ACL 2025, anthology `2025.findings-acl.697`, pages 13552–13562, DOI `10.18653/v1/2025.findings-acl.697`, arXiv:2502.11276 | **not in bib** |

HoPE disambiguation: `chen2025hope` = ACL 2025 high-frequency rotary. Do **not** cite Dai et al. hyperbolic HoPE (`dai2025hyperbolicrope`, arXiv:2509.05218) or Li et al. VLM HoPE (`li2025hope`, arXiv:2505.20444) in the same sentence.

### SHOULD cite

| Key | Title | Authors | Source | Live status |
| --- | --- | --- | --- | --- |
| `urrutia2026decoupling` | Decoupling Positional and Symbolic Attention Behavior in Transformers | Felipe Urrutia, Jorge Salas, Alexander Kozachinskiy, Cristian Buc Calderon, Hector Pasten, Cristobal Rojas | ICLR 2026 poster 2026-04-23, OpenReview `V38yAoqddQ`, arXiv:2511.11579. ICLR virtual page drops the word “Behavior” and shortens Buc Calderon → Calderon; **keep the PDF title and bib names**. | in bib, uncited |
| `deepseekv2` | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | DeepSeek-AI | arXiv:2405.04434 | in bib, uncited. MLA identity only. Do not claim a DeepSeek reproduction (`d_{\mathrm{rope}}=32` ≠ production 64). |

### OPTIONAL (one clause max; drop first if overfull)

| Key | Title | Authors | Source |
| --- | --- | --- | --- |
| `tang2026jetlong` | Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE | Haozhan Tang, Zerui Wang, Yuxian Gu, Song Han, Han Cai | arXiv:2607.07740, 2026-07-08/10. No venue. Range operator, not \(z\). Do not bake off RULER. |

### Already cited — do not add aliases

| Key | Title | Authors | Source |
| --- | --- | --- | --- |
| `oka2026fmrope` | Frequency Bands in RoPE: Base Frequency and Context Length Shape the Interpolation–Extrapolation Trade-off | Yui Oka, Itsumi Saito, Kyosuke Nishida, Kuniko Saito | ICLR 2026 poster, OpenReview `PR1PPxvG9Q` |
| `oka2026frequencyentropy` | Probing Rotary Position Embeddings through Frequency Entropy | Yui Oka, Kentaro Hanafusa, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito | ICLR 2026, OpenReview `1JZuEDq62N` |
| `tian2026mrrope` | MrRoPE: Mixed-Radix Rotary Position Embedding | Qingyuan Tian, Wenhong Zhu, Xiaoran Liu, Xiaofeng Wang, Rui Wang | ICLR 2026 oral, OpenReview `1J63FJYJKg` |
| `barbero2025round` | Round and Round We Go! What Makes Rotary Positional Encodings Useful? | Federico Barbero, Alex Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu, Petar Veličković | ICLR 2025. p-RoPE **removes the slowest** frequencies. Live related-work sentence is correct. |
| `karypis2026lerope` | LeRoPE: Learnable RoPE Frequencies Improve Language Modeling | Petros Karypis, Sean O'Brien, Shreyas Kadekodi, Rui Zhu, Julian McAuley | arXiv:2607.10134, 2026-07-11. **No venue.** 63.6% vs 10.4% is LeRoPE Table 2 (217M frozen-table vs p-RoPE). Compatible evidence only. |
| `wang2026adarope` | AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally | Shaowen Wang, Yuke Zheng, Tansheng Zhu, Shuang Chen, Shaofan Liu, Suncong Zheng, Jian Li | ICML 2026 (arxiv comments; icml.cc poster 60704, 2026-07-06). arXiv:2607.19363 |
| `li2026repo` | RePo: Language Models with Context Re-Positioning | Huayang Li, Tianyu Zhao, Deng Cai, Richard Sproat | ICML 2026 (arxiv comments). arXiv:2512.14391. Already cited. |
| `zhang2026grape` | Group Representational Position Encoding | Yifan Zhang, Zixiang Chen, Yifeng Liu, Zhen Qin, Huizhuo Yuan, Kangping Xu, Yang Yuan, Quanquan Gu, Andrew Chi-Chih Yao | ICLR 2026 |
| `movahedi2026selectiverope` | Selective Rotary Position Embedding | Sajad Movahedi, Timur Carstensen, Arshia Afzal, Frank Hutter, Antonio Orvieto, Volkan Cevher | ICLR 2026, OpenReview `AQo1SEElNb` |

ICLR 2027 Author Guidelines (live 2026-08-26): cite related arXiv in the third person. **No four-month contemporaneous omission clause.** Do not skip Wu.

---

## 3. Citation gaps

Process: ICLR rewrite kept these keys in `refs/references.bib` and stopped citing them when related work was compressed. FMRoPE was added; Xu / Chen-HoPE / CoPE were dropped.

| Gap | Where | Why it is a score issue |
| --- | --- | --- |
| `xu2024base` uncited | Range paragraph | Support bound on the scalar base. Without it, \((a,R)\) looks invented. |
| `liu2024scaling` missing | Same sentence as Xu | ICLR 2024 critical dimension. Range paragraph jumps PI/YaRN → FMRoPE. |
| `wu2026datashapes` uncited | Range paragraph | Usage of a **fixed geometric grid** (\(\theta^\star\sim 1/W\), PI = scale matching). Closest language collision with “spectral budget.” |
| `chen2025hope` uncited | Frequency-use paragraph | High-frequency-only construction: replace slow RoPE components. |
| `li2026copeclipped` uncited | Same | Soft-clip slow frequencies, zero learned parameters. |
| `wertheimer2026frayed` uncited | Same | ICLR 2026 geometry + high-frequency subset (RoPE-ID). 2027 analogue of missing Oka. |
| `chiang2025rotary` missing | Same, next to “not dead” | Wide-angle/**high-frequency** rotations inefficient; **low-frequency dims of retrieval heads** load-bearing. Live “does not make a channel dead” cites nothing. Do not paraphrase as Barbero. |
| `deepseekv2` uncited | §4 MLA paragraph | Architecture identity. |
| Frayed bib type | `references.bib` | Still `@article` arXiv; paper is ICLR 2026. |

Do **not** restore ALiBi, XPOS, KERPLE, PPE, PoPE, STRING, nD-RoPE, hyperbolic HoPE, CARoPE.

---

## 4. Novelty (one sentence each)

Defended object: \(x_k=a+Rz_k\). Knob 1 = support \((a,R)\). Knob 2 = interior \(z\) at pinned endpoints. Geometry \(23/46/r_2=2.00\) is a budget account, **not** dead/unused. Cosh unique **only** for the stated surrogate. Freeze `0.56/60.47/61.04` is geometric / **derived** / ramp, **not** Cosh. Target-matched FMRoPE wins.

| Paper | They own | We own | Attack if unnamed | Live TeX |
| --- | --- | --- | --- | --- |
| Oka FMRoPE ICLR 2026 | \(\theta=L_{\mathrm{train}}\), uniform grid; slow dims below the band weakly used | pinned-support move of \(z\); explicit loss to target-aware FMRoPE | “this is FMRoPE” | **Blocked in §2.1** |
| Wu 2607.07678 | learned **usage** of a geometric grid; \(\theta^\star\sim 1/W\) | changing the **supply** of interior \(z\) | “they already have the spectral budget” | **Open** |
| Chen HoPE ACL 2025 | keep high-freq rotary; replace slow with position-independent | full \(K\)-pair table, reallocate \(z\) | “this is HoPE / partial RoPE” | **Open** |
| CoPE 2602.05258 | soft-clip slow frequencies | interiors moved, not clipped | “closed-form slow-band fix already exists” | **Open** |
| Frayed / RoPE-ID ICLR 2026 | sink-token geometry; high-freq on a **subset** | full-table interior \(z\) | “ICLR 2026 already did geometry + high-freq subset” | **Open** |
| Chiang Findings 2025 | high-freq inefficient; low-freq of retrieval heads load-bearing | subspace collapse \(\neq\) unused | “your dead-channel hedge is false / uncited” | word “dead” blocked; **cite missing** |
| Barbero ICLR 2025 | p-RoPE drops **slowest** rotations | do not remove planes | “slow channels already known” | **Blocked** if HoPE/CoPE/RoPE-ID glued into the same operator-scope sentence |
| LeRoPE 2607.10134 | learned table; frozen-table 63.6% vs p-RoPE 10.4% | closed-form table; fixed-support ID | “just learn it” | **Blocked** (compatible, not matched) |
| AdaRoPE ICML 2026 | per-head learned frequencies | one session-static shared table | “they learn allocation finer” | **Blocked** |
| MrRoPE ICLR 2026 | training-free radix conversion | training-time \(z\) | “training-free remap exists” | **Blocked** |
| Xu NeurIPS 2024 | geometric-family base lower-bounds context | knob 2 at fixed base | “the variable is the base” | **Open** |
| Jet-Long 2607.07740 | bifocal range operator | static table / pinned-support ID | “zero-shot long RoPE already exists” | **Open** (optional clause) |

Compatible: slow **positional functions** can collapse and still be the long-wavelength coordinates retrieval reads. Do not write “this contradicts collapse.” Do not restyle freeze `0.56\to60.47` as recovering dead channels.

---

## 5. TeX (only these files)

### 5.1 `paper-2027/sections/02_related.tex` — Range paragraph

**Replace** the FMRoPE/MrRoPE sentences (current L16–18) with:

```tex
The RoPE base bounds attainable context length in the geometric family
\citep{xu2024base}, and scaling-law analyses of RoPE extrapolation
identify a critical dimension for that bound \citep{liu2024scaling}.
FMRoPE instead moves $(a,R)$ through a scalar base while retaining a
uniform exponent grid \citep{oka2026fmrope}.
Wu et al.\ study learned \emph{usage} of that same geometric grid:
query/key energy concentrates on frequencies matched to a data-induced
dependency width $W$ ($\theta^\star\sim 1/W$), and they interpret
position interpolation as scale matching \citep{wu2026datashapes}.
That is a theory of how weights select among given channels; it does
not pin sampled support and move interior $z$.
MrRoPE gives a distinct training-free mixed-radix conversion of
pretrained frequencies \citep{tian2026mrrope}.
```

Keep the closing sentence: these comparisons change support or realised phases; the control pins endpoints and log-span before moving \(z\).

Optional, only if still under 9 pages:

```tex
Jet-Long dynamically remaps remote positions with a bifocal operator;
it is range transport, not a pinned-support interior-allocation control
\citep{tang2026jetlong}.
```

### 5.2 `paper-2027/sections/02_related.tex` — Frequency-use paragraph

**Replace** current L22–30 with:

```tex
\paragraph{Frequency use and operator scope.}
Trained RoPE exhibits frequency-localized query/key norms; partial RoPE
removes the slowest rotations \citep{barbero2025round}, while
\mbox{Frequency Entropy} measures rotational-pair utilisation
\citep{oka2026frequencyentropy}.
High-frequency rotary HoPE replaces slow components with
position-independent encodings \citep{chen2025hope}; RoPE-ID restricts
rotation to a high-frequency subset \citep{wertheimer2026frayed}; CoPE
soft-clips low-frequency bands \citep{li2026copeclipped}.
Chiang \& Yogatama find that wide-angle (high-frequency) rotations can
be inefficient, while low-frequency dimensions of retrieval heads remain
load-bearing \citep{chiang2025rotary}.
We instead measure phase-invariant effective dimension of the full
two-dimensional pair subspaces; redundancy in this metric does not make
a channel dead or unused.
Those constructions change operator scope---removing, clipping, or
restricting rotation---rather than reallocating a full $K$-pair table at
fixed endpoints.
GRAPE generalises the planar group action and Selective RoPE learns
input-dependent angles \citep{zhang2026grape,movahedi2026selectiverope};
our intervention keeps the canonical planes and one session-static
frequency table.
```

Write “high-frequency rotary HoPE” so it cannot be read as Dai. Do not add Urrutia unless the page still has one free clause after compile.

### 5.3 Learning-tables paragraph

No MUST additions. Keep LeRoPE 63.6% / 10.4% and the “not a matched comparator” sentence.

### 5.4 `paper-2027/sections/05_discussion.tex`

After the LeRoPE sentence (current L48–52), one fence:

```tex
Dropping, clipping, or restricting rotation to a high-frequency subset
changes operator support \citep{chen2025hope,li2026copeclipped,wertheimer2026frayed};
the fixed-support intervention instead keeps both endpoints and moves only
interior $z$.
```

### 5.5 `paper-2027/sections/04_experiments.tex` — MLA

```tex
a $432$M MLA transformer \citep{deepseekv2} trains from scratch
```

### 5.6 Do not edit

Abstract (no `FMRoPE`, no `61.04`, no target-matched vector). Identification Q1–Q3 science. Theorems. `paper/`. Ethics / AI-use.

If overfull: cut Jet-Long, then Urrutia, then the intro cluster. **Never** cut Xu, Liu, Wu, Chen-HoPE, CoPE, Frayed, Chiang.

---

## 6. Bibliography (`paper-2027/refs/references.bib`)

Cite existing keys as-is: `xu2024base`, `chen2025hope`, `wu2026datashapes`, `li2026copeclipped`, `deepseekv2`. Optional: `tang2026jetlong`, `urrutia2026decoupling`.

**Add:**

```bibtex
@inproceedings{liu2024scaling,
  title     = {Scaling Laws of {RoPE}-based Extrapolation},
  author    = {Liu, Xiaoran and Yan, Hang and An, Chenxin and Qiu, Xipeng and Lin, Dahua},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=JO7k0SJ5V6},
  note      = {arXiv:2310.05209}
}

@inproceedings{chiang2025rotary,
  title     = {The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval},
  author    = {Chiang, Ting-Rui and Yogatama, Dani},
  booktitle = {Findings of the Association for Computational Linguistics: {ACL} 2025},
  pages     = {13552--13562},
  year      = {2025},
  doi       = {10.18653/v1/2025.findings-acl.697},
  note      = {arXiv:2502.11276}
}
```

**Replace** the existing `wertheimer2026frayed` `@article` with:

```bibtex
@inproceedings{wertheimer2026frayed,
  title     = {Frayed {RoPE} and Long Inputs: A Geometric Perspective},
  author    = {Wertheimer, Davis and Zhang, Aozhong and Liu, Derrick and Yin, Penghang and Wang, Naigang},
  booktitle = {International Conference on Learning Representations ({ICLR})},
  year      = {2026},
  url       = {https://openreview.net/forum?id=W8ZXfNaqku},
  note      = {arXiv:2603.18017}
}
```

CoPE, Wu, Jet-Long, LeRoPE: arXiv only.

---

## 7. Forbidden

- “we beat FMRoPE / YaRN”; “geometric FMRoPE”; “dead / unused / reclaimable”
- Cosh unique beyond the stated surrogate; “global optimum”; “basin”
- “LeRoPE validates EVQ”; any unrun CoPE/HoPE/RoPE-ID/Jet-Long bake-off
- “the same fourfold support move”; EVQ-Cosh as subject of `0.56\to60.47`
- “Wu already owns the spectral budget”; “this contradicts retrieval heads”
- Dai hyperbolic HoPE; ICML 2026 VLM CoPE; inventing venues

Keep: exact-range 3/3 = \(z\) at fixed training support vs paper-faithful FMRoPE; target-matched FMRoPE wins; \(r_2=2.00\) is not an LM predictor; freeze is derived/ramp not Cosh.

---

## 8. Compile

`paper-2027/compile.sh`. Need: 9-page body; zero undefined citations; `chen2025hope` in the PDF is ACL 2025 Chen/Lv/Luan/Wang/Liu; FMRoPE still named in identification §6.1; abstract still omits FMRoPE and `61.04`; `paper/main.pdf` SHA-256 unchanged `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`.
