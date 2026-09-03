# ICLR 2027 narrative optimization plan (Codex execution)

> **ARCHIVED AUGUST 2026 CYCLE (2026-08-29):** this file records a closed
> revision cycle. It has no current authority over the manuscript, narrative,
> experiment priority, compute, or edit order. Do not execute or enforce it.
> Current work follows `AGENTS.md` → `INDEX.md` → `paper-2027/HANDOFF.md` →
> current TeX and the routed canonical owner.

- **Date:** 2026-08-26
- **Historical scope:** the Codex wording/layout pass planned on 2026-08-26
- **Not:** a numerical owner, a new experiment, a GPU authorization, or
  manuscript prose to copy blindly when it conflicts with a canonical owner
- **Objective:** maximize ICLR 2027 acceptance by making a 30-minute PE
  reviewer *believe one causal chain*. The evidence is already large. The
  failure mode is that mixed signs look like contradictions unless the
  interpretive key is in their hands before they see the tables.
- **Immutable:** never edit, compile, move, or regenerate `paper/`.

If a number in this file disagrees with its canonical owner, **stop and
report**. Do not average, round into a new display string, or splice protocols.

Related-work names are owned by
[`ICLR2027_CITATION_NOVELTY_AUDIT_20260826.md`](../../audits/ICLR2027_CITATION_NOVELTY_AUDIT_20260826.md).
Keep the three-paragraph classifier. Do not flatten it to four nouns, and do
not delete Xu / Chen-HoPE / CoPE / Frayed / Chiang / Wu if that audit has
already inserted them.

---

## 0. How to use this file

Execute §§4–8 in order. Obey §2 locks and §3 forbiddens. Compile with
`paper-2027/compile.sh`. Do not commit or push unless the user asks.

This plan supersedes
[`ICLR2027_MANUSCRIPT_OPTIMIZATION_AND_SIMULATED_REVIEW_20260826.md`](ICLR2027_MANUSCRIPT_OPTIMIZATION_AND_SIMULATED_REVIEW_20260826.md)
as the **current TeX edit order** at the time of writing (this file was itself
superseded as the edit order on 2026-08-27 — see banner). That memo remains the
simulated-review record; most of its P0/P1 items are already in live TeX. Do
not re-apply it.

---

## 1. Reviewer memory (optimize for this sentence)

After abstract + Figure 1 + §2, a PE reviewer should be able to tell the AC:

> A geometric 4K RoPE head spends 46 slow dimensions on 2.00 positional
> dimensions. Pin the spectral endpoints and move only the interior
> allocation: trained OOD loss changes in 3/3 seeds, and that is not a
> scalar-base change. EVQ-Cosh is a closed-form table on that axis. On a
> released 1.485B checkpoint the same *coordinate* — not the Cosh curve —
> moves 16K retrieval from 0.56% to 60.47%; a coarse ramp recovers too.

If they instead remember “another parameter-free RoPE table, mixed results,
huge frozen jump,” the rewrite failed.

### 1.1 Objection order (write the paper in this order)

A real reviewer objects in this sequence. Answer *before* they invent a
reject sentence.

1. RoPE is just the base.
2. Slow channels are already known to be dead (Barbero / Oka).
3. This is FMRoPE (move the spectrum, keep a geometric grid).
4. You only beat untuned FMRoPE (target-matched you lose).
5. Then Cosh is the method — but a ramp matches, so why Cosh?
6. Better static geometry should mean better PPL.
7. Small-model diagnostics; where is 1B / RULER / capability?

Live science already answers all seven. Live *order* answers 3 before 1 is
finished (intro names FMRoPE/LeRoPE before 3/3 lands) and concatenates 5’s
construction with 7’s frozen number, so 5 and 7 collapse into “Cosh did
0.56→60.47, but a ramp also did, so the method is optional.”

---

## 2. Dialectical decisions (do not re-litigate)

Three adversarial passes (hostile PE specialist, conservationist
identification reviewer, TeX surgeon) agreed on the Q&A and disagreed on
mechanics. Resolved as follows. Codex must not pick the discarded side.

| Proposal | Decision | Why (reviewer, not author convenience) |
| --- | --- | --- |
| Name `FMRoPE` in the abstract | **No.** Name it in §2.1 protocol, first clause after 3/3. | Abstract-as-FMRoPE-control starts the overlap trial at t=0, before pin/30-interiors. Locked names also forbid the alias “geometric FMRoPE.” |
| Put target-matched `+0.060/+0.227/+0.460` in the abstract | **No.** Keep in §2.1 body. | Abstract-only readers will recite “the method loses.” Body readers must still see the two-knob proof. |
| Put `61.04` in the abstract | **No.** Keep “a coarse ramp reproduces” without the number. Keep `61.04` in Fig. 1c, §5 freeze, discussion. | `61.04` next to Cosh in the abstract reads “the construction is unnecessary.” Freeze ramp is **derived-profile** parity, not a Cosh ablation. |
| Write “the same fourfold support move” linking 151.9M and freeze | **Forbidden.** | Different models, baselines, and \(z\): train-length \(\theta\) retarget vs Native-4K→16K factor-four on unseen-nine. Two knobs, two estimands. Point with “a later frozen corollary (Fig. 1c / §5),” not “the same move.” |
| Replace Fig. 1b MLA with \(r_2=2.00\) heatmap | **No.** Keep three empirical panels: exact-range / MLA / freeze. | \(r_2=2.00\) is already intro sentence 2. Putting it under “consequential” teaches “better rank ⇒ better LM,” which Table 1 falsifies (`7.14→76.20`). Dropping MLA from page 1–2 recreates “small diagnostic paper” (NeurIPS AC.2) while freeze is a *checkpoint* intervention, not from-scratch training. Collapse heatmap already lives in Fig. 3. |
| Flatten related work to four nouns | **No.** Compress the *intro parade*; keep a **classifier with named mechanism sentences**. | zWsa’s missing-citation reject was FMRoPE, not a 25-name deficit. Flattening MrRoPE into “YaRN-style range transport” or dropping Frequency Entropy / LeRoPE 63.6% recreates “insufficient related work” for a 2026 PE specialist. |
| Delete 50.9M factorial from the body | **No.** One sentence, **no** `8/12 7/12 10/12 9/12` as displayed results. | Those fractions on page 3 read “sometimes fails” next to 3/3. Deleting the study with no pointer looks like protocol shopping (27bE asked for \(\tau\) and non-Cosh schedules; this *is* that answer). |
| Lead abstract with “Cosh is only a witness” + freeze `61.04` | **No.** Abstract: Cosh is the closed-form **drop-in table** on the identified axis; uniqueness stays in theory; freeze is a **derived** \(z\). | PE posters that got in had a drop-in object (MrRoPE-Pro, Selective RoPE). Analysis-only is a valid ICLR genre; this paper is not that genre if it still ships a table. |
| Add official YaRN `7.94%` to Fig. 1 / abstract | **No.** | Same-support geometric is the control. Featuring `7.94%` becomes “we beat YaRN,” which the repository already forbids as a headline. Appendix already has the row. |
| Cut Theorem 1, Table 1, 432M/750M/1.485B/8B, or target-matched from the body | **No.** | Volume stays. Routing changes. Identification and systems breadth are co-equal. |

---

## 3. Hard locks

### 3.1 Numbers (display strings; owners win)

| Item | Display | Owner |
| --- | --- | --- |
| Exact-range NLL (anchored EVQ-Cosh − FMRoPE) | `+0.026/-0.281/-0.176/-0.146` at \(1\times/2\times/4\times/8\times\); OOD **3/3** | `EXACT_RANGE_151M_3SEED_RESULT_20260820` |
| Target-matched | `+0.060/+0.227/+0.460`; **0/3** (FMRoPE favoured) | same, §4 |
| Freeze OLMo 16K unseen-nine, 20 rows/task | geometric **0.56%**, derived **60.47%**, coarse ramp **61.04%**; paired **59.92** points; interval **[54.88, 64.80]** | `SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823` §3.2 |
| 50M crossing PPL | `7.14 / 76.20 / 23.05 / 7.16` | `table_coadapt.tex` / full-RoPE report |
| MLA 432M, \(K=16\) | 8K `35.4/35.8`; 16K `138.8/95.6`; **31.1%** | `data/curated/table18_mla_3seed_aggregate.json` |
| Collapse instance | \(L=4096\), \(b=5\times10^5\), \(K=64\); **23** pairs, **46** dims, \(r_2=\mathbf{2.00}\) (standard grid \(u_k=k/K\); not inclusive 24/48) | full-RoPE report §3.2 |

Do not splice `+0.026` into the target-matched vector. Do not promote Qwen `0.6175`. Body Qwen, if kept, stays `57.75%/66.50%`.

### 3.2 Nomenclature

- `Geo` ≠ `Native` ≠ `FMRoPE`.
- Exact-range arms: **FMRoPE** at \(\theta=L_{\mathrm{train}}=256\), geometric exponents; **anchored EVQ-Cosh** (\(\tau=4\)), same extrema and log-span, 30 interiors.
- Freeze arms: same-support **geometric / derived / coarse ramp**. Derived **is not** EVQ-Cosh.
- `\rs{}` / YaRN-style ≠ official YaRN.
- EVQ-Cosh unique **only** for the stated convex surrogate. Finite \(\tau\) is a fallible operating prior, not a basin or global optimum.
- Slow bands: redundant **as positional functions** in the stated metric; they can still carry content. Never “dead,” “unused,” or “reclaimable.”
- RULER freeze: unseen-nine confirmation set. Official 13-family `2.02/31.63` is a different protocol. Do not pool. Do not write bare “RULER macro” in the abstract; write “16K RULER (nine-task, 20 rows).”

### 3.3 Forbidden sentences

Do not write:

- “geometric FMRoPE” (prefixed alias)
- “the same fourfold support move”
- “axis not the Cosh curve” using freeze-ramp numbers
- “changing only \(z\)” in a sentence whose subject is EVQ-Cosh and whose object is `0.56→60.47`
- “we beat FMRoPE” / “we beat YaRN”
- “dead channels”
- additive support + allocation law
- generic “statistically indistinguishable” without the evaluation-row unit (already fixed in live freeze paragraph; keep it)

---

## 4. Page budget

Live body already ends on page 9. **Every added line names its deleted line.**

**Pay (delete/shorten):**

1. Intro competitor parade `01_intro.tex` L31–41 (~11 lines) → 2–3 sentences, no name list before 3/3.
2. Intro Cosh construction L65–74 → 3–4 lines (drop 55/64 quantiles and cosine-kernel appendix teaser from the intro).
3. Intro geometry recap L95–108 → 3–4 lines pointing at Thm. 1 / Prop. 2 / Table 1. Do not reprint the identity.
4. Identification factorial L34–43 → one sentence, no `8/12`.
5. Related-work name parade and the duplicate **EVQ-Cosh** paragraph L80–89 → merge; see §7.
6. If `page:bodyend` becomes 10: shorten `05_discussion.tex` overlap with intro first, then leftover related-work clauses. **Never** cut appendix proofs, Theorem 1, Table 1, target-matched numbers, freeze triple, or any completed scale paragraph.

**Spend:**

- Five-sentence abstract (same or fewer words than now).
- Fig. 1 caption role-labels (not a new panel).
- Identification as three short Q&As (should be shorter than current protocol+result+factorial).
- One role-tag clause per experiments paragraph (~6 half-sentences).

---

## 5. File-by-file surgery

| File | Operation |
| --- | --- |
| `paper-2027/sections/00_abstract.tex` | **Replace** with §6. |
| `paper-2027/sections/01_intro.tex` | **Rewrite in place**, §8. Keep Eq. (1), collapse instance, 3/3 numbers, freeze triple, contributions (iii) split. |
| `paper-2027/sections/02_identification.tex` | **Replace** with §9. Keep labels `sec:identification` and `sec:exp-identify`. |
| `paper-2027/sections/02_related.tex` | **Shorten** to §7 classifier. Do not add PPE. |
| `paper-2027/sections/03_theory.tex` | **Do not touch** theorems/Table 1/Fig. 2–3. |
| `paper-2027/sections/04_experiments.tex` | **Do not cut numbers.** Prepend the role tags in §10. Freeze paragraph: keep Geometric/derived/ramp; add “nine-task, 20 rows” if missing. |
| `paper-2027/sections/05_discussion.tex` | Touch only if page 10. Do not add claims. |
| `06_ethics.tex`, `07_reproducibility.tex`, `08_ai_use.tex` | **Do not touch.** |
| `paper-2027/main.tex` | **Do not touch** (`page:bodyend`, macros, section order). |
| `paper-2027/figs/make_fig_evidence_overview.py` | **Caption/title-only** unless a label on panel (c) still says EVQ-Cosh. Keep three panels. Keep asserts. `pdf.fonttype=42`. |
| `tables/*.tex`, `appendix/*.tex` | **Do not touch** (M4 fractions stay in App. Table 9). |
| `paper/` | **BLOCKED.** |
| `AGENTS.md`, `INDEX.md`, `HANDOFF.md` | Do not edit in the Codex paper pass; routing already points here. |

---

## 6. Target abstract (replace `00_abstract.tex`)

Grammar may be tightened for the 9-page line break; **do not change the numbers or the identity split.** Five sentences, reviewer order 1–3 then 5 then 7. No `FMRoPE` token. No `61.04`. No target-matched vector. No “architecture, continuation, and scale” glue.

```latex
RoPE gives each attention head a finite set of rotary frequencies, but
geometric spacing conflates the sampled log-frequency range with how those
channels are allocated inside it. In a representative $4$K head, $23$ slow
pairs---$46$ nominal dimensions---span only $2.00$ block-whitened
R\'enyi-$2$ effective positional dimensions. We write
$x_k=-\log\omega_k=a+Rz_k$ to separate sampled support $(a,R)$ from interior
allocation $z$. A paired three-seed intervention pins both endpoints and the
log-span, moves only $30$ interior frequencies, and improves OOD loss at
$2\times$, $4\times$, and $8\times$ in every seed
($+0.026/-0.281/-0.176/-0.146$), so $z$ is not a disguised base change.
A stated convex surrogate then supplies \evq{}, a closed-form table with no
learned positional parameters, unique for that surrogate rather than a
universal language-model optimum. Separately, on a released $1.485$B OLMo
checkpoint, with support, amplitude, data, decoder, and weights fixed,
changing only interior $z$ raises nine-task $16$K RULER (20 rows/task) from
$0.56\%$ to $60.47\%$ (paired difference $59.92$ points, interval
$[54.88,64.80]$); a coarse ramp reproduces the recovery, so the identified
axis matters more than one particular curve.
```

Checks after writing: (i) EVQ-Cosh is not the grammatical subject of `0.56→60.47`; (ii) freeze is “changing only interior \(z\)”; (iii) “reproduces the recovery” has no `61.04`; (iv) no competitor names.

---

## 7. Related work (classifier, not a parade)

Keep a body section. Target \(\approx 0.6\)–\(0.8\) page so theory can start on page 4.

**Opening (keep, tighten):** prior work changes realised phases, support \((a,R)\), or interior \(z\); the coordinates interact.

**Keep named mechanism sentences:**

| Citation | One-line job |
| --- | --- |
| FMRoPE | Moves \((a,R)\) via scalar base; **retains a uniform exponent grid**. |
| Barbero + Frequency Entropy | Slow-band *utilization*; we measure phase-invariant 2D \(r_2\), not “dead/unused.” Glue these to \(46\to 2.00\). |
| YaRN / PI / LongRoPE | Range transport of a learned spectrum. One cluster. |
| MrRoPE | Mixed-radix conversion of pretrained frequencies. **Own sentence** (do not collapse into YaRN). Training-free extension, not pinned-support \(z\). |
| LeRoPE | Learned table; 217M frozen-table ablation **63.6% vs 10.4%** is compatible evidence that a *fixed* table carries value, **not** validation of EVQ and not a matched comparator. |
| GRAPE / Selective RoPE | Operator generalizations; this paper keeps canonical planes and a session-static 1D table. One sentence for both. |
| Resonance | Analytic non-geometric snap for interpolation; different objective, not a fixed-support \(z\) test. |
| DAPE / FIRE / FoPE / RePo | Larger functional object than a fixed RoPE table. Keep the existing one sentence. |

**May cut from body** (appendix or “also”): SelfExtend, PoSE, CLEX, Jet-Long, CoPE, RoPE-ID, MHRoPE, MRoPE-I, Xu base bounds, Gu deconstructing, Wu data-shapes, AdaRoPE as a separate paragraph (fold into LeRoPE).

**Merge/delete** the standalone `\paragraph{EVQ-Cosh.}` block (L80–89). One contrast vs LeRoPE in the learning paragraph is enough: analytic table vs optimization run; range methods can still operate on the substrate.

Do not add PPE. Do not expand RePo.

---

## 8. Intro (`01_intro.tex`)

Keep this skeleton, in this order:

1. **L1–20 keep.** Eq. (1), geometric \(z_k=k/(K-1)\), central question, EVQ as constructive witness not universal optimum.
2. **L22–29 keep.** 23/46/\(r_2=2.00\); “can still carry content”; spectral-budget problem.
3. **Replace L31–41** with ~3 sentences, *after* the collapse, *before* “It can”:

   > Long-context tables usually move sampled support and interior placement together, so they do not isolate \(z\) at fixed endpoints. Section~\ref{sec:related} places range transport, scalar-base geometric grids, and learned tables on that map. The experiment below pins both endpoints and the log-span.

   Do **not** name FMRoPE, YaRN, LeRoPE, or AdaRoPE here. The overlap trial starts in §2.1 with a control identity, not a literature exam.

4. **Keep L44–51 (3/3)** as the next paragraph. This is Q&A 3. Optional six words: “The geometric control is named in §2.”
5. **Keep L53–63 freeze numbers.** Must already say **tested derived allocation**, not Cosh. Keep ramp `61.04%` and the evaluation-row interval here **or** in Fig. 1c + §5, not in the abstract. If L53–63 still lets a reader think the freeze table is Cosh, insert “an independently derived same-support \(z\), not the Cosh table.”
6. **Shorten L65–74** to: closed-form inverse-CDF of a stated convex surrogate; zero learned positional parameters; operator unchanged; uniqueness only for that surrogate. Drop 55/64 and the cosine-kernel teaser from the intro.
7. **Fig. 1: keep three panels.** Retitle caption as in §11. Do not change the Python data unless panel (c) labels are wrong.
8. **Replace L90–93 scale list** with one sentence:

   > Architecture, continuation, scale, and capability studies in §\ref{sec:experiments} test whether the same coordinate remains consequential; none replaces the three-seed identification.

9. **Shorten L95–108** to: each frequency is a 2D subspace; Thm. 1 converts mean redundancy into \(r_2\); 50M crossing shows static rank does not rank trained models; obstruction theorem is the exact frozen-map limit. Do not redisplay the identity if Theorem 1 still does.
10. **Keep contributions (i)–(iii).** (iii) already splits Cosh vs frozen controls — do not re-splice. Optional (i) clause: “at pinned training support; not an additive gain over target-aware bases.”

---

## 9. Identification (`02_identification.tex`)

Three short Q&As. Keep the protocol facts (151.9M, three seeds, `499,974,144` tokens, 32 anchors, FMRoPE §6.1, \(\theta=256\), \(\tau=4\), \(K-2=30\)).

**Q1. Does \(z\) change trained behaviour at fixed \((a,R)\)?**

Protocol: name **FMRoPE** here as the geometric control at pinned training support (paper-faithful §6.1, geometric exponents). Anchored EVQ-Cosh moves only 30 interiors.
Result: `+0.026/-0.281/-0.176/-0.146`, OOD 3/3 (Fig. 1a). No geometric base reproduces this once support is pinned.

**Q2. Does that beat target-aware FMRoPE?**

No. `+0.060/+0.227/+0.460` at \(2\times/4\times/8\times\), FMRoPE favoured 3/3 (App.~\ref{sec:identification-details}). The fixed-support result identifies \(z\); it does not establish an additive gain over target-aware support selection.

**Q3. Does support selection exhaust the design?**

No. A later frozen corollary, on a released checkpoint with support and amplitude already matched, still moves only interior \(z\): geometric / derived / coarse ramp score `0.56% / 60.47% / 61.04%` (Fig. 1c, §\ref{sec:exp-frozen-support}). That contrast is not the 151.9M protocol and not the Cosh table.

**Shape breadth (one sentence, no fractions):**

> A pre-specified 50.9M factorial versus Geo, including scaled Cosh and a deformation-matched exponential, supports the allocation axis without identifying a unique shape or strength (App.~Table~\ref{tab:m4}).

---

## 10. Experiments role tags

Do not change any number. First clause of each paragraph = the reviewer question.

| Paragraph | Opening clause |
| --- | --- |
| Scarce-channel MLA | Interior allocation still matters when the rotary budget is only \(K=16\): … |
| Range composition | With the later index-space rule held fixed, the training-time \(z\) changes what that rule recovers: … |
| Continuation / 1.485B | The same in-window/long-length crossover survives full-parameter continuation and same-initialisation pretraining through \(1.485\)B: … |
| Frozen checkpoints | After pretraining, with support and amplitude matched, interior \(z\) still moves frozen behaviour, and a coarse ramp shows it is not Cosh-specific: … |
| 1.485B task-family | After matched Q/K adaptation, the table changes remote answer overlap, not only tail perplexity: … |
| 8B remote-source | At \(8\)B the long-context gain is causal use of the remote block, not a local readout artifact: … |
| Video DiT | The allocation principle also appears under bidirectional 3D RoPE: … |

Keep 8B labeled LoRA / adaptation, not from-scratch scale. Keep 2Wiki/RULER as task-family, not unseen-task transfer.

---

## 11. Figure 1

**Keep** `make_fig_evidence_overview.py` three panels and numeric asserts.

| Panel | Data (unchanged) | Caption job |
| --- | --- | --- |
| (a) | Exact-range per-seed \(\Delta\)NLL | Identification: pinned support, 3/3 OOD |
| (b) | MLA 432M PPL change | Scarce-channel **consequence**, not identification |
| (c) | `0.56 / 60.47 / 61.04` | Frozen same-support **geometric / derived / coarse ramp**. Never EVQ-Cosh |

Caption skeleton:

> Interior allocation is identifiable and consequential.
> (a) Moving only 30 interior frequencies at pinned support improves every
> OOD length in all three 151.9M seeds.
> (b) In a three-seed 432M MLA model with \(K=16\), 16K PPL falls by 31.1%
> with a small in-window cost (architecture consequence, not the
> identification).
> (c) On a frozen 1.485B OLMo checkpoint, same-support geometric / derived /
> coarse ramp score 0.56% / 60.47% / 61.04% at 16K.

Regenerate the PDF only if titles/labels in the Python change. Keep
`pdf.fonttype=42`. Do not lift the collapse heatmap into this figure.

---

## 12. What “done” looks like (reviewer test, not author test)

Read pages 1–3 at normal zoom, 90 seconds, then 8 minutes.

**Pass if:**

- You can recite the memory sentence in §1 without naming MLA, 8B, or canonical correlations.
- You cannot honestly say the paper forgot FMRoPE (it is the §2.1 control).
- You cannot honestly say freeze `60.47%` is EVQ-Cosh.
- You cannot honestly say the authors hid `+0.060/+0.227/+0.460`.
- Page 3 does not display `8/12`.
- Fig. 1 still has a from-scratch systems panel (MLA) and a 1.485B panel (freeze).

**Fail if:**

- Abstract names FMRoPE, prints `61.04`, or makes Cosh the subject of the RULER jump.
- Intro still lists PI/YaRN/LongRoPE/FMRoPE/LeRoPE/AdaRoPE before 3/3.
- Related work is four nouns, or still a 25-name parade through page 4.
- Any completed scale paragraph was deleted to buy space.
- `paper/` was compiled or edited.

---

## 13. Verification (Codex must run)

From repository root. Python/pytest through Conda `aidemo`. Never compile `paper/`.

```bash
cd paper-2027 && ./compile.sh
```

Required: body page 9; undefined refs/cites 0; worst overfull \(\le 5\)pt (target 0pt); `\iclrfinalcopy` commented; Type-3 fonts 0; US Letter.

If figure Python changed:

```bash
conda run --no-capture-output -n aidemo python paper-2027/figs/make_fig_evidence_overview.py
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_repository_navigation.py tests/test_rope_core.py -q
```

Do not run the supplement packager unless asked.

Report: files changed; `page:bodyend`; worst overfull; Type-3 count; that `paper/` was untouched; any lock in §3 you could not satisfy.

---

## 14. Out of scope for this pass

- New experiments, seeds, \(\tau\), tables, or GPU.
- Appendix A.11 \(\tau_*\) display cleanup (optional, theory-credibility only; do it only if the body pass is green and page 9 still holds; do **not** import EVQ-Cosh-R / Nyström).
- Author roster, OpenReview, reciprocal reviewing, dual-submission Sep 24 branch.
- HANDOFF hash updates unless the user asks after a green compile.
- Git commit/push.
