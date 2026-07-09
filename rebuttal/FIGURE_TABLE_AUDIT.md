# Figure/Table Consistency Audit

日期：2026-06-10

范围：核对 fable 材料指出的 QuALITY Figure/Table mismatch。重点是当前 `paper/main.pdf` 和旧 `paper/EVQ-Cosh_NeurIPS2026.pdf` 中的 QuALITY downstream figure/table。

## 0. Verdict

fable 里的 Figure/Table mismatch 是真实问题。该问题已在本轮修复：`paper/figs/fig5_downstream_qa.pdf/png` 已重画为 Gold-NLL figure，`paper/main.pdf` 已用 Tectonic 重新编译，并在 page 36 视觉核验通过。

当前 PDF 状态：

- `paper/main.pdf`：41 pages，current working PDF，已包含修复后的 Figure 8。
- `paper/EVQ-Cosh_NeurIPS2026.pdf`：29 pages，older/submission-like PDF，仍可解释 fable 为何指出该问题。

原始问题：

1. Pre-fix `paper/main.pdf` page 35 had Table 21. The table was internally coherent: it reported both accuracy and Gold NLL. NLL deltas were `-1.7%`, `-30.1%`, `-8.1%`, `-21.4%`.
2. Pre-fix `paper/main.pdf` page 36 had Figure 8. The figure panels plotted QA accuracy and EVQ-minus-Geo accuracy delta in percentage points, but the caption said “Gold-answer NLL across context lengths” and described NLL advantages.
3. Therefore the source/PDF mismatch was not just a stale fable observation. It was a current trust bug.
4. The applied fix was to regenerate `paper/figs/fig5_downstream_qa.pdf/png` as a Gold-NLL figure matching the caption and Table 21.

Current verified status:

- Regeneration sources: `scripts/figures/fig5_downstream_qa_nll.tex` and `scripts/figures/build_fig5_downstream_qa.sh`.
- Generated assets: `paper/figs/fig5_downstream_qa.pdf`, `paper/figs/fig5_downstream_qa.png`.
- Compile command: `cd paper && mkdir -p build_tectonic && tectonic -X compile main.tex --outdir build_tectonic && cp build_tectonic/main.pdf main.pdf`.
- PDF check: `paper/main.pdf` has 41 pages; pypdf font-object scan found no Type3 fonts; References starts on page 10; rendered page 36 shows Figure 8 as Gold-NLL / EVQ relative NLL change.

## 1. Evidence

### 1.1 Table 21

Rendered `paper/main.pdf` page 35:

- Table caption: “QuALITY QA downstream evaluation (454M, n=2086, single seed). Accuracy is near random for all configs; gold-answer NLL reveals EVQ’s advantage in probability space.”
- Rows:
  - 4K raw: Accuracy 26.1 / 26.8, Gold NLL 2.220 / 2.182, NLL delta `-1.7%`.
  - 8K raw: Accuracy 24.6 / 26.8, Gold NLL 3.202 / 2.239, NLL delta `-30.1%`.
  - 8K YaRN: Accuracy 26.5 / 26.6, Gold NLL 2.389 / 2.195, NLL delta `-8.1%`.
  - 16K raw: Accuracy 24.1 / 23.7, Gold NLL 7.915 / 6.220, NLL delta `-21.4%`.

The `.tex` source matches this table:

- `paper/appendix/a3_supporting_results.tex:71-87`

### 1.2 Fixed Figure 8

Rendered `paper/main.pdf` page 36 now shows:

- Figure title in image: “454M QuALITY QA: EVQ lowers gold-answer NLL at extrapolated lengths”.
- Panel (a): “Gold-answer NLL”.
- Panel (b): “EVQ relative NLL change”.
- Panel (a) bars match Table 21 Gold NLL values.
- Panel (b) bars show `-1.7%`, `-30.1%`, `-8.1%`, `-21.4%`.
- Caption says: “Downstream QA: Gold-answer NLL across context lengths ... EVQ’s NLL advantage grows from -1.7% ... to -30.1% ...”.

The current figure asset confirms this:

- `paper/figs/fig5_downstream_qa.png` plots NLL and relative NLL deltas, not accuracy.
- `paper/appendix/a3_supporting_results.tex:92-98` includes this figure under an NLL caption.

### 1.3 Older PDF

`paper/EVQ-Cosh_NeurIPS2026.pdf` still shows the old mismatch pattern, though with different numbering:

- Page 25 has Table 17 and Figure 10.
- Figure 10 panel is QA Accuracy / EVQ-Geo Delta, while caption says Gold-answer NLL.

This likely explains why fable identified the bug even though numbering differs across builds.

## 2. Rebuttal Implication

This issue was not a substantive mechanism failure, but it was a reviewer-trust failure. It is now fixed in the working PDF.

Safe rebuttal posture:

> We thank the reviewer for catching the stale/mislabeled QuALITY figure. The table values and text use gold-answer NLL; the figure panel was an older accuracy visualization and should not have been captioned as NLL. We have replaced the figure with a Gold-NLL plot consistent with Table 21, and we do not use QuALITY accuracy as a primary claim.

Do not write:

- “The reviewer misread the figure.”
- “The figure and table are equivalent.”
- “The accuracy deltas prove downstream improvement.”
- “This does not matter.” It matters for trust, even if not for the mechanism claim.

## 3. Applied Fix

Applied fix:

1. Regenerated `paper/figs/fig5_downstream_qa.pdf` and `.png` as a two-panel NLL figure:
   - Panel (a): Gold-answer NLL, Geo vs EVQ, by length/mode.
   - Panel (b): relative NLL delta, EVQ vs Geo, using `-1.7%`, `-30.1%`, `-8.1%`, `-21.4%`.
2. Kept the current caption with minor wording intact:
   - “Gold-answer NLL across context lengths.”
   - The figure footer mentions that accuracy remains near random.
3. Recompiled `paper/main.pdf`.
4. Re-ran PDF visual check for page containing Figure 8.

## 4. Response Priority

Priority: P0 trust fix, now resolved in the working tree.

If reviewers mention Figure 8/Table 21, response should say the stale/mislabeled figure has been corrected and the NLL table values were the source of truth.
