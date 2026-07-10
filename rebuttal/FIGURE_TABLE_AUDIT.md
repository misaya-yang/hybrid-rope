# Submitted Figure/Table Consistency Audit

审计日期：2026-07-10

范围：只审计 reviewer-facing submission 中的 Figure 8 / Table 21 与 Figure 9 / Table 20。当前 rebuttal pass **没有修改 PDF、LaTeX 或 figure assets**。

## Verdict

两处 reviewer 指出的矛盾都成立：

- Figure 8 是旧的 \(n=200\) accuracy pilot，但 caption 描述 Gold-answer NLL；Table 21 则是 \(n=2086\) full evaluation。
- Figure 9 的约 \(-81\%\) 来自不同的 progressive-training setting，不能作为 Table 20 中 454M three-seed FineWeb-Edu \(-13.3\%\) 行的可视化。

Rebuttal 中必须主动承认，不能写成 reviewer misunderstanding。由于当前阶段不改 PDF，正确口径是 “we will correct in a revision”，不是 “the PDF has been fixed”。

## Figure 8 / Table 21

### Surviving source of truth

`data/curated/quality_454m_full_eval.json` 保存现存的 \(n=2086\) aggregate：

| Setting | Geo accuracy | EVQ accuracy | Geo Gold NLL | EVQ Gold NLL | Relative NLL change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K raw | 26.1 | 26.8 | 2.220 | 2.182 | -1.7% |
| 8K raw | 24.6 | 26.8 | 3.202 | 2.239 | -30.1% |
| 8K YaRN | 26.5 | 26.6 | 2.389 | 2.195 | -8.1% |
| 16K raw | 24.1 | 23.7 | 7.915 | 6.220 | -21.4% |

### Erratum

Submitted Table 21 的 8K-raw Geo accuracy 写成 26.6%。现存 aggregate 给出 `513 / 2086 = 24.59%`，正确四舍五入是 **24.6%**。Gold-NLL 数值及 NLL 方向性结论不变。

Figure 8 的问题不只是这个 cell：它画的是 superseded \(n=200\) accuracy pilot，并带有 32K 点，却使用了 Gold-NLL caption。因此 rebuttal 不能用 Figure 8 支撑任何定量结论。

Safe response:

> We thank the reviewer for catching the stale/mislabeled QuALITY figure. Figure 8 used a superseded n=200 accuracy pilot under a Gold-NLL caption. The surviving n=2086 aggregate is the source of truth and also shows that the submitted Table 21 8K-raw Geo accuracy should be 24.6% (513/2086), not 26.6%. The Gold-NLL values and conclusions are unchanged. We do not use QuALITY accuracy as a primary claim and will correct the figure and table entry in a revision.

## Figure 9 / Table 20

Reviewer 指出的差异同样成立：

- Figure 9 中 454M long-range gain 约为 \(-81\%\)。
- Table 20 中 three-seed 454M FineWeb-Edu row 是 \(-13.3\%\)。
- 约 \(-81\%\) 对应另一个 progressive-training setting，数据集和训练协议不同。

因此 Figure 9 不是 Table 20 的一致可视化；Table 20 本身也混合多个规模、数据集与训练协议，不能当成 controlled scaling law。

Safe response:

> The reviewer is correct that Figure 9 mixed a progressive-training value with the nominal cross-scale comparison. We withdraw the figure-level scaling inference and rely only on the individually scoped rows in Table 20. In particular, the three-seed 454M FineWeb-Edu row is -13.3%, not approximately -81%. Table 20 is heterogeneous supporting evidence rather than a controlled scaling law; we will correct the figure in a revision.

## Forbidden Wording

- “The reviewer misread the figure.”
- “The current PDF has been fixed.”
- “Figure 8 and Table 21 are equivalent.”
- “Figure 9 proves an 81% 454M scaling gain.”
- “Table 20 establishes a controlled scaling law.”
- “The 26.6% → 24.6% change is merely cosmetic.”
