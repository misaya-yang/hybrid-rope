# final_send/ — 定稿 official comments（2026-08-03）

底稿：`../followup/FU_*.md`（Opus 5 组）＋ GPT 审核意见；所有数字逐一对照
`theory_results/REVIEWER_USABLE_EVIDENCE_LEDGER_20260726.md` 与各 owner 报告核验。
旧的 `../rebuttal_qwen3.8/` 一套（LeRoPE/AdamW 角度）存档不用。

## 发送方案

| 顺序 | 文件 | 对象 | 字符数 | 动作 |
| --- | --- | --- | --- | --- |
| 1 | `FINAL_AC.md` | AC | 3199 | 发 |
| 2 | `FINAL_27bE.md` | 27bE | 3236 | 发（四份中最强，GPT 9/10） |
| 3 | `FINAL_zWsa.md` | zWsa | 1530 | 发（249 词完成状态记录，给 zWsa 也顺便给 AC 看线程） |
| 4 | `FINAL_Dz6s.md` | Dz6s | 579 | **可不发**；要发就发这个超短指针版 |

Dz6s 建议不发的理由：他的线程已包含全部证据（见下），再发任何实质内容都是重复。
若担心"完全不回"显得冷淡，发指针版即可。

## 相对 FU_* 的修改

1. **全部去掉 "Thank you again." 起手** —— 讨论期至今无审稿人回复，感谢无由来；
   与上一轮已确认的"不要 Thank you 起手"规则一致。
2. **FINAL_AC = FU_AC 压缩**（4294 → 3199 字符，约 -25%；GPT 理想值 55–65%，
   再压就要动三个承重块本身，已无冗余可删）：
   - 删 Part D 整段（与 AC 线程 "What we ask" 重复）；
   - 删 "Our claim after all of it is narrower than the submission's"
     与 "experiments ... exist because they were named"（GPT 点名的两句）；
   - 开头 "Two things" 改为 "Three items"（原文数错）；
   - 表格第 3 行补入 **from-scratch 1.485B 数字**（+0.0381/−0.0437/−0.1351）——
     经核查这组数字不在已提交的 AC 线程中，属合法新增；
   - LaTeX 反斜杠全部改为 Unicode（→、φ_τ），避免 OpenReview 渲染问题。
3. **FINAL_zWsa = FU_zWsa 压缩到 ~250 词**（GPT 方案）：四项条件完成状态＋
   fixed-range/retargeting/compose 边界＋一句询问。原稿 3544 字符几乎全重复其线程。
4. **FINAL_27bE = FU_27bE 基本原样**，仅两处：
   - 第二 EVQ seed 的括号改写（原句易被误读成两个数字都属于第二 seed）；
   - 末句按 GPT 建议软化为 "We would appreciate knowing whether any attribution
     concern remains unresolved."
5. **FINAL_Dz6s 重写为指针**（⚠️ 这是对 GPT 方案的一处偏离，理由如下）。

## ⚠️ 对 GPT 方案的偏离：Dz6s

GPT 未对照已提交线程，建议保留 FU_Dz6s 的 remote-block 段和"三个 concern 如何
改变论文"段。但逐行核查 `paste/REVIEWER_Dz6s.md` 后确认：

- remote-block 完整段落（1.506、18.75→64.06、24 packs、三域）**已在第 32 行**；
- "What your review changed" 段（三个 concern 各改变了什么）**已在第 46 行**；
- strict generation、RULER、2Wiki、Q/K-only、from-scratch 也全部在其线程内。

原样或按 GPT 裁剪后发送都构成逐字重复，违反"禁止二次提交废话"红线，
故改为超短指针版，并把决定权留给你（发或不发都可）。

## 查重总表（数字是否已在各已提交线程）

| 证据 | 27bE | Dz6s | zWsa | AC |
| --- | --- | --- | --- | --- |
| 8B RULER 0.295→14.03 | ✗ | ✓ | ✓ | ✓ |
| remote-block 1.506/64.06 | ✗ | ✓ | ✓ | ✗ |
| strict gen 18→98 / 0→60 | ✗ | ✓ | ✓ | ✓ |
| 2Wiki 0→17.5 | ✗ | ✓ | ✓ | ✓ |
| Q/K-only 72.19/42.44 | ✗ | ✓ | ✓ | ✓ |
| from-scratch +0.0381 组 | ✓ | ✓ | ✓ | ✗ |
| seed accounting | ✗ | 部分 | ✓ | ✗ |

→ 27bE 的 follow-up 每一块都是其线程缺的（这就是它最能涨分的原因）；
AC 的表格/remote-block/from-scratch/seed 清单也是其线程缺的；
zWsa、Dz6s 线程已饱和，只能做状态记录或指针。

## 未采用材料（存档）

- LeRoPE 并行工作角度（`../rebuttal_qwen3.8/` 全套）——Opus 5/GPT 路线不带；
  revision 里再引用即可。
- 7-31 保窗路线审计（`INWINDOW_PRESERVATION_ROUTE_AUDIT_20260731.md`）自定
  "讨论期不引入"，遵守。

## 数字核验记录（全部 ✓）

0.295/14.03、94.44/77.60、17.69/21.54、1.919/2.309、4.691/3.181、6.899/4.851、
1.5055→1.506、18.75/64.06、95-18-0/100-98-60、69/67、82.16/37.51、72.19/42.44、
22.0/21.5、0/17.5、0/4.0、+0.0381/−0.0437/−0.1351、122/128、126/128、
138.8±5.5/95.6±4.1、0.478/0.205/0.113、32/32-27/32-22/32、10/12、
+0.061/+0.182/+0.279、0.098/0.529/0.638、176.3/21.5、1942.5/104.3、0%→77.5%。
来源：REVIEWER_USABLE_EVIDENCE_LEDGER_20260726.md、EVQ_8B_ADAPTATION_EVIDENCE_20260724.md、
paste/ 五份已提交稿。
