# 讨论期补充 Official Comment — 定稿

日期:2026-08-03 · 讨论期截止 2026-08-03 AOE · 四条,全部 < 5,000 字符

## 发什么

| 序 | 文件 | 发到 | 唯一职能 | 建议标题 |
|---|---|---|---|---|
| 1 | `01_AC.md` | AC | 把第三条件锚回**原稿 Table 1**(AC 自己夸的那一点),在其每一行下补经验层;+ 远程内容机制;+ 逐结果种子核算;+ 收窄后的正面主张 | `Three additions that did not fit the character limit` |
| 2 | `02_27bE.md` | 27bE | 他**一个 8B/下游数字都没拿到**,而 `R27bE.2` 问的正是产线规模 | `The scale evidence that sits in the other threads` |
| 3 | `03_Dz6s.md` | Dz6s | 他从没见过 FMRoPE 论证与 exact-range 控制 —— 而那是 metareview 第一条件 | `The FMRoPE comparison, which your review did not raise` |
| 4 | `04_zWsa.md` | zWsa | Oka **自己**在 §6.3 承认 L_target 依赖是 "a practical limitation";+ 零参数归因阶梯 | `Two additions, both checkable against the sources` |

## 每条为什么不是重复(已逐数字核验)

对每份补充的每个数字,核对是否已出现在该收件人**已发**的那份里。定稿保留的全部为新增。

| 收件人 | 新增内容 | 已剔除的重复 |
|---|---|---|
| AC | Table 1 锚定、24–92%、0.75×–1.5×、p=0.836、1.506、64.06、4.691、69/100、77.5%、+0.0381 | 三条件总表(与已发正文 100% 重合);+0.0381 在同一份内出现两次 |
| 27bE | 14.03、98/100、82.16、176.3、22.0%、1.506、64.06、4.691、69/100 | +0.0381(其已发 §5 有) |
| Dz6s | 0.478、32/32、+0.061、0.529、Barbero | 1.506、64.06(其已发那份有);"你的三点促成了…"式回顾 |
| zWsa | §6.3 "practical limitation"、−0.256 阶梯、p=0.836、24–92%、1.506 | 0.478、32/32、14.03、98/100、82.16、176.3、94.44、Barbero(其已发那份全有) |

## ⛔ 安全红线(已执行)

- **不得引用三种子 exact-range 的 −0.3159/−0.1949/−0.1674** —— 证据台账标 `CONDITIONAL`(缺 per-seed raw、hash、training-seed CI)。四份中均未出现。
- 不引入任何新实验或未发表结果 —— 全部来自原稿或已发内容。
- 不要求任何人改分或重新权衡他人评审。
- 不主动展开 32 参数基线的历史标签、YaRN 实现版本。被直接问到才答(见 `../DISCUSSION_PHASE_FOLLOWUP_PLAYBOOK.md` §6)。

## 一致性(已核)

- 同一数字在四份中同值,无一处对 A 说了对 B 相反的话
- 四份各自保留自曝边界:retarget 下 FMRoPE 更强(AC/Dz6s/zWsa)、Cosh 非普适(AC/zWsa)、in-window 两种代价(AC/27bE)、证据分层(AC/27bE)
- 四份都开门见山说明"为什么这条现在才出现",不含"我们再补充一下"式无信息开场

## 发送顺序

**AC → 27bE → Dz6s → zWsa。** 决策者先,摇摆票次之。
