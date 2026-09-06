# claude_code_workspace — Claude 工作区（paper-2027 / RoPE 谱预算）

本目录是 Claude 在 ICLR 2027 投稿项目（"RoPE Has a Spectral Budget"，
abstract 2026-09-18 / 全文 2026-09-25）中的工作区：实验代码、预注册、运行手册、
回执与报告。**论文 .tex 由 Codex 负责**；Claude 只做审稿、验证、实验与文档，
不碰 `paper-2027/` 其余部分的 `.tex`。

## 当前状态（2026-09-06 晚）

- **Round 12 已执行并得出结论**：零训练 Z 表在 OLMo 1B/7B 自己的 2×/4× 上
  全赢忠实未微调 YaRN（Y2 全 0/64）；Qwen2.5 官方 YaRN 配方给出外部上下界；
  评测审计确认结论不受 raw/chat 模式影响；E1 判定 128 步训练 DID_NOT_LEARN。
- **7B + LoRA + 500M tokens 主实验就绪未启动**：用户 09-06 指令停止全部实验、
  关机。全链脚本已就绪（数据未冻结、探针未跑）。**恢复清单见
  `round12_20260906/REPORT_ROUND12_20260906.md` §7**。
- **服务器已关机**（AutoDL westc，数据盘保留）。所有核心实验产物在服务器
  `/root/autodl-tmp/claude_round12_20260906/`，本地只存代码与文档，
  非核心的结果 JSON 已按用户指令清理（见 INDEX 清理记录）。

## 目录结构

| 目录 | 内容 | 状态 |
|---|---|---|
| `round12_20260906/` | 当前轮：预注册、RUNBOOK、代码、**总结报告** | **活跃主线** |
| `reports/` | 各轮执行报告（含状态栏） | 历史，已加更正 |
| `round11_20260905_olmo/` | OLMo 1B 跨家族轮（ZC/ZF/ON） | 已被 Round 12 取代 |
| `round10_20260905/` | §10 Qwen LoRA 轮 + 代码包快照 | 已被重置与 Round 12 取代 |
| `code/` | 两方向审计时代诊断脚本 ×3 | 已收官，留档 |
| `runbooks/` | 两方向审计开机手册 | 已作废 |

细目与每文件定位见 `INDEX.md`。

## 不变边界（全程有效）

- `code_release_*`（尤其 release008）与冻结历史 run 目录只读，禁改。
- 失败现场保留，不自动续跑；已暴露确认集不回灌训练、不调容差救场。
- 严格成功 = 完整答案精确匹配 + EOS；官方评分并报，不替代严格判分。
- 单 GPU 进程；诊断不产生新的参数选择权。
- 外推对比在各模型自己的 2×/4× 上做（OLMo 4K→8K/16K；Qwen 32K→64K/128K），
  flash attention 必须开。
- 不重启 351 盲标与 teacher-prefix 面板（用户裁决）。

## 关键入口

- 结论与恢复清单：`round12_20260906/REPORT_ROUND12_20260906.md`
- 本轮代码（与服务器 md5 一致，勿本地单方面改动）：`round12_20260906/code/`
- 冻结表/任务/数据的清单与哈希：`round12_20260906/PREGLUCTION_ROUND12.md` +
  `PREGLUCTION_ROUND12_V2_ADDENDUM.md`
- 解读备忘：`round12_20260906/PHASE0_INTERPRETIVE_MEMO.md`
