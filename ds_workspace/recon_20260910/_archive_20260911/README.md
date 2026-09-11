# 归档说明

这些文档**不是错的**，而是**已被同一 campaign 后续的结果覆盖或修正**。
保留它们是因为它们是"解空间收缩"的证据链——用户要求从失败中学习。

**为什么归档（逐条）**：
- `CORRECTION_QWEN_20260911.md` —— 里面的"Qwen 上 MrRoPE 反超 BM 7.3 点、
  两模型族方向相反"**已被撤回**：归档自己的判决文件是 `NO_LONG_GAIN`（6胜/4负）。
  见 `../CROSS_MODEL_VERDICT_20260911.md`。
- `CORRECTION_KNEE_20260911.md` —— 它更正了"拐点在 mean(m)≈0.63"，
  但它自己给出的替代解释（"约束存在但不在带内"）后来被实测收紧为
  "**带内免费、代价只在带外**"。见 `../CONSTRAINT_IS_SLACK_20260911.md`。
- `PLAN_20260911.md` —— 判决树已被后来跑掉的实验覆盖（KNIFE 2×2 已跑、
  `beta_b3` 被它砍掉却是冠军）。处置表见 `../index.md` 与旧 `INDEX`。
- `KKT_ANALYTIC / BAND_FORCING / CONTRACTION_* / REANCHOR_KKT /
  RESOLUTION_MISSING_TERM / CORRECTIONS_FROM_PRO / TURN_WINDOW_THEORY /
  EVQ_UNIFY / K6_PREFLIGHT / PHASE1_ROUND1 / DOC_SYNTHESIS / RESULT_* /
  WHY / SEP_CAUSE / QWEN_INSTRUMENT` —— 早期（09-10 及 09-11 上午）的分析与规划，
  其结论要么被机器精度验证吸收进 `../FOUR_CORNERS`，要么被后续实测证伪。
- `INDEX_20260911.md` —— 被 `../index.md` 取代。
