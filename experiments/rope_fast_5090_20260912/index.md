# 5090：E2 / E3 冻结模型比较

2026-09-12。优先级低于 [LoRA 能力恢复](../olmo_recovery_20260912/index.md)，独立于另一 session 管理的 S1 三臂训练。当前准备范围及服务器状态见 [运行记录](../../docs/research/next_stage_20260912/5090_recovery_preparation.md)。

- `e2_prepare.py` / `e2_run.py` / `e2_score.py`：原 778 条自然 QA 的七配置匹配复评，631 长输入五任务等权统计、gain 交互和短窗成本。输入池已被历史实验使用；身份是强基线补齐。
- `e3_tables.py`：按计划精确重建 C42/C42V24，记录实际 FP32 tensor、相同总位移与质心。
- `e3_prepare.py` / `e3_validate.py`：原七任务族的新 700 条 16K、280 条 4K 输入，核对旧面板及来源身份。
- `e3_run.py` / `e3_postprocess.py`：共享输入的两表完整生成及原始输出审计。
- CPU 检查：`e2_test.py` 和 `tests/test_e3_rope_fast_5090.py`；它们不提供模型能力证据。

执行入口默认 dry-run；显式 `--execute` 才使用 GPU。资格检查、正式完整面板、训练和历史结果分别标记。
