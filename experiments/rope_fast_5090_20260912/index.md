# 5090：E2 / E3 与 Q/K GPU 诊断

2026-09-12。优先级低于 [LoRA 能力恢复](../olmo_recovery_20260912/index.md)，独立于另一 session 管理的 S1 三臂训练。当前准备范围及服务器状态见 [运行记录](../../docs/research/next_stage_20260912/5090_recovery_preparation.md)。

- `e2_prepare.py` / `e2_run.py` / `e2_score.py`：原 778 条自然 QA 的七配置匹配复评，631 长输入五任务等权统计、gain 交互和短窗成本。输入池已被历史实验使用；身份是强基线补齐。
- `e2_extra_score.py`：将追加单臂与已完成Native/BM/MrPro按同一631条长输入做配对文档簇bootstrap；追加比较属于开发证据。
- `e3_tables.py`：按计划精确重建 C42/C42V24，记录实际 FP32 tensor、相同总位移与质心。
- `e3_prepare.py` / `e3_validate.py`：原七任务族的新 700 条 16K、280 条 4K 输入，核对旧面板及来源身份。
- `e3_run.py` / `e3_postprocess.py`：共享输入的两表完整生成及原始输出审计。
- `e3_strong_baselines_postprocess.py`：复用同一980条输入，合并C42/C42V24与BM/MrPro强基线，验证四臂完整性并输出任务等权配对差和分层bootstrap。
- `qk_diagnostic.py`：G1，复用E2五任务各2条长输入，实际Q/K系数及同gain相位重放；20次backbone forward，不等4080产物。
- CPU 检查：`e2_test.py` 和 `tests/test_e3_rope_fast_5090.py`；它们不提供模型能力证据。

执行入口默认 dry-run；显式 `--execute` 才使用 GPU。资格检查、正式完整面板、训练和历史结果分别标记。

- `supplement_queue.py`：追加队列入口；默认列任务，空闲GPU执行单项。
- `p1_factorial.py`：距离/局部间隔/干扰数，384次生成。
- `source_counterfactual.py`：4K/16K双来源world，768次生成。
- `layer_group_probe.py`：OLMo前中后层BM/MrPro交换，360次新生成并复用E2。
