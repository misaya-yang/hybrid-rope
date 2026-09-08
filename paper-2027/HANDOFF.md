# 当前交接

更新：2026-09-08。阶段目标见 [README](../README.md#现阶段核心任务)。

## 当前状态

- 从 MrRoPE-Pro 出发做零训练改进，尚未取得本阶段的新能力结果。
- [MrPro-BM](../docs/research/ROPE_MRPRO_BM_PROTOCOL_20260908.md)已完成数组、代码及工作机 CPU 准备，尚未做 GPU 评测。当前准备使用 OLMo-2-0425-1B-Instruct；数学检查不能证明能力提升。
- [短评测准备](../docs/research/ROPE_OLMO_FAST_SCREEN_20260908.md)是四类自造任务，尚未落实作者要求的 RULER 浓缩子集。现有输入与评分不能直接当作该子集启动。
- 此前候选范围为最多十个有理论依据并经对抗审查的方案，目前只有 BM 准备完成；这是后续候选范围，不是先凑齐十个才能实验。
- 准备代码、实验材料和诊断脚本已随 `8ec9d1f` 提交并推送到 `main_0726_09_06`。

## 下一步

复用现有 [RULER 子集入口](../scripts/experiments/scale_transport/ruler_prepare.py)，将短评测改为能区分方法效果的 RULER 浓缩子集，完成必要的输入、解码和评分核对。随后在实际运行条件允许时，完成一次 MrPro 基线和 BM 比较；同条件基线供后续候选复用。根据真实得失决定下一项改进，不默认扩成完整矩阵。

## 工作机入口

- SSH：`ssh -p 27741 root@connect.westc.seetacloud.com`；Python：`/root/miniconda3/bin/python`。
- 准备目录：`/root/autodl-tmp/olmo_fast_screen_20260908/`；`code/` 为已上传代码，`prepared_bm_02/` 为旧自造任务输入，需按上述要求更新。
- 最近记录为无 GPU 模式；本次交接整理未连接服务器，电源、进程与可用资源需现场核实。
- [清理记录](../docs/overview/SERVER_STORAGE_CLEANUP_20260908.md)：旧语料及大缓存已清理，模型权重保留；历史路径不保证仍可重放。

## 历史边界

旧 L/P 诊断、近邻/远程、finite-phase replay、全量确认及 LoRA 队列均不属于当前接续工作。相关事实和纠正保留在 [失败复盘](../docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md)及原始记录中，按需查证。
