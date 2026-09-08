# 当前交接

只保留最新且接手必需的信息；状态变化时替换旧内容，不追加历史流水账。

- **目标：** 从 MrRoPE-Pro 出发做零训练位置编码改进，在同条件比较中取得真实的长上下文能力增益。
- **已有准备：** [MrPro-BM 候选与实现](../docs/research/ROPE_MRPRO_BM_PROTOCOL_20260908.md)，使用 OLMo-2-0425-1B-Instruct；代码和 CPU 准备已完成，尚无 GPU 评测结果。
- **下一步：** 把现有自造短评测改为 RULER 浓缩子集。可复用 [RULER 输入入口](../scripts/experiments/scale_transport/ruler_prepare.py)和[短评测运行框架](../scripts/experiments/olmo_fast_screen/)。核对输入与评分后，完成一次 MrPro 基线与 BM 比较，分析结果并推进改进；同条件基线复用。
- **工作机：** `ssh -p 27741 root@connect.westc.seetacloud.com`；Python 为 `/root/miniconda3/bin/python`。准备目录 `/root/autodl-tmp/olmo_fast_screen_20260908/`，其中 `prepared_bm_02/` 仍是自造任务输入。最近记录为无 GPU 模式，接手时核实现场状态。
