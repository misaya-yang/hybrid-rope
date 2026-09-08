# 当前交接

只保留最新且接手必需的信息；状态变化时替换旧内容，不追加历史流水账。

- **目标：** 从 MrRoPE-Pro 出发做零训练位置编码改进，在同条件比较中取得真实的长上下文能力增益。
- **已有准备：** [MrPro-BM 候选与实现](../docs/research/ROPE_MRPRO_BM_PROTOCOL_20260908.md)，使用 OLMo-2-0425-1B-Instruct；代码和 CPU 准备已完成，尚无 GPU 评测结果。
- **当前工作：** 作者已要求完成并运行混合短评测；[六任务 RULER 面板](../docs/research/ROPE_OLMO_FAST_SCREEN_20260908.md)代码已改为4K/16K共36条、官方任务与评分；工作机相关20项CPU测试通过。输入生成按作者要求已停止，完整面板及GPU比较尚未完成。
- **工作机：** `ssh -p 27741 root@connect.westc.seetacloud.com`；Python 为 `/root/miniconda3/bin/python`。准备目录 `/root/autodl-tmp/olmo_fast_screen_20260908/`，其中 `prepared_bm_02/` 仍是自造任务输入。现场为无GPU模式，作者限定该模式只处理代码和下载，不运行分词生成或模型评测。旧输入准备 `prepared_ruler_01` 至 `03` 未完成；下一 session 的准备与启动命令见短评测协议“下一 session 启动”。作者计划在新 session 进行约一小时自主实验，本次未开卡或启动计时。
