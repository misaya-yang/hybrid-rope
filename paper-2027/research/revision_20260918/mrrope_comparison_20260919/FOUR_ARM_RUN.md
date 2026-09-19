# Llama3 / Llama3.1 × MrRoPE / TailSpline

2026-09-19，作者授权在新克隆 `connect.westd.seetacloud.com:42853` 上依次运行四臂。
旧端点上的数据已完整克隆；不重建RULER、不重新校验模型大文件。
Llama3曾按清理授权删除，本轮已并行恢复下载，2026-09-19 06:22 UTC确认全部完成。

## 固定合同

顺序：Llama3.1–MrPro → Llama3.1–TailSpline → Llama3–MrPro → Llama3–TailSpline。
每臂相同13任务各10条、共130条已存prompt_ids；源顺序、references、seed一致。
全部BF16无量化，单张RTX PRO 6000 Blackwell Server Edition（96GB）。

作者明确要求FlashAttention2。使用官方 `flash-attn 2.8.3.post1` 预编译包，
CUDA12、torch2.8、Python3.12、CXX11 ABI TRUE。安装包与通过gh CLI取得的官方包
SHA-256一致，且已通过实际BF16 causal GPU kernel检查。**正式四臂不使用SDPA。**
先前尝试在读取配置时失败，没有生成任何SDPA样本；旧启动日志保留。

MrPro调用会议补充材料的原始构表与forward。TailSpline读取既有gate的
`tables/tailspline.json`中冻结的`table.values_float32`和gain，使用同一个rotary forward。
两个检查点都用base500000、64对、8192参考、S16；Llama3.1真实128K配置仍如实记录。

四臂共享Llama3.1的generation_config，官方评测函数覆盖temperature=0.7、30新token、
num_beams=1、EOS或换行停止；不依检查点改变采样设置。
输入适配器保留旧gate token IDs，decode与stop-token查询使用各自tokenizer。
这不是原作者HF预生成数据的逐项复现，也不是旧gate decoder的重跑。

评分沿用官方QA任一reference命中、其他任务reference命中比例，最终13任务等权。
队列完成后核对四臂逐行task、row_id、input_ids身份、references、decoder kwargs和seed一致。
不以尚未完成的子任务分数作Full13结论。

## 入口与状态

[执行器](official_single_arm.py)、[顺序队列](four_arm_queue.py)。
远端根：`/root/autodl-tmp/mrrope_official_20260919`。

- 代码：`code/official_single_arm.py`、`code/four_arm_queue.py`。
- 队列：`results/four_arm_fa2/queue_status.json`、根目录`four_arm_queue.log`与`.pid`。
- 逐臂：`results/four_arm_fa2/<family>_<method>/`下protocol、runtime、generations和report。
- 全部完成：`results/four_arm_fa2/comparison.json`。
- Llama3恢复下载：根目录`llama3_download_status.json`和`llama3_download.log`。

队列使用文件锁防止重复启动；每臂成功完成才进入下一臂，任何失败都会停止。
进入Llama3前要求其下载完成。样本逐行flush；支持同协议中断续跑。

四臂均已完成，每臂130条、13任务齐全；远端queue_status为COMPLETE。
本地同步520条原始输出后独立重评分、配对身份及频率表检查全部通过。

| 检查点 | MrPro | TailSpline |
|---|---:|---:|
| Llama‑3.1‑8B | 54.15% | 49.54% |
| Llama‑3‑8B | 0.51% | 18.59% |

完整结果、逐任务差异、旧协议比较及来源边界见[最终实验报告](FOUR_ARM_REPORT.md)。
[原始汇总](four_arm_results/comparison.json)、[离线审计](four_arm_results/audit.json)。
15分钟监控已暂停；没有新增或重跑实验，机器未由本任务关闭。
启动与内核回执作为历史记录保留：[队列启动](four_arm_start_receipt.json)、[FA2内核](fa2_readiness.json)。

## 软件兼容

Transformers5.15把RoPE参数移入`rope_parameters`，执行器仅恢复官方loader期待的
`rope_theta`／`rope_scaling`属性别名，保持参数值不变。
该版本可能没有`hf_device_map`；从实际parameter设备记录并拒绝CPU/meta参数。
两个启动阶段的失败均无生成样本，失败protocol保留在`results/startup_failures/`。
正式运行期间不修改执行器。
