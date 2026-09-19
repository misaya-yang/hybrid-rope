# Llama3.1 官方 MrRoPE-Pro 单臂准备

2026-09-19。作者本轮授权在无卡机器写好代码，克隆 GPU 机器后再下载 Llama3.1。
**代码、CPU 检查和模型下载已完成；新克隆已清理旧资产，未运行预训练模型或 GPU 任务。**
最新作者要求：使用现有128K gate的完整13项、每项10条，总共130条。

入口：[official_single_arm.py](official_single_arm.py)。默认仅打印离线计划，
`--self-test` 为 CPU 随机小模型／数值检查，`--execute` 才加载预训练模型并评测。
脚本使用现有 Python 环境，不自动安装依赖、不自动下载模型。

## 科学合同

本轮只运行 **Llama-3.1-8B-Instruct + MrRoPE-Pro**，沿用官方方法与decoder，
输入改为作者指定的现有128K gate。这既不是原作者完整配方复现，也不是“保留旧 gate 输入与
decoder、只换权重”的配对诊断：旧gate的输出预算和停止规则与官方decoder不同，
不能把分数差完全归因于checkpoint。
上一份 [CPU 检查与诊断方案](CPU_TABLE_AND_CHECKPOINT_PLAN.md)保留作历史记录。

| 项目 | 固定配置 |
| --- | --- |
| 权重 | 本地 Llama-3.1-8B-Instruct，真实配置保留 131072 / llama3 scaling |
| 方法 | 会议补充材料 `LlamaMrRoPE.pro()`；不是 evalr.sh 的 Llama YaRN 示例 |
| RoPE | 原生参考8192、S16、base500000、dim128、32/1 turns、原始gain |
| 安装 | 调用官方 loader / patch，完整替换 `model.model.rotary_emb` |
| 精度／attention | 官方 BF16、无量化、FlashAttention2、device_map=auto；拒绝CPU/disk卸载 |
| 数据 | 现有 `tailspline_llama_s16_128k_gate/assets/full13/inputs.jsonl`，完整130条 |
| 数据版本 | gate上游RULER `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`；保留原manifest与source_parts |
| 覆盖 | 完整13项，每项源顺序10条；实际输入118900–130905 token |
| 分词 | 直接传入保存的prompt_ids，不重新分词、不添加模板／padding、不截断 |
| 解码 | 原始函数：30新token、num_beams=1、temperature=0.7、EOS或换行停止 |
| 采样 | 保留 checkpoint generation_config 的 do_sample，不强制greedy |
| 计分 | 原始包含匹配；QA任一reference命中，其余按reference比例；13任务等权 |

默认 `--samples 10`。结果必须注明每项10条，不能写成官方100条配置的复现。
官方示例默认只有7项，论文写13项；这里显式补齐13项，不声称恢复了作者未公开的历史命令。
作者未固定随机种子；本入口为每条输入固定seed以支持续跑，记录实际seed。
任务排序将NIAH放在前面；不按输出好坏选样，不增加第二个方法臂。

核心评测直接从归档zip中提取并执行原始 `evaluate_one_task`，每次传入一条源样本；
wrapper不重写生成参数；StoredInputTokenizer替换输入分词步骤以保留旧gate精确token序列，
decode和EOS／换行token查询仍委托新检查点tokenizer。分数从全部已存原始预测重聚合，避免单条四舍五入。
补充材料全部Python模块在临时目录展开，loader与patch不做兼容性改写。

可选 `--official-dataset` 仍支持官方HF预生成包，固定revision
`bb8903217d901fb534cc1ca615bd3245f3c1c39a`；当前不下载或运行该数据分支。

## 部署位置与运行

当前准备端点：`ssh -p 14600 root@connect.westd.seetacloud.com`，已确认无GPU。
可随数据盘克隆的代码目录：`/root/autodl-tmp/mrrope_official_20260919/code/`。
权重下载目录：`/root/autodl-tmp/models/Llama-3.1-8B-Instruct`。
Hugging Face不可直连、HF镜像返回403；使用ModelScope
`LLM-Research/Meta-Llama-3.1-8B-Instruct`的公开仓库，每个文件固定其API返回的revision。
仅下载Transformers格式的4个BF16 safetensors分片与配置／tokenizer／许可证，不重复下载原始pth。
后台下载状态保存在实验根的 `model_download_status.json`，完成状态需以该回执为准。
2026-09-19下载已完成：4个分片共约14.97GiB（含配置和tokenizer），291个BF16张量，
参数量8030261248；header与权重索引映射全部通过，原生长度131072。
本地回执：[下载状态](model_download_status.json)、[模型准备检查](model_readiness.json)。
已核对[两个仓库公布的文件元数据](model_source_identity.json)：4个权重分片的大小和SHA-256全部一致。
这一步只比较发布元数据，没有对16GB本地权重重复计算hash。

保留数据位于实验根的
`preserved_ruler/today_rope_plan_20260914/tailspline_llama_s16_128k_gate/`；入口默认指向其中的130条输入。
旧gate及其manifest、source_parts和结果一起保留，迁移映射见[清理回执](cleanup_receipt.json)。
[输入检查](inputs_readiness.json)确认130条、13任务各10条、token ID在Llama3.1词表内。

在该代码目录下：

```sh
python official_single_arm.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python official_single_arm.py --self-test
```

后续GPU与模型、依赖就绪后才执行：

```sh
python official_single_arm.py --execute \
  --model /root/autodl-tmp/models/Llama-3.1-8B-Instruct \
  --output /root/autodl-tmp/mrrope_official_20260919/results/full13_128k
```

中断续跑增加 `--resume`，要求配置、依赖版本、种子和样本数相同。
每行输出后立即flush，报告原子替换；如果系统中断导致末行JSON残缺，脚本会拒绝读取，
保留文件后检查并处理该残行，不能把损坏输出视作有效样本。
既有完整结果直接返回；不重新加载模型重跑已完成臂。

输出：`protocol.json`记录计划、checkpoint配置与generation配置；`runtime.json`记录
实际频率表、gain、设备和attention实现；`generations.jsonl`保存任务、源序号、数据版本指纹、
输入token身份／长度、原始输出token、预测、reference、停止参数和耗时；`report.json`保存逐任务结果。
未完成13项全量时不发布Full13平均分。

## 已完成验证与后续依赖

本地torch2.8.0 / transformers4.57.6及远端torch2.8.0+cu128 / transformers5.15.1均通过：

- 官方模块替换有原生llama3 scaling的随机小Llama；64项FP32频率及gain与已审计官方表精确一致。
- 位置0、8191、65535、131071的BF16 cos/sin一致。
- 小Llama完整前向与KV缓存分段前向一致；仅随机小模型，不是预训练模型实测。
- spy实际经过官方分词、generate、停止与decode路径，验证未添加do_sample覆盖。
- 新增输入适配器检查：保存的prompt_ids逐项一致，decode／stop委托不变。
- QA／多答案评分与官方函数相同；缺样本时Full13汇总保持为空。

当前服务器缺少 `flash_attn`，FlashAttention2需按后续GPU架构选择兼容构建。
默认保存输入分支不依赖datasets；只有可选HF数据分支需要该包。
CUDA预训练模型、FlashAttention2核函数、128K显存尚未实测。
作者明确授权清理后，删除新克隆上的旧资产78项，保留RULER目录27项、实验代码、运行环境与登录配置；
数据盘空闲从3994726400增至86438535168字节（下载模型之前），未操作旧端点上的资产。
模型下载后仍有70368653312字节（约65.5GiB）可用。
`--execute`会在加载大模型前检查依赖，避免把缺包拖到正式推理阶段。

高分支持“Llama3.1权重搭配官方MrPro与decoder具备高分能力”的假设；不单凭这一臂证明历史86.6实际使用的权重。
低分也如实报告，不调整任务、种子、stop rule来追逐86.6。
