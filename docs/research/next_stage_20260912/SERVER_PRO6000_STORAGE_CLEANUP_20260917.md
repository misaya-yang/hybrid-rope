# Pro 6000服务器清理与Mistral资产（2026-09-17）

服务器：`connect.westd.seetacloud.com:51638`。

## 清理结果

- 数据盘：`82/87 GB`（94%）降至`65/87 GB`（75%），释放约16.76 GiB；
- 系统盘：删除本机已有4080副本的GLM-4-9B后，从69%降至11%，空余约27 GB；
- 保留Llama-3-70B NF4、Llama-3-8B、全部128K/256K CPU评测资产、tokenized
  panels、InfiniteBench、正式逐行输出和当前Git代码。

删除内容限定为旧激活、训练checkpoint、停用Qwen-1.5B权重、missing队列、
buggy/aborted/canary目录和重复代码压缩包。`fixed_rope.../panels`、
`tailspline_*128k*/assets`、`official_yarn_full13/panels`以及confirm40 CPU资产均未删。

GLM删除前身份回执：`config.json` SHA256
`98794d8150da03ca7baec63208584810d01caf43427e376071c00e770f4b4fa7`；
同一模型仍保留在4080服务器。

## 本地raw镜像

只同步论文核心的128K/256K与Llama-3-70B结果，不同步prompt正文、权重或激活：

- 本地ignored目录：`results/paper_positive_raw_20260917/pro6000/`；
- 大小约10 MiB；422个紧凑文件；72个逐行raw；
- 远端/本地SHA256：72/72一致；
- `ARCHIVE_MANIFEST.json` SHA256：
  `a558eeac43905d0a90be8e2a31962d22bc7f8d17b37fb138f7b172cfc802ca34`。

覆盖Qwen/GLM 128K三臂、Qwen 256K、Llama S16 128K、InfiniteBench，以及
Llama-3-70B的32K RULER/Natural-QA/PPL和128K PPL/NIAH边界输出。

## Mistral下载身份

下载在后台串行执行；本记录不把“已启动”写成“已完成”。

| 角色 | checkpoint / revision | 路径 | 权重 |
|---|---|---|---|
| 原始基座 | `mistralai/Mistral-7B-v0.1` / `27d67f1b5f57dc0953326b2601d68371d40ea8da` | `/root/autodl-tmp/models/Mistral-7B-v0.1` | safetensors，两片约13.49 GiB |
| YaRN继续训练 | `NousResearch/Yarn-Mistral-7b-128k` / `d09f1f8ed437d61c1aff94c1beabee554843dcdd` | `/root/models/Yarn-Mistral-7b-128k` | PyTorch bin，两片约13.49 GiB |

大权重通过ModelScope镜像串行下载；其对象路径对应HF公布的LFS对象。config、
tokenizer和YaRN remote code从固定revision的HF镜像下载。后台流程最后对四个权重
运行SHA256；日志位于
`/root/autodl-tmp/mistral_download_20260917/download.log`。

预期HF权重SHA256：

- base shard 1：`9742cb4764964155b7a5f35eefad651f590006091ddeb536863d6c5865cca1b9`；
- base shard 2：`9bcf56354ec0c68b5f8e97b4f3b02d16af899a65b0868d6dba5a51c1b30f01cb`；
- YaRN shard 1：`ac92349639a9b81ff8d3f4af00a36729895fb46c54d014f1e95d9c2f64ee1539`；
- YaRN shard 2：`2a8faed6f7a31940b678ce98d986ed5d5fbfb8c1bc8e7dc00159bae92e019757`。
