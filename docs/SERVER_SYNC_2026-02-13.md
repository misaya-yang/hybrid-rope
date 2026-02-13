# 服务器同步记录（2026-02-13）

## 目标

将两台服务器的实验产物统一同步到本地仓库，便于直接上传 GitHub 和给导师复核。

## 来源服务器

1. A100: `117.50.192.217:23`
2. A800: `117.50.220.66:23`

## 同步范围

1. `/opt/dfrope/results/**` 中的日志、JSON、文本、脚本辅助文件
2. `/opt/dfrope/*.py` 和关键 `.sh/.log`
3. `logs/` 目录（A800）

## 排除规则

1. 权重文件：`*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.ckpt`, `*.gguf`, `*.onnx`
2. 大体积 token cache：`*.u16`, `*.npy`, `*.npz`

## 关键处理动作

1. 发现 A800 同脚本重复启动（两个 PID），已保留一个、终止一个，避免重复写日志与资源竞争。
2. 清理 A100 同步出的超大缓存文件（约 900MB+），保留可复现必需材料。
3. 生成按机器分层归档目录：
   - `/Users/misaya.yanghejazfs.com.au/dfrope/github_bundle/server_artifacts_2026-02-13/a100`
   - `/Users/misaya.yanghejazfs.com.au/dfrope/github_bundle/server_artifacts_2026-02-13/a800`

## 当前进度快照

1. A100 `run_100m_scaling.py`：`geo_500k` 已完成，`hybrid_a0.2_t100k` 训练中。
2. A800 `run_llama3_hybrid_lora_v3.py`：单进程持续训练中，GPU 持续占用。

## 后续追加（同日更新）

1. A100 新增 `run_50m_theta_shape_factorial.py`，用于修复 baseline 公平性争议：
   - 扫描 `geo_100k/200k/300k/500k`；
   - 对照 `hybrid_a0.2_t100k` 与 `hybrid_a0.2_t500k`；
   - 每组 3 seeds，统一训练/评测设置。
2. A800 LoRA 微调继续推进，loss 持续下降，进度约超过 1/3。

## 已同步文件规模

1. 归档目录总大小约 `3.1M`
2. 文件总数 `120+`

## 备注

如果后续要做最终论文材料打包，建议在这个目录基础上再追加：

1. 自动生成图表（PPL 曲线、对比柱图）
2. 一页式结论摘要（结论 + 风险 + 下一步）
