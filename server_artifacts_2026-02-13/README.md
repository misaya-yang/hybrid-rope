# 服务器归档快照（2026-02-13）

这个目录用于 GitHub 上传前的“可复现实验材料归档”，来源于两台训练服务器：

- `a100/`：`117.50.192.217`（A100 80GB）
- `a800/`：`117.50.220.66`（A800 80GB）

## 归档原则

- 仅同步：代码、日志、JSON 结果、配置与说明文件。
- 明确排除：模型权重与大体积二进制缓存（例如 `.pt`、`.bin`、`.safetensors`、`.u16`）。
- 目录按机器拆分，保留 `results/` 原始层级，方便回溯。

## 目录说明

- `a100/scripts/`：A100 端运行脚本快照（含 `run_100m_scaling.py` 等）
- `a100/results/`：A100 端实验日志与结果
- `a100/logs/`：A100 顶层监控日志
- `a100/meta/`：A100 清单信息
- `a800/scripts/`：A800 端脚本快照（含 `run_llama3_hybrid_lora_v3.py`）
- `a800/results/`：A800 端实验日志与结果
- `a800/logs/`：A800 下载/运行日志
- `a800/meta/`：A800 清单信息

## 状态快照（同步时）

- A100：`run_100m_scaling.py` 正在跑 `hybrid_a0.2_t100k`，`geo_500k` 已完成并写入 `results.json`。
  - `geo_500k`: `PPL@2048=6.457`, `PPL@16384=10.888`
- A800：`run_llama3_hybrid_lora_v3.py` 仅保留单进程运行（已去重），LoRA 训练持续推进。

## 补充（同日后续）

- A100 已新增 `run_50m_theta_shape_factorial.py`（位于 `a100/scripts/`）：
  - 目标：做 theta 与 shape 的因子分离公平对照；
  - 组合：`geo_100k/200k/300k/500k` + `hybrid_a0.2_t100k/t500k`；
  - seeds：`[42, 123, 7]`。

## 建议上传方式

优先上传本目录下：

- `a100/scripts/*`, `a100/results/*`
- `a800/scripts/*`, `a800/results/*`
- `a100/meta/*`, `a800/meta/*`

并在 commit message 中注明：`exclude weights and token cache`.
