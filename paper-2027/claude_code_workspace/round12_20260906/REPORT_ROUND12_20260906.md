# Round 12 实验报告（2026-09-05 ~ 09-06，用户指令中止于 09-06）

> **Review status, 2026-09-06:** the new cross-audit questions Y2 fidelity and several score/causal interpretations below. Local Y2 and pinned-equation code use different ramp/amplitude conventions; executed tensor/scorer parity and raw receipts still require reconciliation. Preserve these historical numbers as reported; do not treat the faithful-baseline/winner/ceiling labels as newly validated. The recovery list is historical; [current REVISION_BRIEF](../../REVISION_BRIEF.md) defines current planning and grants no restart authorization.

机器：AutoDL westc 4080 SUPER 32GB。工作目录 `/root/autodl-tmp/claude_round12_20260906`（B12）。
成功标准（用户定义，两次修正）：在各模型**自己的** 2×/4× 上赢未微调 YaRN（忠实版 = Y2 臂）。
OLMo（4K 训练）→ 8K/16K；Qwen2.5（原生 32K）→ 64K/128K。flash attention 开。

## 1. 核心结果：Z 表（零训练）在 YaRN 失效处全赢

Track A 全矩阵 {OLMo 1B, 7B} × {N, Z, Y, M, Y2}，冻结静态表（manifest ROUND12_STATIC_TABLES_FROZEN_V1，hash 校验）。

**Y2 = 忠实未微调 YaRN（√t 增益 + ramp，按论文实现）在 OLMo 上全 0：**
| 臂 | 1B near@8K | 1B far@16K | 7B near@8K | 7B far@16K |
|---|---|---|---|---|
| Y2 | 0/64 | 0/64 | 0/64（EOS 0.203） | 0/64（EOS 0.445, em 0） |
| **Z** | **29/64** | 2/64 | **37/64 组**（90/128 行） | 4/64 组（13/128 行，EOS 0.35） |
| Y/M | 20/64 | 0 | 20/64 | 0（M EOS 0.0，过锐化随规模加重） |
| N | 0 | 0 | 0 | 0 |

按成功标准：**Z 在 1B/7B、2×/4× 全赢 Y2**。2× 强（无训练即救起），4× 仍有裕度缺口。
7B Z compact 28/32，ruler 单键 4096 strict 18/32。

## 2. Qwen2.5 外部对照（原生 32K + 官方 HF YaRN，ruler_official/32）

同任务同判分，全部 0 OOM（128K 峰值 25.6G，base 分块 prefill chunk=2048）：
- 7B：4× = 26（81%），2× = 31（97%）；native32k sanity 7/8
- 1.5B：4× = 29（91%），2× = 30（94%）
- 0.5B：4× = 12（38%），2× = 20（63%）

结论：出货配方在 ≥1.5B 上很强（外部上界）；0.5B 4× 退化到 38% → 谱预算随容量/训练量走。
同任务在 OLMo 上 Y2 全 0 → 我们的方法赢在 YaRN 失效处。

## 3. 评测模式审计（raw vs chat-template，用户质疑项）

Qwen2.5-Instruct 在 raw completion 模式原生 32K 也崩（复述背景、无 EOS）→ 全家族已改
chat-template 模式复跑（上表即 template 结果）。**对 OLMo 补同式审计**：

| 格 | compact | near@8K | far@16K |
|---|---|---|---|
| 1B N：raw → chat | 32→31/32 | 0→0 | 0→0 |
| 1B Z：raw → chat | 27→24/32 | 29→22/64 组 | 2→1/64 组（行 6/128，EOS 0.516） |
| 7B N：raw → chat | 28/32（行 58/64） | 0→0 | 0→0 |
| 7B Z：chat | 用户中止（~10%，部分样本已存） | — | — |

**结论：OLMo 无 Qwen 式 raw 崩溃**；chat 模式分数略低（模板信封 ~29 tok），
Track A 结论全部成立且方向保守（raw 略乐观）。

## 4. E1 诊断（Pro 统一方案 §4.3 分支读数）

- 交叉格：W_ZF×T_0 far/near 全 0（compact 0.84 保留）→ 学到的行为锁表；
  W_ON×T_Z near 0.3125、far 0.09375 → Z 表部分赋能未适配权重。
- 拟合轨迹（ZF）：train-far strict 12→66/256、val-far 5→19/128、EOS 0.48→0.92，
  但 worst_margin 仍 −2.4、frac_positive 仅 14%（val）。拟合 ON 全 0。
- **判定：DID_NOT_LEARN**（Pro §4.4 梯度陷阱）——干预须改变监督到达的内容，
  不只是换表/换权重。与用户"之前欠训练"判断收敛 → 500M 实验动机。
- 台账修正：e1_fit_readout.py 陈旧契约常数（不对应任何产物）已改为参考记录 +
  根因注记（`CONTRACT_CHECK_ANNOTATION.json`），判定不受影响。

## 5. 分叉/分解诊断（09-05，恒等式全过）

- 换表效应 97.6% 落在背景位；L0 纯直接相位，深层 98% 为表示漂移；
  ON direct 恒 0、Z0≡ZF direct（LoRA 不进旋转算子）。
- §5.2：native 把 98% 注意力质量给插入背景（mass_S=1.8%）；Z 表回收到 22%、写差 −57%，
  但 far strict 仍 1-4/32 → 质量回收了、筛选没学会（与 E1 一致）。
- 分叉判别对序 30/32（E=6.1-6.9）；ZF 增益全部是 margin 翻正、对序不变。
- far@16K 卡的是 4× 总长，不是证据距离（near@16K 仅 3/32）。

## 6. 500M 主实验（就绪未启动，用户中止）

配方 `R12_7B_CPT_500M_V1` = V1 配方 + 恰好三处工程改动（零配方改动）：
1. **教师预缓存**：native_pool_v3 512 行全带固定 prediction_positions（已验 512/512），
   KL 目标离线算好（~0.6GB bf16）→ 训练不载教师，7B 32G 可装（探针把关）。
2. **数据**：500,129,328 tokens = 30,592 段 × 16,384（PG19-only，每书 ≤8 连续头部段，
   避开 V1 val 分片，validation 复用 V1 冻结 npy；tokenizer 1B≡7B sha 已验）。
3. **检查点**：里程碑 + 16 步滚动 resume（含优化器）。
Phase A 7,648 updates + Phase B 64 SFT；估墙钟 ~5-6 天。

中止时状态：脚本全完成并同步（data_prep_cpt_v2 / build_kl_cache / track_b_train_v2 /
probe_7b_train / chain_500m*，md5 已核）；PG19 已下 14/15 片（第 15 片半截，可续传）；
数据未冻结、探针未跑、训练未开始。各日志末尾均有 `USER_HALT` 标记。

## 7. 恢复清单（下次开机按序）

1. 开机验资产：`B12/{tables,tasks,data/cpt,datasets/pg19,track_a_audit}`、
   OLMo-2-1B/7B 模型、`runs/Z_CPT_500M`。
2. 补下载：`bash code/dl_pg19_more.sh`（hf 自动续传；日志出 `PG19_DL_DONE` 即成）。
3. 补审计 7B Z chat 格（单独跑 `track_a_eval.py --arm Z --chat-template ...`，
   参考 `driver_audit_olmo_chat.sh` 第 4 步参数；部分样本已在 `olmo7b_Z_chat/`）。
4. 冻结数据：`wait_and_prep_500m.sh`（或手动调 data_prep_cpt_v2.py）→
   检查 `CPT_DATA_FROZEN_V2_500M` 后 `rm cpt_500m/partial_train.bin`（磁盘）。
   磁盘决策待定：代码含 125M 里程碑，预算文件说删之——开机先核对（空闲 ~10G）。
5. `chain_500m_waiter.sh`：探针（≤29G / 16K micro≤60s）→ KL 缓存 → 训练。
   探针失败备案：数据重建 8K 序列（~1h，CPU）。

## 8. 遗留

- 4× 裕度缺口：所有臂 ≤2-4/32-64（零训练上限已现；需 500M 训练验证）。
- compact 保留代价：Z 系 24-26/32 < ON 27/32（1-2 组）。
- 1B 数据缩放曲线（33M/150M/500M）未启动。
- 盲标 351 例 / teacher-prefix 诊断：用户裁决不重启，保持 pending。
