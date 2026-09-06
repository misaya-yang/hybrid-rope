# Round 12 预注册（2026-09-06 执行日）

状态：**PREGLUCTION_ROUND12_V1**（执行前冻结；执行中不得修改下列承诺）
依据：Pro 两份文件 `HYBRID_ROPE_NEXT_DAY_PLAN_20260906.md`、`RoPE_ICLR2027_Research_Guidance_20260905.md`
模型：OLMo-2-0425-1B-Instruct（已就绪）+ OLMo-2-1124-7B-Instruct（今晚下载中）

## 0. 几何事实（已核验）
1B 与 7B：head_dim=128、rope_theta=500000、L0=4096、vocab=100352、rope_scaling=null
→ 默认频率谱逐位相同（native fp32 sha `dde15c31…`），同一套数值表可用于两模型。
`rope_tables.py verify` 已于今晚（2026-09-06 凌晨，CPU）通过双模型 config.json
断言后冻结 `ROUND12_STATIC_TABLES_FROZEN_V1`（服务器
`/root/autodl-tmp/claude_round12_20260906/tables/manifest_round12.json`）：
- N: `dde15c31…`（=冻结引擎 native sha，torch HF 配方；注意 numpy pow 有 1 ulp 差，禁用）
- Z: `56ddfae2…`（round-11 逐字节，amp 1.102585782722872）
- Y: `cc9da456…`（round-11 逐字节，amp 1.138629436111989）
- M: `ed0120e3…`（MrRoPE-Pro §3.2 推导，band d_l=14（32-turn）、d_h=32（1-turn）、
  n=18，YaRN 振幅 1.138629436111989）

## 1. 路线 A：零训练静态表矩阵（预算 ≤3 GPU-h）
- 矩阵：{1B, 7B} × {N, Z, Y(YaRN-s4), M(MrRoPE-Pro-s4)}，每臂一张静态表+固定 gain，
  全请求长度同表，无 native routing，无增益扫描，无头选择（冻结后追加扫描=违规）。
- 表冻结规则：N=HF 默认（sha 必须等于冻结引擎 native sha）；Z/Y=round-11
  fixed_controls 逐字节复制（双哈希+amplitude 核验）；M=按 MrRoPE ICLR2026 §3.2
  公式推导（32-turn/1-turn 带界、ε_j 权重、累积乘积除法、YaRN 1/t 振幅），不拟合任何基准。
- 任务：RULER 单键/多键各 32 实例 ×4K/8K/16K（种子 12 一次性生成）+ round-11
  single_evidence 自然 QA 原样复用（仅 split=train、worlds 0/1、按首次出现序，
  2048: 32 实例×2 世界=64 行；8192/16384: 各 64 实例×2 世界=128 行）
  + 7B 先跑 64 条 compact 资格，再跑 4K/8K/16K。
- 评分双轨并列：strict（完整答案精确+EOS，group 全 world）与官方
  （RULER substring、QA EM/F1）同时记录；round-11 已有的逐条 receipt 格式不变。
- 精确 receipt 复用许可：与既有冻结结果逐位相同的格子允许引用旧 receipt，不重跑。

## 2. 路线 B：双臂训练（预算 ≤12 GPU-h，仅 1B）
新配方命名：**R12_DUAL_ARM_CPT_SFT_V1**（不复用冻结引擎名）。
- 两臂：Y-CPT 与 Z-CPT。臂的静态表+gain 安装后全程冻结；可训练部分仅为
  LoRA r16/α16/dropout0（全部 7 类 linear）+ 全部 RMSNorm affine + input embedding。
  冻结：原 linear、LM head、rotary（inv_freq 与 gain）。
- 优化器：AdamW lr 2e-5、betas (0.9,0.95)、wd 0、clip 1.0、5% warmup→cosine 2e-6；
  Phase B 边界重置优化器。
- Phase A（CPT）：512 更新 ×4×16K 真实文档（PG19 1024 + FWE 1024，文档级互斥划分
  train/val），token 上限 33,554,432 硬断言；每更新 ≤2K native replay，全词表
  teacher KL（采样位置），权重 1；teacher=同权重 native（amplitude 1.0）。
- Phase B（SFT）：64 更新 ×8（2×QA@8K + 2×QA@16K + 4×binding 双世界）；
  binding 世界黄金同步；损失仅在答案+EOS。
- 存档：CPT128 / CPT256 / CPT512 / SFT64；adapter + norm/embedding 增量同存 +
  manifest（sha、步数、tokens、挂钟）。
- 候选选择规则（预注册）：以 CPT512/SFT64 为候选；先比 Native（不得整体回退），
  再比生成质量；暴露后不得重选检查点。
- Replay 池：native_pool_v3（rows_sha256 `282e5740…`，4 组×128 train）。

## 3. 数据
- PG19：liyucheng/pg19 train parquet（deepmind 原版 GCS 不可达，弃用）；
  FWE：`fineweb_edu/sample/10BT`。文档 <16385 token 直接丢弃，不填充伪造长度。
- 工程注记（不改变上述选择规则）：无卡容器 cgroup=2GiB 内存+0.5 核，
  CPT 制备改为三段低资源管道（duckdb 向量化扫描×2 → 候选 jsonl 落盘 →
  tokenizers-only 可续跑 encode）；tokenizers 与 AutoTokenizer 的
  `encode(add_special_tokens=False)` 已用 72 条真实/合成文本验证逐位一致
  （`data/cpt/equiv_result.json`，ids_sha256 `1c738c7f…`）。
- SFT 视图 512 条 = 256 QA（round-11 transport_views single_evidence，split=train、
  worlds 0/1，8K/16K 各 64 实例×2 世界=128 视图，原样流式复制）+ 256 binding（种子 12，
  本轮新合成，双世界黄金同步）。
- 注意：round-11 16K single_evidence 在 test split 有同 (实例,世界) 重复行；
  本轮一律只取 split=train，避免 test 行混入训练或重复计分。

## 4. 评估与第一天通过线
- Native：2 文本域×64 + PIQA/ARC-Easy/HellaSwag×256 + 旧 strict 套件回归；
  通过线：整体 ≥88%，逐项披露；低于线则该臂判负、不得以“接近”开脱。
- 16K：生成内容相对未训练基座增加（双轨分数并列呈现）。
- compact 27→25（92.59%）不构成否决（Pro 已撤回该附加失败条件）。

## 5. 预算
Day 1 ≤20 GPU-h：A ≤3h、B ≤12h、其余 ≤5h（调试/评估）。项目总 ≤100 GPU-h。
单 GPU 进程；失败现场保留不自动续跑；无卡模式今晚只做下载与代码。

## 6. 禁止清单（本轮生效）
改表、加扫描、换检查点、改评分口径替代 strict、把官方分数当唯一口径、
跨轮混用未冻结视图、超预算不报告、对 7B 做训练（本轮仅零训练）。
