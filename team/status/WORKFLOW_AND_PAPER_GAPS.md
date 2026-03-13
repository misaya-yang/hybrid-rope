# EVQ 项目：工作流防空转方案 + 论文核心矩阵加固

> 日期: 2026-03-12
> 背景: 项目积累了约 200-250 GPU 小时的空转浪费，根因是计划→执行链条缺乏结构化验证。同时论文需要补强理论-实验-图表三大矩阵。

---

# 第一部分：GPU 防空转方案

## 一、问题诊断：为什么 AI Handoff 不够

AI_HANDOFF_PITFALLS.md 是一份**被动文档**——它要求执行者主动去读、去记、去遵守。但 LLM agent 的特点是：

1. **上下文遗忘**: 长 session 后容易忘记早期约束
2. **过度自信**: 觉得"差不多就行"，不会主动对照 checklist
3. **路径依赖**: 看到一段旧代码就直接复制，不验证是否已被 deprecate
4. **缺乏反馈**: 跑了 5 小时 GPU 才发现参数错了，没有 early fail 机制

**AI Handoff 的根本缺陷**: 它是给人看的文档，但执行者是 AI。需要把约束从"文档"变成"代码"。

## 二、解决方案：三层防御体系

### 第一层：实验配置文件化 (Config-Driven)

**核心思路**: 不再让 AI 自己拼命令行参数，而是用结构化 YAML 配置文件定义实验，配置文件自身包含验证逻辑。

```yaml
# experiments/phase21e_quality_clean.yaml
experiment:
  name: "Phase 21E QuALITY Clean Protocol"
  phase: "21e"
  date: "2026-03-13"

model:
  tier: "454m"
  hidden_size: 1024
  num_heads: 16
  head_dim: 64  # DERIVED: hidden_size / num_heads, 不允许手动设置

rope:
  method: "evq"  # geo | evq
  base: 500000
  tau: null  # null = auto-compute from tau_formula
  tau_formula: "head_dim / sqrt(seq_len)"  # 强制声明公式

training:
  seq_len: 8192
  steps: 25000
  lr: 1.0e-5
  warmup: 500
  dropout: 0.1
  micro_batch_size: 1
  grad_accum: 4
  init_checkpoint: "${EVQ_INIT}"  # 环境变量引用
  checkpoint_type: "progressive"  # progressive | retrofit — retrofit 会触发警告

yarn:
  train_time: false
  eval_time: true
  implementation: "progressive_smoothstep"  # 只允许这个值，ntk_aware 会被拒绝
  scale_formula: "eval_seq_len / seq_len"

eval:
  seq_lengths: [8192, 16384]
  scoring_mode: "options_nll"  # 不允许 "first_char_match"
  protocol: "standard_no_padding"  # standard_no_padding | distractor_padded
  eval_samples: 200

# 自动验证规则（执行前必须全部通过）
validation_rules:
  - "head_dim == hidden_size // num_heads"
  - "tau == head_dim / math.sqrt(seq_len) if tau is null"
  - "checkpoint_type != 'retrofit'"
  - "yarn.implementation == 'progressive_smoothstep'"
  - "scoring_mode != 'first_char_match'"
  - "steps >= 10000"  # 防止训练量不足
```

**好处**:
- AI agent 只需要填 YAML，不需要记住所有约束
- 验证规则在配置文件里，执行前自动检查
- 配置文件可以 git 追踪，出了问题可以 diff 回溯
- 不同 session 的 AI agent 看到的是同一份配置，不会各自理解不同

### 第二层：Preflight 验证脚本 (Zero-GPU Gate)

**核心思路**: 在任何 GPU 训练开始之前，必须通过一个 0-GPU 的验证脚本。验证失败 → 训练被阻断。

```python
# scripts/preflight_check.py
"""
Preflight validation — 在 GPU 训练前运行，检查所有已知踩坑点。
运行方式: python preflight_check.py --config experiments/phase21e.yaml
通过才允许启动训练。
"""

import yaml, math, sys, os, torch

def run_preflight(config_path):
    cfg = yaml.safe_load(open(config_path))
    errors = []
    warnings = []

    # ═══ 1. τ 参数验证 ═══
    head_dim = cfg['model']['hidden_size'] // cfg['model']['num_heads']
    if head_dim != cfg['model']['head_dim']:
        errors.append(f"head_dim 不一致: 声明 {cfg['model']['head_dim']}, "
                      f"实际 {head_dim} (hidden_size/num_heads)")

    expected_tau = head_dim / math.sqrt(cfg['training']['seq_len'])
    actual_tau = cfg['rope']['tau'] or expected_tau
    if abs(actual_tau - expected_tau) / expected_tau > 0.05:
        errors.append(f"τ 偏差过大: 声明 {actual_tau:.4f}, "
                      f"理论值 {expected_tau:.4f} (head_dim/√seq_len)")

    # ═══ 2. YaRN 实现验证 ═══
    if cfg['yarn'].get('implementation') == 'ntk_aware':
        errors.append("YaRN 实现被设为 ntk_aware — 这是已知 bug，"
                      "会摧毁 EVQ 频率结构。必须用 progressive_smoothstep")

    # ═══ 3. 评测协议一致性 ═══
    if cfg['eval']['scoring_mode'] == 'first_char_match':
        errors.append("评分模式 first_char_match 已废弃 — "
                      "83.84% 选项首字符不唯一。必须用 options_nll")

    # ═══ 4. Checkpoint 血统检查 ═══
    ckpt_type = cfg['training'].get('checkpoint_type', 'unknown')
    if ckpt_type == 'retrofit':
        warnings.append("Checkpoint 类型是 retrofit（Geo pretrain → EVQ finetune）。"
                        "这不是 clean pair，EVQ 继承了 Geo 的 attention pattern，会打折扣。"
                        "建议用 progressive checkpoint。")

    # ═══ 5. 训练量合理性 ═══
    steps = cfg['training']['steps']
    bs = cfg['training']['micro_batch_size'] * cfg['training']['grad_accum']
    total_samples = steps * bs
    if total_samples < 50000:
        warnings.append(f"总训练样本 {total_samples} 偏少（FIRE 用了 3.2M）。"
                        f"考虑增加步数或 batch size。")

    # ═══ 6. inv_freq 数值验证 ═══
    # 实际构建 inv_freq 并检查范围
    from run_evq_sweep import evq_cosh_inv_freq
    inv_freq = evq_cosh_inv_freq(
        head_dim=head_dim, tau=actual_tau,
        base=cfg['rope']['base']
    )
    if inv_freq.min() < 1e-8:
        warnings.append(f"inv_freq 最小值 {inv_freq.min():.2e} 过小，"
                        f"可能导致 float32 精度丢失")
    if torch.isnan(inv_freq).any():
        errors.append("inv_freq 包含 NaN — 检查 τ 和 base 的组合")

    # ═══ 7. 已废弃脚本检查 ═══
    deprecated_scripts = [
        'phase21b_quality_eval.py',  # YaRN bug
    ]
    # 检查 config 中是否引用了废弃脚本
    config_str = str(cfg)
    for script in deprecated_scripts:
        if script in config_str:
            errors.append(f"配置引用了已废弃脚本 {script}")

    # ═══ 输出结果 ═══
    print("=" * 60)
    print(f"PREFLIGHT CHECK: {config_path}")
    print("=" * 60)

    if errors:
        print(f"\n❌ BLOCKED — {len(errors)} 个致命错误:")
        for e in errors:
            print(f"  ✗ {e}")
        print("\n训练被阻断。修复以上错误后重新运行 preflight。")
        sys.exit(1)

    if warnings:
        print(f"\n⚠️  {len(warnings)} 个警告:")
        for w in warnings:
            print(f"  ! {w}")

    print(f"\n✅ PASSED — 所有检查通过。τ={actual_tau:.4f}, "
          f"head_dim={head_dim}, base={cfg['rope']['base']}")
    print("可以启动 GPU 训练。")
```

**使用流程**:
```bash
# 1. AI agent 先生成/编辑配置文件
# 2. 运行 preflight（0 GPU，几秒钟）
python scripts/preflight_check.py --config experiments/phase21e.yaml

# 3. 通过后才启动训练
python scripts/core_text_phases/phase21b_scrolls_finetune.py \
    --config experiments/phase21e.yaml
```

### 第三层：Dry-Run 模式 (1-GPU-Minute Gate)

**核心思路**: 用 1 分钟 GPU 时间跑一个 micro-experiment，验证整个 pipeline 端到端能跑通。

```python
# 在 phase21b_scrolls_finetune.py 中加入 --dry_run 模式
if args.dry_run:
    print("═══ DRY RUN MODE ═══")
    args.steps = 10          # 只跑 10 步
    args.eval_samples = 5    # 只评 5 个样本
    # 跑完后验证:
    # 1. loss 不是 NaN
    # 2. 生成的文本不是乱码
    # 3. eval metric 有合理值（不是 0 或 100%）
    # 4. checkpoint 能保存和加载
    # 5. 打印完整配置摘要供人类确认
```

**使用流程**:
```bash
# 1. Preflight 通过后
# 2. Dry run（~1 分钟 GPU）
python phase21b_scrolls_finetune.py --config phase21e.yaml --dry_run

# 3. 人类确认 dry run 输出合理
# 4. 正式训练
python phase21b_scrolls_finetune.py --config phase21e.yaml
```

## 三、流程改造总结

### 旧流程（当前）

```
你写计划(md) → AI读计划 → AI拼命令行 → GPU跑几小时 → 发现bug → 重来
                ↑ 可能没读完    ↑ 参数可能错      ↑ 浪费
```

### 新流程（改造后）

```
你写计划(md)
    ↓
AI 生成 YAML 配置文件（结构化，有约束）
    ↓
Preflight 验证（0 GPU，自动检查 τ/YaRN/metric/checkpoint）
    ↓ 通过
Dry Run（1分钟 GPU，端到端验证）
    ↓ 通过
你确认 dry run 输出
    ↓ 确认
正式训练（GPU 全速）
    ↓
自动生成 report.md（归档）
```

**关键改变**: 在 GPU 空转之前加了两道 gate。第一道 0 成本，第二道 1 分钟成本。只有通过这两道 gate 才允许正式训练。

### 防护覆盖矩阵

| 历史踩坑 | Preflight 检查 | Dry Run 检查 | 配置文件约束 |
|---------|---------------|-------------|------------|
| τ 算错 | ✅ 公式验证 | — | ✅ tau_formula 字段 |
| d_head 搞混 | ✅ 自动计算 | — | ✅ 自动推导 |
| YaRN 用了 NTK-aware | ✅ 拒绝 ntk_aware | — | ✅ 只允许 progressive |
| 评分用首字符匹配 | ✅ 拒绝 first_char | — | ✅ 只允许 options_nll |
| Checkpoint 是 retrofit | ⚠️ 警告 | — | ⚠️ 标注类型 |
| 训练量太少 | ⚠️ 警告 | — | — |
| inv_freq NaN | ✅ 数值检查 | ✅ loss 检查 | — |
| 生成截断 | — | ✅ 生成长度检查 | — |
| eval 协议不匹配 | ✅ 配置一致性 | ✅ 端到端验证 | ✅ 声明式配置 |
| 脚本已废弃 | ✅ 黑名单检查 | — | — |

---

# 第二部分：论文核心矩阵加固

## 一、理论矩阵现状与缺口

### 已完成的理论（可以写 Theorem/Proposition）

| ID | 内容 | 数学状态 | 论文位置 |
|----|------|---------|---------|
| T1 | 频率分配是变分逆问题 | ✅ 精确定理 | §4.1 |
| T3 | φₖ(τ) = 1-(1/τ)arcsinh((1-uₖ)sinhτ) 是闭式解 | ✅ Euler-Lagrange | §4.1 |
| T4 | Geometric 是 τ=0 边界情况 | ✅ 精确推论 | §4.1 |
| T6 | Sub-cycle 单调减少 Δρ>0 | ✅ sinh 凸性证明 | §4.5 |

### 需要补强的理论（当前不够严格）

| ID | 内容 | 当前状态 | 需要做什么 | 优先级 |
|----|------|---------|-----------|--------|
| T2 | 宽带投影 K≈αI+βA⁻¹ | R²>0.99 在特定条件下 | **写成正式 Lemma，明确声明边界条件**：D(Δ)∝1/Δ, base∈[8K,100K], L≥4096 | 高 |
| T5 | Waterbed 不等式 | 定性描述，未正式证明 | 写 Appendix 证明（Jensen 不等式 + 约束条件） | 中 |
| T7 | τ*=d_head/√L | 纯经验公式 | **必须改措辞**："empirical scaling conjecture" 而非 "scaling law" | **高** |
| T8 | Fisher → attention 桥接 | 局部启发式 | 降级为 "Remark" 或 "Intuitive interpretation" | 低 |

### 具体行动

**行动 1（高优先）**: τ* 措辞修正

当前问题: mainstory.md 里把 τ*=d_head/√L 叫做 "parameter-free scaling law"，听起来像定理。

修正为:
> **Empirical Scaling Conjecture.** We propose τ* = d_head/√L as a near-optimal default. This conjecture is validated across 99 independent runs (27 configurations × 3 seeds), achieving exact optimality in 3/9 configurations and top-3 ranking in 8/9. We do not derive this formula from first principles; its theoretical status remains an open question.

**行动 2（高优先）**: 宽带投影边界条件形式化

当前问题: §4.1 中宽带投影是核心假设，但边界没有写成定理形式。

写成:
> **Lemma 2.1 (Broadband Projection).** Under the conditions: (i) distance prior D(Δ) ∝ 1/Δ, (ii) RoPE base ∈ [8K, 100K], and (iii) context length L ≥ 4096, the phase-collision kernel admits the rank-2 projection K(φ₁,φ₂) ≈ αI + βA⁻¹ with coefficient of determination R² > 0.99 in the mid-frequency band.

**行动 3（中优先）**: Waterbed 不等式 Appendix 证明

写一个 1-page appendix:
- 从 Jensen 不等式出发
- 证明 ∫ln E(φ) dφ ≥ ln b - ln c
- 给出短程代价上界和长程收益下界的显式表达

## 二、实验矩阵现状与缺口

### Claim × Evidence 矩阵（精简版）

| Claim | 核心证据 | 统计强度 | 缺口 | 优先修复 |
|-------|---------|---------|------|---------|
| **C1**: Geo 是特例 | 数学证明 | A+ | 无 | — |
| **C2**: τ* 近最优 | 99 runs, 3 seeds | A+ | τ* 措辞需改 | 高 |
| **C3**: EVQ+YaRN 协同 | 454M 48K PPL<3.3 | A- (单 seed) | **Stage 2-3 多 seed 确认** | **关键** |
| **C4**: 渐进训练超线性 | 34.6→52→81.2% | B+ (单 seed) | **同上** | **关键** |
| **C5**: Waterbed 下游验证 | 13 task NLL ±4.4% + QuALITY Gold NLL -30% | A | 无关键缺口 | — |
| **C6**: 零参数优于学习式 | 125M 3-seed vs DAPE | A | 无 | — |
| **跨尺度一致性** | 50M-750M | A | 无 | — |

### 关键实验缺口

**缺口 1（关键）: Phase 17c 多 seed 确认**

状态: Stage 1 (seeds 43/44) ✅ → Stage 2 🔄 进行中 → Stage 3 ❌ 待跑

这是论文最大的单点风险。如果 Stages 2-3 不确认方向，C3/C4 从 A 降到 B+。

时间估算: ~1-2 周（取决于 GPU 可用性）

**缺口 2（补强）: Inference Latency 测量**

当前: 论文声称 "zero inference overhead" 但没有实测数据。

需要做: 一个简单的 wall-clock benchmark:
```python
# 10 次前向传播取平均
for method in ['geo', 'evq', 'geo_yarn', 'evq_yarn']:
    latencies = benchmark_forward(model, method, seq_len=8192, n_runs=10)
    print(f"{method}: {mean(latencies):.2f}ms ± {std(latencies):.2f}ms")
```

时间: ~1 小时 GPU + 1 小时代码

**缺口 3（补强）: 置信区间补全**

Phase 17c 报告 EVQ+YaRN@48K = 2.63 但没有不确定性。需要:
- 如果单 seed: 至少报告不同 chunk 的 PPL 方差
- 如果多 seed: 报告 seed 间方差

## 三、图表矩阵现状与缺口

### 主论文图表预算（9 页正文）

| 位置 | 内容 | 状态 | 占页 |
|------|------|------|------|
| Fig 1 | 频率动力学（training loss + passkey + PPL） | ✅ 存在 | 0.5 |
| Fig 2 | **EVQ+YaRN 协同**（4线 PPL 图 2K-48K） | ✅ **核心图** | 0.7 |
| Fig 3 | PE 主导极端外推 | ✅ 存在 | 0.5 |
| Fig 5 | QuALITY Gold NLL + accuracy | ✅ 存在 | 0.5 |
| Fig 6 | τ* 验证（99-run） | ✅ 存在 | 0.5 |
| Table 1 | 跨尺度 PPL 一致性 | ✅ 就绪 | 0.3 |
| Table 2 | EVQ+YaRN 主结果 | ✅ 就绪 | 0.4 |
| Table 8 | 13 task NLL 反转 | ✅ 就绪 | 0.4 |
| Table 10 | QuALITY n=2086 | ✅ 就绪 | 0.3 |
| **总计** | | | **~4.1** |

### 缺失的关键图表

| 图表 | 内容 | 为什么需要 | 数据来源 | 工时 | 优先级 |
|------|------|-----------|---------|------|--------|
| **Collision-block 图** | base vs collision fraction，解释 base=10K 为什么失败 | **Reviewer 防御必备** — 唯一的负面结果需要理论解释 | Phase 18 + collision formula | 2-3h | **关键** |
| NLL 任务分解图 | QA vs non-QA 的 NLL gap 可视化 | 比表格更直观，强化 "EVQ 优势集中在 QA" 的叙事 | Phase 21a 数据 | 1h | 高 |
| τ sweep heatmap | 多配置 PPL gap vs τ/τ* | 视觉化 "shallow basin" | Phase 16 数据 | 2h | 中 |
| Training loss 曲线 | Geo vs EVQ 训练过程 | 说明短程 waterbed 在训练中可见 | 训练日志 | 1h | 中 |
| Inference latency 对比 | 前向传播时间 Geo/EVQ/+YaRN | 证明 zero overhead | 待测 | 1h | 中 |

## 四、综合优先级栈

### P0（阻塞提交）

1. **17c 多 seed Stage 2-3** — 论文最大单点风险
2. **Collision-block 图** — base=10K 负面结果的理论解释
3. **τ* 措辞修正** — "scaling law" → "empirical conjecture"
4. **宽带投影形式化** — Lemma 2.1 + 显式边界条件

### P1（强 accept → spotlight）

5. **NLL 任务分解可视化** — 强化 Claim 5 叙事
6. **Inference latency 实测** — 确认 zero overhead
7. **Waterbed 不等式 Appendix 证明** — 补齐理论严格性
8. **置信区间补全** — PPL 主结果的不确定性

### P2（锦上添花）

9. **τ sweep heatmap** — 视觉化 shallow basin
10. **Training loss 曲线** — 训练过程中的 waterbed
11. **Reproducibility appendix** — 超参数卡
12. **Phase 18 多 seed** — base 泛化确认

### P3（有余力再做）

13. **1.5B 验证** — Phase 20 路线
14. **更多下游任务** — SCROLLS 其他子集
15. **代码开源准备** — GitHub repo

## 五、实施建议

### 关于工作流改造

1. **立即**: 把 preflight_check.py 写好，下次实验前必须跑
2. **本周**: 把现有的 Phase 21E 配置写成 YAML，作为模板
3. **持续**: 每个新实验都先生成 YAML → preflight → dry run → 正式跑

### 关于论文加固

1. **本周**: τ* 措辞修正 + 宽带投影 Lemma（纯写作，0 GPU）
2. **本周**: Collision-block 图（Phase 18 数据已有，只需画图）
3. **等 GPU**: 17c Stage 2-3 多 seed（最高优先，不可缩短）
4. **穿插**: Inference latency 测量 + NLL 任务分解图

---

*文档版本: v1.0 | 2026-03-12*
