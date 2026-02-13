# 实验状态报告 (2026-02-13)

## 已完成实验

### ✅ 1. Evidence Chain 50M (3 Config × 3 Seed)
- **路径**: `results/evidence_chain_50m_3cfg3seed/`
- **结论**: Hybrid RoPE (α=0.2, θ=100k) 长序列外推最佳

### ✅ 2. Cross-Model WikiText Evaluation
- **路径**: `results/cross_model_wikitext_v1/`
- **结论**: 
  - LLaMA+geo 16k崩溃 (22x)
  - Sigmoid稳定LLaMA (1.08x)
  - Qwen YARN最佳 (PPL@16k < PPL@2k)

### ✅ 3. Qwen Hybrid LoRA
- **路径**: `results/qwen_hybrid_lora/`
- **结论**: LoRA后PPL上升但检索能力保持100%

### ✅ 4. 350M Final Training
- **路径**: `results/350m_final/`
- **训练**: 500M tokens
- **结论**: Hybrid在所有超参长度上优于geo

---

## 服务器状态

| 服务器 | GPU | 状态 | 用途 |
|--------|-----|------|------|
| AutoDL RTX 6000 | 98GB Blackwell | 🟢 在线 | 大规模实验 |

**连接命令**: `C:\Users\Admin\.ssh\connect_seetacloud.bat`

---

## 待办事项 / 下一步实验建议

### 选项 A: LLaMA Sigmoid 深入实验
- 目标：验证sigmoid对LLaMA的稳定效果
- 实验量：中等
- 预期收益：高 (22x -> 1.08x的改进值得深入)

### 选项 B: Qwen YARN 逆向工程
- 目标：分析Qwen为何PPL@16k < PPL@2k
- 实验量：低 (主要是分析)
- 预期收益：高 (理解SOTA长上下文方案)

### 选项 C: 1.5B 规模化实验
- 目标：在更大模型上验证Hybrid效果
- 实验量：高
- 预期收益：中高 (需要H100资源)

### 选项 D: 消融实验
- 目标：α和θ参数扫描
- 实验量：中等
- 预期收益：中 (优化超参)

---

## 实验规范

### 文件命名
```
{experiment_name}/
├── results.json    # 必需：结构化结果
├── run.log         # 必需：运行日志
├── summary.json    # 可选：训练摘要
└── README.md       # 可选：实验说明
```

### JSON格式
```json
{
  "timestamp": "YYYY-MM-DD_HHMMSS",
  "experiment": "name",
  "metadata": {
    "model": "...",
    "dataset": "...",
    "server": "...",
    "tokens": "..."
  },
  "results": { ... },
  "summary": { ... }
}
```

### 提交规范
- 每完成一个实验：更新 `results/README.md`
- 重大发现：更新 `docs/RESULTS.md`
- 新方法：更新 `docs/METHODOLOGY.md`

---

## 联系方式

- 仓库：https://github.com/misaya-yang/hybrid-rope