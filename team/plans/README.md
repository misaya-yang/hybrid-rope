# Experiment Plans Index

本目录包含所有已设计的实验计划。每个计划回答三个问题:
1. 测试什么 claim?
2. 已有什么证据?
3. 下一步跑什么实验?

---

## 计划清单

| 文件 | 状态 | 描述 | 论文关联 |
|------|------|------|---------|
| `phase22_qa_mix_experiment.md` | 📋 设计完成 | QA-weighted 下游混合训练 | Fig 5 downstream |
| `phase21_scrolls_downstream.md` | ✅ 已执行 | SCROLLS + QuALITY 下游评估 | Fig 5, Table A1 |
| `phase20_1_5b_spotlight.md` | ⏸️ 搁置 | 1.5B scale-up spotlight | 扩展证据 (如资源允许) |
| `phase19_progressive_chain_125m.md` | ⏸️ 搁置 | 125M progressive 训练链 | 方法论验证 |
| `phase17b_454m_512_to_1024_continue.md` | ✅ 已执行 | 454M L=512→1024 续训 | Fig 4 Stage 2 |
| `capacity_compensation_hypothesis.md` | 📋 设计完成 | Scale/training sufficiency 假说 | §6 Limitations |
| `DSR.md` | 📋 设计完成 | Distance sensitivity ratio 检索 | 潜在新实验 |
| `base_generalization_sweep.md` | 📋 设计完成 | RoPE base (8K-100K) sweep | Appendix ablation |
| `theory_strengthening_roadmap.md` | 📋 进行中 | 理论 polish 路线图 | §3 Theory |

### 状态说明
- ✅ 已执行: 实验已完成，结果在 `docs/exp/` 中
- 📋 设计完成: 方案已设计，等待 GPU 时间
- ⏸️ 搁置: 因资源/优先级暂停

### 优先级建议
1. **P0**: Phase 22 QA-mix (如果仍需下游差异化)
2. **P1**: Theory strengthening (投稿前必须完成)
3. **P2**: Base generalization sweep (reviewer 可能问到)
4. **P3**: 1.5B spotlight, DSR (camera-ready 或 follow-up)
