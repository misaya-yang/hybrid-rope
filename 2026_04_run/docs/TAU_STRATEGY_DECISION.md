# Progressive Training: tau 策略决策文档

> 核心问题：分阶段训练时，tau 应该随 L_train 变化（retarget）还是保持不变（delayed）？

---

## 结论（先说答案）

**必须用 Retarget 协议**，理由：
1. 论文 Phase 17C (seed=42) 的主结果就是用 retarget 跑的
2. 3-seed 复现必须与 seed=42 保持完全一致的实验协议
3. 高 token 预算 (500M+/stage) 下，retarget 是论文的实际配置

---

## 两种协议对比

### Delayed (EVQ-D): tau 固定不变
```
Stage 1 (L=512):  tau = d_head/√512  = 2.828
Stage 2 (L=1024): tau = 2.828 (不变)
Stage 3 (L=2048): tau = 2.828 (不变)
```
- 优点：频率分配一致性，不引入 stage 间的突变
- 缺点：Stage 3 时 tau 对 L=2048 偏大（最优应是 1.414），浪费 in-range PPL

### Retarget (EVQ-R): tau 按当前 L_train 重算
```
Stage 1 (L=512):  tau = d_head/√512  = 2.828
Stage 2 (L=1024): tau = d_head/√1024 = 2.000
Stage 3 (L=2048): tau = d_head/√2048 = 1.414
```
- 优点：每个 stage 的 tau 都接近最优，in-range PPL 代价最小
- 缺点：stage 间 inv_freq 发生变化，模型需要重新适应

---

## 实验证据

### EXP-4 (350M, M4, 低 token)

| Stage | 最长外推 | EVQ-D | EVQ-R | Winner |
|-------|---------|-------|-------|--------|
| S1 @4096 | 8× | 204.9 | 212.2 | **D (+3.6%)** |
| S2 @8192 | 8× | 104.6 | 108.2 | **D (+3.5%)** |
| S3 @8192 | 4× | 133.9 | 132.2 | R (-1.3%) |

**但 EXP-4 有严重缺陷**:
- Token 预算极低 (18M/9M/9M, 共 36M)
- Phase 17C 用了 500M/stage = **28× 更多 tokens**
- 低 token 结果不能外推到高 token 场景
- Stage 3 两者差距仅 1.3%，在噪声范围内

### Phase 17C (454M, GPU, 高 token) — 论文主结果

Phase 17C 脚本明确使用:
```python
TAU = d_head / math.sqrt(SEQ_LEN)  # retarget per stage
```

seed=42 的结果:
- EVQ+YaRN@16K: -34.6% → -52.0% → -81.2% (monotonic amplification)
- EVQ+YaRN@48K: PPL=2.63
- Passkey@8K+YaRN: 100%

**这就是 retarget 协议在高 token 下的实际表现。**

---

## 为什么 Retarget 在高 token 下更好？

物理直觉：
1. tau* = d_head/√L 是该 L 下的最优频率分配
2. 高 token 预算下，模型有足够训练量适应新的 inv_freq
3. Delayed 的 tau=2.828 在 L=2048 下过大，浪费 ~2-3% in-range PPL
4. 高 token 下这个 in-range 代价无法被外推收益补回

低 token 下的反直觉结果（Delayed 赢）可能是因为：
- 模型没有足够训练量适应 retarget 后的新频率
- 保持一致的 inv_freq 在低 token 下更稳定

---

## 3-seed 复现的具体 tau 值

| Seed | Stage 1 (L=512) | Stage 2 (L=1024) | Stage 3 (L=2048) |
|------|-----------------|------------------|------------------|
| 42 | tau=2.828 ✅ done | tau=2.000 ✅ done | tau=1.414 ✅ done |
| 43 | tau=2.828 ✅ done | tau=2.000 ❌ TODO | tau=1.414 ❌ TODO |
| 44 | tau=2.828 ✅ done | tau=2.000 ❌ TODO | tau=1.414 ❌ TODO |

**关键**：每个 stage 的 env var 设置:
```bash
# Stage 2: PHASE17C_SEQ_LEN=1024 → tau=64/√1024=2.0
# Stage 3: PHASE17C_SEQ_LEN=2048 → tau=64/√2048≈1.414
```
脚本内部自动计算 tau = d_head/√SEQ_LEN，不需要手动指定 tau。
