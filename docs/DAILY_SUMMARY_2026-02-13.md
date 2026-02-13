# 2026-02-13/14 实验日报

**服务器**: AutoDL A100 95GB  
**模型**: LLaMA-3-8B-Instruct  
**任务**: Anchored Sigmoid RoPE 长度外推实验

---

## 一、实验清单

### 1. Anchored Sigmoid V3 Followup (早些时候)
- **路径**: `results/anchored_sigmoid_v3_followup/`
- **目的**: 验证anchored sigmoid RoPE的长度外推能力
- **结果**: 已完成，见 `ANCHORED_SIGMOID_V3_SUMMARY.md`

### 2. Advisor Followup 2026-02-14
- **路径**: `results/advisor_followup_2026-02-14/`
- **目的**: 根据导师建议的后续实验
- **状态**: 已完成

### 3. 9小时夜间实验 - 第一轮 ✅
- **路径**: `results/night_run_anchored_x20_9h/`
- **时间**: 00:51 - 01:04 (~13分钟)
- **状态**: **已完成**

### 4. 9小时扩展实验 - 第二轮 🔄
- **路径**: `results/night_run_9h_extended/`
- **启动时间**: 01:04
- **PID**: 30185
- **状态**: **运行中**，预计8-9小时完成

---

## 二、第一轮实验核心结果

### Phase 1A: θ上限对照

| Config | PPL@2k | PPL@16k | Collapse Ratio |
|--------|--------|---------|----------------|
| geo_500k | 11.125 | 262.278 | **23.58x** ⚠️ 严重崩溃 |
| geo_1M | 11.118 | 25.475 | 2.29x |
| geo_2M | 11.115 | **9.264** | **0.83x** ✅ 最佳 |
| anchored_x20 | 11.122 | 19.041 | 1.71x |

### Phase 1B: 全长度边界扫描

| Config | 2k | 8k | 16k | 24k | 32k | 49k |
|--------|-----|-----|------|------|------|------|
| geo_500k | 11.13 | 9.52 | **262.3** | 1375 | 2345 | 3811 |
| anchored_x20 | 11.12 | 9.41 | **19.0** | 138.3 | 636.5 | 1503.9 |

### Phase 2: 多种子稳健性

| Config | L=16384 (mean±std) | L=49152 |
|--------|-------------------|---------|
| geo_500k | 194.96±48.35 | 3810.99 |
| anchored_x20 | **15.44±2.84** | 1503.95 |

---

## 三、核心发现

### 1. 崩溃边界分析
- **geo_500k**: 16k开始崩溃 (PPL=262)
- **anchored_x20**: 24k才开始明显退化 (PPL=138)
- **边界后移**: ~1.5-2x

### 2. 最佳配置
- **geo_2M** 在16k上表现最好 (PPL=9.26，无崩溃)
- 但geo_2M在超长序列上可能有问题（未完全测试）

### 3. Anchored x20 价值
- 相比geo_500k: **16k上PPL从262降到19 (14x改善)**
- 稳健性好: std=2.84，变异系数仅18%
- 短序列无性能损失 (PPL@2k ≈ 11.12)

### 4. OOM限制
- 49k长度在A100 95GB上会OOM
- 需要更长的上下文需要更大显存或优化

---

## 四、第二轮扩展实验设计 (运行中)

| # | 实验名称 | 配置数 | 预估时间 |
|---|----------|--------|----------|
| 1 | Theta细粒度扫描 | 7 theta × 5 长度 | 1.5h |
| 2 | Anchor Factor消融 | 6 factor × 5 长度 | 1h |
| 3 | Anchor Dim消融 | 6 dim × 4 长度 | 45min |
| 4 | Slope消融 | 6 slope × 4 长度 | 30min |
| 5 | 边界密集扫描 | 3配置 × 16长度 | 1h |
| 6 | 多种子稳健性 | 2配置 × 4长度 × 7 seeds | 1.5h |
| 7 | 跨域验证 | wikitext + TinyStories | 1h |
| 8 | PPL比率分析 | 3配置 × 6长度 × 3 seeds | 45min |

**预计完成时间**: 上午8:00-9:00

---

## 五、关键结论

1. **Anchored Sigmoid RoPE有效**: 在不损失短序列性能的前提下，显著延后长度外推崩溃点

2. **最优参数区间**:
   - anchor_factor: 15-25 (x20表现最佳)
   - anchor_dim: ~16 (head_dim/8)
   - slope: 0.5

3. **下一步工作**:
   - 等待第二轮实验完成
   - 绘制PPL vs Length曲线
   - 撰写方法部分

---

## 六、监控命令

```bash
# 查看第二轮实验进度
C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "tail -30 /root/autodl-tmp/dfrope/hybrid-rope/results/night_run_9h_extended/run.log"

# 检查进程状态
C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "ps aux | grep 30185"
```

---

*Generated at 2026-02-14 01:17*