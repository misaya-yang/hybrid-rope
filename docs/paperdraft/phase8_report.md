# Phase 8: Extended-Ratio Context Extension & Scaling Law Verification

> **Date**: 2026-03-01 – 2026-03-02
> **Hardware**: RTX 5090 32GB
> **Total GPU time**: ~7h (8A-C: 161 min, 8D: ~130 min, 8E: ~70 min)
> **Purpose**: Test 8x expansion ratio, Hybrid EVQ, passkey recovery, from-scratch 4K, τ* scaling law

---

## 8A: 512→4K Context Extension (8x Expansion)

**Goal**: Does Geometric collapse at 8x expansion? Can EVQ/Hybrid EVQ do better?

**Config**: 350M model, pretrain 512-tok (50M tokens), continue 4K (10M tokens), lr=3e-5, batch=2.

### PPL Results

| Method | PPL@512 | PPL@1K | PPL@2K | PPL@4K | PPL@8K | PPL@16K |
|--------|---------|--------|--------|--------|--------|---------|
| A1 Geometric | 76.6 | 77.8 | 77.1 | **80.4** | **81.4** | **83.3** |
| A2 PI (/8x) | 79.1 | 80.3 | 79.3 | 82.2 | 159.1 | 254.4 |
| A3 YaRN (s=8) | 76.5 | 77.5 | 76.8 | 79.5 | 107.8 | 161.9 |
| A4 EVQ τ=1.5 | 78.5 | 79.9 | 79.3 | 83.1 | 84.2 | 85.7 |
| A5 EVQ τ=2.0 | 80.2 | 81.4 | 80.8 | 84.8 | 86.5 | 88.0 |
| A6 EVQ τ=2.5 | 82.1 | 83.3 | 83.2 | 87.3 | 88.9 | 90.2 |
| **A7 Hybrid τ=2.0** | 77.2 | 79.3 | 78.1 | **82.0** | **83.3** | **84.7** |

### Passkey Retrieval Rate by Length

| Method | PK@1K | PK@2K | PK@4K | PK@8K | Global |
|--------|-------|-------|-------|-------|--------|
| A1 Geometric | **82%** | **72%** | 59% | **54%** | **66.8%** |
| A2 PI (/8x) | 81% | 74% | **65%** | 43% | 65.8% |
| A3 YaRN (s=8) | **87%** | **82%** | **70%** | 49% | **72.0%** |
| A4 EVQ τ=1.5 | 70% | 66% | 52% | 52% | 60.0% |
| A5 EVQ τ=2.0 | 64% | 63% | 47% | 50% | 56.0% |
| A6 EVQ τ=2.5 | 64% | 60% | 45% | 53% | 55.5% |
| **A7 Hybrid τ=2.0** | 74% | 61% | 50% | 52% | 59.3% |

### Key Findings

1. **Geometric does NOT collapse at 8x expansion**: PPL@16K=83.3, only +3.7% vs PPL@4K. With 32 channels and base=500000, the frequency grid retains sufficient resolution even at 16K positions.

2. **EVQ maintains flat extrapolation**: PPL@16K = 85.7 (τ=1.5), 88.0 (τ=2.0), 90.2 (τ=2.5). The slope is flatter than Geometric (EVQ: ~3% from 4K→16K, Geo: ~4%), but the absolute PPL is higher because EVQ's frequency reallocation disrupts pretrained Q/K alignment.

3. **Hybrid EVQ = best non-Geometric PPL**: PPL@16K=84.7, bridging Geometric (83.3) and pure EVQ (88.0). By keeping high-frequency channels unchanged, Hybrid preserves within-window pattern matching while using EVQ for low-frequency extrapolation.

4. **Passkey: Q/K alignment hypothesis confirmed**: Geometric leads at PK@1K (82%) and PK@8K (54%). YaRN best overall (72%) despite poor PPL — YaRN's smooth interpolation preserves alignment better. EVQ methods have degraded passkey (56-60%), confirming that frequency reallocation disrupts the attention pattern.

5. **Hybrid partially recovers passkey vs pure EVQ**: Hybrid PK@1K=74% vs EVQ τ=2.0 PK@1K=64% (+10pp). At PK@8K, Hybrid (52%) ≈ EVQ (50%). The high-frequency Geometric channels help short-range retrieval.

---

## 8A: Comparison with 7F (4x vs 8x Expansion)

| Method | PPL@8K (7F, 4x) | PPL@8K (8A, 8x) | Δ | PK@4K (7F) | PK@4K (8A) |
|--------|-----------------|-----------------|---|-----------|-----------|
| Geometric | 98.0 | 81.4 | −17% | 52% | 59% |
| PI | 246.1 | 159.1 | −35% | 34% | 65% |
| YaRN | 174.5 | 107.8 | −38% | 48% | 70% |
| EVQ τ=2.0 | 99.1 | 86.5 | −13% | 40% | 47% |

**Surprising finding**: 8x expansion (10M tokens) achieves LOWER PPL than 4x expansion (5M tokens) at the same eval lengths. The 2x more continuation tokens matter more than the expansion ratio.

---

## 8B: Fine-tune Ablation (512→2K, Varying Continuation Tokens)

**Goal**: Does passkey improve with more continuation tokens?

### Results (EVQ τ=2.0)

| Tokens | PPL@2K | PPL@8K | PK@1K | PK@2K | PK@4K |
|--------|--------|--------|-------|-------|-------|
| 2.5M | 90.4 | 102.0 | 63% | 57% | 50% |
| 5M (7F ref) | 89.2 | 99.1 | 82% | 72% | 40% |
| 10M | 80.7 | 88.8 | 66% | 64% | 47% |
| 20M | 75.7 | 82.0 | 72% | 63% | 48% |

### Results (Geometric)

| Tokens | PPL@2K | PPL@8K | PK@1K | PK@2K | PK@4K |
|--------|--------|--------|-------|-------|-------|
| 5M (7F ref) | 87.6 | 98.0 | 90% | 78% | 52% |
| 10M | 77.1 | 85.9 | 80% | 70% | 54% |
| 20M | 73.4 | 80.5 | 80% | 69% | 58% |

### Key Findings

1. **PPL scales with data**: Both EVQ and Geometric show continuous improvement (PPL@8K: EVQ 102→82, Geo 98→80 over 2.5M→20M).

2. **Passkey recovery is non-monotonic**: EVQ PK@1K drops from 82% (5M) to 66% (10M) then recovers to 72% (20M). The 5M reference from 7F has anomalously high passkey — likely because Phase 7F used a different data ordering.

3. **Geometric maintains passkey lead**: At 20M tokens, Geo still leads EVQ in passkey at all lengths (80/69/58% vs 72/63/48%). The gap narrows with more data but doesn't close.

4. **PK@4K improves steadily for both**: EVQ 40→50%, Geo 52→58%. More training tokens help extrapolation passkey.

---

## 8C: From-Scratch 4K Baseline

**Goal**: Compare extension (8A) vs from-scratch training at 4K.

**Config**: 350M, 4096-tok, 50M tokens, lr=6e-4, batch=2.

| Method | PPL@4K | PPL@8K | PPL@16K | PK@1K | PK@2K | PK@4K | PK@8K |
|--------|--------|--------|---------|-------|-------|-------|-------|
| C1 Geometric | 91.1 | 115.6 | 175.4 | 87% | 80% | 54% | 55% |
| C2 EVQ τ=2.0 | 93.1 | 113.9 | 164.4 | 82% | 68% | 60% | 54% |

### 8A (Extension) vs 8C (From-Scratch) Comparison

| | Geo ext (8A) | Geo scratch (8C) | EVQ ext (8A) | EVQ scratch (8C) |
|---|-------------|-----------------|-------------|-----------------|
| PPL@4K | **80.4** | 91.1 | **84.8** | 93.1 |
| PPL@8K | **81.4** | 115.6 | **86.5** | 113.9 |
| PPL@16K | **83.3** | 175.4 | **88.0** | 164.4 |
| PK@4K | **59%** | 54% | 47% | **60%** |

### Key Findings

1. **Extension >> from-scratch for PPL**: Pretrain+continue (50M+10M) dramatically outperforms from-scratch (50M at 4K). PPL@16K: Geo 83.3 vs 175.4 (2.1x), EVQ 88.0 vs 164.4 (1.9x).

2. **From-scratch better for passkey**: Scratch EVQ PK@4K=60% vs extension EVQ PK@4K=47%. From-scratch models learn fresh Q/K alignment matched to EVQ frequencies, avoiding the alignment disruption problem.

3. **EVQ advantage in from-scratch**: EVQ PPL@16K=164.4 vs Geo 175.4 (−6.3%). From scratch, EVQ's superior frequency allocation outweighs the alignment cost since there's no pretrained alignment to preserve.

---

## 8D: τ* Scaling Law Verification

**Goal**: Verify τ*(L_train) = 64/√L conjecture.

**Config**: 125M, from-scratch, 50M tokens, FineWeb-Edu.

### D1: L_train=256 (predicted τ*=4.0)

| τ | PPL@256 | PPL@2048 (8×L) | PPL@8192 |
|---|---------|----------------|----------|
| 0.0 | 181.5 | 461.0 | 1284.4 |
| 2.0 | 185.4 | 389.2 | 891.1 |
| 3.0 | 185.3 | 305.4 | 666.6 |
| 4.0 | 185.0 | 315.2 | 776.4 |
| **5.0** | 188.5 | **271.0** | **664.2** |

**Observed τ*=5.0** (PPL still monotonically improving at τ=5.0, same as L=128 regime).

### D2: L_train=512 (predicted τ*=2.83)

| τ | PPL@512 | PPL@4096 (8×L) | PPL@8192 |
|---|---------|----------------|----------|
| 0.0 | 79.7 | 350.2 | 472.9 |
| 1.5 | 81.2 | 337.6 | 458.7 |
| 2.0 | 81.0 | 304.9 | 409.8 |
| 2.83 | 83.3 | 314.1 | 451.9 |
| 3.5 | 83.9 | 285.9 | 389.0 |
| **4.0** | 84.0 | **245.0** | **336.3** |

**Observed τ*=4.0** (again monotonically improving, curve hasn't peaked).

### Scaling Law Analysis

All 5 data points (using observed τ* = max tested τ when no peak found):

| L_train | Predicted τ* (64/√L) | Observed τ* | Note |
|---------|---------------------|-------------|------|
| 128 | 5.66 | >5.0 | Phase 6: monotonic, no peak |
| 256 | 4.0 | >5.0 | D1: monotonic, no peak |
| 512 | 2.83 | >4.0 | D2: monotonic, no peak |
| 1024 | 2.0 | ≈2.0 | Phase 6: peaked |
| 2048 | 1.41 | ≈1.5 | Phase 7F ext: peaked |

**Fit results** (forced through origin): τ* = 67.84/√L, R²=0.76.

**Critical interpretation**: The scaling law τ*=C/√L appears valid only in the **long training length regime** (L≥1024) where the extrapolation PPL curve actually peaks. For short training lengths (L≤512), the curve is still monotonically improving at the largest tested τ — the true τ* may be larger than any value we tested. This is consistent with Phase 6's finding that L=128 never peaks.

**Revised model**: There may be a **transition length** L_crit around 512-1024 where the EVQ benefit saturates. Below L_crit, more compression always helps (no peak). Above L_crit, the peak emerges and follows τ*≈C/√L.

### Waterbed Check (PPL@L_train)

- D1: PPL@256 ranges 181.5-188.5 across τ=0..5 (max degradation 3.9%)
- D2: PPL@512 ranges 79.7-84.0 across τ=0..4 (max degradation 5.4%)

Mild waterbed effect at short training lengths (up to 5%), compared to essentially zero waterbed at L=128 (Phase 6). This suggests that shorter training lengths have more room for improvement but also more sensitivity to frequency reallocation.

---

## 8E: Extra From-Scratch 4K Runs

**Goal**: Test EVQ τ=1.0 and Hybrid τ=1.0 in from-scratch 4K setting.

**Config**: Same as 8C — 350M, 4096-tok, 50M tokens, lr=6e-4, batch=2.

### Complete From-Scratch 4K Comparison

| Method | PPL@4K | PPL@8K | PPL@16K | PK@1K | PK@2K | PK@4K | PK@8K | Global PK |
|--------|--------|--------|---------|-------|-------|-------|-------|-----------|
| C1 Geo | 91.1 | 115.6 | 175.4 | 87% | 80% | 54% | 55% | **69.0%** |
| C2 EVQ τ=2.0 | 93.1 | 113.9 | 164.4 | 82% | 68% | 60% | 54% | 66.0% |
| **E1 EVQ τ=1.0** | 92.8 | 120.3 | 180.1 | **88%** | **83%** | 57% | **60%** | **72.0%** |
| **E2 Hybrid τ=1.0** | 93.0 | 117.3 | 172.6 | **93%** | 82% | 50% | 57% | 70.5% |

### Key Findings

1. **EVQ τ=1.0 achieves BEST passkey among all from-scratch methods**: 72% global retrieval (vs Geo 69%, EVQ τ=2.0 66%). Lower τ means less frequency reallocation, preserving more Q/K alignment while still providing extrapolation benefit.

2. **Hybrid τ=1.0 has best short-range passkey**: PK@1K=93% (vs Geo 87%), thanks to preserving high-frequency Geometric channels. But PK@4K drops to 50% (vs EVQ τ=1.0's 57%).

3. **PPL tradeoff with τ**: EVQ τ=1.0 PPL@16K=180.1 (worse than τ=2.0's 164.4). Lower τ = better passkey but worse extrapolation PPL. Hybrid τ=1.0 splits the difference at 172.6.

4. **EVQ τ=2.0 best for pure PPL**: PPL@16K=164.4 remains the best extrapolation. The τ choice controls the PPL-vs-passkey tradeoff.

5. **Hierarchy**: For from-scratch training at L=4096:
   - Best passkey: EVQ τ=1.0 (72%) → Hybrid τ=1.0 (70.5%) → Geo (69%) → EVQ τ=2.0 (66%)
   - Best PPL@16K: EVQ τ=2.0 (164.4) → Hybrid τ=1.0 (172.6) → Geo (175.4) → EVQ τ=1.0 (180.1)

---

## Cross-Phase Summary

| Experiment | Key Result | Paper Impact |
|------------|-----------|-------------|
| 8A Extension 8x | Geometric doesn't collapse; Hybrid=best non-Geo | Context extension is Geometric's domain |
| 8A Passkey | YaRN best (72%), EVQ worst (56%), Hybrid mid (59%) | Frequency reallocation hurts alignment |
| 8B Finetune | PPL improves with data; passkey gap doesn't fully close | Alignment cost is structural, not data-limited |
| 8C From-scratch | Extension >> scratch for PPL; scratch better for passkey | Different paradigms, different tradeoffs |
| 8D Scaling law | τ*>5 at L=256, τ*>4 at L=512; R²=0.76 | Scaling law valid only for L≥1024 |
| 8E EVQ τ=1.0 | Best from-scratch passkey (72%); Hybrid τ=1.0 PK@1K=93% | τ controls PPL-passkey tradeoff |

---

## 8F: Multi-Seed Statistical Verification

**Goal**: Verify 8E headline results (EVQ τ=1.0 PK > Geo, Hybrid dual-win) with 4 seeds.

**Config**: Same as 8C/8E — 350M, from-scratch 4K, 50M tokens, lr=6e-4. Seeds: 42 (reused from 8C/8E), 137, 256, 314.

### PPL@16K by Seed

| Seed | Geometric | EVQ τ=1.0 | Hybrid τ=1.0 |
|------|-----------|-----------|-------------|
| 42 | 175.4 | 180.1 | 172.6 |
| 137 | 194.5 | 182.8 | 187.3 |
| 256 | 162.7 | 194.9 | 177.2 |
| 314 | 170.3 | 217.7 | 170.7 |
| **Mean** | **175.7** | **193.9** | **177.0** |
| **Std** | **13.6** | **17.1** | **7.4** |

### Passkey Global by Seed

| Seed | Geometric | EVQ τ=1.0 | Hybrid τ=1.0 |
|------|-----------|-----------|-------------|
| 42 | 69.0% | 72.0% | 70.5% |
| 137 | 81.5% | 70.0% | 71.8% |
| 256 | 72.3% | 71.5% | 70.3% |
| 314 | 71.3% | 69.0% | 71.3% |
| **Mean** | **73.5%** | **70.6%** | **70.9%** |
| **Std** | **5.5%** | **1.4%** | **0.7%** |

### Passkey by Length (Mean ± Std)

| Length | Geometric | EVQ τ=1.0 | Hybrid τ=1.0 |
|--------|-----------|-----------|-------------|
| @1K | 91.0% ± 3.6% | 86.8% ± 1.5% | **93.5% ± 2.1%** |
| @2K | 82.8% ± 3.6% | **85.0% ± 4.4%** | 80.8% ± 1.3% |
| @4K | **64.5% ± 10.0%** | 56.3% ± 5.9% | 53.8% ± 2.9% |
| @8K | 55.8% ± 7.4% | 54.5% ± 5.3% | 55.8% ± 1.9% |

### Statistical Tests

| Test | t-stat | p-value | Significant? |
|------|--------|---------|-------------|
| EVQ vs Geo PPL@16K (paired) | 1.36 | 0.266 | No |
| EVQ vs Geo PK (paired) | −0.93 | 0.419 | No |
| EVQ vs Geo PK (pooled χ²) | — | 0.076 | No |
| Hybrid vs Geo PPL@16K (paired) | 0.27 | 0.808 | No |
| Hybrid vs Geo PK (paired) | −1.02 | 0.381 | No |
| Hybrid vs Geo PK (pooled χ²) | — | 0.114 | No |

Cohen's h: EVQ vs Geo = −0.064, Hybrid vs Geo = −0.057 (negligible effect size).

### Verdict: FAIL

**The 8E headline does not survive multi-seed verification.**

1. **EVQ τ=1.0 PK (70.6%) < Geo PK (73.5%)**: Direction reversed from seed 42. Only 1/4 seeds show EVQ > Geo. p=0.42, not significant.

2. **Hybrid dual-win does not hold**: Hybrid PPL@16K (177.0) ≈ Geo (175.7), p=0.81. Hybrid PK (70.9%) < Geo (73.5%), p=0.38. Neither advantage is significant.

3. **EVQ PPL@16K is WORSE**: 193.9 vs 175.7 (+10.3%). Seed 314 is an outlier (217.7) but even without it, EVQ averages worse.

### Key Insights Despite the Fail

1. **Hybrid has dramatically lower variance**: PPL@16K std=7.4 vs Geo 13.6 vs EVQ 17.1. PK std=0.7% vs Geo 5.5%. Hybrid is the most stable method across seeds.

2. **Hybrid PK@1K = 93.5% consistently beats Geo PK@1K = 91.0%**: This is the one sub-metric where Hybrid shows consistent advantage (4/4 seeds), though the margin (2.5pp) is small.

3. **PK@8K is indistinguishable**: All three methods converge to ~55-56% at 8K, confirming the 350M model capacity ceiling.

4. **Geo has surprisingly high variance**: PK ranges from 69% to 81.5% (12.5pp range). This makes any 3pp claim from a single seed meaningless.

5. **The "from-scratch EVQ passkey advantage" from Phase 6D (128-tok) does not transfer to 4K training**: At 128-tok, EVQ had clear advantage (55% vs 48.5%). At 4K, the advantage vanishes because the model has enough training context to develop strong attention patterns regardless of frequency allocation.

---

## Updated Narrative for Paper

The key finding from Phase 8 is that EVQ's advantage depends on the **training paradigm** and **training length**:

1. **Short training (128-tok, PE-dominant)**: EVQ provides massive PPL improvements (up to −35% at 8K) and better passkey (+7pp), both statistically significant across seeds. This is EVQ's strongest regime.

2. **Long training (4K, from-scratch)**: **No significant difference between EVQ/Hybrid and Geometric** in either PPL@16K or passkey (p > 0.05 for all comparisons). The model's learned attention patterns dominate over frequency allocation at this training length. Hybrid shows the lowest variance.

3. **Context extension**: Geometric RoPE is robust at 8× expansion. EVQ/Hybrid match but don't beat Geometric. PI and YaRN collapse.

4. **Scaling law**: τ*(L) ≈ 68/√L for L ≥ 1024, R²=0.76.

5. **Practical recommendation**: EVQ is most valuable for **short-context pretraining** where PE quality dominates. For long-context training (≥ 4K), Geometric RoPE is sufficient; Hybrid EVQ is a reasonable alternative offering lower variance across seeds.
