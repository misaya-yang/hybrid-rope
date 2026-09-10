# -*- coding: utf-8 -*-
"""D2 (c3) 双端风险泛函拟合 v2：
R = U + α·H + β·Φ  （A 归一为 1；α=B/A>0, β=C/A>0 两个有效自由度）
U = 长端未覆盖暴露（逐证据行、桥内 arc 钟、带 [d/band, d]、D_j<d 计数和）
H = 短端洞风险：Σ_g (ρ_g − ρ0)_+^p，g∈23..39（超容差周期洞惩罚）
Φ = bank 忠实度失真：Σ_{j∈bank} r_j·(4^{m_j}−1)（bank 边缘相位位移，单位=原生圈）
核心检验：∃ α,β>0 使严格排序 R(s28)<R(P2)<R(LBS)<R(MrPro)<R(Smooth) 成立；
统计可行 (α,β,ρ0,p,band,arc) 区域大小（近唯一性），并用中位参数做留出法预测。
"""
import json, math, itertools
import numpy as np

ROOT = '/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope'
d = json.load(open(f'{ROOT}/analysis/unify_20260910/tables/ground_truth_tables.json'))
M = d['methods']
W, S, L = 32768, 4.0, 131072
lnb64 = math.log(1e6) / 64
NAT = math.exp(lnb64)

FIVE = ['MrPro', 'E1_s28_less', 'LongBridgeSlower', 'FullLagP2_Transfer3B', 'Smooth_MrBudget']
ORDER = ['E1_s28_less', 'FullLagP2_Transfer3B', 'LongBridgeSlower', 'MrPro', 'Smooth_MrBudget']
SCORE = {'E1_s28_less': 83.3333, 'FullLagP2_Transfer3B': 81.6667, 'LongBridgeSlower': 80.0694,
         'MrPro': 78.1250, 'Smooth_MrBudget': 68.3333}
HOLD = ['E1_pair28_29', 'LongBridgeFaster', 'E1_s29_more', 'MrProBM', 'MrUni',
        'E7_local_projection', 'GapCapped']
PANEL128 = dict(SCORE, **{'E1_pair28_29': 73.9583, 'LongBridgeFaster': 73.9583,
                          'E1_s29_more': 77.9167, 'MrProBM': 70.8333, 'MrUni': 73.3333,
                          'E7_local_projection': 68.6111, 'GapCapped': 62.1528})
ROWS = [88726.0, 95676.0, 105880.0, 117487.0, 127000.0]

def derived(name):
    m = M[name]['m_j']
    Tn = [2 * math.pi * 10 ** (6 * j / 64) for j in range(64)]
    r = [W / Tn[j] for j in range(64)]
    T = [None if m[j] is None else Tn[j] * 4 ** m[j] for j in range(64)]
    D = [None if m[j] is None else W * 4 ** m[j] for j in range(64)]
    rho = [None if (T[g] is None or T[g + 1] is None) else T[g + 1] / T[g] for g in range(63)]
    return dict(m=m, r=r, T=T, D=D, rho=rho)

DER = {n: derived(n) for n in FIVE + HOLD + ['StackFrontBack', 'MrProN16', 'MrProN15']}

def feats(nm, rho0, p, band, arc, bank=(24, 29)):
    x = DER[nm]
    U = 0
    for dd in ROWS:
        for j in range(24, 40):
            if x['m'][j] is None or x['m'][j] >= 1 - 1e-6 or x['r'][j] >= arc:
                continue
            if x['T'][j] <= dd <= x['T'][j] * band and x['D'][j] < dd:
                U += 1
    H = sum((max(0.0, x['rho'][g] - rho0)) ** p for g in range(23, 40) if x['rho'][g])
    F = sum(x['r'][j] * (4 ** x['m'][j] - 1) for j in range(bank[0], bank[1] + 1)
            if x['m'][j] is not None)
    return U, H, F

# ---- 特征表（默认参数）----
print('===== 特征 (U,H,Φ) @ ρ0=1.40, p=1, band=4, arc=8, bank=24..29 =====')
for nm in FIVE:
    U, H, F = feats(nm, 1.40, 1.0, 4.0, 8.0)
    print(f'  {nm:26s} U={U:2d} H={H:8.4f} Φ={F:7.3f}  score={SCORE[nm]}')

# ---- 严格排序可行域扫描 ----
# ∃α,β>0: R(t1)<R(t2)<R(t3)<R(t4) 即 4 个线性不等式 (α,β)
# 对网格点计可行比例 + 记录每条不等式松紧（margin）
def feasible_frac(rho0, p, band, arc, bank):
    Fv = {nm: feats(nm, rho0, p, band, arc, bank) for nm in FIVE}
    # 不等式：R(ORDER[i]) + gap < R(ORDER[i+1])，用 margin 网格
    us, hs, ps = [], [], []
    for i in range(4):
        a, b = ORDER[i], ORDER[i + 1]
        us.append(Fv[b][0] - Fv[a][0]); hs.append(Fv[b][1] - Fv[a][1]); ps.append(Fv[b][2] - Fv[a][2])
    # 网格 (α,β)∈(0.02,4]×(0.002,0.4]，看满足全部 4 条的比例
    A = np.linspace(0.02, 4.0, 200); B = np.linspace(0.002, 0.4, 200)
    tot = 0; ok = 0
    sample = []
    for aa in A:
        for bb in B:
            tot += 1
            if all(us[k] + aa * hs[k] + bb * ps[k] > 1e-9 for k in range(4)):
                ok += 1
                sample.append((aa, bb))
    return ok / tot, sample, Fv

print('\n===== (ρ0,p,band,arc) 网格下的可行 (α,β) 比例 =====')
feas = []
for rho0 in [1.36 + 0.004 * i for i in range(12)]:
    for p in [0.75, 1.0, 1.5, 2.0]:
        for band in [2.0, 4.0, 8.0]:
            for arc in [8.0]:
                for bank in [(24, 29), (24, 28), (26, 30)]:
                    fr, smp, Fv = feasible_frac(rho0, p, band, arc, bank)
                    if fr > 0:
                        feas.append((rho0, p, band, arc, bank, fr, smp))
                        print(f'  ρ0={rho0:.3f} p={p} band={band} bank={bank}: 可行比例 {fr*100:.2f}%  '
                              f'(U,H,Φ) s28={Fv["E1_s28_less"]} ... Sm={Fv["Smooth_MrBudget"]}'
                              .replace('(', '(').replace('[', ''))
print(f'共 {len(feas)} 个 (结构参数) 组合存在可行 (α,β) 区域，'
      f'比例中位数 {np.median([f[5] for f in feas])*100:.2f}%（网格窗 α∈(0,4],β∈(0,0.4]）')

# ---- 取可行比例最高的组合，报告 (α,β) 区域与近唯一性 ----
if feas:
    best = max(feas, key=lambda t: t[5])
    rho0, p, band, arc, bank, fr, smp = best
    print(f'\n===== 最大可行组合: ρ0={rho0:.3f} p={p} band={band} arc={arc} bank={bank} '
          f'(比例 {fr*100:.1f}%) =====')
    al = np.array([s[0] for s in smp]); be = np.array([s[1] for s in smp])
    print(f'  α 范围 [{al.min():.3f},{al.max():.3f}] 中位 {np.median(al):.3f}')
    print(f'  β 范围 [{be.min():.4f},{be.max():.4f}] 中位 {np.median(be):.4f}')
    am, bm = float(np.median(al)), float(np.median(be))
    Fv = {nm: feats(nm, rho0, p, band, arc, bank) for nm in FIVE + HOLD +
          ['StackFrontBack', 'MrProN16', 'MrProN15']}
    R = {nm: Fv[nm][0] + am * Fv[nm][1] + bm * Fv[nm][2] for nm in Fv}
    # lstsq: score ≈ s0 − λR
    names = FIVE
    X = np.array([[1.0, -R[nm]] for nm in names]); y = np.array([SCORE[nm] for nm in names])
    (s0, lam), *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = {nm: s0 - lam * R[nm] for nm in names}
    resid = max(abs(pred[nm] - SCORE[nm]) for nm in names)
    print(f'  拟合 score ≈ {s0:.2f} − {lam:.3f}·R   最大残差 {resid:.2f}')
    print('  ===== 5 点 + 留出法 + 队列预测 =====')
    for nm in names + HOLD:
        tag = 'fit ' if nm in names else 'HOLD'
        print(f'  [{tag}] {nm:26s} U={Fv[nm][0]:2d} H={Fv[nm][1]:7.3f} Φ={Fv[nm][2]:6.2f} '
              f'R={R[nm]:6.2f} pred={s0-lam*R[nm]:6.2f} actual={PANEL128[nm]:6.2f} '
              f'err={s0-lam*R[nm]-PANEL128[nm]:+6.2f}')
    for nm in ['StackFrontBack', 'MrProN16', 'MrProN15']:
        print(f'  [预测未执行] {nm:16s} U={Fv[nm][0]:2d} H={Fv[nm][1]:7.3f} Φ={Fv[nm][2]:6.2f} '
              f'R={R[nm]:6.2f}  （MrPro R={R["MrPro"]:.2f} 为参照）')
    # 稳健性：所有可行 (α,β) 网格点上，留出法名次是否一致
    from collections import Counter
    def ranks(aa, bb):
        Rs = {nm: Fv[nm][0] + aa * Fv[nm][1] + bb * Fv[nm][2] for nm in FIVE + HOLD}
        order = tuple(sorted(Rs, key=lambda n: Rs[n]))
        return order
    cnt = Counter(ranks(a_, b_) for (a_, b_) in smp)
    top, n = cnt.most_common(1)[0]
    print(f'\n  留出法稳健性：{n}/{len(smp)} 个可行点给出同一全序；全序 = {" < ".join(t[:12] for t in top)}')
    c2 = Counter(tuple(sorted(o[:5])) for o in cnt.elements()) if hasattr(cnt, 'elements') else None
