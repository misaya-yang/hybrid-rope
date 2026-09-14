# -*- coding: utf-8 -*-
"""D2 (c3) v3：三特征双端风险泛函
R = u(U) + α·H + β·Φ
U = Σ_rows min(#暴露桥钟, cap)   （长端；行级风险封顶）
H = Σ_rows Σ_{g: T_g ≤ d} (ρ_g − ρ0)_+^p   （短端洞，按行距离累计：洞在 d 之下即伤该行梯级）
Φ = Σ_{j∈bank} r_j(4^{m_j}−1)   （bank 忠实度位移，原生圈）
扫描 (ρ0,p,cap,band,arc,bank窗口)，报告严格排序可行 (α,β) 区域面积 + Chebyshev 裕量，
中位参数 lstsq 拟合分数幅度，留出法 & 队列预测。"""
import json, math
import numpy as np
from collections import Counter

ROOT = '/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope'
d = json.load(open(f'{ROOT}/analysis/unify_20260910/tables/ground_truth_tables.json'))
M = d['methods']
W = 32768.0
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

def der(n):
    m = M[n]['m_j']
    Tn = [2 * math.pi * 10 ** (6 * j / 64) for j in range(64)]
    r = [W / Tn[j] for j in range(64)]
    T = [None if m[j] is None else Tn[j] * 4 ** m[j] for j in range(64)]
    D = [None if m[j] is None else W * 4 ** m[j] for j in range(64)]
    rho = [None if (T[g] is None or T[g + 1] is None) else T[g + 1] / T[g] for g in range(63)]
    return dict(m=m, r=r, T=T, D=D, rho=rho)

DER = {n: der(n) for n in FIVE + HOLD + ['StackFrontBack', 'MrProN16', 'MrProN15']}

def feats(nm, rho0, p, cap, band, arc, bank):
    x = DER[nm]
    U = 0
    for dd in ROWS:
        c = 0
        for j in range(24, 40):
            if x['m'][j] is None or x['m'][j] >= 1 - 1e-6 or x['r'][j] >= arc:
                continue
            if x['T'][j] <= dd <= x['T'][j] * band and x['D'][j] < dd:
                c += 1
        U += min(c, cap)
    U = U ** 0.5 if cap == 0 else U          # cap==0 变体: sqrt(Σcount)
    H = 0.0
    for dd in ROWS:
        for g in range(23, 40):
            if x['rho'][g] and x['T'][g] <= dd:
                H += (max(0.0, x['rho'][g] - rho0)) ** p
    F = sum(x['r'][j] * (4 ** x['m'][j] - 1) for j in range(bank[0], bank[1] + 1)
            if x['m'][j] is not None)
    return U, H, F

def region(rho0, p, cap, band, arc, bank):
    """返回可行 (α,β) 网格点 + 面积比例（窗 α∈(0,12],β∈(0,1.5]）+ 最大归一裕量"""
    Fv = {nm: feats(nm, rho0, p, cap, band, arc, bank) for nm in FIVE}
    inc = []                                   # 不等式: u + αh + βf > 0 （R(next)-R(prev)）
    for i in range(4):
        a, b = ORDER[i], ORDER[i + 1]
        inc.append(tuple(Fv[b][k] - Fv[a][k] for k in range(3)))
    A = np.linspace(0.02, 12.0, 240); B = np.linspace(0.002, 1.5, 150)
    pts = []
    for aa in A:
        for bb in B:
            sl = [u + aa * h + bb * f for (u, h, f) in inc]
            m0 = min(sl)
            if m0 > 1e-12:
                nrm = min(sl[k] / (abs(inc[k][0]) + aa * abs(inc[k][1]) + bb * abs(inc[k][2]) + 1e-12)
                          for k in range(4))
                pts.append((aa, bb, nrm))
    return pts, Fv

print('===== 变体扫描（cap: 行级暴露封顶；0=sqrt 全和）=====')
best = None
for cap in (1, 2, 3, 4, 0):
    for rho0 in [1.35, 1.36, 1.371, 1.38, 1.40, 1.42]:
        for p in (1.0, 1.5, 2.0):
            for band in (4.0,):
                for arc in (8.0,):
                    for bank in [(24, 29), (24, 31), (26, 33)]:
                        pts, Fv = region(rho0, p, cap, band, arc, bank)
                        if not pts:
                            continue
                        area = len(pts) / (240 * 150)
                        cen = max(pts, key=lambda t: t[2])
                        if best is None or cen[2] > best[0][2]:
                            best = (cen, (rho0, p, cap, band, arc, bank), pts, area, Fv)
                        print(f'cap={cap} ρ0={rho0} p={p} bank={bank}: 可行面积 {area*100:5.2f}%  '
                              f'最大裕量 {cen[2]*100:4.1f}% @ α={cen[0]:.2f} β={cen[1]:.3f}')
if best is None:
    raise SystemExit('全变体不可行')

cen, par, pts, area, Fv = best
rho0, p, cap, band, arc, bank = par
print(f'\n===== 选定（Chebyshev 裕量最大）: ρ0={rho0} p={p} cap={cap} band={band} arc={arc} bank={bank} '
      f'裕量={cen[2]*100:.1f}% 面积={area*100:.2f}% =====')
aa, bb = cen[0], cen[1]
al = np.array([q[0] for q in pts]); be = np.array([q[1] for q in pts])
print(f'α ∈ [{al.min():.2f},{al.max():.2f}]（中位 {np.median(al):.2f}）  '
      f'β ∈ [{be.min():.3f},{be.max():.3f}]（中位 {np.median(be):.3f}）')
print('\n  特征与 R（α,β 取 Chebyshev 中心）:')
ALLNAMES = FIVE + HOLD + ['StackFrontBack', 'MrProN16', 'MrProN15']
FR = {nm: feats(nm, rho0, p, cap, band, arc, bank) for nm in ALLNAMES}
R = {nm: FR[nm][0] + aa * FR[nm][1] + bb * FR[nm][2] for nm in ALLNAMES}
for nm in FIVE:
    U, H, F = FR[nm]
    print(f'  {nm:26s} U={U:4.1f} H={H:6.3f} Φ={F:6.2f} R={R[nm]:6.2f} score={SCORE[nm]}')
X = np.array([[1.0, -R[nm]] for nm in FIVE]); y = np.array([SCORE[nm] for nm in FIVE])
(s0, lam), *_ = np.linalg.lstsq(X, y, rcond=None)
print(f'\n  幅度拟合 score ≈ {s0:.2f} − {lam:.3f}·R ;  残差: ' +
      ' '.join(f'{nm.split("_")[0][:6]}:{s0-lam*R[nm]-SCORE[nm]:+.2f}' for nm in FIVE))
print('\n  ===== 留出法 =====')
okc = 0
for nm in HOLD:
    U, H, F = FR[nm]
    pred = s0 - lam * R[nm]
    err = pred - PANEL128[nm]
    okc += abs(err) < 8
    print(f'  {nm:22s} U={U:4.1f} H={H:6.3f} Φ={F:6.2f} R={R[nm]:6.2f} '
          f'pred={pred:6.2f} actual={PANEL128[nm]:6.2f} err={err:+6.2f}')
# 全序（5+7）与实测名次一致性
meas = sorted(FIVE + HOLD, key=lambda n: -PANEL128[n])
predo = sorted(FIVE + HOLD, key=lambda n: R[n])
print(f'\n  实测降序: {" > ".join(n.split("_")[0][:8] for n in meas)}')
print(f'  预测降序: {" > ".join(n.split("_")[0][:8] for n in predo)}')
ktau = sum(1 for i in range(len(meas)) for j in range(i + 1, len(meas))
           if (R[meas[i]] - R[meas[j]]) < 0) / (len(meas) * (len(meas) - 1) / 2)
print(f'  Kendall τ (R vs 分数, 12 点) = {2*ktau-1:+.3f}')
print('\n  ===== 留出全序稳健性（可行区内 400 随机点）=====')
idx = np.random.RandomState(0).choice(len(pts), min(400, len(pts)), replace=False)
cnt = Counter()
for i in idx:
    a2, b2, _ = pts[i]
    R2 = {nm: FR[nm][0] + a2 * FR[nm][1] + b2 * FR[nm][2] for nm in FIVE + HOLD}
    cnt[tuple(sorted(FIVE + HOLD, key=lambda n: R2[n]))] += 1
top, n = cnt.most_common(1)[0]
print(f'  最常见前 5 序: {" < ".join(t.split("_")[0][:8] for t in top[:5])}  ({n}/{len(idx)})')
c5 = Counter(tuple(o[:5] for o in cnt))  # placeholder
c5 = Counter({k[:5]: v for k, v in cnt.items()})
print(f'  前 5 名（拟合段）保持面板序的比例: {sum(v for k,v in c5.items() if k==tuple(ORDER))/len(idx)*100:.0f}%')
print(f'  12 点全序唯一化比例: {n/len(idx)*100:.0f}%')
print('\n  ===== 队列候选（未执行，仅结构预测）=====')
for nm in ['StackFrontBack', 'MrProN16', 'MrProN15']:
    U, H, F = FR[nm]
    print(f'  {nm:15s} U={U:4.1f} H={H:6.3f} Φ={F:6.2f} R={R[nm]:6.2f} '
          f'(参照 MrPro R={R["MrPro"]:.2f} → 预测 {"优于" if R[nm]<R["MrPro"] else "劣于"} MrPro, '
          f'pred score {s0-lam*R[nm]:.1f})')
print('\n  ===== 逐行暴露/洞明细（选定参数）=====')
for nm in FIVE:
    x = DER[nm]
    lines = []
    for dd in ROWS:
        e = [j for j in range(24, 40)
             if x['m'][j] is not None and x['m'][j] < 1 - 1e-6 and x['r'][j] < arc
             and x['T'][j] <= dd <= x['T'][j] * band and x['D'][j] < dd]
        hv = sum((max(0.0, x['rho'][g] - rho0)) ** p for g in range(23, 40)
                 if x['rho'][g] and x['T'][g] <= dd)
        lines.append(f'd={dd/1000:.0f}K:e{len(e)}{e}h{hv:.3f}')
    print(f'  {nm:24s} ' + ' | '.join(lines))
