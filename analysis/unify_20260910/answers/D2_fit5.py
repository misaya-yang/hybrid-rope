# -*- coding: utf-8 -*-
"""D2 (c3) v5 final: R = U + α·H + β·Φ  精细可行域 + 留出法 + 敏感性"""
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
    T = [Tn[j] * 4 ** m[j] for j in range(64)]
    D = [W * 4 ** m[j] for j in range(64)]
    rho = [T[g + 1] / T[g] for g in range(63)]
    return dict(m=m, r=r, T=T, D=D, rho=rho)

ALLN = FIVE + HOLD + ['StackFrontBack', 'MrProN16', 'MrProN15']
DER = {n: der(n) for n in ALLN}

def feats(nm, rho0, p, band=4.0, arc=8.0, bank=(24, 29)):
    x = DER[nm]
    U = 0
    perrow = []
    for dd in ROWS:
        e = [j for j in range(24, 40)
             if x['m'][j] < 1 - 1e-6 and x['r'][j] < arc
             and x['T'][j] <= dd <= x['T'][j] * band and x['D'][j] < dd]
        perrow.append(e); U += len(e)
    H = sum((max(0.0, x['rho'][g] - rho0)) ** p
            for dd in ROWS for g in range(23, 40) if x['T'][g] <= dd)
    F = sum(x['r'][j] * (4 ** x['m'][j] - 1) for j in range(bank[0], bank[1] + 1))
    return U, H, F, perrow

def feasible(rho0, p, band, arc, bank, ng=400):
    Fv = {nm: feats(nm, rho0, p, band, arc, bank)[:3] for nm in FIVE}
    inc = []
    for i in range(4):
        a, b = ORDER[i], ORDER[i + 1]
        inc.append(tuple(Fv[b][k] - Fv[a][k] for k in range(3)))
    A = np.linspace(0.05, 10, ng); B = np.linspace(0.02, 5, ng)
    pts = []
    for aa in A:
        for bb in B:
            sl = [inc[k][0] + aa * inc[k][1] + bb * inc[k][2] for k in range(4)]
            if min(sl) > 1e-12:
                nrm = min(sl[k] / (abs(inc[k][0]) + aa * abs(inc[k][1]) + bb * abs(inc[k][2]) + 1e-12)
                          for k in range(4))
                pts.append((aa, bb, nrm))
    return pts, Fv, inc

print('===== 精细网格：可行 (ρ0,p) 与 (α,β) 区域 =====')
rows_out = []
for rho0 in [1.360 + 0.002 * i for i in range(16)]:
    for p in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pts, Fv, inc = feasible(rho0, p, 4.0, 8.0, (24, 29))
        if not pts:
            continue
        ar = np.array(pts)
        cen = max(pts, key=lambda t: t[2])
        rows_out.append((rho0, p, ar[:, 0].min(), ar[:, 0].max(), ar[:, 1].min(), ar[:, 1].max(),
                         len(pts) / 400 / 400, cen))
        print(f'  ρ0={rho0:.3f} p={p}: α∈[{ar[:,0].min():.2f},{ar[:,0].max():.2f}] '
              f'β∈[{ar[:,1].min():.2f},{ar[:,1].max():.2f}] 面积={len(pts)/160000*100:.2f}% '
              f'Cheb=(α={cen[0]:.2f},β={cen[1]:.2f}) 裕量={cen[2]*100:.1f}%')
# 取裕量最大的结构参数
best = max(rows_out, key=lambda t: t[7][2])
rho0, p = best[0], best[1]
pts, Fv, inc = feasible(rho0, p, 4.0, 8.0, (24, 29), ng=600)
ar = np.array(pts); cen = max(pts, key=lambda t: t[2])
aa, bb = cen[0], cen[1]
print(f'\n===== 选定 ρ0={rho0:.3f}, p={p}（Chebyshev 裕量最大）; 权重 α={aa:.3f}, β={bb:.3f} =====')
print(f'  可行窗：α∈[{ar[:,0].min():.2f},{ar[:,0].max():.2f}]（宽度 {(ar[:,0].max()/ar[:,0].min()-1)*100:.0f}%）'
      f' β∈[{ar[:,1].min():.2f},{ar[:,1].max():.2f}]')
print(f'  4 条相邻不等式中绑定（裕量<20%）的：' +
      ' ; '.join(f'{ORDER[i][:8]}<{ORDER[i+1][:8]}: {min((inc[i][0]+q*0 for q in [0]),0)}' for i in range(0)))
marg = [min(sl for sl in [inc[k][0] + aa*inc[k][1] + bb*inc[k][2]]) for k in range(4)]
names = [f'{ORDER[i][:10]}≺{ORDER[i+1][:10]}' for i in range(4)]
print(f'  各相邻裕量 @中心: ' + ' | '.join(f'{names[i]}: R+= {marg[i]:.3f}' for i in range(4)))

FR = {nm: feats(nm, rho0, p)[:3] for nm in ALLN}
R = {nm: FR[nm][0] + aa * FR[nm][1] + bb * FR[nm][2] for nm in ALLN}
print('\n  ===== 5 点（拟合）=====')
for nm in ORDER:
    U, H, F = FR[nm]
    print(f'  {nm:26s} U={U:3d} H={H:6.2f} Φ={F:5.2f} R={R[nm]:7.2f} score={SCORE[nm]}')
X = np.array([[1.0, -R[nm]] for nm in FIVE]); y = np.array([SCORE[nm] for nm in FIVE])
(s0, lam), *_ = np.linalg.lstsq(X, y, rcond=None)
print(f'  幅度拟合: score ≈ {s0:.2f} − {lam:.3f}·R   残差 ' +
      ' '.join(f'{nm.split("_")[0][:6]}:{s0-lam*R[nm]-SCORE[nm]:+.2f}' for nm in ORDER))
print('\n  ===== 留出法（7 臂未参与定权）=====')
for nm in HOLD:
    U, H, F = FR[nm]
    pred = s0 - lam * R[nm]
    print(f'  {nm:22s} U={U:3d} H={H:6.2f} Φ={F:5.2f} R={R[nm]:7.2f} '
          f'pred={pred:6.2f} actual={PANEL128[nm]:6.2f} err={pred-PANEL128[nm]:+6.2f}')
meas = sorted(FIVE + HOLD, key=lambda n: -PANEL128[n])
predo = sorted(FIVE + HOLD, key=lambda n: R[n])
print(f'\n  实测降序: {" > ".join(n.split("_")[0][:9] for n in meas)}')
print(f'  预测降序: {" > ".join(n.split("_")[0][:9] for n in predo)}')
pairs = [(meas[i], meas[j]) for i in range(len(meas)) for j in range(i+1, len(meas))]
conc = sum(1 for a, b_ in pairs if R[a] < R[b_])
print(f'  Kendall τ (12 点全序) = {2*conc/len(pairs)-1:+.3f}')
print('\n  ===== 可行域内稳健性（1000 随机可行点）=====')
rng = np.random.RandomState(7)
cnt = Counter(); kcnt = Counter()
for i in rng.choice(len(pts), 1000, replace=True):
    a2, b2, _ = pts[i]
    R2 = {nm: FR[nm][0] + a2 * FR[nm][1] + b2 * FR[nm][2] for nm in FIVE + HOLD}
    cnt[tuple(sorted(FIVE + HOLD, key=lambda n: R2[n]))] += 1
    kcnt[tuple(sorted(FIVE, key=lambda n: R2[n]))] += 1
top, n = cnt.most_common(1)[0]
print(f'  12 点全序一致率: {n}/1000')
print(f'  最常见: {" < ".join(t.split("_")[0][:9] for t in top)}')
for k, v in kcnt.most_common(4):
    print(f'  5 点序: {" < ".join(t.split("_")[0][:9] for t in k)}  ({v/10}%)')
print('\n  ===== 队列候选（未执行——仅结构预测）=====')
for nm in ['StackFrontBack', 'MrProN16', 'MrProN15']:
    U, H, F = FR[nm]
    print(f'  {nm:15s} U={U:3d} H={H:6.2f} Φ={F:5.2f} R={R[nm]:7.2f} '
          f'(MrPro R={R["MrPro"]:.2f}) → 预测 {"优于" if R[nm]<R["MrPro"] else "劣于"} MrPro, '
          f'pred {s0-lam*R[nm]:.1f}')
print('\n  ===== 敏感性：band/arc/bank 变体下 ρ0=1.371,p=0.5 是否仍可行 =====')
for band in (2.0, 4.0, 8.0):
    for arc in (6.47, 8.0, 9.96):
        for bank in ((24, 29), (24, 31), (26, 29)):
            pts2, _, _ = feasible(1.371, 0.5, band, arc, bank, ng=200)
            if pts2:
                ar2 = np.array(pts2)
                print(f'  band={band} arc={arc} bank={bank}: 可行 α∈[{ar2[:,0].min():.2f},{ar2[:,0].max():.2f}] '
                      f'β∈[{ar2[:,1].min():.2f},{ar2[:,1].max():.2f}]')
            else:
                print(f'  band={band} arc={arc} bank={bank}: 不可行')
print('\n  ===== 逐行暴露清单（选定参数）=====')
for nm in FIVE:
    _, _, _, perrow = feats(nm, rho0, p)
    print(f'  {nm:24s} ' + ' | '.join(f'{int(dd/1000)}K:{e}' for (dd, e) in
                                      [(dd, [j for j in e]) for dd, e in
                                       [(ROWS[i], perrow[i]) for i in range(5)]]))
# 反事实：p=1 或 ρ0=1.35 为何破 —— 打印违背的不等式
for (rho0t, pt) in [(1.35, 0.5), (1.371, 1.0), (1.40, 0.5)]:
    Fv2 = {nm: feats(nm, rho0t, pt)[:3] for nm in FIVE}
    inc2 = [tuple(Fv2[ORDER[i+1]][k]-Fv2[ORDER[i]][k] for k in range(3)) for i in range(4)]
    # 最优 α,β: 最小化违背
    bestm = (1e9, None)
    for aal in np.linspace(0.05, 12, 400):
        for bbl in np.linspace(0.02, 6, 300):
            v = -min(inc2[k][0] + aal*inc2[k][1] + bbl*inc2[k][2] for k in range(4))
            if v < bestm[0]:
                bestm = (v, (aal, bbl))
    print(f'  反例 ρ0={rho0t} p={pt}: 最小最大违背 {bestm[0]:+.3f}@α={bestm[1][0]:.2f},β={bestm[1][1]:.2f} '
          f'（>0 即无解；违背出现在最紧的不等式）')
