# -*- coding: utf-8 -*-
"""D2 (c3) v4: four-feature risk functional
R = U + a*H + b*Phi + g*G
U = sum_rows min(#exposed bridge arc clocks, 3)
H = sum_rows sum_{g: T_g<=d} (rho_g-rho0)_+^p
Phi = sum_{j in 24..29} r_j (4^m_j - 1)
G = #{g in 23..39: rho_g > rho1}
Strict 5-point ranking -> polytope in (a,b,g); scan (rho0,p,rho1);
Chebyshev center; lstsq magnitudes; holdout; queue predictions.
"""
import json, math
import numpy as np
from collections import Counter

ROOT = '/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope'
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

ALLN = FIVE + HOLD + ['StackFrontBack', 'MrProN16', 'MrProN15']
DER = {n: der(n) for n in ALLN if all(x is not None for x in M[n]['m_j'])}
MISS = [n for n in ALLN if n not in DER]
print('缺全 m_j 的方法(将按可用槽计算或跳过):', MISS)

def feats(nm, rho0, p, rho1, cap=3, band=4.0, arc=8.0, bank=(24, 29)):
    x = DER[nm]
    U = 0
    for dd in ROWS:
        c = 0
        for j in range(24, 40):
            if x['m'][j] >= 1 - 1e-6 or x['r'][j] >= arc:
                continue
            if x['T'][j] <= dd <= x['T'][j] * band and x['D'][j] < dd:
                c += 1
        U += min(c, cap)
    H = 0.0
    for dd in ROWS:
        for g in range(23, 40):
            if x['T'][g] <= dd:
                H += (max(0.0, x['rho'][g] - rho0)) ** p
    F = sum(x['r'][j] * (4 ** x['m'][j] - 1) for j in range(bank[0], bank[1] + 1))
    G = sum(1 for g in range(23, 40) if x['rho'][g] > rho1)
    return U, H, F, G

def scan():
    res = []
    for rho0 in (1.35, 1.36, 1.365, 1.371, 1.375, 1.38, 1.40):
        for p in (1.0, 1.5, 2.0):
            for rho1 in (1.34, 1.35, 1.36, 1.371, 1.38, 1.40, 1.42):
                Fv = {nm: feats(nm, rho0, p, rho1) for nm in FIVE}
                inc = []
                for i in range(4):
                    a, b = ORDER[i], ORDER[i + 1]
                    inc.append(tuple(Fv[b][k] - Fv[a][k] for k in range(4)))
                # grid on simplex a+b+g=1 (positive)
                pts = []
                A = np.linspace(0.01, 0.98, 98)
                B = np.linspace(0.01, 0.98, 98)
                for aa in A:
                    for bb in B:
                        gg = 1 - aa - bb
                        if gg <= 0.01:
                            continue
                        sl = [inc[k][0] + aa * inc[k][1] + bb * inc[k][2] + gg * inc[k][3]
                              for k in range(4)]
                        if min(sl) > 1e-12:
                            nrm = min(sl[k] / (abs(inc[k][0]) + aa * abs(inc[k][1]) +
                                               bb * abs(inc[k][2]) + gg * abs(inc[k][3]) + 1e-12)
                                      for k in range(4))
                            pts.append((aa, bb, gg, nrm))
                if pts:
                    cen = max(pts, key=lambda t: t[3])
                    res.append((cen[3], len(pts) / (98 * 98), rho0, p, rho1, inc, pts))
    return res

res = scan()
if not res:
    print('四特征仍不可行')
    raise SystemExit
res.sort(key=lambda t: -t[0])
print(f'\n可行 (ρ0,p,ρ1) 组合数: {len(res)}；Top-8（按 Chebyshev 裕量）:')
for marg, frac, rho0, p, rho1, inc, pts in res[:8]:
    cen = max(pts, key=lambda t: t[3])
    print(f'  ρ0={rho0} p={p} ρ1={rho1}: 裕量 {marg*100:4.1f}%  面积 {frac*100:4.1f}%  '
          f'中心 (α,β,γ)/归一 = ({cen[0]:.3f},{cen[1]:.3f},{cen[2]:.3f})')

marg, frac, rho0, p, rho1, inc, pts = res[0]
cen = max(pts, key=lambda t: t[3])
aa, bb, gg = cen[0] / cen[2], cen[1] / cen[2], 1.0   # 以 γ=1 归一
print(f'\n===== 选定 ρ0={rho0} p={p} ρ1={rho1}；权重 (α,β) per γ=1: α={aa:.3f} β={bb:.3f} =====')
FR = {nm: feats(nm, rho0, p, rho1) for nm in DER if nm in FIVE + HOLD + ['StackFrontBack', 'MrProN16', 'MrProN15']}
R = {nm: FR[nm][0] + aa * FR[nm][1] + bb * FR[nm][2] + gg * FR[nm][3] for nm in FR}
for nm in FIVE:
    print(f'  {nm:26s} U={FR[nm][0]:3d} H={FR[nm][1]:6.3f} Φ={FR[nm][2]:5.2f} G={FR[nm][3]:2d} '
          f'R={R[nm]:7.2f} score={SCORE[nm]}')
X = np.array([[1.0, -R[nm]] for nm in FIVE]); y = np.array([SCORE[nm] for nm in FIVE])
(s0, lam), *_ = np.linalg.lstsq(X, y, rcond=None)
print(f'\n  幅度拟合 score ≈ {s0:.2f} − {lam:.3f}·R ; 5 点残差: ' +
      ' '.join(f'{nm.split("_")[0][:6]}:{s0-lam*R[nm]-SCORE[nm]:+.2f}' for nm in FIVE))
print('\n  ===== 留出法 =====')
for nm in HOLD:
    if nm not in FR:
        print(f'  {nm:22s} (m_j 不全，跳过)'); continue
    print(f'  {nm:22s} U={FR[nm][0]:3d} H={FR[nm][1]:6.3f} Φ={FR[nm][2]:5.2f} G={FR[nm][3]:2d} '
          f'R={R[nm]:7.2f} pred={s0-lam*R[nm]:6.2f} actual={PANEL128[nm]:6.2f} '
          f'err={s0-lam*R[nm]-PANEL128[nm]:+6.2f}')
full5 = [n for n in FIVE if n in FR]
holdok = [n for n in HOLD if n in FR]
meas = sorted(full5 + holdok, key=lambda n: -PANEL128[n])
predo = sorted(full5 + holdok, key=lambda n: R[n])
print(f'\n  实测降序: {" > ".join(n.split("_")[0][:9] for n in meas)}')
print(f'  预测降序: {" > ".join(n.split("_")[0][:9] for n in predo)}')
pairs = [(meas[i], meas[j]) for i in range(len(meas)) for j in range(i + 1, len(meas))]
conc = sum(1 for a, b in pairs if R[a] < R[b])
print(f'  Kendall τ ({len(meas)} 点) = {2*conc/len(pairs)-1:+.3f}')
print('\n  ===== 队列候选（未执行；仅结构预测，无实测分）=====')
for nm in ['StackFrontBack', 'MrProN16', 'MrProN15']:
    if nm not in FR:
        print(f'  {nm}: m_j 不全，跳过'); continue
    print(f'  {nm:15s} U={FR[nm][0]:3d} H={FR[nm][1]:6.3f} Φ={FR[nm][2]:5.2f} G={FR[nm][3]:2d} '
          f'R={R[nm]:7.2f} → 预测 {"优于" if R[nm] < R["MrPro"] else "劣于"} MrPro；'
          f'pred {s0-lam*R[nm]:.1f} vs MrPro {s0-lam*R["MrPro"]:.1f} (实际)')
# 稳健性: 可行 simplex 点上留出全序稳定性
idx = np.random.RandomState(0).choice(len(pts), min(500, len(pts)), replace=False)
cnt = Counter()
for i in idx:
    a2, b2, g2, _ = pts[i]
    R2 = {nm: FR[nm][0] + a2 * FR[nm][1] + b2 * FR[nm][2] + g2 * FR[nm][3] for nm in full5 + holdok}
    cnt[tuple(sorted(full5 + holdok, key=lambda n: R2[n]))] += 1
top, n = cnt.most_common(1)[0]
print(f'\n  可行域内 {len(idx)} 点：最常见全序占比 {n/len(idx)*100:.0f}%')
print(f'  该序: {" < ".join(t.split("_")[0][:9] for t in top)}')
c5 = sum(v for k, v in cnt.items() if k[:5] == tuple(ORDER))
print(f'  前5名保持面板序比例: {c5/len(idx)*100:.0f}%')
print('\n  ===== 结构阈值敏感性：其余组合下的可行裕量 =====')
for marg, frac, rho0b, pb, rho1b, incb, ptsb in res[1:]:
    print(f'  ρ0={rho0b} p={pb} ρ1={rho1b}: 裕量 {marg*100:4.1f}% 面积 {frac*100:4.1f}%')
