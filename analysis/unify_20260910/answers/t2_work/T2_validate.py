# -*- coding: utf-8 -*-
"""T2 结构验证 + 预测矩阵（纯 CPU，零 GPU 依赖）。

输入：
  analysis/unify_20260910/tables/ground_truth_tables.json  (G1 部署/重建数组)
闭式（与 T1/D4 同一族，D4 §5；d4_scripts/tables.py:5-7）：
  RAMP   m_q = q(q+1)/(N'(N'+1)),  q = clip(j-23, 0, N')
  UNIF   m_j = clip((j-23)/N', 0, 1)
所有输出数字本脚本现算。
"""
import json, math
import numpy as np

ROOT = '/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope'
G1 = json.load(open(f'{ROOT}/analysis/unify_20260910/tables/ground_truth_tables.json'))
M = G1['methods']

W, S, L = 32768.0, 4.0, 131072.0
LN4 = math.log(4.0)                 # 1.3862943611198906
NB = math.log(1e6) / 64.0           # 0.21586735246819178
NAT = math.exp(NB)                  # 1.2409377607517196
TN = [2 * math.pi * 10 ** (6 * j / 64) for j in range(64)]      # 原生周期
R = [W / TN[j] for j in range(64)]                              # 原生窗内圈数
GAIN = 1 + 0.1 * LN4

def ramp_m(n):
    m = []
    for j in range(64):
        q = max(0, min(n, j - 23))
        m.append(q * (q + 1) / (n * (n + 1)))
    return m

def unif_m(n):
    return [min(1.0, max(0.0, (j - 23) / n)) for j in range(64)]

def derived_from_m(m):
    T = [TN[j] * 4 ** m[j] for j in range(64)]
    D = [W * 4 ** m[j] for j in range(64)]
    rho = [T[g + 1] / T[g] for g in range(63)]
    return dict(m=m, T=T, D=D, rho=rho)

DEPLOY = {}   # JSON 数组口径
for name in M:
    m = M[name]['m_j']
    if m is None or any(x is None for x in m):
        continue
    DEPLOY[name] = derived_from_m([float(x) for x in m])

CAND = {}     # 闭式口径（float64 精确构造）
for n in range(13, 18):
    CAND[f'RAMP{n}'] = derived_from_m(ramp_m(n))
    CAND[f'UNIF{n}'] = derived_from_m(unif_m(n))
# ramp16 ⊕ notch（探索变体：m28:=m27，预算留在桥内右移，非冻结队列成员）
_m16 = ramp_m(16); _mn = list(_m16); _mn[28] = _m16[27]
CAND['RAMP16NOTCH'] = derived_from_m(_mn)

print('=' * 78)
print('## 0. T1 闭式一致性对账（自行重建 vs D4 交付表 vs G1 公式重建数组）')
print('=' * 78)
# D4_transport_rule.md §5 交付表（N16）打印值：j, m_j, nu_j, D_j, eps, lam, rho
D4_N16 = {23: (0.0, None, 32768, .007353, 1.0102, 1.2537),
          24: (.007353, None, 33104, .014706, 1.0206, 1.2665),
          25: (.022059, None, 33786, .022059, 1.0311, 1.2795),
          26: (.044118, None, 34835, .029412, 1.0416, 1.2926),
          27: (.073529, None, 36284, .036765, 1.0523, 1.3058),
          28: (.110294, None, 38182, .044118, 1.0631, 1.3192),
          29: (.154412, None, 40590, .051471, 1.0740, 1.3327),
          30: (.205882, None, 43592, .058824, 1.0850, 1.3464),
          31: (.264706, None, 47295, .066176, 1.0961, 1.3602),
          32: (.330882, None, 51840, .073529, 1.1073, 1.3741),
          33: (.404412, None, 57402, .080882, 1.1187, 1.3882),
          34: (.485294, None, 64213, .088235, 1.1301, 1.4024),
          35: (.573529, None, 72569, .095588, 1.1417, 1.4168),
          36: (.669118, 1.6678e-4, 82851, .102941, 1.1534, 1.4313),
          37: (.772059, 1.1653e-4, 95560, .110294, 1.1652, 1.4460),
          38: (.882353, 8.0588e-5, 111347, .117647, 1.1771, 1.4608),
          39: (1.0, 5.5168e-5, 131072, 0, 1.0000, 1.2409)}
x = CAND['RAMP16']
worst_m = worst_D = worst_rho = worst_nu = 0.0
for j, (m0, nu0, D0, eps0, lam0, rho0) in D4_N16.items():
    worst_m = max(worst_m, abs(x['m'][j] - m0))
    worst_D = max(worst_D, abs(x['D'][j] - D0))
    worst_rho = max(worst_rho, abs(x['rho'][j - 23 + 23] - rho0) if j >= 24 else 0.0)
    omj = 10 ** (-j / 64.0 * 6 / 6 * 1) * 1  # omega_j = b^{-j/64} = 10^{-6j/64}
    omj = 10 ** (-6.0 * j / 64.0)
    nu = omj * 4 ** (-x['m'][j])
    if nu0 is not None:
        worst_nu = max(worst_nu, abs(nu / nu0 - 1))
print(f'RAMP16 vs D4 交付表: max|Δm|={worst_m:.2e}  max|ΔD|={worst_D:.2f}  '
      f'max|Δρ|={worst_rho:.2e}  max ν 相对差={worst_nu:.2e} (D4 表精度 1e-4 级 → 一致)')
# vs G1 JSON 公式重建数组
for cname, jname in [('RAMP16', 'MrProN16'), ('RAMP15', 'MrProN15'), ('RAMP17', 'MrPro')]:
    dm = max(abs(CAND[cname]['m'][j] - DEPLOY[jname]['m'][j]) for j in range(64))
    print(f'{cname} vs G1 `{jname}` m_j: max|Δm|={dm:.2e}')
# G1 GR§6 队列修正值锚点
anch = {'MrProN16': dict(m28=.110294, m36=.6691, m37=.7721, m38=.8824, m39=1.0, hole=(1.461, 38)),
        'MrProN15': dict(m28=.125, m36=.7583, m37=.875, m38=1.0, m39=1.0, hole=(1.476, 37)),
        'StackFrontBack': dict(m28=.065359, m36=.6252, m37=.7295, m38=.8465, m39=.9799, hole=(1.493, 35))}
for nm, a in anch.items():
    y = DEPLOY[nm]
    hs = [(g, y['rho'][g]) for g in range(23, 40)]
    gmx, rmx = max(hs, key=lambda t: t[1])
    dev = max(abs(y['m'][j] - a[f'm{j}']) for j in (28, 36, 37, 38, 39))
    print(f'{nm}: GR§6 锚点复算 max|Δm|={dev:.2e}  max洞 {rmx:.4f}@g{gmx}（文档 {a["hole"][0]}@g{a["hole"][1]}）'
          f'  |Δρ|={abs(rmx - a["hole"][0]):.2e}')

print()
print('=' * 78)
print('## (a) 恒等式自检（闭式 float64；阈值 1e-12）')
print('=' * 78)
def identities(tag, x, Np, form):
    m, rho = x['m'], x['rho']
    # 1) Σ 过渡段超原生预算 = ln4·(m40−m23)
    s1 = sum(math.log(rho[g]) - NB for g in range(23, 40))
    d1 = abs(s1 - LN4 * (m[40] - m[23]))
    # 1b) 直接 ε 和（m 单位）
    eps = [m[g + 1] - m[g] for g in range(23, 40)]
    d1b = abs(sum(eps) - 1.0)
    # 2) 单调性
    dmin = min(m[j + 1] - m[j] for j in range(63))
    # 3) D_j(j>=j*) >= L
    jstar = min(j for j in range(24, 64) if x['D'][j] >= L - 1e-9)
    dmin_star = min(x['D'][j] for j in range(jstar, 64)) - L
    # 4) ρ_max 恒等
    hs = [(g, rho[g]) for g in range(23, 40)]
    gmx, rmx = max(hs, key=lambda t: t[1])
    lam_pred = 4 ** (1.0 / Np) if form == 'UNIF' else 4 ** (2.0 / (Np + 1))
    rho_pred = NAT * lam_pred
    lg, lr = min(hs, key=lambda t: t[1])
    d4 = abs(rmx - rho_pred)
    d4a = abs(rmx / NAT - lam_pred)
    # 5) 全表水床 Σ_all
    s_all = sum(math.log(rho[g]) - NB for g in range(63))
    d5 = abs(s_all - LN4 * (m[63] - m[0]))
    flag = 'PASS' if (d1 < 1e-12 and d1b < 1e-12 and dmin >= -1e-15 and dmin_star >= -1e-9
                      and d4 < 1e-12 and d5 < 1e-12) else 'FAIL'
    print(f'{tag:12s} j*={jstar:2d} | Σ桥−ln4={d1:.2e} ΣΔm−1={d1b:.2e} | minΔm={dmin:+.2e} | '
          f'minD_j*(≥L)−L={dmin_star:+.4f} | ρmax={rmx:.4f}@g{gmx} vs 恒等式 {rho_pred:.4f} Δ={d4:.2e} '
          f'| λmax≡4^(1/N′)仅均分: Δλ={d4a:.2e} | Σall Δ={d5:.2e} | {flag}')
for n in range(13, 18):
    identities(f'RAMP{n:2d}', CAND[f'RAMP{n}'], n, 'RAMP')
for n in range(13, 18):
    identities(f'UNIF{n:2d}', CAND[f'UNIF{n}'], n, 'UNIF')
identities('RAMP16NOTCH', CAND['RAMP16NOTCH'], 16, 'RAMP')
print()
print('闭式 λ_max（=4^(1/N′)，均分）与 ρ_max=ρ_nat·4^(1/N′)（修正式，D2 c1）：')
for n in range(13, 18):
    print(f'  N′={n}: λ=4^(1/N′)={4**(1/n):.6f}  ρ_nat·λ={NAT*4**(1/n):.6f}  | '
          f'ramp 族末步 ε=2/(N′+1)={2/(n+1):.6f} λ={4**(2/(n+1)):.6f} ρ={NAT*4**(2/(n+1)):.6f}')
print()
print('部署(fp32反演)/公式重建数组同检（fp32 精度，阈值不同，如实打印）：')
def identities_json(nm):
    x = DEPLOY[nm]; m, rho = x['m'], x['rho']
    s1 = sum(math.log(rho[g]) - NB for g in range(23, 40))
    dmin = min(m[j + 1] - m[j] for j in range(63))
    jstar = min(j for j in range(24, 64) if x['D'][j] >= L - 1e-9)
    dstar = min(x['D'][j] for j in range(jstar, 64)) - L
    print(f'  {nm:26s} Σ桥−ln4={s1-LN4:+.2e}  minΔm={dmin:+.2e}  minD_j*−L={dstar:+.2e}  j*={jstar}')
for nm in ['MrPro', 'E1_s28_less', 'LongBridgeSlower', 'Smooth_MrBudget', 'E1_pair28_29',
           'LongBridgeFaster', 'MrUni', 'StackFrontBack', 'MrProN16', 'MrProN15']:
    identities_json(nm)
print(f'  （P2 Σ桥={sum(math.log(DEPLOY["FullLagP2_Transfer3B"]["rho"][g])-NB for g in range(23,40))-LN4:+.6f} '
      '——已知 I1 漂移 −0.000775，GR §4-2）')
g_ok = all(M[nm].get('gain') in (None, GAIN) for nm in ['MrPro', 'StackFrontBack', 'MrProN16', 'MrProN15'])
print(f'  gain 恒等：官方 a=1+0.1·ln4={GAIN!r}；JSON gain 字段 '
      f'MrPro={M["MrPro"]["gain"]} Stack={M["StackFrontBack"].get("gain")} '
      f'N16={M["MrProN16"].get("gain")} N15={M["MrProN15"].get("gain")}  一致={g_ok}')

print()
print('=' * 78)
print('## (b) 距离-覆盖矩阵：候选 vs MrPro，证据距离行 89K/96K/106K/117K/127K')
print('=' * 78)
ROWS = [('mk_2@89K', 88726.0, 'MrPro 0.0 / s28 1.0 / LBS 1.0 / P2 0.0'),
        ('mk_1@96K', 95676.0, 'MrPro 1.0 / LBS 0.0 / P2 0.0'),
        ('vt_0@106K', 105880.0, 'MrPro 0.2 / s28 0.2 / LBS 1.0 / P2 0.8'),
        ('mk_0@117K', 117487.0, '全员 1.0'),
        ('vt_1@127K', 127000.0, '全员 1.0')]
SETS = {'MrPro': DEPLOY['MrPro'], 'N16': CAND['RAMP16'], 'N15': CAND['RAMP15'],
        'Stack': DEPLOY['StackFrontBack'], 'N16notch': CAND['RAMP16NOTCH']}

def exposure(x, dd, band=4.0, arc=8.0):
    e, resp, safe = [], [], 0
    for j in range(24, 40):
        if x['m'][j] >= 1 - 1e-9 or R[j] >= arc:
            continue
        if x['T'][j] <= dd <= x['T'][j] * band:
            resp.append(j)
            if x['D'][j] < dd:
                e.append(j)
    for j in range(24, 40):
        if x['D'][j] >= dd and x['T'][j] <= dd and x['r'][j] if False else False:
            pass
    n_safe = sum(1 for j in range(24, 40) if x['D'][j] >= dd and x['T'][j] <= dd and x['m'][j] < 1 - 1e-9)
    return e, resp, n_safe

hdr = f'{"row":11s}' + ''.join(f'{k:>14s}' for k in SETS)
print(hdr)
matrix = {}
for lbl, dd, note in ROWS:
    line = f'{lbl:11s}'
    for k, x in SETS.items():
        e, rp, ns = exposure(x, dd)
        line += f'{str(e):>14s}'
        matrix[(k, lbl)] = (e, rp, ns)
    print(line + f'   实测行分: {note}')
print()
print('覆盖翻转（相对 MrPro：暴露→覆盖 的槽，及裕量 D_j−d；负=仍缺）：')
for lbl, dd, _ in ROWS:
    out = []
    for j in range(24, 40):
        if R[j] >= 8:
            continue
        eM = j in matrix[('MrPro', lbl)][0]
        for k in ['N16', 'N15', 'Stack', 'N16notch']:
            ek = j in matrix[(k, lbl)][0]
            if ek and not eM:
                out.append(f'{k}:+j{j}')
            if eM and not ek:
                dj = SETS[k]['D'][j] - dd
                out.append(f'{k}:−j{j}(Δ={dj:+,.0f})')
    print(f'  {lbl}: ' + ('; '.join(out) if out else '无变化'))
print()
print('洞位置（max ρ 及其部署周期带），供受损行推断：')
for k, x in SETS.items():
    hs = [(g, x['rho'][g], x['T'][g], x['T'][g + 1]) for g in range(23, 40)]
    gmx, rmx, t1, t2 = max(hs, key=lambda t: t[1])
    over = [(g, round(r, 4)) for g, r, _, _ in hs if r > 1.372]
    print(f'  {k:10s} maxρ={rmx:.4f}@g{gmx} 带 T {t1:,.0f}→{t2:,.0f}  超ρ0=1.372 的洞: {over}')

print()
print('=' * 78)
print('## (c) 36 行面板回归：R=U+αH+βΦ 复算（独立重扫可行域）+ 8 方法排序')
print('=' * 78)
FIVE = ['E1_s28_less', 'FullLagP2_Transfer3B', 'LongBridgeSlower', 'MrPro', 'Smooth_MrBudget']
EIGHT = FIVE + ['E1_pair28_29', 'LongBridgeFaster', 'MrUni']
P128 = {'MrPro': 78.1250, 'E1_s28_less': 83.3333, 'LongBridgeSlower': 80.0694,
        'FullLagP2_Transfer3B': 81.6667, 'Smooth_MrBudget': 68.3333,
        'E1_pair28_29': 73.9583, 'LongBridgeFaster': 73.9583, 'MrUni': 73.3333}
P32 = {'MrPro': 87.2222, 'E1_s28_less': 87.2222, 'LongBridgeSlower': 80.5556,
       'FullLagP2_Transfer3B': 72.9167, 'Smooth_MrBudget': 87.2222,
       'E1_pair28_29': 87.2222, 'LongBridgeFaster': 87.2222, 'MrUni': 64.5833}
DIST = {'MrPro': DEPLOY['MrPro'], 'E1_s28_less': DEPLOY['E1_s28_less'],
        'LongBridgeSlower': DEPLOY['LongBridgeSlower'],
        'FullLagP2_Transfer3B': DEPLOY['FullLagP2_Transfer3B'],
        'Smooth_MrBudget': DEPLOY['Smooth_MrBudget'], 'E1_pair28_29': DEPLOY['E1_pair28_29'],
        'LongBridgeFaster': DEPLOY['LongBridgeFaster'], 'MrUni': DEPLOY['MrUni'],
        'StackFrontBack': DEPLOY['StackFrontBack'], 'MrProN16': CAND['RAMP16'],
        'MrProN15': CAND['RAMP15'], 'RAMP16NOTCH': CAND['RAMP16NOTCH']}

def feats(x, rho0, p):
    U = sum(len(exposure(x, dd)[0]) for _, dd, _ in ROWS)
    H = 0.0
    for _, dd, _ in ROWS:
        for g in range(23, 40):
            if x['T'][g] <= dd:
                H += max(0.0, x['rho'][g] - rho0) ** p
    Phi = sum(R[j] * (4 ** x['m'][j] - 1) for j in range(24, 30))
    return U, H, Phi

def cheb(rho0, p, ng=600):
    Fv = {nm: feats(DIST[nm], rho0, p) for nm in FIVE}
    inc = [tuple(Fv[FIVE[i + 1]][k] - Fv[FIVE[i]][k] for k in range(3)) for i in range(4)]
    A = np.linspace(0.05, 10.0, ng)[:, None]
    B = np.linspace(0.02, 5.0, ng)[None, :]
    ok = np.ones((ng, ng), bool)
    for c in inc:
        ok &= (c[0] + c[1] * A + c[2] * B) > 1e-12
    if not ok.any():
        return None
    marg = np.full((ng, ng), 9e9)
    for i, c in enumerate(inc):
        den = np.abs(c[0]) + np.abs(c[1]) * A + np.abs(c[2]) * B + 1e-12
        marg = np.minimum(marg, (c[0] + c[1] * A + c[2] * B) / den)
    marg[~ok] = -9e9
    ii = np.unravel_index(np.argmax(marg), marg.shape)
    aA = np.linspace(0.05, 10.0, ng)
    bB = np.linspace(0.02, 5.0, ng)
    return dict(rho0=rho0, p=p, alpha=float(aA[ii[0]]), beta=float(bB[ii[1]]),
                margin=float(marg[ii]), area_frac=float(ok.mean()),
                amin=float(aA[np.argmax(ok, axis=0)][0]) if ok.any() else None,
                Fv=Fv)

best = None
scan = []
for rho0 in [1.360 + 0.001 * i for i in range(31)]:
    for p in [0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.7]:
        r2 = cheb(rho0, p)
        if r2:
            scan.append((rho0, p, r2['margin'], r2['alpha'], r2['beta'], r2['area_frac']))
            if best is None or r2['margin'] > best[1]['margin']:
                best = (rho0, r2)
print(f'可行 (ρ0,p) 组合数: {len(scan)} / {31*7}（ρ0 步长 .001、αβ 网格 600²；fit5 用 .002/400² 故其窗更窄）')
for row in scan:
    print(f'  ρ0={row[0]:.3f} p={row[1]} margin={row[2]*100:.1f}% α={row[3]:.2f} β={row[4]:.2f} 面积占比={row[5]*100:.2f}%')
# fit5 精确复现检验
Fv_chk = {nm: feats(DIST[nm], 1.372, 0.5) for nm in FIVE}
inc_chk = [tuple(Fv_chk[FIVE[i + 1]][k] - Fv_chk[FIVE[i]][k] for k in range(3)) for i in range(4)]
a_ok, b_ok = 3.920, 1.234
sl = [c[0] + c[1] * a_ok + c[2] * b_ok for c in inc_chk]
print('fit5 发表点检验 (ρ0=1.372,p=0.5,α=3.920,β=1.234): 4 条相邻裕量 = '
      + ' '.join(f'{s:+.3f}' for s in sl) + (' → 严格排序复现 PASS' if min(sl) > 0 else ' → 不成立 FAIL'))
A_ = np.linspace(0.05, 10.0, 400); B_ = np.linspace(0.02, 5.0, 400)
okm = np.ones((400, 400), bool)
for c in inc_chk:
    okm &= (c[0] + c[1] * A_[:, None] + c[2] * B_[None, :]) > 1e-12
if okm.any():
    aA = np.where(okm.any(axis=1))[0]; bB = np.where(okm.any(axis=0))[0]
    print(f'  ng=400 复现窗: α∈[{A_[aA.min()]:.2f},{A_[aA.max()]:.2f}] β∈[{B_[bB.min()]:.2f},{B_[bB.max()]:.2f}] '
          f'面积={okm.mean()*100:.2f}%（fit5 发表: α∈[3.70,4.02] β∈[0.55,1.33] 0.04%）')
print('  可行 ρ0 范围:', min(s[0] for s in scan), '..', max(s[0] for s in scan),
      ' p 取值:', sorted({s[1] for s in scan}))
rho0c, cc = best[0], best[1]
al, be = cc['alpha'], cc['beta']
print(f'选定 Chebyshev 中心: ρ0={rho0c:.3f} p={cc["p"]} α={al:.3f} β={be:.3f} 裕量={cc["margin"]*100:.1f}%')
print('（D2 fit5 发表值: ρ0=1.372 p=0.5 α=3.920 β=1.234；对照见上）')
Rv = {nm: sum(cc['Fv'][nm]) * 0 for nm in FIVE}
FRv = {nm: feats(DIST[nm], rho0c, cc['p']) for nm in DIST}
Rr = {nm: FRv[nm][0] + al * FRv[nm][1] + be * FRv[nm][2] for nm in DIST}
for nm in FIVE:
    U, H, F = FRv[nm]
    print(f'  {nm:24s} U={U:3d} H={H:6.2f} Φ={F:5.2f} R={Rr[nm]:7.2f} 128K={P128[nm]}')
# 幅度直线（5 点）
Xq = np.array([[1.0, -Rr[nm]] for nm in FIVE]); yq = np.array([P128[nm] for nm in FIVE])
(s0, lam), *_ = np.linalg.lstsq(Xq, yq, rcond=None)
print(f'幅度拟合(5点): score ≈ {s0:.2f} − {lam:.3f}·R  残差 ' +
      ' '.join(f'{nm.split("_")[0][:6]}:{s0-lam*Rr[nm]-P128[nm]:+.2f}' for nm in FIVE))
print()
print('8 方法 128K 排序复现：')
meas = sorted(EIGHT, key=lambda n: -P128[n])
pred = sorted(EIGHT, key=lambda n: Rr[n])
print('  实测降序: ' + ' > '.join(n.split("_")[0][:9] for n in meas))
print('  预测降序: ' + ' > '.join(n.split("_")[0][:9] for n in pred))
pairs = [(meas[i], meas[j]) for i in range(len(meas)) for j in range(i + 1, len(meas))
         if P128[meas[i]] > P128[meas[j]]]
conc = sum(1 for a, b_ in pairs if Rr[a] < Rr[b_])
print(f'  可比较序对(排除并列): {len(pairs)}  一致 {conc}  Kendall τ = {2*conc/len(pairs)-1:+.3f}')
for a, b_ in pairs:
    if Rr[a] >= Rr[b_]:
        print(f'  漏点: {a}({P128[a]}) vs {b_}({P128[b_]})  R {Rr[a]:.2f} vs {Rr[b_]:.2f} 违背')
print()
print('32K 侧（含 Smooth/pair 反例结构）：')
for nm in EIGHT:
    U, H, F = FRv[nm]
    xs = DIST[nm]
    # 短端特征：bank 负载 Φ 与 T≤32K 带洞 H32
    H32 = sum(max(0.0, xs['rho'][g] - rho0c) ** cc['p'] for g in range(23, 40) if xs['T'][g] <= 32768.0)
    BL = sum(xs['m'][j] for j in range(24, 29))
    print(f'  {nm:24s} 32K={P32[nm]:6.2f}  Φ={F:5.2f} BL={BL:.4f} H32={H32:5.3f} '
          f'ρmax(T≤32K)={max([xs["rho"][g] for g in range(23,40) if xs["T"][g]<=32768]+[0]):.4f}')
# 可行域弱序检验: s32 = t0 − aΦ − b·H32（网格最小违背）
feats32 = {nm: (FRv[nm][2], sum(max(0.0, DIST[nm]['rho'][g] - rho0c) ** cc['p']
               for g in range(23, 40) if DIST[nm]['T'][g] <= 32768.0)) for nm in EIGHT}
t32 = np.array([P32[nm] for nm in EIGHT])
Ph = np.array([feats32[nm][0] for nm in EIGHT]); Hh = np.array([feats32[nm][1] for nm in EIGHT])
bestv = (1e9, None)
for a in np.linspace(0, 3.0, 61):
    for b in np.linspace(0, 6.0, 61):
        predv = t32.mean() - a * (Ph - Ph.mean()) - b * (Hh - Hh.mean())
        v = float(np.max(predv - t32))  # 只罚“预测高于实测”过冲? 用双向:
        v2 = float(np.max(np.abs(predv - t32 - np.mean(predv - t32))))
        resid = predv - t32; resid -= resid.mean()
        if np.max(np.abs(resid)) < bestv[0]:
            bestv = (float(np.max(np.abs(resid))), (a, b))
print(f'  最小最大残差拟合 score32≈c−aΦ−bH32: |resid|max={bestv[0]:.2f}pp @a={bestv[1][0]:.2f},b={bestv[1][1]:.2f}'
      f'  （128K 行单位 4.17pp；>4.17pp 即至少错一行）')

print()
print('队列候选 R 与预测（[假设]，未执行）：')
for nm in ['StackFrontBack', 'MrProN16', 'MrProN15', 'RAMP16NOTCH']:
    U, H, F = FRv[nm]
    print(f'  {nm:14s} U={U:3d} H={H:6.2f} Φ={F:5.2f} R={Rr[nm]:7.2f} '
          f'({s0-lam*Rr[nm]:6.1f}pp 幅度外推，超内插域不可信) vs MrPro R={Rr["MrPro"]:.2f} → '
          f'{"优于" if Rr[nm] < Rr["MrPro"] else "劣于"}')
print(f'  D2 fit5 发表: Stack R=46.55 / N16 51.15 / N15 51.51; 本复算 '
      f'{Rr["StackFrontBack"]:.2f} / {Rr["MrProN16"]:.2f} / {Rr["MrProN15"]:.2f}')

print()
print('=' * 78)
print('## (d) 预测矩阵支撑数字')
print('=' * 78)
for k in ['N16', 'N15', 'Stack', 'N16notch']:
    x = SETS[k]
    print(f'-- {k}: m24..29={", ".join(f"{x[chr(109)][j]:.4f}" for j in range(24,30))}')
    for j in range(30, 40):
        print(f'   j={j:2d} m={x["m"][j]:.6f} D={x["D"][j]:9,.0f} T={x["T"][j]:9,.0f} ρ_j={x["rho"][j]:.4f}')
print()
print('与 D2/D4 交叉一致性: ')
print(f'  N16: D_38={CAND["RAMP16"]["D"][38]:,.0f} (GR§6 111,347?) — 见上; '
      f'D_37={CAND["RAMP16"]["D"][37]:,.0f} vs mk_1 行 95,676 → 裕量 {CAND["RAMP16"]["D"][37]-95676:+,.0f}')
print(f'  N15: D_37={CAND["RAMP15"]["D"][37]:,.0f} → 裕量 {CAND["RAMP15"]["D"][37]-95676:+,.0f}; '
      f'D_36={CAND["RAMP15"]["D"][36]:,.0f} vs mk_2 88,726 → {CAND["RAMP15"]["D"][36]-88726:+,.0f}')
print(f'  Stack: D_39={DEPLOY["StackFrontBack"]["D"][39]:,.0f} vs vt_1 127,000 → '
      f'{DEPLOY["StackFrontBack"]["D"][39]-127000:+,.0f}; D_38={DEPLOY["StackFrontBack"]["D"][38]:,.0f}')
print(f'  MrPro: D_38={DEPLOY["MrPro"]["D"][38]:,.0f} (106K 行缺 {105880-DEPLOY["MrPro"]["D"][38]:,.0f} tokens)')
# U 合计核对 (D2 a2: MrPro/s28 17, LBS/Smooth 13, P2 0)
for nm in ['MrPro', 'E1_s28_less', 'LongBridgeSlower', 'FullLagP2_Transfer3B', 'Smooth_MrBudget',
           'MrProN16', 'MrProN15', 'StackFrontBack', 'RAMP16NOTCH']:
    print(f'  U({nm}) = {FRv[nm][0]}')
