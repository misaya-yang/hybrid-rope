# -*- coding: utf-8 -*-
"""D2 中频关键性：覆盖矩阵 + 双端风险泛函拟合（纯 CPU 复算）。
输入：analysis/unify_20260910/tables/ground_truth_tables.json（部署反演 m_j）。
常数：W=32768, S=4, ln S=1.3862943611198906, L=131072, b=1e6, Dr=64。
所有数字本脚本现算，不抄表格。
"""
import json, math, itertools

ROOT = '/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope'
d = json.load(open(f'{ROOT}/analysis/unify_20260910/tables/ground_truth_tables.json'))
M = d['methods']
W, S, L = 32768, 4.0, 131072
lnS = math.log(4.0)
lnb64 = math.log(1e6) / 64
NAT = math.exp(lnb64)          # 原生周期比 1.2410...

# 证据距离行（出处：digest_panel-results §5.4；vt_1 精确值本地无出处，取 127000 标[假设]）
ROWS = [('mk2_89K', 88726.0, 'NONGEOMETRIC:348'),
        ('mk1_96K', 95676.0, 'NONGEOMETRIC:443'),
        ('vt0_106K', 105880.0, 'GLM:33'),
        ('mk0_117K', 117487.0, 'NONGEOMETRIC:443'),
        ('vt1_127K', 127000.0, 'UNIFIED §4 [假设精确值]')]
# 逐行得分（ruler.jsonl 复核值，digest §5.4）
SCORES = {'MrPro': dict(mk2=0.0, mk1=1.0, vt0=0.2, mk0=1.0, vt1=1.0),
          'E1_s28_less': dict(mk2=1.0, mk1=1.0, vt0=0.2, mk0=1.0, vt1=1.0),
          'LongBridgeSlower': dict(mk2=1.0, mk1=0.0, vt0=1.0, mk0=1.0, vt1=1.0),
          'FullLagP2_Transfer3B': dict(mk2=0.0, mk1=0.0, vt0=0.8, mk0=1.0, vt1=1.0)}

FIVE = ['MrPro', 'E1_s28_less', 'LongBridgeSlower', 'FullLagP2_Transfer3B', 'Smooth_MrBudget']
HOLD = ['E1_pair28_29', 'LongBridgeFaster', 'E1_s29_more', 'MrProBM', 'MrUni',
        'E7_local_projection', 'GapCapped']
PANEL128 = {'MrPro': 78.1250, 'E1_s28_less': 83.3333, 'LongBridgeSlower': 80.0694,
            'FullLagP2_Transfer3B': 81.6667, 'Smooth_MrBudget': 68.3333,
            'E1_pair28_29': 73.9583, 'LongBridgeFaster': 73.9583, 'E1_s29_more': 77.9167,
            'MrProBM': 70.8333, 'MrUni': 73.3333, 'E7_local_projection': 68.6111,
            'GapCapped': 62.1528}
PANEL32 = {'MrPro': 87.2222, 'E1_s28_less': 87.2222, 'LongBridgeSlower': 80.5556,
           'FullLagP2_Transfer3B': 72.9167, 'Smooth_MrBudget': 87.2222,
           'E1_pair28_29': 87.2222, 'LongBridgeFaster': 87.2222, 'E1_s29_more': 95.5556,
           'MrProBM': 91.6667, 'MrUni': 64.5833, 'E7_local_projection': 90.0,
           'GapCapped': 84.4444}

def derived(name):
    m = [x if x is not None else None for x in M[name]['m_j']]
    Tn = [2 * math.pi * 10 ** (6 * j / 64) for j in range(64)]      # 原生周期
    r = [W / Tn[j] for j in range(64)]                              # 原生窗内圈数
    Tdep = [None if m[j] is None else Tn[j] * 4 ** m[j] for j in range(64)]
    D = [None if m[j] is None else W * 4 ** m[j] for j in range(64)]
    rho = []  # 洞比 ρ_g = T_{g+1}/T_g = exp(gap_g)，按部署周期定义（UNIFIED §1-2）
    for g in range(63):
        if Tdep[g] is None or Tdep[g + 1] is None:
            rho.append(None)
        else:
            rho.append(Tdep[g + 1] / Tdep[g])
    return dict(m=m, Tn=Tn, r=r, T=Tdep, D=D, rho=rho)

DER = {n: derived(n) for n in FIVE + HOLD + ['StackFrontBack', 'MrProN16', 'MrProN15',
                                             'Native', 'HighGapToLong', 'YaRN_linear_official']}

# 自检：与 JSON 自带数组对账
for n in FIVE:
    for j in (24, 28, 32, 36, 39):
        a, b = DER[n]['D'][j], M[n]['D_j'][j]
        assert a is not None and abs(a - b) <= 1e-3 * max(1.0, abs(b)), (n, j, a, b)
print('[自检] D_j 重算 vs JSON 存储值 一致（1e-3 相对容差）')

# ============ (a) 逐槽地平线矩阵：过渡槽 24–39 ============
print('\n===== (a1) 过渡带逐槽 D_j（K=1000 tokens）与 T_j^dep、r_j =====')
hdr = 'slot |  r_j   |' + ''.join(f' {nm[:9]:>11} |' for nm in FIVE)
print(hdr)
for j in range(23, 41):
    row = f'{j:4d} | {DER["MrPro"]["r"][j]:6.2f} |'
    for nm in FIVE:
        D = DER[nm]['D'][j]
        row += f' {D/1000:9.1f}K |' if D else f' {"—":>11} |'
    row += '   T:' + ','.join(f'{DER[nm]["T"][j]/1000:.2f}' for nm in FIVE) + 'K'
    print(row)

print('\n===== (a2) MrPro 危险区地平线与证据距离行的关系 =====')
for nm in FIVE:
    Ds = [DER[nm]['D'][j] for j in (36, 37, 38, 39) if DER[nm]['D'][j]]
    print(f'{nm:26s} D36..39 = ' + ', '.join(f'{x:,.0f}' for x in Ds))

def exposure(nm, d, band=4.0, arc_cut=8.0, bridge=(24, 39)):
    """行距离 d：责任槽 = 桥内 arc 时钟(r<arc_cut) 且 T^dep∈[d/band, d]；
    暴露 = 其中 D_j < d；返回 (暴露集合, 责任集合, 安全桥钟数)"""
    x = DER[nm]
    resp, exp, safe = [], [], []
    for j in range(bridge[0], bridge[1] + 1):
        if x['m'][j] is None or x['m'][j] >= 1.0 - 1e-6 or x['r'][j] >= arc_cut:
            continue
        if x['T'][j] <= d <= x['T'][j] * band:
            resp.append(j)
            if x['D'][j] < d:
                exp.append(j)
    n_safe = sum(1 for j in range(bridge[0], bridge[1] + 1)
                 if x['D'][j] and x['D'][j] >= d and x['T'][j] <= d)
    return exp, resp, n_safe

print('\n===== (a3) 覆盖矩阵：暴露的桥内 arc 时钟（D_j < d）/ 责任槽 =====')
for _, dd, src in ROWS:
    line = f'd={dd:,.0f} ({src}):'
    for nm in FIVE:
        e, rp, ns = exposure(nm, dd)
        line += f'\n    {nm:24s} 责任{rp} 暴露{e} 安全桥钟={ns}'
    print(line)

print('\n===== (a4) 得分联表复算校验（vs digest §5.4） =====')
for nm in ['MrPro', 'E1_s28_less', 'LongBridgeSlower', 'FullLagP2_Transfer3B']:
    sc = SCORES[nm]
    ex = {}
    for key, dd in [('mk2', 88726.0), ('mk1', 95676.0), ('vt0', 105880.0)]:
        e, rp, ns = exposure(nm, dd)
        ex[key] = (len(e), len(rp))
    print(f'{nm:24s} scores={sc}  暴露数/责任数 mk2={ex["mk2"]} mk1={ex["mk1"]} vt0={ex["vt0"]}')

# LBS 手术前后关键穿越
print('\n===== (a5) LBS 补哪段 / s28 补哪段（数值穿越） =====')
for j in (36, 37, 38, 39):
    a, b = DER['MrPro']['D'][j], DER['LongBridgeSlower']['D'][j]
    print(f'D_{j}: MrPro {a:,.0f} -> LBS {b:,.0f}  (x{b/a:.4f})')
print(f'vt_0 行距离 105,880: LBS D_38 = {DER["LongBridgeSlower"]["D"][38]:,.0f}'
      f'  裕量 = {DER["LongBridgeSlower"]["D"][38] - 105880:,.0f} tokens')
print(f'MrPro D_38 = {DER["MrPro"]["D"][38]:,.0f} 差 {105880 - DER["MrPro"]["D"][38]:,.0f} tokens 未覆盖')
x28p, x28s = DER['MrPro'], DER['E1_s28_less']
for nm, x in [('MrPro', x28p), ('s28', x28s)]:
    m28 = x['m'][28]
    shift3k = 3000 / x['Tn'][28] * (1 - 4 ** (-m28))   # d=3K 处相位移动（圈）
    print(f'{nm}: m28={m28:.6f} T28^dep={x["T"][28]:,.1f} 局部相位移动@3K={shift3k:.4f} 圈'
          f' = {shift3k*2*math.pi:.3f} rad')

# ============ (b) r_j 阶梯：窗内容忍 / 窗外恶性双重性 ============
print('\n===== (b1) 槽 22–41 的 r_j 阶梯（r=W/T^nat；二分法边界 r=8 与 r=1） =====')
for j in range(22, 42):
    r = DER['MrPro']['r'][j]
    tag = 'BANK(r>=8)' if r >= 8 else ('SLOPE(r<1)' if r < 1 else 'ARC(1..8)')
    if 36 <= j <= 39 and r <= 2.25:
        tag += ' DANGER(r 1.15~2.2)'
    print(f'  j={j:2d} r={r:6.2f} T^nat={DER["MrPro"]["Tn"][j]:9,.0f}  {tag}')

print('\n===== (b2) 32K 窗内容忍证据：桥形不同、32K 分数 bit-相同 =====')
for nm in ['MrPro', 'E1_s28_less', 'Smooth_MrBudget', 'E1_pair28_29', 'LongBridgeFaster']:
    print(f'  {nm:20s} 32K={PANEL32[nm]}  Σm_24..39={sum(DER[nm]["m"][24:40]):.4f} '
          f' m36..39={",".join(f"{DER[nm]["m"][j]:.4f}" for j in (36,37,38,39))}')
print('  （对照：动桥但伤及别的机制的 LBS 80.56 / P2 72.92 / BM 91.67 —— 见 (c) 泛函分解）')

# ============ (c) 洞预算与双端风险泛函 ============
print('\n===== (c1) 均分界：ρ_max ≥ ρ_nat · 4^{1/N′} =====')
for Np in (17, 16, 15, 8, 1):
    print(f'  N′={Np:2d}: 均分界 ρ_minmax = {NAT * 4 ** (1/Np):.4f}'
          f'  (MrPro 实测 max ρ = 1.4477@N′=17)')

print('\n===== (c2) 各表逐 gap 洞比（g=23..39） =====')
for nm in FIVE + ['E1_pair28_29', 'MrProBM']:
    x = DER[nm]
    gaps = [(g, x['rho'][g]) for g in range(23, 40)]
    mx = max(gaps, key=lambda t: t[1])
    above = [(g, round(r, 4)) for g, r in gaps if r > 1.35]
    print(f'  {nm:22s} max ρ={mx[1]:.4f}@g{mx[0]}  >1.35 的洞: {above}')

def components(nm, rho0, p, bank=(24, 29)):
    x = DER[nm]
    U = sum(len(exposure(nm, dd)[0]) for _, dd, _ in ROWS)          # 长端：行暴露计数和
    H = sum((max(0.0, x['rho'][g] - rho0)) ** p for g in range(23, 40)
            if x['rho'][g] is not None)                              # 短端：超容洞惩罚
    F = sum(x['m'][j] for j in range(bank[0], bank[1] + 1) if x['m'][j] is not None)
    return U, H, F

import numpy as np
print('\n===== (c3) 5 点精确拟合（5×5 线性解 s0 − (A·U+B·H+C·F)，网格 ρ0,p）=====')
def solve(rho0, p, bank=(24, 29)):
    X, y = [], []
    for nm in FIVE:
        U, H, F = components(nm, rho0, p, bank)
        X.append([1.0, U, H, F]); y.append(PANEL128[nm])
    X = np.array(X); y = np.array(y)
    try:
        coef = np.linalg.solve(X, y)
    except np.linalg.LinAlgError:
        return None
    resid = float(np.max(np.abs(X @ coef - y)))
    return coef, resid, {nm: components(nm, rho0, p, bank) for nm in FIVE}

region = []
for rho0 in [1.30 + 0.002 * i for i in range(80)]:
    for p in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        for band in [2.0, 4.0, 8.0]:
            for arc in [8.0, 9.96]:
                # 参数化 exposure 定义影响 U —— 通过闭包重定义
                pass
# exposure 依赖全局 band/arc：改为显式重算
def exposure_p(nm, dd, band, arc, bridge=(24, 39)):
    x = DER[nm]; e = []
    for j in range(*[bridge[0], bridge[1] + 1]):
        if x['m'][j] is None or x['m'][j] >= 1 - 1e-6 or x['r'][j] >= arc:
            continue
        if x['T'][j] <= dd <= x['T'][j] * band and x['D'][j] < dd:
            e.append(j)
    return e

def solve2(rho0, p, band, arc, bank=(24, 29)):
    X, y = [], []
    comps = {}
    for nm in FIVE:
        x = DER[nm]
        U = sum(len(exposure_p(nm, dd, band, arc)) for _, dd, _ in ROWS)
        H = sum((max(0.0, x['rho'][g] - rho0)) ** p for g in range(23, 40) if x['rho'][g])
        F = sum(x['m'][j] for j in range(bank[0], bank[1] + 1))
        comps[nm] = (U, H, F)
        X.append([1.0, U, H, F]); y.append(PANEL128[nm])
    X = np.array(X); y = np.array(y)
    try:
        coef = np.linalg.solve(X, y)
    except np.linalg.LinAlgError:
        return None
    resid = float(np.max(np.abs(X @ coef - y)))
    return coef, resid, comps

good = []
for band in (2.0, 4.0, 8.0):
    for arc in (8.0, 9.96):
        for bank in ((24, 29), (24, 28), (24, 30)):
            for rho0 in [1.2410 + 0.001 * i for i in range(220)]:
                for p in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
                    r2 = solve2(rho0, p, band, arc, bank)
                    if r2 is None:
                        continue
                    coef, resid, comps = r2
                    s0, A, B, C = coef
                    if resid < 1e-9 and A > 0 and B > 0 and C > 0:
                        good.append((band, arc, bank, rho0, p, s0, A, B, C))
print(f'  可行参数点（5 点残差<1e-9 且 A,B,C>0）: {len(good)} / 共扫描 '
      f'{3*2*3*220*7} 点')
if good:
    import statistics as st
    for i, keyf in enumerate(['A(长端)', 'B(洞)', 'C(bank)']):
        vs = [g[6 + i] for g in good]
        print(f'  {keyf}: min={min(vs):.4f} med={st.median(vs):.4f} max={max(vs):.4f}')
    print('  ρ0 分布: min={:.4f} med={:.4f} max={:.4f}'.format(
        min(g[3] for g in good), st.median([g[3] for g in good]), max(g[3] for g in good)))
    print('  band/arc/bank 组合计数:', {k: sum(1 for g in good if (g[0], g[1], g[2]) == k)
                                        for k in sorted({(g[0], g[1], g[2]) for g in good})})

# 中位参数 → 留出法检验
if good:
    med = lambda xs: st.median(xs)
    rho0m = med([g[3] for g in good]); pm = med([g[4] for g in good])
    Am = med([g[6] for g in good]); Bm = med([g[7] for g in good]); Cm = med([g[8] for g in good])
    s0m = med([g[5] for g in good])
    bandm = med([g[0] for g in good]); arcm = med([g[1] for g in good])
    bankm = (24, 29)
    print(f'\n  中位参数: s0={s0m:.3f} A={Am:.3f} B={Bm:.3f} C={Cm:.3f} rho0={rho0m:.4f} p={pm} band={bandm} arc={arcm}')
    print('  ===== 留出法（未参与拟合的已测臂）预测 vs 实际 128K =====')
    for nm in HOLD + FIVE:
        x = DER[nm]
        U = sum(len(exposure_p(nm, dd, bandm, arcm)) for _, dd, _ in ROWS)
        H = sum((max(0.0, x['rho'][g] - rho0m)) ** pm for g in range(23, 40) if x['rho'][g])
        F = sum(x['m'][j] for j in range(bankm[0], bankm[1] + 1))
        pred = s0m - (Am * U + Bm * H + Cm * F)
        tag = 'fit' if nm in FIVE else 'HOLD'
        err = pred - PANEL128[nm]
        print(f'  [{tag}] {nm:24s} U={U:2d} H={H:7.4f} F={F:.4f} pred={pred:6.2f} '
              f'actual={PANEL128[nm]:6.2f} err={err:+6.2f}')
    print('  ===== 队列候选预测（未执行，无分数，仅结构预测） =====')
    for nm in ['StackFrontBack', 'MrProN16', 'MrProN15']:
        x = DER[nm]
        U = sum(len(exposure_p(nm, dd, bandm, arcm)) for _, dd, _ in ROWS)
        H = sum((max(0.0, x['rho'][g] - rho0m)) ** pm for g in range(23, 40) if x['rho'][g])
        F = sum(x['m'][j] for j in range(bankm[0], bankm[1] + 1))
        Rr = Am * U + Bm * H + Cm * F
        print(f'  {nm:15s} U={U:2d} H={H:7.4f} F={F:.4f} R={Rr:6.3f} '
              f'（越低越好；MrPro R={s0m-PANEL128["MrPro"]:.3f}）')

print('\n===== (c4) P2 洞带在 89K 行的“梯级丢失”复算 =====')
x = DER['FullLagP2_Transfer3B']; y0 = DER['MrPro']
for nm, xx in [('P2', x), ('MrPro', y0)]:
    rung = [(j, xx['T'][j], 88726.0 / xx['T'][j]) for j in range(24, 40)
            if xx['T'][j] and 3 <= 88726.0 / xx['T'][j] <= 40]
    ts = [t for _, t, _ in rung]
    holes = [(ts[i + 1] / ts[i], ts[i], ts[i + 1]) for i in range(len(ts) - 1)]
    worst = max(holes, default=(1, 0, 0))
    print(f'  {nm}: 89K 行 3~40 圈梯级 {len(rung)} 根，最大相邻比 {worst[0]:.3f} '
          f'(T {worst[1]:,.0f}->{worst[2]:,.0f})')

print('\n===== (c5) 逐行“危险带”包含关系复算（UNIFIED §4 声明的数字核验） =====')
m3639 = [DER['MrPro']['D'][j] for j in (36, 37, 38, 39)]
print(f'  MrPro 危险区地平线带 = [{min(m3639):,.0f}, {max(m3639):,.0f}]')
for label, dd, res in [('mk2', 88726, 'MrPro 0.0 / s28,LBS 1.0'), ('vt0', 105880, 'MrPro 0.2 / LBS 1.0'),
                       ('mk1', 95676, 'MrPro 1.0（反例）'), ('mk0', 117487, '全员 1.0（反例带外仍对）'),
                       ('vt1', 127000, '全员 1.0')]:
    inside = min(m3639) <= dd <= max(m3639)
    e = len(exposure_p('MrPro', dd, 4.0, 8.0))
    print(f'  {label}@{dd:,.0f}: 带内={inside} MrPro暴露桥钟数={e}  得分: {res}')
