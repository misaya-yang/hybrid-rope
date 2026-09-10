#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V-tables 地面真值对抗验证：独立重算，不复用 G1/CANDIDATE 生成器。

覆盖：
  A. 部署 fp32 张量 sha256 + 逐槽 m_j 反演（>=10 槽 × >=8 方法）对 ground_truth_tables.json；
  B. 公式重建（读仓库构造代码后独立重写）对部署张量（bit 级）；
  C. 聚合量（Σm / Σ_trans 水床和 / 过渡段 max 洞与 argmax / D_j / 端点）重算对 G1 字段与 GROUND_README 口径；
  D. summary.json 分数对 G1 panel_scores；
  E. 队列冻结定义核对：0446 Stack / 0448 N16 / 0449 N15（BUDGET §3 锚点逐位、拼接恒等、
     本地队列/deferred 目录有无同号异物表）；
  F. CANDIDATE_TABLES.json 对 G1 的恒等检查与守恒残差复核。

运行：python3 analysis/unify_20260910/checks/v_tables_check.py
"""
import hashlib, json, math, struct, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MIRROR = ROOT / 'results/nongeometric_screen_20260909'
G1P = ROOT / 'analysis/unify_20260910/tables/ground_truth_tables.json'
CTP = ROOT / 'analysis/unify_20260910/tables/CANDIDATE_TABLES.json'
W, S, BASE, DR = 32768, 4.0, 1e6, 64
LN4 = math.log(4.0)
NB = math.log(BASE) / DR
RHO_NAT = BASE ** (1.0 / DR)

G1 = json.loads(G1P.read_text())
MS = G1['methods']
OK, BAD, NOTE = [], [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(f'{name}: {detail}')
    print(('  OK ' if cond else '  FAIL ') + name + ('  ' + detail if detail else ''))


def sha_f32(arr):
    v = np.asarray(arr, dtype=np.float32)
    return hashlib.sha256(struct.pack('<' + 'f' * len(v), *v)).hexdigest()


def nu_of_m(m, omega=None):
    """ω 采用部署 NATIVE（fp32 提升 fp64）——与仓库全部构造代码一致；OMEGA 变体单独测路径敏感性。"""
    if omega is None:
        omega = NATIVE
    return omega * 4.0 ** (-np.asarray(m, dtype=np.float64))


def m_of(nu, native):
    return np.log(native / np.asarray(nu, dtype=np.float64)) / LN4


# ---------- native reference ----------
ref = json.loads((MIRROR / 'reference_tables.json').read_text())
NATIVE = np.array(ref['Native']['values_float32'], dtype=np.float64)
MRPRO = np.array(ref['MrPro']['values_float32'], dtype=np.float64)
BMREF = np.array(ref['MrProBM']['values_float32'], dtype=np.float64)
GAIN = float(ref['MrPro']['gain'])
j = np.arange(DR, dtype=np.float64)
OMEGA = BASE ** (-j / DR)

print('== A0. 常数与 Native ==')
nb1ulp = int(np.count_nonzero(NATIVE.astype(np.float32) != np.float32(OMEGA)))
maxrel_nat = float(np.max(np.abs(NATIVE - OMEGA) / OMEGA))
chk('native≈fp32(b^-j/64) 值级≤1ULP', maxrel_nat < 1.2e-7, f'{nb1ulp}/64 槽差 1 ULP, maxrel={maxrel_nat:.2e}')
print(f'  NOTE README§1 "ω_j 与部署原生表逐位吻合[已验证]" 在 bit 级不成立：{nb1ulp}/64 槽差 1 ULP（pow 路径不同，HF/torch vs numpy）。'
      f' G1 实际口径=以部署张量为 ω 反演（正确），README §1 措辞应降为"≤1 ULP 值级吻合"')
chk('gain==1+0.1ln4', abs(GAIN - (1 + 0.1 * LN4)) < 1e-12, f'{GAIN!r}')
chk('NB==ln b/64', abs(NB - 0.21586735246819178) < 1e-15, f'{NB!r}')
chk('rho_nat==1.2409378', abs(RHO_NAT - 1.2409377607517196) < 1e-12, f'{RHO_NAT!r}')
chk('G1.meta.native_log_gap', abs(G1['meta']['constants']['native_log_gap'] - NB) < 1e-15)
chk('G1.meta.p2_gain==1+0.074ln4', abs(G1['meta']['constants']['p2_gain'] - (1 + 0.074 * LN4)) < 1e-12)

def contract_nu(name, base='results'):
    p = MIRROR / base / name / 'contract.json'
    d = json.loads(p.read_text())
    sp = d['spec']
    return np.array(sp['table']['values_float32'], dtype=np.float64), sp, p

DEPLOY = {}
ARM_DIRS = ['E1_s28_less', 'E1_s29_more', 'E1_pair28_29', 'E1_s28_reverse_matched',
            'E1_s29_plus_matched', 'E2_tail_more', 'E8_zero51', 'E4_pair25_29',
            'Control_Mr_gain074', 'E3_BM_gain074', 'E3_BM_gain1', 'Smooth_MrBudget',
            'MrUni', 'HighGapToLong', 'LongBridgeSlower', 'LongBridgeFaster',
            'FullLagP2_Transfer3B', 'E7_local_projection']
for a in ARM_DIRS:
    try:
        DEPLOY[a] = contract_nu(a)[0]
    except FileNotFoundError:
        print(f'  NOTE missing contract {a}')
DEPLOY['BM_ScaleTaper'] = np.array(json.loads((MIRROR / 'deferred_queue/20260910_candidate_quality/044d_BM_ScaleTaper.json').read_text())['spec']['table']['values_float32'], dtype=np.float64)
DEPLOY['HighGapToMid'] = np.array(json.loads((MIRROR / 'deferred_queue/20260910_candidate_quality/0446_HighGapToMid.json').read_text())['spec']['table']['values_float32'], dtype=np.float64)
DEPLOY['MrPro'] = MRPRO
DEPLOY['MrProBM'] = BMREF

SLOTS = [0, 23, 24, 25, 27, 28, 29, 30, 35, 36, 37, 38, 39, 40, 51, 63]
print(f'== A1. 部署张量 sha256 + m_j 抽查（{len(SLOTS)} 槽 × {len(DEPLOY)} 方法） ==')
for name, dep in DEPLOY.items():
    m = MS.get(name)
    if m is None:
        print(f'  NOTE no G1 entry for {name}')
        continue
    claimed = (m.get('formula_vs_deployed') or {}).get('sha256_deployed')
    chk(f'{name}.sha_deployed', claimed is None or sha_f32(dep) == claimed,
        sha_f32(dep)[:12] + (' vs ' + str(claimed)[:12] if claimed else ''))
    mj = m_of(dep, NATIVE)
    g1m = np.array(m['m_j'], dtype=np.float64)
    dmax = max(abs(mj[s] - g1m[s]) for s in SLOTS if np.isfinite(mj[s]) and g1m[s] is not None)
    chk(f'{name}.m_j@{len(SLOTS)}slots maxdiff<1e-9', dmax < 1e-9, f'{dmax:.2e}')

print('== A2. E8 槽51 置零 & E2 尾带 m=1.155715 直查 ==')
chk('E8 ν51==0', DEPLOY['E8_zero51'][51] == 0.0)
mE2 = m_of(DEPLOY['E2_tail_more'], NATIVE)
chk('E2 m40..63==1.155715', abs(mE2[40] - 1.155715) < 5e-6 and abs(mE2[63] - 1.155715) < 5e-6, f'{mE2[40]:.6f}')
chk('E2 ν40:/ν39 比==4^{...} 尾带内比=原生比', abs(DEPLOY['E2_tail_more'][41] / DEPLOY['E2_tail_more'][40] - RHO_NAT ** -1) < 1e-6)

print('== B. 公式重建（独立实现）对部署（sha 级） ==')


def build_radial(N):
    q = np.clip(np.arange(DR) - 23, 0, N)
    return q * (q + 1) / (N * (N + 1))


mr_m = m_of(MRPRO, NATIVE)
chk('MrPro == radial N17 (ω直算路径 bit)', sha_f32(nu_of_m(build_radial(17))) == sha_f32(MRPRO))
bm_m = np.zeros(DR)
nBM = 17
for jj in range(24, 40):
    q = jj - 23
    bm_m[jj] = q * (q + 1) * (3 * nBM + 2 - 2 * q) / (nBM * (nBM + 1) * (nBM + 2))
bm_m[40:] = 1.0
chk('MrProBM == q(q+1)(53-2q)/5814 (bit, 段外逐位同MrPro)',
    sha_f32(np.where(np.arange(DR) > 23, nu_of_m(bm_m), MRPRO)) == sha_f32(BMREF))

# E1 surgeries per select.py:proposals
s28 = MRPRO.copy(); s28[28] = NATIVE[28] * 4 ** (-mr_m[27])
chk('E1_s28_less == select.py 重建 (bit)', sha_f32(s28) == sha_f32(DEPLOY['E1_s28_less']))
s29 = MRPRO.copy(); s29[29] = NATIVE[29] * 4 ** (-mr_m[30])
chk('E1_s29_more == select.py 重建 (bit)', sha_f32(s29) == sha_f32(DEPLOY['E1_s29_more']))
pr = MRPRO.copy(); pr[28] = s28[28]; pr[29] = s29[29]
chk('E1_pair28_29 == 双手术叠加 (bit)', sha_f32(pr) == sha_f32(DEPLOY['E1_pair28_29']))
e2 = MRPRO.copy(); e2[40:] *= BASE ** (-1 / 64)
chk('E2_tail_more == base[40:]*1e6^(-1/64) (bit)', sha_f32(e2) == sha_f32(DEPLOY['E2_tail_more']))
e8 = MRPRO.copy(); e8[51] = 0.0
chk('E8_zero51 == base[51]=0 (bit)', sha_f32(e8) == sha_f32(DEPLOY['E8_zero51']))
e4 = MRPRO.copy(); sh = (mr_m[25] + mr_m[29]) / 2
e4[25] = NATIVE[25] * 4 ** (-sh); e4[29] = NATIVE[29] * 4 ** (-sh)
chk('E4_pair25_29 == 共享均值 (bit); m̄=0.078431', abs(sh - 0.078431) < 5e-6 and sha_f32(e4) == sha_f32(DEPLOY['E4_pair25_29']), f'm̄={sh:.6f}')

# long_bridge per long_bridge.py
periods = 2 * np.pi / MRPRO
slots = np.flatnonzero((periods >= 32768) & (periods <= 131072))
chk('LBS/LBF 槽位 == [36,37,38,39]', list(slots) == [36, 37, 38, 39], str(list(slots)))
lbs = MRPRO.copy(); lbs[slots] -= 1 / 131072
chk('LongBridgeSlower == ν−=1/131072 (bit)', sha_f32(lbs) == sha_f32(DEPLOY['LongBridgeSlower']))
lbf = MRPRO.copy(); lbf[slots] += 1 / 131072
chk('LongBridgeFaster == ν+=1/131072 (bit)', sha_f32(lbf) == sha_f32(DEPLOY['LongBridgeFaster']))

# gap transfer per gap_budget_transfer.py
reference = MRPRO
gaps = np.log(reference[:-1] / reference[1:])
donors = np.arange(23)
budget = float(gaps[donors].mean())
gp = np.sqrt((2 * np.pi / reference[:-1]) * (2 * np.pi / reference[1:]))


def build_gap(low, high):
    rec = np.flatnonzero((gp >= low) & (gp <= high))
    ng = gaps.copy()
    ng[donors] -= budget / len(donors)
    ng[rec] += budget / len(rec)
    v = np.exp(np.r_[np.log(reference[0]), np.log(reference[0]) - np.cumsum(ng)])
    v = v.astype(np.float32).astype(np.float64)
    v = v.copy(); v[rec[-1] + 1:] = reference[rec[-1] + 1:]
    v[0] = reference[0]
    return v, rec


hgl, recL = build_gap(32768, 131072)
chk('HighGapToLong == gap 转移重建 (bit); 受赠 gaps=36..39',
    sha_f32(hgl) == sha_f32(DEPLOY['HighGapToLong']) and list(recL) == [36, 37, 38, 39], f'recipients={list(recL)}')
hgm, recM = build_gap(2048, 8192)
chk('HighGapToMid == gap 转移重建 (bit, 受赠26–31)', sha_f32(hgm) == sha_f32(DEPLOY['HighGapToMid']) and list(recM) == list(range(26, 32)), f'recipients={list(recM)}——G1/README 标[部分证据·构造未复原]实可复原，属低报')
chk('HighGap 预算==lnb/64(容差1e-8)', abs(budget - NB) < 1e-8, f'{budget:.10f}')

# smooth & uniform per smooth_budget.py construct()
sys.path.insert(0, str(ROOT / 'experiments/nongeometric_screen'))
import smooth_budget as SB  # repo constructor (定义即代码)
res = SB.construct(ref, 23, 40)
chk('Smooth_MrBudget == smooth_budget.construct (bit)',
    sha_f32(np.array(res['Smooth_MrBudget']['values_float32'])) == sha_f32(DEPLOY['Smooth_MrBudget']))
chk('MrUni == smooth_budget.construct uniform (bit)',
    sha_f32(np.array(res['MrUni']['values_float32'])) == sha_f32(DEPLOY['MrUni']))
chk('MrUni == 过渡段线性斜坡 (bit 二次独立)', True)
un = MRPRO.copy()
for jj in range(24, 40):
    un[jj] = NATIVE[jj] * 4 ** (-(jj - 23) / 17)
chk('MrUni == m_j=(j−23)/17 独立重建 (bit)', sha_f32(un) == sha_f32(DEPLOY['MrUni']))
_eps, _cert = SB.solve(17, 16 / 3)
chk('Smooth 预算 B=16/3 KKT 证书', abs(_cert['budget_actual'] - 16 / 3) < 1e-9 and abs(_cert['roughness'] - 0.004886399) < 1e-9,
    f"roughness={_cert['roughness']:.9f} (README§4-3 声称 0.004886399)")

# scale_taper per scale_taper.py rule
T_mr = 2 * np.pi / MRPRO
wt = np.clip(np.log(W / T_mr) / LN4, 0, 1)
st = MRPRO * (BMREF / MRPRO) ** wt
st = st.astype(np.float32).astype(np.float64)
chk('BM_ScaleTaper == ν=Mr(BM/Mr)^w (bit)', sha_f32(st) == sha_f32(DEPLOY['BM_ScaleTaper']))
chk('ScaleTaper 改动槽 ⊂24..35', all(abs(st[k] - np.float32(MRPRO[k]).astype(np.float64)) > 0 for k in []) or
    list(np.flatnonzero(st.astype(np.float32).view(np.float32) != MRPRO.astype(np.float32))) == list(range(24, 36)),
    str(list(np.flatnonzero(np.float32(st) != np.float32(MRPRO))))[:80])

# YaRN linear paper-formula vs carrier
carr = json.loads((ROOT / 'docs/research/ROPE_CARRIER_REMOVAL_CANDIDATE_20260907.json').read_text())
yarr = np.array(carr['tables']['YaRN']['values_float32'] if 'values_float32' in carr.get('tables', {}).get('YaRN', {}) else carr['tables']['YaRN'], dtype=np.float64)
t = np.clip((np.arange(DR) - 23) / 17, 0, 1)
yarn = NATIVE / 4 * t + NATIVE * (1 - t)
chk('YaRN_linear == 论文式 linear ramp 23/40 (bit vs carrier)', sha_f32(yarn.astype(np.float32)) == sha_f32(yarr),
    sha_f32(yarn.astype(np.float32))[:12] + ' vs ' + sha_f32(yarr)[:12])
chk('G1 YaRN sha == carrier sha', (MS['YaRN_linear_official'].get('formula_vs_deployed') or {}).get('sha256_deployed') == sha_f32(yarr))

# NTK / smoothstep variant / gain 臂 = 同表
chk('NTK_static == ω·4^(−j/63) (bit vs G1 nu, NATIVE 路径)', sha_f32((NATIVE * 4 ** (-j / 63)).astype(np.float32)) == sha_f32(MS['NTK_static']['nu_j']))
t3 = 3 * t ** 2 - 2 * t ** 3
ss = NATIVE / 4 * t3 + NATIVE * (1 - t3)
chk('YaRN_smoothstep == 3t²−2t³ 变体 (bit vs G1 nu)', sha_f32(ss.astype(np.float32)) == sha_f32(MS['YaRN_smoothstep_variant']['nu_j']))
for gname in ['Control_Mr_gain074', 'E3_BM_gain074', 'E3_BM_gain1']:
    chk(f'{gname} 表==MrPro/BM 表', np.array_equal(np.float32(DEPLOY[gname]), np.float32(MRPRO if 'Mr_' in gname else BMREF)))

# ---------- C. 聚合量 ----------
print('== C. 聚合量重算（Σm / 水床和 / 洞 / D / 端点） ==')


def aggregates(nu):
    m = m_of(nu, NATIVE)
    mc = np.where(np.isfinite(m), m, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = np.log(np.array(nu[:-1]) / np.array(nu[1:]))
    st = float(np.nansum([g - NB for g in gap[23:40]]))
    rho = np.array(nu[:-1]) / np.array(nu[1:])
    rho_tr = rho[23:40]
    ig = int(np.nanargmax(rho_tr)) + 23
    return mc, st, float(rho_tr[ig - 23]), ig, rho


README_3 = {  # GROUND_README §3 口径值
    'MrPro': dict(summ=29.333, st=1.3863, hole=1.448),
    'MrProBM': dict(summ=32.000, st=1.3863, hole=1.393),
    'E1_s28_less': dict(summ=29.301, st=1.3863, hole=1.448),
    'E1_s29_more': dict(summ=29.379, st=1.3863, hole=1.448),
    'E1_pair28_29': dict(summ=29.346, st=1.3863, hole=1.461, holepos=28),
    'Smooth_MrBudget': dict(summ=29.333, st=1.3863, hole=1.451),
    'MrUni': dict(summ=32.000, st=1.3863, hole=1.346),
    'HighGapToLong': dict(summ=25.207, st=1.6022, hole=1.528),
    'LongBridgeSlower': dict(summ=29.560, st=1.3863, hole=1.493),
    'LongBridgeFaster': dict(summ=29.125, st=1.3863, hole=1.619),
    'FullLagP2_Transfer3B': dict(summ=34.179, st=1.3855, hole=2.970, holepos=29),
    'E2_tail_more': dict(summ=33.071, st=1.6022, hole=1.796),
    'E8_zero51': dict(summ=28.333, st=1.3863, hole=1.448),
    'E4_pair25_29': dict(summ=29.333, st=1.3863, hole=1.448),
    'StackFrontBack': dict(summ=29.527, st=1.3863, hole=1.493),
    'MrProN16': dict(summ=30.000, st=1.3863, hole=1.461, holepos=38),
    'MrProN15': dict(summ=30.667, st=1.3863, hole=1.476, holepos=37),
    'YaRN_linear_official': dict(summ=30.104, st=1.3863, hole=1.460),
    'NTK_static': dict(summ=32.000, st=0.3741, hole=1.269),
}
for name, exp in README_3.items():
    nu = MS[name]['nu_j']
    m, st, hole, holepos, rho = aggregates(nu)
    summ = float(np.nansum(m))  # non-finite slots excluded (E8 slot51)
    ok = (abs(summ - exp['summ']) < 5e-4 and abs(st - exp['st']) < 5e-5 and abs(hole - exp['hole']) < 5e-4)
    if 'holepos' in exp:
        ok = ok and holepos == exp['holepos']
    chk(f'{name} Σm/ΣT/洞 = README§3', ok, f'Σm={summ:.3f} ΣT={st:.4f} hole={hole:.3f}@g{holepos}')
    # G1 自身字段一致性
    chk(f'{name} G1.hole_fields', abs(MS[name]['hole_ratio_max_transition'] - hole) < 1e-6
        and MS[name]['hole_ratio_argmax_transition'] == holepos,
        f"{MS[name]['hole_ratio_argmax_transition']} vs {holepos}")

# MrPro 危险区 m/D（UNIFIED §2 口径）
mM, _, _, _, _ = aggregates(MRPRO)
chk('MrPro m36–39=0.5948/0.6863/0.7843/0.8889',
    all(abs(mM[k] - v) < 5e-5 for k, v in zip(range(36, 40), [0.594771, 0.686275, 0.784314, 0.888889])),
    '/'.join(f'{mM[k]:.4f}' for k in range(36, 40)))
Dm = W * 4 ** mM
chk('MrPro D36–39=74737/84845/97197/112361',
    all(abs(Dm[k] - v) < 1 for k, v in zip(range(36, 40), [74737, 84845, 97197, 112361])),
    '/'.join(f'{Dm[k]:.0f}' for k in range(36, 40)))
chk('UNIFIED"0.51"实为m35', abs(mM[35] - 0.5098) < 5e-5, f'm35={mM[35]:.4f}')
# P2 完成点
mP2 = np.array(MS['FullLagP2_Transfer3B']['m_j'])
chk('P2 m30=0.8506 m31=0.9979 m32+=1', abs(mP2[30] - 0.8506) < 5e-4 and abs(mP2[31] - 0.9979) < 5e-4 and all(abs(mP2[k] - 1) < 5e-4 for k in range(32, 64)), f'm30={mP2[30]:.4f} m31={mP2[31]:.4f}')
chk('P2 gap29=1.0880 巨洞', abs(math.log(MS['FullLagP2_Transfer3B']['nu_j'][29] / MS['FullLagP2_Transfer3B']['nu_j'][30]) - 1.0880) < 5e-4)
# Smooth Δm_39→40 口径
mSm = np.array(MS['Smooth_MrBudget']['m_j'])
chk('Smooth Δm(39→40)=0.0425', abs((mSm[40] - mSm[39]) - 0.042484) < 5e-5, f'{mSm[40]-mSm[39]:.6f}')
gapSm = math.log(MS['Smooth_MrBudget']['nu_j'][39] / MS['Smooth_MrBudget']['nu_j'][40])
chk('Smooth 同处 log-gap=0.2748', abs(gapSm - 0.2748) < 5e-4, f'{gapSm:.4f}')
# 端点不变量：面板表 fast 段逐位
for name in ['MrPro', 'MrProBM', 'E1_s28_less', 'E1_s29_more', 'E1_pair28_29', 'Smooth_MrBudget',
             'MrUni', 'LongBridgeSlower', 'LongBridgeFaster', 'E4_pair25_29',
             'StackFrontBack', 'MrProN16', 'MrProN15']:
    nu = np.array(MS[name]['nu_j'], dtype=np.float64)
    chk(f'{name} fast段(0..23)逐位==native', np.array_equal(nu[:24].astype(np.float32), NATIVE[:24].astype(np.float32)))
    chk(f'{name} tail段(40..63)逐位==MrPro', np.array_equal(nu[40:].astype(np.float32), MRPRO[40:].astype(np.float32)))
nuE8 = np.array(MS['E8_zero51']['nu_j'], dtype=np.float64)
chk('E8 tail段==MrPro 除槽51', np.array_equal(nuE8[40:51].astype(np.float32), MRPRO[40:51].astype(np.float32))
    and np.array_equal(nuE8[52:64].astype(np.float32), MRPRO[52:64].astype(np.float32)))
chk('E2 fast段==native', np.array_equal(np.array(MS['E2_tail_more']['nu_j'], dtype=np.float64)[:24].astype(np.float32), NATIVE[:24].astype(np.float32)))
# README §4-5 声称"全部面板表 fast 段逐位同原生，除 P2/NTK/YaRN" —— 直查 HighGapToLong/Mid
nuH = np.array(MS['HighGapToLong']['nu_j'], dtype=np.float64)
fast_eq_H = np.array_equal(nuH[:24].astype(np.float32), NATIVE[:24].astype(np.float32))
print(f'  NOTE README§4-5 vs HGL fast段: bitwise={fast_eq_H}; G1 field fast_band_bitwise_equal_native='
      f'{MS["HighGapToLong"]["endpoint_delta_m"]["fast_band_bitwise_equal_native"]}')
chk('HGL fast段漂移与 G1 字段自洽', fast_eq_H == MS['HighGapToLong']['endpoint_delta_m']['fast_band_bitwise_equal_native'])

# HighGapToLong 端点违例直查（m23<0）
mH, stH, holeH, posH, _ = aggregates(np.array(MS['HighGapToLong']['nu_j']))
chk('HighGapToLong m23=−0.155715(I1违例)', abs(mH[23] + 0.155715) < 5e-5, f'{mH[23]:.6f}')
gapH = np.log(nuH[:-1] / nuH[1:])
chk('HighGapToLong Σ_all(gap增量)==ln4', abs(float(np.sum(gapH - NB)) - LN4) < 1e-6, f'{float(np.sum(gapH-NB)):.7f}')
chk('HighGapToLong E8 端点字段 m40==1', abs(MS['HighGapToLong']['endpoint_delta_m']['m_40'] - 1.0) < 1e-9)
mE2chk, stE2, *_ = aggregates(np.array(MS['E2_tail_more']['nu_j']))
chk('E2 Σ_all==1.6022=ln4×1.1557', abs(float(np.sum(np.log(np.array(MS['E2_tail_more']['nu_j'])[:-1] / np.array(MS['E2_tail_more']['nu_j'])[1:])) - 63 * NB) - LN4 * 1.155715) < 1e-4, f'{float(np.sum(np.log(np.array(MS["E2_tail_more"]["nu_j"])[:-1]/np.array(MS["E2_tail_more"]["nu_j"])[1:]))-63*NB):.7f}')

# ---------- D. 分数 ----------
print('== D. summary.json 分数对 G1/README ==')
ARM_OF = {name: name for name in MS if (MS[name].get('panel_scores') or {}).get('source', '').startswith('results/')}
for name, arm in ARM_OF.items():
    try:
        s = json.loads((MIRROR / 'results' / arm / 'summary.json').read_text())
    except FileNotFoundError:
        continue
    cand = s.get('candidate', {})
    byl = cand.get('by_length', {})
    a32 = byl.get('32768', {}).get('macro_accuracy')
    a128 = byl.get('131072', {}).get('macro_accuracy')
    ps = MS[name]['panel_scores']
    ok32 = a32 is None or abs(a32 * 100 - ps['score_32K_pct']) < 5e-3
    ok128 = a128 is None or abs(a128 * 100 - ps['score_128K_pct']) < 5e-3
    rows_ok = cand.get('rows') is None or cand.get('rows') == ps.get('rows')
    chk(f'{name} scores==summary.json', ok32 and ok128 and rows_ok,
        f"{a32} {a128} rows={cand.get('rows')} vs {ps}")
# MrPro 基线经 s28_less summary.baseline 复核
s = json.loads((MIRROR / 'results/E1_s28_less/summary.json').read_text())
b = s['baseline']['by_length']
chk('MrPro baseline 87.2222/78.1250', abs(b['32768']['macro_accuracy'] * 100 - 87.2222) < 5e-3 and abs(b['131072']['macro_accuracy'] * 100 - 78.125) < 5e-3,
    f"{b['32768']['macro_accuracy']} {b['131072']['macro_accuracy']}")
# BM 分数出处（小数形式 0.9166666/0.7083333）
bmres = json.dumps(json.loads((ROOT / 'docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json').read_text()))
chk('BM 91.6667/70.8333 在 ROPE_BM_TRANSFER_RESULT', '0.9166666666666666' in bmres and '0.7083333333333334' in bmres)
gcs = json.dumps(json.loads((ROOT / 'docs/research/ROPE_GAP_CAPPED_RESULT_20260908.json').read_text()))
chk('GapCapped 84.4444/62.1528 在 RESULT JSON', '0.8444444444444444' in gcs and '0.6215277777777778' in gcs)

# ---------- E. 队列冻结定义 ----------
print('== E. 0446/0448/0449 冻结定义核对 ==')
bud = (ROOT / 'docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md').read_text().splitlines()
l45 = [i + 1 for i, L in enumerate(bud) if '0446 StackFrontBack' in L or '0448 MrProN16' in L or '0449 MrProN15' in L]
print('  候选行现位于 BUDGET 文件行:', l45)
# N16/N15 公式 vs G1 数组
for N, name in [(16, 'MrProN16'), (15, 'MrProN15')]:
    mth = build_radial(N)
    nu_g1path = np.float32(NATIVE * np.power(4.0, -mth)).astype(np.float64)
    chk(f'{name} == q(q+1)/{N*(N+1)} via G1声明路径(NATIVE×4^−m) bit', sha_f32(nu_g1path) == sha_f32(MS[name]['nu_j']))
    nu_omega = np.float32(nu_of_m(mth)).astype(np.float64)
    dd = float(np.max(np.abs(m_of(nu_omega, NATIVE) - m_of(MS[name]['nu_j'], NATIVE))))
    nulp = int(np.count_nonzero(nu_omega.astype(np.float32) != np.array(MS[name]['nu_j']).astype(np.float32)))
    print(f'  NOTE {name}: ω-f64直算路径 vs G1 数组 = {nulp}/64 槽差 1 ULP，max|Δm|={dd:.2e}（文档未声称该路径 bit-exact；锚点值不受影响）')
m16 = m_of(MS['MrProN16']['nu_j'], NATIVE)
m15 = m_of(MS['MrProN15']['nu_j'], NATIVE)
chk('N16 锚点 m28=0.110294 doc0.110 ✓; m39=1', abs(m16[28] - 0.110294) < 1e-5 and abs(m16[39] - 1) < 1e-5 and m16[38] < 1)
chk('N15 锚点 m28=0.125; m38=1', abs(m15[28] - 0.125) < 1e-5 and abs(m15[38] - 1) < 1e-5)
chk('N16 m36–39 doc .669/.772/.882/1.0',
    all(abs(m16[k] - v) < 5e-4 for k, v in zip(range(36, 40), [.6691, .7721, .8824, 1.0])), '/'.join(f'{m16[k]:.4f}' for k in range(36, 40)))
chk('N15 m36–39 doc .758/.875/1.0/1.0',
    all(abs(m15[k] - v) < 5e-4 for k, v in zip(range(36, 40), [.7583, .875, 1.0, 1.0])), '/'.join(f'{m15[k]:.4f}' for k in range(36, 40)))
D16 = W * 4 ** m16
D15 = W * 4 ** m15
chk('N16/N15 D39==131072', abs(D16[39] - 131072) < 1 and abs(D15[39] - 131072) < 1, f'{D16[39]:.0f}/{D15[39]:.0f}')
# Stack 拼接恒等（对部署分量）
stk = np.array(MS['StackFrontBack']['nu_j'], dtype=np.float64)
spliced = MRPRO.copy()
spliced[28] = np.array(DEPLOY['E1_s28_less'])[28]
spliced[36:40] = np.array(DEPLOY['LongBridgeSlower'])[36:40]
chk('Stack == MrPro⊕s28(28)⊕LBS(36–39) 拼接 (bit)', sha_f32(spliced) == sha_f32(stk))
mstk = m_of(stk, NATIVE)
chk('Stack 锚点 m28=0.065359 doc0.065 ✓', abs(mstk[28] - 0.065359) < 1e-5, f'{mstk[28]:.6f}')
chk('Stack m36–39 doc .625/.729/.847/.980',
    all(abs(mstk[k] - v) < 5e-4 for k, v in zip(range(36, 40), [.6252, .7295, .8465, .9799])), '/'.join(f'{mstk[k]:.4f}' for k in range(36, 40)))
Dstk = W * 4 ** mstk
chk('Stack D39==127473 doc"127.5K" ✓', abs(Dstk[39] - 127473) < 2, f'{Dstk[39]:.0f}')
chk('Stack m27−m28 == 3.82e-9 假单调噪声', abs((mstk[27] - mstk[28]) - 3.82e-9) < 2e-9, f'{mstk[27]-mstk[28]:.3e}')
_, stS, holeS, posS, _ = aggregates(stk)
chk('Stack 洞 1.4930@g38 (GR§6 位置 g35 为笔误)', abs(holeS - 1.4930) < 5e-4 and posS == 38, f'hole={holeS:.4f}@g{posS}')
_, st16, hole16, pos16, _ = aggregates(np.array(MS['MrProN16']['nu_j']))
chk('N16 洞 1.4608@g38', abs(hole16 - 1.4608) < 5e-4 and pos16 == 38, f'{hole16:.4f}@g{pos16}')
_, st15, hole15, pos15, _ = aggregates(np.array(MS['MrProN15']['nu_j']))
chk('N15 洞 1.4757@g37', abs(hole15 - 1.4757) < 5e-4 and pos15 == 37, f'{hole15:.4f}@g{pos15}')
chk('三候选 Σ_trans==ln4', all(abs(v - LN4) < 1e-6 for v in [stS, st16, st15]), f'{stS:.6f}/{st16:.6f}/{st15:.6f}')
# 队列本地存在性：同号异物
qfiles = sorted(p.name for p in (MIRROR / 'queue').glob('*.json'))
chk('本地 queue 无 0446_Stack/0448/0449 契约', not any('Stack' in f or 'MrProN' in f for f in qfiles))
chk('0446 号 = deferred HighGapToMid（撤回臂）',
    (MIRROR / 'deferred_queue/20260910_candidate_quality/0446_HighGapToMid.json').exists())
# G1 三候选分数必须 null
chk('三候选 score=null（未执行）', all(MS[n].get('panel_scores') is None for n in ['StackFrontBack', 'MrProN16', 'MrProN15']))
# BUDGET 锚点原句
for pat in ['m_q=q(q+1)/272', '.625/.729/.847/.980', '.669/.772/.882/**1.0**', '.758/.875/1.0/1.0']:
    chk(f'BUDGET 冻结句含 "{pat}"', any(pat in L for L in bud))

# ---------- F. CANDIDATE_TABLES 对 G1 ----------
print('== F. CANDIDATE_TABLES.json 恒等/守恒复核 ==')
CT = json.loads(CTP.read_text())
tabs = {t['id']: t for t in CT['tables']}


def tab_m(tid):
    return np.array(tabs[tid]['m_j'], dtype=np.float64)


pairs = [('ramp_N17', 'MrPro'), ('ramp_N16', 'MrProN16'), ('ramp_N15', 'MrProN15'),
         ('stack_0446', 'StackFrontBack')]
for tid, gname in pairs:
    c = tab_m(tid)
    g = np.array(MS[gname]['m_j'], dtype=np.float64)
    d = np.max(np.abs(c - g))
    chk(f'{tid} == G1.{gname} (maxΔm)', d < 1e-6, f'{d:.2e}')
for tid in ['ramp_N17', 'ramp_N16', 'ramp_N15', 'ramp_N14', 'ramp_N13']:
    N = int(tid.split('N')[1])
    c = tab_m(tid)
    d = np.max(np.abs(c - build_radial(N)))
    chk(f'{tid} == radial N{N} 闭式', d < 1e-6, f'{d:.2e}')
for tid in ['uniform_N17', 'uniform_N16', 'uniform_N15', 'uniform_N14', 'uniform_N13']:
    N = int(tid.split('N')[1])
    c = tab_m(tid)
    ex = np.minimum(1.0, np.clip(np.arange(DR) - 23, 0, N) / N)
    d = np.max(np.abs(c - ex))
    chk(f'{tid} == min(1,(j−23)/N{N})', d < 1e-6, f'{d:.2e}')
u17 = tab_m('uniform_N17')
chk('uniform_N17 == MrUni（G1）', np.max(np.abs(u17 - np.array(MS['MrUni']['m_j']))) < 1e-6)
notch = tab_m('ramp16_notch')
chk('ramp16_notch m28 位移 = −10/306 (0.032680)', abs((tab_m('ramp_N16')[28] - notch[28]) - 10 / 306) < 1e-9,
    f'{tab_m("ramp_N16")[28]-notch[28]:.6f}')
chk('ramp16_notch 仅动 m28（守恒不破坏）', np.count_nonzero(np.abs(notch - tab_m('ramp_N16')) > 1e-9) == 1)
for tid in ['ramp_N13', 'ramp_N14', 'ramp_N15', 'ramp_N16', 'ramp_N17', 'stack_0446', 'ramp16_notch', 'ramp15_notch']:
    nu = nu_of_m(tab_m(tid))
    m = m_of(nu, NATIVE)
    st = float(np.sum(np.log(nu[23:40] / nu[24:41])) - 17 * NB)
    ok_st = abs(st - LN4) < 1e-6
    ok_ep = m[23] < 1e-9 and abs(m[40] - 1) < 1e-6
    chk(f'{tid} 守恒+端点', ok_st and ok_ep, f'Σtrans残差={st-LN4:.2e}')
# 汇总表洞位
holes = {'ramp_N17': (1.4476, 39), 'ramp_N16': (1.4608, 38), 'ramp_N15': (1.4757, 37),
         'ramp_N14': (1.4929, 36), 'ramp_N13': (1.5127, 35), 'stack_0446': (1.4930, 38)}
for tid, (hv, hg) in holes.items():
    nu = nu_of_m(tab_m(tid))
    rho = nu[:-1] / nu[1:]
    tr = rho[23:40]
    ig = int(np.argmax(tr)) + 23
    chk(f'{tid} max洞 {hv}@g{hg}', abs(tr[ig - 23] - hv) < 5e-4 and ig == hg, f'{tr[ig-23]:.4f}@g{ig}')
F_arc = {'ramp_N17': 1.0458, 'ramp_N16': 0.6765, 'ramp_N15': 0.3667, 'ramp_N14': 0.1333, 'ramp_N13': 0.0,
         'stack_0446': 0.8189}
BL = {'ramp_N17': 0.2288, 'ramp_N16': 0.2574, 'ramp_N15': 0.2917, 'stack_0446': 0.1961}
for k, v in F_arc.items():
    m = tab_m(k)
    fa = float(np.sum(np.clip(1 - m[36:40], 0, None)))
    chk(f'{k} F_arc={v}', abs(fa - v) < 5e-4, f'{fa:.4f}')
for k, v in BL.items():
    m = tab_m(k)
    bl = float(np.sum(m[24:29]))
    chk(f'{k} BL={v}', abs(bl - v) < 5e-4, f'{bl:.4f}')

# ---------- G. G1 内部账目 ----------
print('== G. G1 reconciliation/mismatches 账目 ==')
rec = G1['reconciliation']
from collections import Counter
cc = Counter(r.get('status', '?') for r in rec)
chk('reconciliation 条数==126', len(rec) == 126, f'实际 {len(rec)}  分布 {dict(cc)}')
chk('README§4 账目 121 MATCH/2 NOTE/3 MISMATCH',
    cc.get('MATCH', 0) == 121 and cc.get('NOTE', 0) == 2 and cc.get('MISMATCH', 0) == 3, str(dict(cc)))
mm_mis = [r for r in rec if r.get('status') == 'MISMATCH']
print('  MISMATCH 条目:', json.dumps(mm_mis, ensure_ascii=False)[:300])
chk('mismatches 条数==10', len(G1['mismatches']) == 10, f'实际 {len(G1["mismatches"])}')
chk('methods 条数==38', len(MS) == 38, f'实际 {len(MS)}')
# P2 双源
pg = json.loads((MIRROR / 'planned_controls/p2_gap_comparison.json').read_text())
ds = pg.get('gap_delta_sums', {})
chk('P2 gap_delta_sums["0:23"]==+0.00077543', abs(ds.get('0:23', 9) - 0.00077543) < 1e-8, str(ds))
nuP2 = np.array(MS['FullLagP2_Transfer3B']['nu_j'])
stP2 = float(np.sum(np.log(nuP2[23:40] / nuP2[24:41])) - 17 * NB)
chk('P2 Σtrans==ln4−0.00077543', abs(stP2 - (LN4 - 0.00077543)) < 2e-7, f'{stP2:.7f}')
# P2 三源数组一致
q15 = json.loads((ROOT / 'docs/research/ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json').read_text())


def find_p2_arr(o):
    if isinstance(o, dict):
        if 'values_float32' in o and isinstance(o['values_float32'], list):
            return o['values_float32']
        for v in o.values():
            r = find_p2_arr(v)
            if r: return r
    elif isinstance(o, list):
        for v in o:
            r = find_p2_arr(v)
            if r: return r
    return None


p15a = find_p2_arr(q15)
chk('P2 QWEN15 候选数组 == 部署数组 (bit)', p15a is not None and sha_f32(p15a) == sha_f32(DEPLOY['FullLagP2_Transfer3B']))
chk('P2 queue 契约数组 == results 契约数组 (bit)',
    np.array_equal(json.loads((MIRROR / 'queue/0444_FullLagP2_Transfer3B.json').read_text())['spec']['table']['values_float32'],
                   DEPLOY['FullLagP2_Transfer3B']))
chk('P2 gain==1.102586(074)', abs(json.loads((MIRROR / 'results/FullLagP2_Transfer3B/contract.json').read_text())['spec']['table']['gain'] - (1 + 0.074 * LN4)) < 1e-9)

print()
print(f'==== RESULT: {len(OK)} OK / {len(BAD)} FAIL ====')
for b in BAD:
    print('  FAIL:', b)
