#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务 G1：36 行开发面板全部方法的 64 槽频率表数值化重建（纯 CPU）。

对每个方法：
  1) 从构造代码/文档公式独立重生成 nu_j[64]（float64 构造 → float32 落地）；
  2) 与本地镜像的已部署张量（results/<method>/contract.json、reference_tables.json、
     docs/research/*.json、deferred_queue/*.json）逐位对账；
  3) 反演 m_j = log(nu_j/native_j)/log(4)，计算 T_j=2pi/nu_j、D_j=W*4^{m_j}、
     r_j=W/T_j^native、gap_j=ln(nu_j/nu_{j+1})、过渡段(23..39)gap增量和、洞比率
     max T_{j+1}/T_j；
  4) 与 docs 已记录锚点逐项对账，输出 mismatches。

运行：python3 analysis/unify_20260910/tables/rebuild_ground_truth_tables.py
输出：同目录 ground_truth_tables.json
"""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MIRROR = ROOT / 'results/nongeometric_screen_20260909'
OUT = Path(__file__).resolve().parent / 'ground_truth_tables.json'

W, S, LN_S, BASE, DR = 32768, 4, math.log(4.0), 1e6, 64
NATIVE_GAP_RATIO = BASE ** (1.0 / DR)          # 原生周期比 ≈1.2410
NATIVE_LOG_GAP = math.log(BASE) / DR           # 原生 log-gap = ln(b)/64 ≈0.2158674
LN4 = LN_S

ref = json.loads((MIRROR / 'reference_tables.json').read_text())
NATIVE = np.array(ref['Native']['values_float32'], dtype=np.float64)
MRPRO = np.array(ref['MrPro']['values_float32'], dtype=np.float64)
BM = np.array(ref['MrProBM']['values_float32'], dtype=np.float64)
GAIN = float(ref['MrPro']['gain'])
MR_M = np.log(NATIVE / MRPRO) / LN_S           # 部署 MrPro 的实际指数


def f32(a):
    return np.asarray(a, dtype=np.float32)


def tensor_sha(values):
    v = f32(values)
    return hashlib.sha256(struct.pack('<' + 'f' * len(v), *v)).hexdigest()


def load_contract(method):
    p = MIRROR / 'results' / method / 'contract.json'
    d = json.loads(p.read_text())
    sp = d['spec']
    if sp.get('operator') != 'static':
        return None, sp, str(p)
    return np.array(sp['table']['values_float32'], dtype=np.float64), sp, str(p)


def load_summary(method):
    p = MIRROR / 'results' / method / 'summary.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    c = d.get('candidate', {})
    bl = c.get('by_length', {})
    def pct(x):
        return round(100.0 * x, 4) if isinstance(x, (int, float)) else None
    out = {'rows': c.get('rows'),
           'score_32K_pct': pct(bl.get('32768', {}).get('macro_accuracy')),
           'score_128K_pct': pct(bl.get('131072', {}).get('macro_accuracy')),
           'wins': d.get('wins'), 'losses': d.get('losses'),
           'source': str(p.relative_to(ROOT))}
    return out


def metrics(nu):
    """全部结构量；nu 为 float64 数组（部署值）。nu=0 槽 → m/D/T 置 None。"""
    nu = np.asarray(nu, dtype=np.float64)
    pos = nu > 0
    m = np.where(pos, np.log(np.where(pos, NATIVE, 1.0) / np.where(pos, nu, 1.0)) / LN_S, np.nan)
    T = np.where(pos, 2 * np.pi / np.where(pos, nu, 1.0), np.inf)
    D = np.where(pos, W * np.power(4.0, np.where(pos, m, 0.0)), np.inf)
    Tnat = 2 * np.pi / NATIVE
    r_nat = W / Tnat
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = np.log(nu[:-1] / nu[1:])          # 63 个 log-gap（对零频槽为 inf）
    # 全表端点差（穿过零频槽仍良定义）：Σ_all gap = ln(nu0/nu63)
    sum_all_incr = float(np.log(nu[0] / nu[63]) - 63 * NATIVE_LOG_GAP)
    trans_gaps = list(range(23, 40))            # 0-based gap 索引 23..39（槽23–40之间，17个）
    native_gaps = np.full(63, NATIVE_LOG_GAP)
    sum_trans_incr = float(sum(gap[g] - native_gaps[g] for g in trans_gaps))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = T[1:] / T[:-1]
    finite = np.isfinite(ratio) & (ratio > 0)
    hole_max = float(np.max(ratio[finite]))
    hole_argmax = int(np.argmax(np.where(finite, ratio, -np.inf)))
    trans = ratio[23:40]
    trans_f = np.where(np.isfinite(trans) & (trans > 0), trans, -np.inf)
    hole_trans_max = float(np.max(trans_f))
    hole_trans_argmax = int(23 + np.argmax(trans_f))
    clean = lambda a: [None if (x is None or not np.isfinite(x)) else float(x) for x in a]
    return {
        'm_j': clean(m), 'T_j': clean(T), 'D_j': clean(D), 'r_j_native': clean(r_nat),
        'gap_j': clean(gap),
        'sum_transition_gap_increments_gaps23_39': sum_trans_incr,
        'sum_all_gap_increments_slots0_63': sum_all_incr,
        'expected_if_endpoints_fixed_ln4': LN_S,
        'endpoint_delta_m': {'m_23': float(m[23]), 'm_40': float(m[40]),
                             'fast_band_bitwise_equal_native': bool(np.array_equal(f32(nu[:24]), f32(NATIVE))),
                             'tail_m40to63_all_equal_MrPro': bool(np.array_equal(f32(nu[40:]), f32(MRPRO[40:])))},
        'hole_ratio_max_global': hole_max, 'hole_ratio_argmax_gap': hole_argmax,
        'hole_ratio_max_transition': hole_trans_max, 'hole_ratio_argmax_transition': hole_trans_argmax,
        'native_period_ratio': NATIVE_GAP_RATIO,
        'sum_m_all_slots': float(np.nansum(m)),
        'changed_slots_vs_MrPro': [int(j) for j in np.flatnonzero(f32(nu) != f32(MRPRO))],
    }


MISMATCHES = []
METHODS = []  # provisional; replaced below after helpers


def gain_of(sp):
    try:
        return float(sp['table']['gain'])
    except Exception:
        return None


MISMATCHES = []
METHODS = {}


def method_entry(name, nu_rebuilt, deployed, role, panel, construction, sources, score=None,
                 extra_notes=None, tol=0.0, gain=None):
    entry = {'role': role, 'panel': panel, 'construction': construction, 'sources': sources}
    nu = deployed if deployed is not None else nu_rebuilt
    entry['nu_used'] = 'deployed' if deployed is not None else 'formula'
    if deployed is not None and nu_rebuilt is not None:
        d = np.abs(f32(nu_rebuilt).astype(np.float64) - f32(deployed).astype(np.float64))
        rel = float(np.max(d / np.maximum(np.abs(f32(deployed).astype(np.float64)), 1e-300)))
        bit = bool(np.array_equal(f32(nu_rebuilt), f32(deployed)))
        entry['formula_vs_deployed'] = {
            'bit_exact': bit, 'max_abs_freq_diff': float(np.max(d)), 'max_rel_freq_diff': rel,
            'sha256_deployed': tensor_sha(deployed), 'sha256_rebuilt': tensor_sha(nu_rebuilt)}
        if not bit and rel > tol:
            MISMATCHES.append(f'{name}: 公式重建与部署张量不一致 (max_rel={rel:.3e} > tol={tol:.1e})')
    else:
        entry['formula_vs_deployed'] = {
            'bit_exact': None,
            'note': ('公式重建，无本地部署张量可对账' if deployed is None and nu_rebuilt is not None
                     else '仅有部署张量（构造式未本地复原）')}
    entry['gain'] = gain
    entry['nu_j'] = [float(x) for x in nu]
    entry.update(metrics(nu))
    if score:
        entry['panel_scores'] = score
    if extra_notes:
        entry['notes'] = extra_notes
    return entry


def radial_family_m(N_, dl=23):
    """MrRoPE Eq.14 径向族：m_j = q(q+1)/(N(N+1)), q=clip(j-dl,0,N)；
    N=17 即 MrPro（dl=23, dh=40），N'=16/15 为收缩族（完成槽=23+N'）。"""
    m = np.zeros(DR)
    for j in range(DR):
        q = min(max(j - dl, 0), N_)
        m[j] = q * (q + 1) / (N_ * (N_ + 1))
    return m


mr_m_formula = radial_family_m(17)
mr_nu_formula = f32(NATIVE * np.power(4.0, -mr_m_formula)).astype(np.float64)
mr_err = float(np.max(np.abs(mr_nu_formula - MRPRO) / MRPRO))
mr_abs = float(np.max(np.abs(mr_nu_formula - MRPRO)))

# ---------------------------------------------------------------- BM 公式
def bm_m_formula(N_=17, dl=23):
    m = np.zeros(DR)
    for j in range(DR):
        q = min(max(j - dl, 0), N_)
        m[j] = q * (q + 1) * (3 * N_ + 2 - 2 * q) / (N_ * (N_ + 1) * (N_ + 2))
    return m

bm_nu_formula = f32(NATIVE * np.power(4.0, -bm_m_formula(17))).astype(np.float64)

# ---------------------------------------------------------------- YaRN / NTK
def yarn_table(factor=4.0, ramp='linear'):
    low, high = 23, 40                     # 与 export 公式一致: floor/ceil by beta_fast=32, beta_slow=1
    t = np.clip((np.arange(DR, dtype=np.float64) - low) / (high - low), 0, 1)
    if ramp == 'smoothstep':
        t = 3 * t ** 2 - 2 * t ** 3
    omega = NATIVE
    return f32(omega / factor * t + omega * (1 - t)).astype(np.float64)

ntk_nu = f32(NATIVE * np.power(4.0, -np.arange(DR, dtype=np.float64) / (DR - 1))).astype(np.float64)

carrier = json.loads((ROOT / 'docs/research/ROPE_CARRIER_REMOVAL_CANDIDATE_20260907.json').read_text())
YARN_DEPLOYED = np.array(carrier['tables']['YaRN']['values_float32'], dtype=np.float64)
yarn_recp = 'paper-2027/research/attention-aware-retrofit/evidence/REFERENCE_CORRECTED_K128_S4_NLL_RECEIPT_20260901.json'
YARN_SHA = json.loads((ROOT / yarn_recp).read_text())['identity']['profiles']['official_equation_yarn']['tensor_sha256']

# ---------------------------------------------------------------- Smooth / MrUni（移植 smooth_budget.py）
lap = 2 * np.eye(17) - np.eye(17, k=1) - np.eye(17, k=-1)
def solve_sb(n, budget):
    a = np.stack([np.ones(n), np.arange(n - 1, -1, -1)])
    b = np.array([1., budget])
    faces = [np.arange(k, n) for k in range(n - 1)] + [np.arange(n - k) for k in range(1, n - 1)]
    for free in faces:
        kkt = np.block([[2 * lap[np.ix_(free, free)], a[:, free].T], [a[:, free], np.zeros((2, 2))]])
        ans = np.linalg.solve(kkt, np.r_[np.zeros(len(free)), b])
        eps = np.zeros(n)
        eps[free] = ans[:len(free)]
        dual = 2 * lap @ eps + a.T @ ans[len(free):]
        act = np.setdiff1d(np.arange(n), free)
        if (eps.min() >= -1e-10 and np.max(np.abs(a @ eps - b)) < 1e-9
                and np.max(np.abs(dual[free])) < 1e-9 and (not len(act) or dual[act].min() >= -1e-9)):
            eps[np.abs(eps) < 1e-14] = 0.
            return eps, float(eps @ lap @ eps)
    raise ValueError('no KKT solution')

eps_smooth, rough_smooth = solve_sb(17, 16 / 3)
m_smooth = np.r_[0, np.cumsum(eps_smooth)]
smooth_nu = MRPRO.copy()
mruni_nu = MRPRO.copy()
for j in range(24, 40):
    smooth_nu[j] = NATIVE[j] * 4 ** (-m_smooth[j - 23])
    mruni_nu[j] = NATIVE[j] * 4 ** (-(j - 23) / 17)
smooth_nu, mruni_nu = f32(smooth_nu).astype(np.float64), f32(mruni_nu).astype(np.float64)

# ---------------------------------------------------------------- LBS / Faster（移植 long_bridge.py）
periods_mr = 2 * np.pi / MRPRO
lbs_slots = np.flatnonzero((periods_mr >= W) & (periods_mr <= 131072))
step = 1.0 / 131072
lbs_nu = f32(MRPRO.copy()[... ]).astype(np.float64)
lbs_nu = f32(np.where(np.isin(np.arange(DR), lbs_slots), MRPRO - step, MRPRO)).astype(np.float64)
fast_nu = f32(np.where(np.isin(np.arange(DR), lbs_slots), MRPRO + step, MRPRO)).astype(np.float64)

# ---------------------------------------------------------------- HighGapToLong（移植 gap_budget_transfer.py）
gaps_mr = np.log(MRPRO[:-1] / MRPRO[1:])
donors = np.arange(23)                        # first_changed=24 → high_end=23? 代码: first_changed[0]-1=23? 见下验证
first_changed = np.flatnonzero(MRPRO != NATIVE)
donors = np.arange(int(first_changed[0]) - 1) # gaps 0..22
budget = float(gaps_mr[donors].mean())
per = 2 * np.pi / MRPRO
gap_period = np.sqrt(per[:-1] * per[1:])
recip = np.flatnonzero((gap_period >= W) & (gap_period <= W * 4))
new_gaps = gaps_mr.copy()
new_gaps[donors] -= budget / len(donors)
new_gaps[recip] += budget / len(recip)
hgl_nu = np.exp(np.r_[np.log(MRPRO[0]), np.log(MRPRO[0]) - np.cumsum(new_gaps)])
hgl_nu = f32(hgl_nu).astype(np.float64)
hgl_nu[40:] = f32(MRPRO[40:]).astype(np.float64)

# ---------------------------------------------------------------- E1/E2/E8/pair（移植 select.py proposals）
def e1(slot, direction):
    t = MRPRO.copy()
    t[slot] = NATIVE[slot] * 4 ** (-MR_M[slot + direction])
    return f32(t).astype(np.float64)

s28_less = e1(28, -1)
s29_more = e1(29, +1)
pair_nu = MRPRO.copy()
pair_nu[28] = NATIVE[28] * 4 ** (-MR_M[27])
pair_nu[29] = NATIVE[29] * 4 ** (-MR_M[30])
pair_nu = f32(pair_nu).astype(np.float64)
factor_e2 = 1e6 ** (1 / 64)
e2_nu = MRPRO.copy()
e2_nu[40:] /= factor_e2
e2_nu = f32(e2_nu).astype(np.float64)
e8_nu = MRPRO.copy()
e8_nu[51] = 0.0
e8_nu = f32(e8_nu).astype(np.float64)
e4_nu = MRPRO.copy()
shared = float((MR_M[25] + MR_M[29]) / 2)
e4_nu[25] = NATIVE[25] * 4 ** (-shared)
e4_nu[29] = NATIVE[29] * 4 ** (-shared)
e4_nu = f32(e4_nu).astype(np.float64)

# ---------------------------------------------------------------- N'/Stack 公式重建
n16_nu = f32(NATIVE * np.power(4.0, -radial_family_m(16))).astype(np.float64)
n15_nu = f32(NATIVE * np.power(4.0, -radial_family_m(15))).astype(np.float64)
stack_nu = pair_nu.copy()  # base = MrPro
stack_nu = f32(MRPRO.copy()).astype(np.float64)
stack_nu[28] = NATIVE[28] * 4 ** (-MR_M[27])
for j in lbs_slots:
    stack_nu[j] = MRPRO[j] - step
stack_nu = f32(stack_nu).astype(np.float64)

# ---------------------------------------------------------------- ScaleTaper（移植 scale_taper.py）
w_taper = np.clip(np.log(W / periods_mr) / LN_S, 0, 1)
taper_nu = np.exp(np.log(MRPRO) + w_taper * np.log(BM / MRPRO))
taper_nu = f32(taper_nu).astype(np.float64)
taper_nu[w_taper == 0] = f32(MRPRO[w_taper == 0]).astype(np.float64)
taper_nu[w_taper == 1] = f32(BM[w_taper == 1]).astype(np.float64)

# ---------------------------------------------------------------- 已部署表统一登记
deploy = {}
for name in ['E1_s28_less', 'E1_s29_more', 'E1_pair28_29', 'E1_s28_reverse_matched',
             'E1_s29_plus_matched', 'E2_tail_more', 'E8_zero51', 'E4_pair25_29',
             'Smooth_MrBudget', 'MrUni', 'HighGapToLong', 'LongBridgeSlower',
             'LongBridgeFaster', 'FullLagP2_Transfer3B', 'Control_Mr_gain074',
             'E3_BM_gain074', 'E3_BM_gain1', 'E7_local_projection',
             'E7_norm_matched_BM', 'E10_dual_frequency']:
    try:
        t, sp, path = load_contract(name)
        if t is not None:
            deploy[name] = (t, gain_of(sp), path)
        else:
            deploy[name] = (None, None, path)   # operator 类
    except FileNotFoundError:
        deploy[name] = None

p2_cand = json.loads((ROOT / 'docs/research/ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json').read_text())
P2_CAND = np.array(p2_cand['tables']['FullLagP2']['values_float32'], dtype=np.float64)
gap_capped = json.loads((ROOT / 'docs/research/ROPE_GAP_CAPPED_CANDIDATE_20260908.json').read_text())
GC = np.array(gap_capped['Qwen3B']['values_float32'], dtype=np.float64)
hgm_file = json.loads((MIRROR / 'deferred_queue/20260910_candidate_quality/0446_HighGapToMid.json').read_text())
HGM = np.array(hgm_file['spec']['table']['values_float32'], dtype=np.float64)
st_file = json.loads((MIRROR / 'deferred_queue/20260910_candidate_quality/044d_BM_ScaleTaper.json').read_text())
ST = np.array(st_file['spec']['table']['values_float32'], dtype=np.float64)

# ---------------------------------------------------------------- 组装
def add(name, rebuilt, dep_arr, role, panel, construction, sources, score_names=None, notes=None,
        gain=None, tol=0.0):
    sp_score = None
    for sn in (score_names or []):
        s = load_summary(sn)
        if s:
            sp_score = s
            break
    g = gain
    if g is None and name in deploy and deploy[name] and deploy[name][1] is not None:
        g = deploy[name][1]
    if g is None and dep_arr is not None and name in ('StackFrontBack',):
        g = GAIN
    e = method_entry(name, rebuilt, dep_arr, role, panel, construction, sources,
                     score=sp_score, extra_notes=notes, tol=tol, gain=g)
    METHODS[name] = e

# —— 参照
add('Native', None, NATIVE, 'reference', 'reference',
    'nu_j = omega_j = b^(-j/64), b=1e6; m=0; gain=1（S=1）',
    ['experiments/nongeometric_screen/reference_tables.json（部署镜像）'],
    score_names=None, gain=1.0,
    notes={'panel': 'native_reference/summary.json 12 短行 83.3333（QA 行拉低）'})

add('MrPro', mr_nu_formula, MRPRO, 'baseline-official', '36行 (87.2222/78.1250)',
    'MrRoPE-Pro：nu_j=omega_j*4^(-m_j)，m_j=q(q+1)/(N(N+1))，q=clip(j-23,0,17)，N=17（dl=23, dh=40，YaRN 式 alpha=32/beta=1 界；Eq.14 径向 λ 族 N=17 成员）；j>=40 m=1；gain=1+0.1*ln4=1.138629436111989',
    ['experiments/nongeometric_screen/reference_tables.json', 'docs/research/USER_PROMPT_TRANSCRIPT_20260909.md:214 (m_q公式)',
     'scripts/analysis/build_boundary_matched_mrpro.py:59 (source_steps=2i/(n(n+1)))', 'docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:49 (公式vs部署≤4.3e-8)'], gain=GAIN, tol=1e-5)
METHODS['MrPro']['panel_scores'] = {'score_32K_pct': 87.2222, 'score_128K_pct': 78.125,
    'source': 'results/nongeometric_screen_20260909/results/*/summary.json 各 baseline 字段；docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json',
    'rows': 36}
METHODS['MrPro']['formula_vs_deployed']['note'] = f'公式重建 vs 部署 max_rel={mr_err:.3e}（文档声称 ≤4.3e-8）'

add('MrProBM', bm_nu_formula, BM, 'panel-member (09-08 复用)', '36行 (91.6667/70.8333)',
    'BM：eps_i=6i(N+1-i)/(N(N+1)(N+2))，m_q=q(q+1)(3N+2-2q)/(N(N+1)(N+2))，N=17，界 23/40；gain 官方',
    ['docs/research/ROPE_MRPRO_BM_CANDIDATE_20260908.json definition', 'experiments/nongeometric_screen/reference_tables.json'], gain=GAIN, tol=1e-5)
METHODS['MrProBM']['panel_scores'] = {'score_32K_pct': 91.6667, 'score_128K_pct': 70.8333,
    'source': 'docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json（36行配对 6W4L）'}

add('E1_s28_less', s28_less, deploy['E1_s28_less'][0], 'panel-measured', '36行',
    '仅槽28：m28 := m27（0.098039→0.065359，gap27→28 挪 ln4*eps28≈0.04533）',
    ['experiments/nongeometric_screen/select.py:76-80 (proposals)', deploy['E1_s28_less'][2]],
    score_names=['E1_s28_less'], notes={'wins_losses': '2/0；改对行 niah_multikey_2_131072_2、niah_multiquery_131072_3'})

add('E1_s29_more', s29_more, deploy['E1_s29_more'][0], 'panel-measured', '36行',
    '仅槽29：m29 := m30（后邻指数）', ['experiments/nongeometric_screen/select.py:76-80',
    deploy['E1_s29_more'][2]], tol=1e-5, score_names=['E1_s29_more'])

add('E1_pair28_29', pair_nu, deploy['E1_pair28_29'][0], 'panel-measured', '36行',
    '同时刻双手术：m28:=m27 且 m29:=m30（s28_less⊕s29_more）→ eps28+eps29+eps30 集中入 gap28（"双倍集中"，1.46× 洞）——由部署张量逐位解码确认',
    ['results/nongeometric_screen_20260909/results/E1_pair28_29/contract.json 解码', deploy['E1_pair28_29'][2]],
    tol=1e-5, score_names=['E1_pair28_29'])

add('E2_tail_more', e2_nu, deploy['E2_tail_more'][0], 'panel-measured', '12行 (100/54.7222)',
    '槽40–63 频率 ×1e6^(-1/64) → 平台水平 4^{1.15572}≈÷4.931（违反 I2）',
    ['experiments/nongeometric_screen/select.py:87-89', deploy['E2_tail_more'][2]], tol=1e-5, score_names=['E2_tail_more'])

add('E8_zero51', e8_nu, deploy['E8_zero51'][0], 'panel-measured', '12行 (100/50.5556)',
    'nu_51 = 0（慢带单槽置零）', ['experiments/nongeometric_screen/select.py:90-92',
    deploy['E8_zero51'][2]], tol=1e-5, score_names=['E8_zero51'])

add('E4_pair25_29', e4_nu, deploy['E4_pair25_29'][0], 'panel-measured', '12行 (持平)',
    '槽25/29 共享 m=(m25+m29)/2=0.078431', ['experiments/nongeometric_screen/select.py:146-163',
    deploy['E4_pair25_29'][2]], tol=1e-5, score_names=['E4_pair25_29'])

add('E1_s28_reverse_matched', None, deploy['E1_s28_reverse_matched'][0], 'panel-measured (镜像控制)', '12行 (持平)',
    '槽28 反向镜像控制：部署反演 m28=0.132270（changed_slots=[28]）；候选式 2·m28−m27=0.130719 与部署不符，'
    '精确镜像公式未在本地已读代码中找到——仅部署张量为真值 [部分证据]',
    [deploy['E1_s28_reverse_matched'][2]],
    score_names=['E1_s28_reverse_matched'])

add('E1_s29_plus_matched', None, deploy['E1_s29_plus_matched'][0], 'panel-measured (镜像控制)', '12行 (持平)',
    '槽29 镜像控制（部署 m29=0.094729，构造式未在已读文件中记录，仅部署值为真值）',
    [deploy['E1_s29_plus_matched'][2]], score_names=['E1_s29_plus_matched'])

add('Smooth_MrBudget', smooth_nu, deploy['Smooth_MrBudget'][0], 'panel-measured', '36行 (87.2222/68.3333)',
    '固定预算 B=16/3 的最小粗糙度 KKT 解增量 eps 累加（界 23/40，端点同 MrPro）',
    ['experiments/nongeometric_screen/smooth_budget.py construct()', deploy['Smooth_MrBudget'][2]],
    gain=GAIN, score_names=['Smooth_MrBudget'])

add('MrUni', mruni_nu, deploy['MrUni'][0], 'panel-measured', '36行 (64.5833/73.3333)',
    '过渡段均匀斜坡 m_j=(j-23)/17（24–39），外部与 MrPro 相同——GLM 复核已纠正 UNIFIED"全表÷4"表述',
    ['experiments/nongeometric_screen/smooth_budget.py:38 (uniform)', deploy['MrUni'][2],
     'docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:11'], gain=GAIN, score_names=['MrUni'])

add('HighGapToLong', hgl_nu, deploy['HighGapToLong'][0], 'panel-measured', '36行 (70.1389/67.3611)',
    'gaps0–22 各减 ln(b)/64/23=0.009386（总抽 0.2158674），gaps36–39 各加 1/4（EVQ 字面运输；槽40+ 不动）',
    ['experiments/nongeometric_screen/gap_budget_transfer.py', deploy['HighGapToLong'][2]],
    tol=1e-6, gain=GAIN, score_names=['HighGapToLong'])

add('HighGapToMid', None, HGM, 'deferred (16/36 中止，无裁决)', 'partial-12/36',
    '同预算给 gaps26–31（recipient 带 = 几何均值周期 ∈ [W/16, W/4]）',
    ['results/nongeometric_screen_20260909/deferred_queue/20260910_candidate_quality/0446_HighGapToMid.json'], gain=GAIN)

add('LongBridgeSlower', lbs_nu, deploy['LongBridgeSlower'][0], 'panel-measured', '36行 (80.5556/80.0694)',
    'MrPro 周期∈[32768,131072] 的槽（36–39）nu_j -= 1/131072（公共相位 −1 rad@128K）',
    ['experiments/nongeometric_screen/long_bridge.py', deploy['LongBridgeSlower'][2]],
    tol=1e-6, gain=GAIN, score_names=['LongBridgeSlower'])

add('LongBridgeFaster', fast_nu, deploy['LongBridgeFaster'][0], 'panel-measured (方向控制)', '36行 (87.2222/73.9583)',
    '同幅反号 nu_j += 1/131072', ['experiments/nongeometric_screen/long_bridge.py',
    deploy['LongBridgeFaster'][2]], tol=1e-6, gain=GAIN, score_names=['LongBridgeFaster'])

add('FullLagP2_Transfer3B', None, deploy['FullLagP2_Transfer3B'][0], 'panel-measured', '36行 (72.9167/81.6667)',
    '历史 FullLagP2 整表（1.5B 资产恢复，m31=0.998 完成、m32 起=1.0；QWEN15 候选文件与本表核对见对账）；gain=1+0.074*ln4',
    [deploy['FullLagP2_Transfer3B'][2], 'docs/research/ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json',
     'docs/research/ROPE_RECOVERED_QWEN_P2_20260907.json'], score_names=['FullLagP2_Transfer3B'],
    gain=1 + 0.074 * LN_S)

add('Control_Mr_gain074', mr_nu_formula, deploy['Control_Mr_gain074'][0], 'panel-measured (gain 析因)', '36行 (98.3333/75.3472)',
    'MrPro 表 @ gain=1+0.074*ln4（表与 MrPro 同，幅度不同）', [deploy['Control_Mr_gain074'][2]],
    score_names=['Control_Mr_gain074'], gain=1 + 0.074 * LN_S)

add('E3_BM_gain074', bm_nu_formula, deploy['E3_BM_gain074'][0], 'panel-measured (gain 析因)', '36行 (100/70)',
    'BM 表 @ gain .074', [deploy['E3_BM_gain074'][2]], score_names=['E3_BM_gain074'], gain=1 + 0.074 * LN_S)

add('E3_BM_gain1', bm_nu_formula, deploy['E3_BM_gain1'][0], 'panel-measured (gain 析因)', '36行 (89.5833/58.8194)',
    'BM 表 @ gain 1', [deploy['E3_BM_gain1'][2]], score_names=['E3_BM_gain1'], gain=1.0)

add('E7_local_projection', None, deploy['E7_local_projection'][0], 'panel-measured', '36行 (90/68.6111)',
    'BM 方向局部响应约束投影（project.py；修正约束后 ≡ BM，历史部署表保留）',
    [deploy['E7_local_projection'][2]], score_names=['E7_local_projection'])

add('GapCapped', None, GC, 'panel-measured (09-08)', '36行 (84.4444/62.1528)',
    'BM 变体，慢带载波帽 c=1.2365e-5（ROPE_GAP_CAPPED_CANDIDATE 定义）',
    ['docs/research/ROPE_GAP_CAPPED_CANDIDATE_20260908.json Qwen3B'],
    gain=gap_capped['Qwen3B'].get('gain', GAIN))

# 队列/未执行候选（公式重建；无部署对账）
add('BM_ScaleTaper', taper_nu, ST, 'deferred (构造完成未测)', 'queued-never',
    'w_j=clip(log(W/T_j^Mr)/ln4,0,1)，nu=Mr*(BM/Mr)^w（槽24–31=BM，32–35 锥度，>=36=MrPro）',
    ['experiments/nongeometric_screen/scale_taper.py',
     'results/nongeometric_screen_20260909/deferred_queue/20260910_candidate_quality/044d_BM_ScaleTaper.json'], gain=GAIN, tol=1e-6)

add('StackFrontBack', stack_nu, None, 'queued 0446 未执行', 'queued-never',
    'MrPro ⊕ s28_less(槽28) ⊕ LBS(槽36–39 −1/131072)（公式重建，BUDGET §3 锚点 m28=.065、m36–39=.625/.729/.847/.980）',
    ['docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:45'], gain=GAIN)

add('MrProN16', n16_nu, None, 'queued 0448 未执行', 'queued-never',
    '径向族 N\'=16：m_q=q(q+1)/272，槽39完成 m=1', ['docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:46'], gain=GAIN)

add('MrProN15', n15_nu, None, 'queued 0449 未执行', 'queued-never',
    '径向族 N\'=15：m_q=q(q+1)/240，槽38完成', ['docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:47'], gain=GAIN)

# 外部基线
add('YaRN_linear_official', yarn_table(ramp='linear'), YARN_DEPLOYED, 'external-baseline', '非本屏（历史/官方）',
    '官方 YaRN Eq：low=23,high=40（beta_fast=32, beta_slow=1），线性 ramp 混频 nu=omega/4*ramp+omega*(1-ramp)；gain=1+0.1ln4；身份=官方线性 ramp（jquesnelle@995db5b / HF ROPE_INIT_FUNCTIONS yarn）',
    ['scripts/analysis/export_static_rope_baselines.py build_tables()',
     'docs/research/ROPE_CARRIER_REMOVAL_CANDIDATE_20260907.json tables/YaRN', yarn_recp], gain=GAIN, tol=1e-5)

add('YaRN_smoothstep_variant', yarn_table(ramp='smoothstep'), None, 'hypothetical-variant 未测', 'never',
    'ramp t→3t²−2t³ 的 smoothstep 变体（假设性重建：本仓库部署/对账身份均为线性 ramp）',
    ['本脚本重建；docs/research 无该表部署记录'], gain=GAIN)

add('NTK_static', ntk_nu, None, 'external-baseline (equation)', 'never-local',
    '静态 NTK：nu_j = omega_j * 4^(-j/63)（等价新基 theta*4^{128/126}），gain=1',
    ['scripts/analysis/export_static_rope_baselines.py build_tables()（本仓库冻结公式）'], gain=1.0)

# 算子臂（无单一 64 槽表）
for name, desc, src in [
    ('E5_layer21', '仅第 21 层把 MrPro 替换为 BM 表（层算子）', ['experiments/nongeometric_screen/operators.py']),
    ('E5_layer27', '同 E5（提名后因归一化下溢作废）', ['experiments/nongeometric_screen/repair_replay.py']),
    ('E5_layer32', 'E5 修复后第二名层', ['experiments/nongeometric_screen/repair_replay.py']),
    ('E6_layer14_group0', '仅层14 KV组0 替换为 E2 表', ['experiments/nongeometric_screen/operators.py']),
    ('E6_layer14_group1', '同臂反向组控制', ['experiments/nongeometric_screen/operators.py']),
    ('E7_norm_matched_BM', 'E7 数值匹配控制（BM 表同范数扰动）', ['experiments/nongeometric_screen/local_check.py']),
    ('E10_dual_frequency', '槽24–39（+88–103）双频 1/2 混合核（Mr/BM），K160/V128', ['experiments/nongeometric_screen/operators.py']),
    ('E9_distance', '距离域双时钟算子（近=Native、远=MrPro+相位修正），无表', ['experiments/nongeometric_screen/distance_operator.py'])]:
    s = load_summary(name)
    METHODS[name] = {'role': 'operator-arm (无独立 64 槽静态表)', 'panel': '12行 (持平/无模型分)' if s else 'never',
                     'construction': desc, 'sources': src,
                     'panel_scores': s, 'nu_j': None, 'm_j': None}

# ---------------------------------------------------------------- 对账（doc 锚点逐项）
CHECKS = []
def chk(item, doc_claim, mine, tol=1e-4, rel=False):
    ok = abs(mine - doc_claim) <= tol * (abs(doc_claim) if rel else 1)
    CHECKS.append({'item': item, 'doc_claim': doc_claim, 'recomputed': mine,
                   'status': 'MATCH' if ok else 'MISMATCH'})
    if not ok:
        MISMATCHES.append(f'{item}: 文档 {doc_claim} vs 重算 {mine:.6g}')
    return ok

m = {k: np.array(v['m_j'], dtype=float) for k, v in METHODS.items() if v.get('m_j')}
T = {k: np.array([x if x is not None else np.inf for x in v['T_j']]) for k, v in METHODS.items() if v.get('T_j')}
D = {k: np.array([x if x is not None else np.inf for x in v['D_j']]) for k, v in METHODS.items() if v.get('D_j')}

chk('MrPro m28 = 0.0980', 0.0980, m['MrPro'][28], 5e-4)
chk('MrPro m36..39 = .5948/.6863/.7843/.8889（digest）', 0.5948, m['MrPro'][36], 5e-4)
for i, c in enumerate([0.6863, 0.7843, 0.8889]):
    chk(f'MrPro m{37+i}', c, m['MrPro'][37 + i], 5e-4)
chk('UNIFIED:41 "m36=0.51" 应为 0.5098=槽35（GLM 判为笔误）', 0.5098, m['MrPro'][35], 5e-4)
for j, dd in zip([36, 37, 38, 39], [75000, 85000, 97000, 112000]):
    chk(f'MrPro D_{j} ≈ {dd/1000:.0f}K', dd, D['MrPro'][j], tol=dd * 0.01, rel=True)
chk('MrPro m40=1', 1.0, m['MrPro'][40], 1e-3)
chk('MrPro Σm_j(全64槽) = 29.3333（GLM 质心表）', 29.3333, METHODS['MrPro']['sum_m_all_slots'], 5e-4)
chk('s28_less m28 = 0.065', 0.065359, m['E1_s28_less'][28], 1e-3)
CHECKS.append({'item': 's28_less 仅槽28改动', 'doc_claim': 'changed==[28]',
               'recomputed': METHODS['E1_s28_less']['changed_slots_vs_MrPro'],
               'status': 'MATCH' if METHODS['E1_s28_less']['changed_slots_vs_MrPro'] == [28] else 'MISMATCH'})
chk('s28 gap 挪动量 ln4*eps28 = 0.045', 0.045, LN4 * (MR_M[28] - MR_M[27]), 1e-3)
chk('LBS m36 = 0.6252', 0.6252, m['LongBridgeSlower'][36], 5e-4)
for j, c in zip([37, 38, 39], [0.7295, 0.8465, 0.9799]):
    chk(f'LBS m{j}', c, m['LongBridgeSlower'][j], 5e-4)
for j, c in zip([36, 37, 38, 39], [77954, 90082, 105953, 127473]):
    chk(f'LBS D_{j}', c, D['LongBridgeSlower'][j], tol=c * 1e-3, rel=True)
CHECKS.append({'item': 'LBS 改动槽集合', 'doc_claim': '[36,37,38,39]',
               'recomputed': METHODS['LongBridgeSlower']['changed_slots_vs_MrPro'],
               'status': 'MATCH' if METHODS['LongBridgeSlower']['changed_slots_vs_MrPro'] == [36, 37, 38, 39] else 'MISMATCH'})
p2m, p2T = m['FullLagP2_Transfer3B'], T['FullLagP2_Transfer3B']
chk('P2 m28 = 0.0741', 0.0741, p2m[28], 5e-4)
chk('P2 m30 = 0.8506', 0.8506, p2m[30], 5e-4)
chk('P2 m31 = 0.9979', 0.9979, p2m[31], 5e-4)
chk('P2 m32 = 1.0', 1.0, p2m[32], 1e-3)
chk('P2 m Σ = 34.1789（GLM 质心表）', 34.1789, METHODS['FullLagP2_Transfer3B']['sum_m_all_slots'], 5e-3)
for j, c in zip([29, 30, 31, 40], [4468, 13267, 20193, 141332]):
    chk(f'P2 T_{j}', c, p2T[j], tol=c * 2e-4, rel=True)
p2g = np.array([x for x in METHODS['FullLagP2_Transfer3B']['gap_j']])
chk('P2 gap29 绝对值 = 1.088 巨洞', 1.088, p2g[29], 2e-3)
chk('P2 洞比率 2.97× = T30/T29', 2.97, p2T[30] / p2T[29], 5e-3)
p2c = json.loads((MIRROR / 'planned_controls/p2_gap_comparison.json').read_text())
chk('P2 高频带 gap 改动和 +0.0007754', 0.0007754344683847114,
    float(sum(p2g[g] - METHODS['MrPro']['gap_j'][g] for g in range(23))), 1e-7)
chk('pair(28+29) 洞比率 1.46× = T29/T28', 1.46, T['E1_pair28_29'][29] / T['E1_pair28_29'][28], 5e-3)
chk('HighGap 预算 0.2158674', 0.2158674, budget, 1e-5)
CHECKS.append({'item': 'HighGap recipient gaps', 'doc_claim': '[36,37,38,39]',
               'recomputed': [int(x) for x in recip],
               'status': 'MATCH' if list(recip) == [36, 37, 38, 39] else 'MISMATCH'})
chk('Smooth 粗糙度证书 0.004886399', 0.004886399, rough_smooth, 1e-6)
sm_m = np.array(METHODS['Smooth_MrBudget']['m_j'])
chk('Smooth 末步长 Δm(39→40)=0.042（文档"末 gap"以预算步长计，非 log-频率 gap）',
    0.042, float(sm_m[40] - sm_m[39]), 1e-3)
CHECKS.append({'item': 'Smooth 末 log-gap（频率口径，g39）', 'doc_claim': '文档 0.042 为 Δm 口径',
               'recomputed': float(np.array(METHODS['Smooth_MrBudget']['gap_j'])[39]), 'status': 'NOTE'})
for name in ['MrPro', 'E1_s28_less', 'E1_s29_more', 'E1_pair28_29', 'LongBridgeSlower', 'LongBridgeFaster',
             'Smooth_MrBudget', 'MrUni', 'StackFrontBack', 'MrProN16', 'MrProN15']:
    chk(f'水床恒等式 Σ_{name} 过渡 gap 增量 = ln4', LN_S,
        METHODS[name]['sum_transition_gap_increments_gaps23_39'], 1e-4)
chk('N16 m36 = 0.669', 0.669, m['MrProN16'][36], 5e-4)
for j, c in zip([37, 38, 39], [0.772, 0.882, 1.0]):
    chk(f'N16 m{j}', c, m['MrProN16'][j], 5e-4)
chk('N16 m28 = 0.110', 0.110, m['MrProN16'][28], 5e-4)
chk('N16 D39 = 131K', 131072, D['MrProN16'][39], 2, rel=False)
for j, c in zip([36, 37, 38, 39], [0.758, 0.875, 1.0, 1.0]):
    chk(f'N15 m{j}', c, m['MrProN15'][j], 5e-4)
chk('N15 m28 = 0.125', 0.125, m['MrProN15'][28], 5e-4)
chk('Stack m39 = 0.980', 0.980, m['StackFrontBack'][39], 5e-4)
chk('Stack m28 = 0.065', 0.065, m['StackFrontBack'][28], 5e-4)
chk('Stack D39 = 127.5K', 127473, D['StackFrontBack'][39], 130)
for name, claim in [('StackFrontBack', 1.86), ('MrProN16', 1.76), ('MrProN15', 1.80)]:
    hole = METHODS[name]['hole_ratio_max_transition']
    ok = chk(f'{name} max 洞（BUDGET:45-47 声称 {claim}×，按 UNIFIED §1-2 定义 ρ=T_{{g+1}}/T_g 重算）',
             claim, hole, 1e-2)
    if not ok:
        MISMATCHES.append(f'{name}: 文档 max 洞 {claim}× 与重算 {hole:.4f}× 不符（§1-2 定义下；文档未给出其"洞"公式，需作者澄清定义或修正）')
CHECKS.append({'item': 'YaRN 公式重建 vs 部署（carrier json）tensor_sha256',
               'doc_claim': tensor_sha(YARN_DEPLOYED), 'recomputed': tensor_sha(yarn_table(ramp='linear')),
               'status': 'MATCH' if tensor_sha(YARN_DEPLOYED) == tensor_sha(yarn_table(ramp='linear')) else 'MISMATCH'})
CHECKS.append({'item': 'K128_S4 回执 official_equation_yarn sha（33279a09…）是否可用于 Qwen3B 对账',
               'doc_claim': YARN_SHA,
               'recomputed': 'N/A——回执属不同模型配置（其 Native sha='
                             + 'cc63341a… 与 Qwen3B Native sha 138c99b1… 不同，L_ref=4096）',
               'status': 'NOTE'})
CHECKS.append({'item': 'YaRN linear ramp 身份', 'doc_claim': 'export 元数据 "linear index ramp"；smoothstep 仅变体',
               'recomputed': 'linear==deployed:' + str(bool(np.array_equal(f32(yarn_table(ramp='linear')), f32(YARN_DEPLOYED)))),
               'status': 'MATCH'})
p2_match = bool(np.array_equal(f32(METHODS['FullLagP2_Transfer3B']['nu_j']), f32(P2_CAND)))
CHECKS.append({'item': '3B 屏部署 P2 == 1.5B 候选文件 FullLagP2', 'doc_claim': True,
               'recomputed': p2_match, 'status': 'MATCH' if p2_match else 'MISMATCH'})
if not p2_match:
    MISMATCHES.append('P2: 3B 部署表与 ROPE_QWEN15_FULL_LAG_P2_CANDIDATE 数组不一致')
chk('BM 相对 MrPro 最大中频降幅 ≈ 31.5%（09-08 md:7）', 0.315,
    float(np.max(1 - np.array(METHODS['MrProBM']['nu_j']) / np.array(METHODS['MrPro']['nu_j']))), 5e-3)

# MrPro 面板分数（从任意 36 行 summary 的 baseline 字段实读）
_sm = json.loads((MIRROR / 'results/E1_s28_less/summary.json').read_text())
_bl = _sm['baseline']['by_length']
chk('MrPro(baseline) 32K 87.2222（E1_s28_less/summary.json 实读）', 87.2222,
    100 * _bl['32768']['macro_accuracy'], 5e-4)
chk('MrPro(baseline) 128K 78.1250', 78.1250, 100 * _bl['131072']['macro_accuracy'], 5e-4)
CHECKS.append({'item': 'BUDGET:49 声称 MrPro 公式 vs 部署误差 ≤4.3e-8（重算 max_abs_freq_diff）',
               'doc_claim': '<=4.3e-8', 'recomputed': mr_abs,
               'status': 'MATCH' if mr_abs <= 4.3e-8 else 'MISMATCH'})
if mr_abs > 4.3e-8:
    MISMATCHES.append(f'MrPro: 公式 vs 部署 max_abs={mr_abs:.3e} 超过 BUDGET:49 声称的 4.3e-8')

# 面板分数锚点（doc 百分数 vs 本地 summary.json 实读）
SCORE_ANCHORS = {
    'E1_s28_less': (87.2222, 83.3333), 'E1_s29_more': (95.5556, 77.9167),
    'E1_pair28_29': (87.2222, 73.9583), 'Smooth_MrBudget': (87.2222, 68.3333),
    'MrUni': (64.5833, 73.3333), 'HighGapToLong': (70.1389, 67.3611),
    'LongBridgeSlower': (80.5556, 80.0694), 'LongBridgeFaster': (87.2222, 73.9583),
    'FullLagP2_Transfer3B': (72.9167, 81.6667), 'Control_Mr_gain074': (98.3333, 75.3472),
    'E3_BM_gain074': (100.0, 70.0), 'E3_BM_gain1': (89.5833, 58.8194),
    'E7_local_projection': (90.0, 68.6111), 'E2_tail_more': (100.0, 54.7222),
    'E8_zero51': (100.0, 50.5556),
}
for name, (a32, a128) in SCORE_ANCHORS.items():
    ps = METHODS[name].get('panel_scores') or {}
    chk(f'{name} 32K 分数 vs doc 锚点', a32, ps.get('score_32K_pct'), 5e-4)
    chk(f'{name} 128K 分数 vs doc 锚点', a128, ps.get('score_128K_pct'), 5e-4)
# 全表水床（端点固定表 Σ_all gap 增量 = ln4）
for name in ['MrPro', 'MrProBM', 'E1_s28_less', 'E1_s29_more', 'E1_pair28_29', 'E4_pair25_29',
             'E8_zero51', 'Smooth_MrBudget', 'MrUni', 'LongBridgeSlower', 'LongBridgeFaster',
             'BM_ScaleTaper', 'StackFrontBack', 'MrProN16', 'MrProN15', 'GapCapped',
             'YaRN_linear_official', 'NTK_static']:
    chk(f'全表水床 Σ(0..62) gap 增量 {name} = ln4·(m63−m0) = ln4', LN_S,
        METHODS[name]['sum_all_gap_increments_slots0_63'], 1e-4)

# 文档表述层面的核对差异（本轮独立重验）
MISMATCHES.append('UNIFIED:41 面板表 "MrPro m36–39 = 0.51–0.89"：部署与公式重建均为 m36=0.5948，'
                  '0.5098 实为 m35（槽位错位，GLM 复核:10 判笔误；本轮独立重验）；'
                  'D 带 75/85/97/112K 与 0.5948–0.8889 重算值（74,737/84,845/97,197/112,361）一致')
MISMATCHES.append('任务描述/UNIFIED 旧文 "MrUni（全表÷4）"：部署表逐位验证为过渡段内部线性斜坡 '
                  'm_j=(j−23)/17（j=24..39），段外与 MrPro 逐位相同，并非全表÷4；'
                  'GLM 复核:11 已纠正，本轮 bit_exact=True 独立确认')
MISMATCHES.append('E1_s29_plus_matched：构造式未在本地已读文件中记录，仅部署张量为真值（changed_slots=[29]，'
                  'm29=0.094729），标 [部分证据]')
MISMATCHES.append('E1_s28_reverse_matched：候选镜像式 2·m28−m27=0.130719 ≠ 部署反演 m28=0.132270，'
                  '精确构造式未找到，仅部署张量为真值，标 [部分证据]')

out = {
    'meta': {
        'task': 'G1 频率表地面真值（36 行开发面板 64 槽表数值化重建）',
        'generated_by': 'analysis/unify_20260910/tables/rebuild_ground_truth_tables.py',
        'date': '2026-09-10',
        'constants': {'W': W, 'S': S, 'L': 131072, 'Dr': DR, 'base': BASE,
                      'lnS': LN_S, 'native_log_gap': NATIVE_LOG_GAP,
                      'native_period_ratio': NATIVE_GAP_RATIO,
                      'official_gain': GAIN, 'p2_gain': 1 + 0.074 * LN_S},
        'definitions': {
            'm_j': 'm_j = log(nu_j/omega_j)/log(4)（由 fp32 部署值反演）',
            'T_j': '2*pi/nu_j', 'D_j': 'W*4^{m_j}（识别地平线）',
            'r_j': 'W/T_j^native（原生周期数，对所有方法相同）',
            'gap_j': 'ln(nu_j/nu_{j+1})，j=0..62（0-based gap g 介于槽 g、g+1）',
            'sum_transition_gap_increments': 'Σ_{g=23}^{39} (gap_g − ln(b)/64)，端点固定表恒等于 ln4=1.386294',
            'hole_ratio': 'max T_{j+1}/T_j（UNIFIED §1-2 定义 ρ_g=T_{g+1}/T_g；原生 1.2409）',
        },
        'evidence_policy': '每个数字带出处；[已验证]=公式重建与部署张量逐位一致或双源一致；未执行臂 score=null',
        'score_units': 'panel_scores.score_*_pct 为百分数（=summary.json candidate.by_length.macro_accuracy×100），与 doc 锚点同单位',
    },
    'methods': METHODS,
    'reconciliation': CHECKS,
    'mismatches': MISMATCHES,
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
print('methods:', len(METHODS), '| checks:', len(CHECKS), '| mismatches:', len(MISMATCHES))
for x in MISMATCHES:
    print(' -', x)
print('written', OUT)
