#!/usr/bin/env python3
"""Feasibility verifier CPU check: regenerate proposal A (0446 StackFrontBack) and
B (0448 MrProN16) m_j tables from the formulas declared in
docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md §3, and run
identity self-checks against deployed fp32 arrays in the local results mirror.
No GPU. Pure CPU, python3 stdlib + json.
"""
import json, math, os

ROOT = '/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/'
MIR = ROOT + 'results/nongeometric_screen_20260909/'
THETA = 1e6; W = 32768; S = 4.0; L = 131072
LN_S = math.log(S); LNB64 = math.log(THETA)/64

def load_arr(path, key=None):
    d = json.load(open(path))
    if key is not None:
        d = d[key]
    for k in ('values_float32','table'):
        if isinstance(d, dict) and k in d:
            d = d[k]
    if isinstance(d, dict) and 'values_float32' in d:
        d = d['values_float32']
    if isinstance(d, dict):
        d = [d[str(i)] for i in range(64)]
    return [float(x) for x in d]

# --- deployed arrays -------------------------------------------------------
Native = load_arr(MIR+'reference_tables.json', 'Native')
MrDep  = load_arr(MIR+'reference_tables.json', 'MrPro')
def contract_arr(m):
    d = json.load(open(MIR+f'results/{m}/contract.json'))
    t = d['spec']['table'] if 'spec' in d else d['table']
    return [float(x) for x in t['values_float32']]
s28Dep = contract_arr('E1_s28_less')
lbsDep = contract_arr('LongBridgeSlower')
ctrDep = contract_arr('Control_Mr_gain074')   # MrPro table @ gain .074
gt = json.load(open(ROOT+'analysis/unify_20260910/tables/ground_truth_tables.json'))

# --- formula reconstructions ----------------------------------------------
omega64 = [THETA**(-j/64.0) for j in range(64)]          # fp64 native
def mrpro_m(j):
    q = min(max(j-23,0),17)
    return q*(q+1)/306.0
MrForm = [omega64[j]*math.exp(-mrpro_m(j)*LN_S) for j in range(64)]

def relerr(a,b):  # max |a-b|/max(|b|,1e-30)
    return max(abs(x-y)/max(abs(y),1e-30) for x,y in zip(a,b))

print('== V0 formula vs deployed ==')
print('native formula vs deployed:', f"{relerr(omega64,Native):.3e}")
print('MrPro formula(q(q+1)/306) vs deployed MrPro:', f"{relerr(MrForm,MrDep):.3e}")
print('MrPro formula vs Control_Mr_gain074 table (same table diff gain):', f"{relerr(MrForm,ctrDep):.3e}")
bit = sum(1 for a,b in zip(MrForm,MrDep) if a==b)
print('bit-exact fp32 slots (MrPro):', bit, '/64 (note: formula fp64 vs fp32 storage)')

# m inversion helper
def m_of(nu, w=None):
    w = w or omega64
    return [math.log(w[j]/nu[j])/LN_S if nu[j]>0 else None for j in range(64)]

mMr = m_of(MrDep)
# s28 deployed: only slot 28 differs from MrPro?
diff28 = [j for j in range(64) if s28Dep[j]!=MrDep[j]]
print('E1_s28_less deployed slots differing from MrPro:', diff28, '| m28_dep=', f"{m_of(s28Dep)[28]:.6f}", 'm27_Mr=', f"{mMr[27]:.6f}")
# LBS deployed construction check: nu := nu - 1/131072 on slots whose native-window period in [W, L]
diffL = [j for j in range(64) if lbsDep[j]!=MrDep[j]]
ok_lbs = all(abs((lbsDep[j]-(MrDep[j]-1.0/L)) ) < 1e-9 for j in diffL)
print('LongBridgeSlower differing slots:', diffL, 'construct nu-1/131072 ok:', ok_lbs,
      'm36-39 =', [f"{x:.6f}" for x in m_of(lbsDep)[36:40]])
mLBS = m_of(lbsDep)

# --- Proposal A: MrPro (deployed base) + s28 stitch at slot 28 + LBS at 36-39
mA = list(mMr)
mA[28] = mMr[27]                      # s28_less: predecessor exponent (== deployed 0.065359)
for j in range(36,40): mA[j] = mLBS[j]  # LBS measured exponents (== deployed array values)
nuA = [omega64[j]*math.exp(-mA[j]*LN_S) for j in range(64)]
# also stitch on deployed arrays directly:
nuA2 = list(MrDep)
nuA2[28] = s28Dep[28]
for j in range(36,40): nuA2[j] = lbsDep[j]

def checks(nu, name):
    m  = m_of(nu)
    gaps = [math.log(nu[j]/nu[j+1]) for j in range(63)]
    rho  = [nu[j]/nu[j+1] for j in range(63)]     # T_{g+1}/T_g = nu_g/nu_{g+1}
    D    = [W*S**m[j] for j in range(64)]
    print(f'== {name} ==')
    print(' m28=%.6f  m36-39=%s' % (m[28], ['%.4f'%m[j] for j in range(36,40)]))
    mono_m = min(m[j+1]-m[j] for j in range(63))
    mono_nu= min(nu[j]-nu[j+1] for j in range(63))
    print(' monotonic m (min diff)          :', f"{mono_m:.3e}", 'OK' if mono_m>=-1e-12 else 'VIOLATION')
    print(' strictly decreasing nu (min diff):', f"{mono_nu:.3e}", 'OK' if mono_nu>0 else 'VIOLATION')
    print(' box m in [0,1]                  :', min(m)>=-1e-12 and max(m)<=1+1e-12, f"min={min(m):.6f} max={max(m):.6f}")
    print(' endpoints m23=%.6f m40=%.6f     :' % (m[23], m[40]), (abs(m[23])<1e-9) and abs(m[40]-1)<1e-9)
    wsum = sum(g-LNB64 for g in gaps[23:40])
    print(' waterbed sum_{g=23..39}(gap-lnb64) = %.9f  (ln4=%.9f, err=%.2e)' % (wsum, LN_S, abs(wsum-LN_S)))
    tot  = sum(gaps[23:40])
    print(' 17-gap total = %.6f  vs 17*lnb64+ln4 = %.6f (err %.2e)' % (tot, 17*LNB64+LN_S, abs(tot-(17*LNB64+LN_S))))
    gmax = max(range(23,40), key=lambda g: rho[g])
    print(' max hole rho (transition) = %.4f @gap%d ; global max %.4f @gap%d' % (rho[gmax], gmax, max(rho), rho.index(max(rho))))
    print(' D36-39 =', ['%.0f'%D[j] for j in range(36,40)], ' D39=%.0f (L=131072)'%D[39])
    print(' Sigma m (all 64) = %.4f ; Sigma m(24..39) = %.4f' % (sum(m), sum(m[24:40])))
    return m, rho, D

mA_out = checks(nuA, 'Proposal A (fp64 formula stitch)')
print(' A stitched-deployed nu vs formula stitch relerr:', f"{relerr(nuA,nuA2):.3e}")
# cross-check vs ground-truth JSON reconstruction
if 'StackFrontBack' in gt['methods']:
    g = gt['methods']['StackFrontBack']
    gnu = [float(x) for x in g['nu_j']]
    print(' A vs ground_truth_tables StackFrontBack nu relerr:', f"{relerr(nuA2,gnu):.3e}")

# --- Proposal B: m_q = q(q+1)/272, q = clip(j-23, 0, 16)
mB = [0.0]*24 + [ (q*(q+1)/272.0) for q in range(1,17) ] + [1.0]*24
# j=24..39 -> q=1..16 (q=16 gives 1.0 at slot 39); j>=40 -> 1.0
nuB = [omega64[j]*math.exp(-mB[j]*LN_S) for j in range(64)]
mB_out = checks(nuB, 'Proposal B (MrProN16 formula)')
if 'MrProN16' in gt['methods']:
    g = gt['methods']['MrProN16']
    gnu=[float(x) for x in g['nu_j']]; gm=[x for x in g['m_j']]
    print(' B vs ground_truth_tables N16 nu relerr:', f"{relerr(nuB,gnu):.3e}", ' m maxdiff:', f"{max(abs(x-y) for x,y in zip(mB,gm if gm[0] is not None else [0]*64)):.3e}")
# claimed values audit
print('== claim audit ==')
mAr = m_of(nuA2); rhoA=[nuA2[j]/nuA2[j+1] for j in range(63)]; rhoB=[nuB[j]/nuB[j+1] for j in range(63)]
print(' A claimed hole 1.86x -> recomputed %.4f ; B claimed 1.76x -> %.4f' % (max(rhoA[23:40]), max(rhoB[23:40])))
print(' A claimed m36-39 .625/.729/.847/.980 -> %.4f/%.4f/%.4f/%.4f' % tuple(mAr[36:40]))
print(' B claimed m28 .110 -> %.6f ; m36-39 .669/.772/.882/1.0 -> %.4f/%.4f/%.4f/%.4f' % (mB[28], mB[36], mB[37], mB[38], mB[39]))
print(' B D39 = %.0f (=L, completion true); A D39 = %.0f (< L by %.0f)' % (W*S**mB[39], W*S**mAr[39], L - W*S**mAr[39]))
# B front-end direction vs s28_less
print(' B m24-28 vs MrPro delta:', [f"{mB[j]-mMr[j]:+.6f}" for j in range(24,29)], '(positive = MORE compressed than MrPro)')
# MrUni endpoint audit (proposal cites as I1 evidence)
mUni = m_of(contract_arr('MrUni'))
print('== MrUni (cited I1 evidence) == m23=%.6f m24=%.6f m28=%.6f m40=%.6f  -> violates I1? %s violates I2? %s' %
      (mUni[23], mUni[24], mUni[28], mUni[40], mUni[23]!=0, mUni[40]!=1.0))
HG = m_of(contract_arr('HighGapToLong'))
print(' HighGapToLong m23=%.6f (I1 violation: %s)' % (HG[23], HG[23]<-1e-9))
E2n = contract_arr('E2_tail_more'); mE2 = m_of(E2n)
print(' E2 m40=%.6f (I2 violation: %s)' % (mE2[40], mE2[40] > 1+1e-6))
E8n = contract_arr('E8_zero51')
print(' E8 nu51=%.1f (slot51 in I2 tail band -> I2-side violation)' % E8n[51])

print('== fp32-cast bit comparison (claim: err <= 4.3e-8) ==')
import struct
def f32(x): return struct.unpack('f', struct.pack('f', x))[0]
diff = sum(1 for j in range(64) if f32(MrForm[j]) == MrDep[j])
print('MrPro formula cast-to-fp32 bit-exact slots vs deployed:', diff, '/64')
# B and A monotonicity at 1e-6 tolerance (fp32 inversion noise band)
print('A min m-diff with 1e-6 tol:', 'OK' if min(mA[j+1]-mA[j] for j in range(63)) > -1e-6 else 'VIOLATION',
      ' (worst %.2e, fp32-inversion noise)' % min(mA[j+1]-mA[j] for j in range(63)))