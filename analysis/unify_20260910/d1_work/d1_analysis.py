# D1 高频冗余 量化分析（纯 CPU，仅用本地地面真值与已存逐行结果）
import json, math, os
ROOT='/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope'
GT=json.load(open(f'{ROOT}/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=GT['methods']; C=GT['meta']['constants']
W,S,L,Dr,b=C['W'],C['S'],C['L'],C['Dr'],C['base']
ln4=math.log(4); ln_gap0=math.log(b)/Dr
OUT={}

# ---------- (a) bank 槽表 ----------
nat=M['MrPro']  # r_j_native 对所有方法相同
rowsA=[]
for j in range(Dr):
    wj=b**(-j/Dr); Tj=2*math.pi/wj; rj=W/Tj
    rowsA.append(dict(j=j,omega=wj,T_nat=Tj,r_nat=rj))
print('--- slot r_nat table (0..40) ---')
for r in rowsA[:41]:
    print(f"j={r['j']:2d} omega={r['omega']:.6e} T_nat={r['T_nat']:12.2f} r_nat={r['r_nat']:10.3f}")
cut8=[r['j'] for r in rowsA if r['r_nat']>=8]
print('r>=8 slots:', min(cut8),'..',max(cut8), ' count', len(cut8))
print('r>=10 slots cutoff:', max(r['j'] for r in rowsA if r['r_nat']>=10))

# per-token step change & winding drift for displacements
def drift(r_nat, m_dep, m_ref=0.0):
    # deployed vs reference table: windings-per-window shift of end-of-window phase
    return r_nat*(math.exp(-ln4*m_dep)-math.exp(-ln4*m_ref))

# HighGapToLong vs Native
hg=M['HighGapToLong']['m_j']; mu=M['MrUni']['m_j']; mp=M['MrPro']['m_j']
p2=M['FullLagP2_Transfer3B']['m_j']
print('\n--- HighGapToLong bank detune (j=0..29), Δm = m_HG - 0 ---')
tot=0; tot10=0
for j in range(30):
    r=rowsA[j]['r_nat']; dm=hg[j]
    dr=drift(r,dm); pct=(math.exp(-ln4*dm)-1)*100
    tot+=abs(dr)
    if j<=23: tot10+=abs(dr)
    print(f"j={j:2d} m={dm:+.4f} per-token step chg={pct:+.2f}% window-end drift={dr:+.2f} turns")
print(f'Sum |drift| slots0-29 = {tot:.1f} turns; slots0-23 = {tot10:.1f}')

print('\n--- MrUni vs MrPro (bank-edge slots 24..31) ---')
for j in range(22,34):
    r=rowsA[j]['r_nat']; dm=mu[j]-mp[j]
    dr=drift(r,mp[j],0)+ -drift(r,mu[j],0)  # additional drift vs MrPro = -(r(e^-ln4 mu - e^-ln4 mp))
    extra=r*(math.exp(-ln4*mp[j])-math.exp(-ln4*mu[j]))
    print(f"j={j:2d} mMr={mp[j]:.4f} mUni={mu[j]:.4f} Δm={dm:+.4f} r={r:7.2f} extra drift={-extra:+.2f} turns stepchg={(math.exp(-ln4*mu[j])-math.exp(-ln4*mp[j]))*100:+.1f}%")

# hole ratio per gap for methods
def hole_ratios(mj):
    out=[]
    for g in range(Dr-1):
        out.append(math.exp(ln_gap0+ln4*(mj[g+1]-mj[g])))
    return out
hrP=hole_ratios(mp)
print('\n--- MrPro hole ratios g=23..40:', [f'g{g}:{hrP[g]:.3f}' for g in range(23,41)])
for nm in ['E1_s28_less','E1_s29_more','E1_pair28_29','Smooth_MrBudget','MrUni','LongBridgeSlower','LongBridgeFaster','FullLagP2_Transfer3B','HighGapToLong']:
    hr=hole_ratios(M[nm]['m_j'])
    mx=max(range(20,45), key=lambda g: hr[g])
    print(nm, 'max hole g=20..44:', f'g{mx}={hr[mx]:.4f}', '| g28=%.3f g29=%.3f'%(hr[28],hr[29]))

# ---------- (b) row-level diffs ----------
RD=f'{ROOT}/results/nongeometric_screen_20260909/results'
def rowdiff(name):
    diffs=[]
    p=f'{RD}/{name}/ruler.jsonl'
    if not os.path.exists(p): return None
    for line in open(p):
        r=json.loads(line)
        if abs(r['correct']-r['baseline_correct'])>1e-9:
            diffs.append((r['row_id'], r['baseline_correct'], r['correct']))
    return diffs
print('\n--- ROW DIFFS vs MrPro baseline ---')
for nm in ['E1_s28_less','E1_s29_more','E1_pair28_29','LongBridgeSlower','LongBridgeFaster','Smooth_MrBudget','HighGapToLong','MrUni','E2_tail_more','E8_zero51','E1_s28_reverse_matched','E1_s29_plus_matched','E4_pair25_29','FullLagP2_Transfer3B','Control_Mr_gain074','E10_dual_frequency','E9_distance']:
    d=rowdiff(nm)
    if d is None: print(nm,'no file'); continue
    print(nm, '->', [(rid, f'{a}->{bb}') for rid,a,bb in d])

# 2x2 factorial on 128K (36 rows)
f=lambda nm,x: {('32'):M[nm]['panel_scores']['score_32K_pct']}
base128=M['MrPro']['panel_scores']['score_128K_pct']; base32=M['MrPro']['panel_scores']['score_32K_pct']
a=M['E1_s28_less']['panel_scores']['score_128K_pct']-base128
bb=M['E1_s29_more']['panel_scores']['score_128K_pct']-base128
ab=M['E1_pair28_29']['panel_scores']['score_128K_pct']-base128
print(f'\n128K factorial: effect_m28_down={a:+.3f} effect_m29_up={bb:+.3f} joint={ab:+.3f} interaction={ab-a-bb:+.3f}')
a3=M['E1_s28_less']['panel_scores']['score_32K_pct']-base32; b3=M['E1_s29_more']['panel_scores']['score_32K_pct']-base32; ab3=M['E1_pair28_29']['panel_scores']['score_32K_pct']-base32
print(f'32K factorial: {a3:+.3f} {b3:+.3f} joint={ab3:+.3f} interaction={ab3-a3-b3:+.3f}')
