import json, math
d=json.load(open('/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=d['methods']
LN4=math.log(4); LNB64=math.log(1e6)/64; RN=math.exp(LNB64); W=32768; L=131072
def line(name):
    v=M[name]; m=v['m_j']
    dg=[m[g+1]-m[g] for g in range(63)]
    def jstar(tol):
        for j in range(24,64):
            if all(m[k]>=1-tol for k in range(j,64)): return j
    js=jstar(1e-6); jse=jstar(1e-2)
    jb=24
    for j in range(24,40):
        if m[j]>1e-6: jb=j; break
    Farc=sum(max(0,1-m[j]) for j in range(36,40))
    BL=sum(m[j] for j in range(24,29))
    tot=sum(dg[23:40]); cen=sum((g-23)*x for g,x in zip(range(23,40),dg[23:40]))/tot
    rmax=max(dg[23:40]); arg=dg[23:40].index(rmax)+23
    s=v.get('panel_scores') or {}
    sc=s.get('score_32K_pct'); s2=s.get('score_128K_pct')
    print(f"{name:26s} jb={jb} j*=({js},{jse}) sum={tot:.4f} cen={cen:.2f} Farc={Farc:.4f} BL={BL:.4f} rho={RN*4**rmax:.3f}@g{arg} m36-39={[round(m[j],4) for j in range(36,40)]} sc=({sc},{s2})")
for k in ['StackFrontBack','MrProN16','MrProN15','YaRN_linear_official','HighGapToLong','E2_tail_more','E7_local_projection','GapCapped','BM_ScaleTaper','E1_s29_more','MrProBM','E1_s28_reverse_matched','E1_s29_plus_matched']: line(k)
# s28 notch transport amount
m=M['E1_s28_less']['m_j']; mp=M['MrPro']['m_j']
print("\ns28 notch: delta m28 =", round(mp[28]-m[28],6), "m-units =", round((mp[28]-m[28])*LN4,6),"nats; gap27->28 shift check: gap27 s28=",round(m[28]-m[27],6))
# HighGap out-of-band injection
mh=M['HighGapToLong']['m_j']
print("HighGapToLong m0,m23:",round(mh[0],4),round(mh[23],4)," sum0..22 extras(m):",round(sum(mh[g+1]-mh[g] for g in range(0,23)),4))
# P2 T29,T30 band and 1-index convention sanity: D36 for MrPro
m=mp; print("MrPro D36..39:", [round(W*4**m[j]) for j in (36,37,38,39)])
print("r_native j36..40:", [round(M['MrPro']['r_j_native'][j],3) for j in range(36,41)])
