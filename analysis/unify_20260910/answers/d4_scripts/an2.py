import json, math
d=json.load(open('/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=d['methods']
LN4=math.log(4); LNB64=math.log(1e6)/64; W=32768; L=131072

def analyze(name,m,scores=None):
    gap=[math.log(1.0) for _ in m]
    dgap=[(m[g+1]-m[g]) for g in range(63)]  # in m units: gap_g - native = (m_{g+1}-m_g)*ln4
    bridge=dgap[23:40]
    tot=sum(bridge)
    # j* : first slot with m>=1-tol, all after
    def jstar(tol): 
        for j in range(24,64):
            if all(m[k]>=1-tol for k in range(j,64)): return j
        return None
    js=jstar(1e-6); jse=jstar(1e-2)
    jb=None
    for j in range(1,64):
        if m[j]>1e-6: jb=j; break
    # extras in m-units per gap; hole rho = 4^dgap * 1.241 (native ratio)
    rho=[math.exp(LNB64+g*LN4) for g in dgap]
    maxhole=max(rho[23:40]); argmax=rho[23:40].index(maxhole)+23
    F_arc=sum(max(0,1-m[j]) for j in range(36,40))
    F_arc35=sum(max(0,1-m[j]) for j in range(35,40))
    BL=sum(m[j] for j in range(24,29))  # bank-edge load slots24-28
    # centroid of bridge mass (m units), weight by extra
    cen=sum((g-23)*x for g,x in zip(range(23,40),bridge))/max(tot,1e-12)
    # mid-band max delta (g 23..38 excluding last bridge gap) and bank band max
    mb=max(bridge[:13]); bb=max(bridge[:6])  # g23..35, g23..28
    tail=max(bridge[13:])  # g36..39
    out=f"{name:14s} jb={jb} j*={js}(eff{jse}) sum={tot:.6f} Farc={F_arc:.4f} Farc35={F_arc35:.4f} BL={BL:.4f} cen={cen:.2f} rho_max={maxhole:.3f}@g{argmax} mid13={mb:.4f} bank6={bb:.4f} tail={tail:.4f}"
    if scores: out+=f" 32K/128K={scores}"
    print(out)
    return dict(jb=jb,jstar=js,F_arc=F_arc,BL=BL,rho_max=maxhole,argmax=argmax,cen=cen,mid13=mb,bank6=bb,tail=tail,scores=scores,m=m)

print("== deployed methods ==")
keys=['MrPro','MrUni','Smooth_MrBudget','E1_s28_less','E1_pair28_29','LongBridgeSlower','LongBridgeFaster','FullLagP2_Transfer3B','StackFrontBack','MrProN16','MrProN15','YaRN_linear_official','MrProBM','E7_local_projection','HighGapToLong','E2_tail_more']
res={}
for k in keys:
    v=M[k]; m=v['m_j']; s=v.get('panel_scores') or {}
    sc=(s.get('score_32K_pct'),s.get('score_128K_pct')) if s else None
    if None in sc: sc=None
    res[k]=analyze(k,m,sc)

print("\n== P2 m24..31:",[round(x,4) for x in res['FullLagP2_Transfer3B']['m'][23:32]])
print("== LBS dgap23..39 m-units:",[round(res['LongBridgeSlower']['m'][g+1]-res['LongBridgeSlower']['m'][g],4) for g in range(23,40)])
print("== Smooth dgap:",[round(res['Smooth_MrBudget']['m'][g+1]-res['Smooth_MrBudget']['m'][g],4) for g in range(23,40)])
print("== Smooth m35..40:",[round(res['Smooth_MrBudget']['m'][j],4) for j in range(35,41)])
print("== LBF dgap:",[round(res['LongBridgeFaster']['m'][g+1]-res['LongBridgeFaster']['m'][g],4) for g in range(23,40)])
print("== pair dgap:",[round(res['E1_pair28_29']['m'][g+1]-res['E1_pair28_29']['m'][g],4) for g in range(23,40)])
print("== s28 m24..29:",[round(res['E1_s28_less']['m'][j],4) for j in range(23,30)])
