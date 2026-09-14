import json, math, itertools
d=json.load(open('/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=d['methods']
def ax(name):
    m=M[name]['m_j']; dg=[m[g+1]-m[g] for g in range(63)]
    Farc=sum(max(0,1-m[j]) for j in range(36,40))
    return dict(BL=sum(m[24:29]), bank=max(dg[23:29]), mid=max(dg[29:37]), Farc=Farc, last=1-m[39])
pts={}
for k,v in M.items():
    if v.get('m_j') is None or any(x is None for x in v['m_j']): continue
    s=v.get('panel_scores') or {}
    a=s.get('score_32K_pct'); b=s.get('score_128K_pct')
    if b is None: continue
    pts[k]=(ax(k),a,b)
show=[k for k in ['MrPro','E1_s28_less','LongBridgeSlower','FullLagP2_Transfer3B','E1_pair28_29','LongBridgeFaster','Smooth_MrBudget','MrUni','MrProBM','E7_local_projection','E1_s29_more','E1_s28_reverse_matched','E1_s29_plus_matched','HighGapToLong','E2_tail_more','E4_pair25_29','Control_Mr_gain074'] if k in pts]
print("missing:", [k for k in ['MrPro','E1_s28_less','LongBridgeSlower','FullLagP2_Transfer3B','E1_pair28_29','LongBridgeFaster','Smooth_MrBudget','MrUni','MrProBM','E7_local_projection','E1_s29_more','E1_s28_reverse_matched','E1_s29_plus_matched','HighGapToLong','E2_tail_more','E4_pair25_29','Control_Mr_gain074'] if k not in pts])
print(f"{'method':26s} BL     bank   mid    Farc   1-m39   32K     128K")
for k in show:
    f=pts[k][0]
    print(f"{k:26s} {f['BL']:.4f} {f['bank']:.4f} {f['mid']:.4f} {f['Farc']:.4f} {f['last']:.4f} {pts[k][1]:7.3f} {pts[k][2]:7.3f}")
keys=['BL','bank','mid','Farc','last']
def check(idx,label):
    viol=[]
    for x,y in itertools.combinations(show,2):
        fx,fy=pts[x][0],pts[y][0]
        le=all(fx[k]<=fy[k]+1e-9 for k in keys); ge=all(fx[k]>=fy[k]-1e-9 for k in keys)
        if le and not ge and pts[x][idx]<pts[y][idx]-1e-9: viol.append((x,y))
        if ge and not le and pts[x][idx]>pts[y][idx]+1e-9: viol.append((y,x))
    print(f"{label} violations:", viol if viol else "NONE")
check(2,'128K'); 
pts32={k:v for k,v in pts.items() if v[1] is not None}
show32=[k for k in show if pts[k][1] is not None]
viol=[]
for x,y in itertools.combinations(show32,2):
    fx,fy=pts[x][0],pts[y][0]
    le=all(fx[k]<=fy[k]+1e-9 for k in keys); ge=all(fx[k]>=fy[k]-1e-9 for k in keys)
    if le and not ge and pts[x][1]<pts[y][1]-1e-9: viol.append((x,y))
    if ge and not le and pts[x][1]>pts[y][1]+1e-9: viol.append((y,x))
print("32K violations:", viol if viol else "NONE")
