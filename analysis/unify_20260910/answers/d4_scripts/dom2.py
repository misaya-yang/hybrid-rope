import json, itertools
d=json.load(open('/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=d['methods']
names=['MrPro','E1_s28_less','LongBridgeSlower','FullLagP2_Transfer3B','E1_pair28_29','LongBridgeFaster','Smooth_MrBudget','MrUni','MrProBM','E7_local_projection','E1_s29_more','E1_s28_reverse_matched','E1_s29_plus_matched','HighGapToLong','E2_tail_more','E4_pair25_29']
def feats(m,b,c):
    dg=[m[g+1]-m[g] for g in range(63)]
    return (sum(m[24:29]), max(dg[23:b+1]), max(dg[b+1:c+1]), max(dg[c:63]), sum(max(0,1-m[j]) for j in range(36,40)), 1-m[39])
bad={}
for b in (28,29,30):
    for c in (36,37,39):
        if c<=b: continue
        viol=[]
        P={n:feats(M[n]['m_j'],b,c) for n in names}
        S={n:(M[n]['panel_scores']['score_32K_pct'],M[n]['panel_scores']['score_128K_pct']) for n in names}
        for x,y in itertools.combinations(names,2):
            fx,fy=P[x],P[y]
            le=all(a<=bb+1e-9 for a,bb in zip(fx,fy)); ge=all(a>=bb-1e-9 for a,bb in zip(fx,fy))
            if le and not ge and S[x][1]<S[y][1]-1e-9: viol.append((x,y,'128'))
            if ge and not le and S[x][1]>S[y][1]+1e-9: viol.append((y,x,'128'))
        bad[(b,c)]=viol
for k,v in bad.items(): print(k, len(v), v[:6])
# with tail axis for E2 & bank-to-29 for s29_plus: pick b=29,c=37 and drop Control (already dropped). show best
