import json, itertools
d=json.load(open('/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=d['methods']
names=['MrPro','E1_s28_less','LongBridgeSlower','FullLagP2_Transfer3B','E1_pair28_29','LongBridgeFaster','Smooth_MrBudget','MrUni','MrProBM','E7_local_projection','E1_s29_more','E1_s28_reverse_matched','E1_s29_plus_matched','HighGapToLong','E2_tail_more','E4_pair25_29']
def feats(m,b,c):
    dg=[m[g+1]-m[g] for g in range(63)]
    return (sum(m[24:29]), max(dg[23:b+1]), max(dg[b+1:c+1]), max(dg[c:63]), sum(max(0,1-m[j]) for j in range(36,40)), 1-m[39])
for b,c in [(29,37)]:
    P={n:feats(M[n]['m_j'],b,c) for n in names}
    for idx,lab in [(0,'32K'),(1,'128K')]:
        viol=[]
        for x,y in itertools.combinations(names,2):
            fx,fy=P[x],P[y]
            le=all(a<=bb+1e-9 for a,bb in zip(fx,fy)); ge=all(a>=bb-1e-9 for a,bb in zip(fx,fy))
            sx=M[x]['panel_scores']['score_32K_pct' if idx==0 else 'score_128K_pct']
            sy=M[y]['panel_scores']['score_32K_pct' if idx==0 else 'score_128K_pct']
            if le and not ge and sx<sy-1e-9: viol.append((x,y,round(sx,2),round(sy,2)))
            if ge and not le and sx>sy+1e-9: viol.append((y,x,round(sy,2),round(sx,2)))
        print(lab, len(viol)); [print('  ',v) for v in viol]
