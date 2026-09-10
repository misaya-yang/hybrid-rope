import json, math
d=json.load(open('/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=d['methods']
LN4=math.log(4); LNB64=math.log(1e6)/64; W=32768; L=131072
print("ln4",LN4,"native gap",LNB64,"native rho",math.exp(LNB64))
# 1) conservation verification across endpoint-fixed tables
print("\n== conservation: sum_{g=23}^{39}(gap_g - lnB/64) ==")
for k,v in M.items():
    if 'm_j' not in v or v['m_j'] is None: continue
    m=v['m_j']; g=v['gap_j']
    if None in g or None in m: continue
    s=sum(x-LNB64 for x in g[23:40])
    endp = abs(m[23])<1e-6 and all(abs(x-1)<1e-6 for x in m[40:64])
    print(f"{k:28s} endfix={endp} sum23-39={s:+.6f} m23={m[23]:.4f} m40={m[40]:.4f}")
