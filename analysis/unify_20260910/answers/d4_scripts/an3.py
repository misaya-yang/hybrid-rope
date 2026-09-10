import json, math
d=json.load(open('/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=d['methods']
LN4=math.log(4); LNB64=math.log(1e6)/64; W=32768; L=131072; RN=math.exp(LNB64)
def show(k):
    v=M[k]; m=v['m_j']
    print(k, "m23..40:", " ".join(f"{m[j]:.4f}" for j in range(23,41)))
    print("   dgap23..39:", " ".join(f"{m[g+1]-m[g]:.4f}" for g in range(23,40)))
    T=[2*math.pi/ (math.exp(-math.log(1e6)*j/64)*4**(-m[j])) for j in range(64)]
    print("   deployed T at g27,28,29,32,35,38,39:", [round(T[g]/1000,1) for g in (27,28,29,32,35,38,39)])
for k in ['MrPro','Smooth_MrBudget','LongBridgeSlower','LongBridgeFaster','FullLagP2_Transfer3B','MrUni']: show(k)
