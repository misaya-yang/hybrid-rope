import json, math
ROOT='/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope'
GT=json.load(open(f'{ROOT}/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=GT['methods']; C=GT['meta']['constants']
W,S,b=C['W'],C['S'],C['base']; ln4=math.log(4); G0=math.log(b)/64
mp=M['MrPro']['m_j']; sm=M['Smooth_MrBudget']['m_j']
# Smooth direct
print('Smooth m24..m41:', [round(sm[j],4) for j in range(24,42)])
print('MrPro  m24..m41:', [round(mp[j],4) for j in range(24,42)])
print('Smooth d_m vs MrPro (24..39):', [round(sm[j]-mp[j],4) for j in range(24,40)])
print('sum m24..39: MrPro=%.4f Smooth=%.4f'%(sum(mp[24:40]), sum(sm[24:40])))
print('MrPro sum m24..28 = %.4f'%sum(mp[24:29]))
print('Smooth m40-m39 = %.6f ; MrPro m40-m39 = %.6f'%(sm[40]-sm[39], mp[40]-mp[39]))
print('Smooth D36..39 vs MrPro:')
for j in range(34,40):
    print(f"  slot{j}: MrPro m={mp[j]:.4f} D={W*4**mp[j]:,.0f} | Smooth m={sm[j]:.4f} D={W*4**sm[j]:,.0f}")
# (a) master table
print('\n--- (a) master table slots 0..31 + 39,40 ---')
tot=0
for j in list(range(0,32))+[39,40]:
    w=b**(-j/64); T=2*math.pi/w; r=W/T
    m=mp[j]; step=(math.exp(-ln4*m)-1)*100
    drift=r*(1-math.exp(-ln4*m))
    beyond=3*r; cT=min(1,r); excess=0 if r>=1 else min(1,4**(1-m)*r)-min(1,r)
    sens=r*ln4
    print(f"j={j:2d} w={w:.4e} T={T:11.2f} r={r:9.3f} cT={cT:.3f} 32K外圈数={beyond:9.1f} 新弧={excess:.3f} MrPro_m={m:.4f} 步进={step:+.2f}% 窗内漂移={drift:.3f}圈 灵敏度={sens:.2f}圈/m")
# HighGap rows check
def rows(name):
    out=[]
    for line in open(f'{ROOT}/results/nongeometric_screen_20260909/results/{name}/ruler.jsonl'):
        r=json.loads(line); out.append((r['row_id'],r['baseline_correct'],r['correct']))
    return out
# weights check
hg=rows('HighGapToLong'); e8=rows('E8_zero51')
d32=sum((c-b) for rid,b,c in hg if '32768' in rid); d128=sum((c-b) for rid,b,c in hg if '131072' in rid)
print(f'\nHighGap 32K rowunits={d32:+.3f} -> {d32*100/12:+.2f}pp ; 128K={d128:+.3f} -> {d128*100/24:+.2f}pp')
e8d=sum((c-b) for rid,b,c in e8)
print(f'E8 12-row units={e8d:+.4f} -> {e8d*100/6:+.2f}pp ; E8 m_j[51] check nu51={M["E8_zero51"]["nu_j"][51]}')
print('E8 m36-41:', [round(M['E8_zero51']['m_j'][j],4) for j in range(49,53)], 'r51=%.4f'%(W/(2*math.pi/b**(-51/64))))
# E2 vs E8 rows
e2=rows('E2_tail_more'); print('E2 rows:', [(rid,b,c) for rid,b,c in e2 if abs(c-b)>1e-9])
# slopes
print('\nSLOPES:')
print(' s28 pair-pt slope: %.1f pp/m'%( (83.3333-78.1250)/(0.0654-0.098039) ))
r28=W/(2*math.pi/b**(-28/64)); 
print(' s28 per-%: %.3f pp per slot%% step (%.2f%% for 0.0326)'%(5.2083/(r28*(math.exp(-ln4*0.0654)-math.exp(-ln4*0.098039))*0 ), 0))
dr=abs(r28*(math.exp(-ln4*0.065359)-math.exp(-ln4*0.098039)))
step=abs((math.exp(-ln4*0.065359)-math.exp(-ln4*0.098039))*100)
print(f'  s28 drift restore={dr:.3f} turns ; step restore={step:.2f}% ; pp/turn={5.2083/dr:.1f} ; pp/(%*slot)={5.2083/step:.3f}')
# MrUni extra drift sum 24..31 & 24..29
mu=M['MrUni']['m_j']
tot=0
for j in range(24,32):
    r=W/(2*math.pi/b**(-j/64)); tot+=abs(r*(math.exp(-ln4*mu[j])-math.exp(-ln4*mp[j])))
print(' MrUni extra |drift| slots24-31 = %.1f turns -> 32K -22.64pp => %.2f pp/turn'%(tot, 22.6389/tot))
# HighGap bulk
hgj=M['HighGapToLong']['m_j']
sumpct=0
for j in range(1,30):
    sumpct+=abs((math.exp(-ln4*hgj[j])-1)*100)
print(' HighGap sum |step%| slots1-29 = %.0f %-slot -> 0.047pp/(%*slot) 32K; 但逐行饱和'%(sumpct))
# LBS/Faster slopes
lb=M['LongBridgeSlower']['m_j']; fb=M['LongBridgeFaster']['m_j']
dmL=sum(lb[j]-mp[j] for j in range(36,40))/4; dmF=sum(fb[j]-mp[j] for j in range(36,40))/4
print(f' LBS avg Δm36-39={dmL:+.4f} -> +1.944pp => per-slot pp/m={1.9444/dmL/4:.1f}; Faster Δm={dmF:+.4f} -> -4.167 => {4.1667/abs(dmF)/4:.1f} pp/m/slot')
# hole ratios table needed
def rho(mj,g): return math.exp(G0+ln4*(mj[g+1]-mj[g]))
for g in [27,28,29,35,38,39]:
    print(f' rho_g{g}: MrPro={rho(mp,g):.3f} s28less={rho(M["E1_s28_less"]["m_j"],g):.3f} s29more={rho(M["E1_s29_more"]["m_j"],g):.3f} pair={rho(M["E1_pair28_29"]["m_j"],g):.3f} P2={rho(M["FullLagP2_Transfer3B"]["m_j"],g):.3f} LBS={rho(M["LongBridgeSlower"]["m_j"],g):.3f} Smooth={rho(sm,g):.3f}')
