# -*- coding: utf-8 -*-
"""V_math part2: D4 closed-form transport rule — independent recomputation.
(1) endpoint-fixed count (UNIFIED '24' vs GR '17'); (2) blind LP global optimum
for three F_arc definitions; (3) family feature tables (uni/ramp 13..17);
(4) §3 per-method F_arc/BL/j*/hole/Σm/centroid vs G1; (5) §4 ceilings H/A/R +
argmin knots; (6) §5 transport arithmetic (nats conversions, lambdas, delivered
RAMP16 table row-by-row, notch/pair)."""
import json, math
ROOT='/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope'
d=json.load(open(f'{ROOT}/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=d['methods']
c=math.log(1e6)/64; ln4=math.log(4); rnat=1e6**(1/64); L=131072; W=32768

def uni(Np): return [min(1.0,max(0.0,(j-23)/Np)) for j in range(64)]
def ramp(Np):
    m=[]
    for j in range(64):
        q=max(0,min(Np,j-23)); m.append(q*(q+1)/(Np*(Np+1)))
    return m
def eps(m): return [m[g+1]-m[g] for g in range(63)]

# ---------- (1) endpoint-fixed counts ----------
print('== (1) endpoint-fixed census ==')
span1=[]; tol16=[]; tol4=[]
for name,e in M.items():
    m=e['m_j']
    if m is None: continue
    span=m[40]-m[23]
    s=sum((m[g+1]-m[g])*ln4 for g in range(23,40))
    fixed = abs(span-1.0)<1e-9
    if fixed: span1.append(name)
    if abs(s-ln4)<1e-6: tol16.append(name)
    if abs(s-ln4)<1e-4: tol4.append(name)
print('methods with nu/m arrays:', sum(1 for e in M.values() if e['m_j'] is not None))
print('exact endpoint-fixed (m40-m23==1):', len(span1))
print('|Σ−ln4|<1e-6 :', len(tol16), '| <1e-4 :', len(tol4))
print('endpoint-fixed list:', sorted(span1))
print('Σ<1e-6 but NOT endpoint-fixed:', sorted(set(tol16)-set(span1)))
print('endpoint-fixed but Σ>=1e-6 (should be none):', sorted(set(span1)-set(tol16)))

# ---------- (2) blind LP min-max max(δmax, F) over full 17-simplex ----------
print('\n== (2) blind LP greedy optimum for 3 F_arc definitions ==')
# greedy: given z, max Σ ε subject ε_i≤z, Σ w_i ε_i ≤ z; fill cheapest w.
# optimum z* from Σ_max(z)=1. weights w_q, q=1..17:
def lp(wts):
    # for given z: maximize Σε s.t. 0≤ε≤z, Σwε≤z (single shared budget z).
    # greedy cheapest-weight-first with cap z per item:
    def summax(z):
        rem=z; tot=0.0
        for w in sorted(wts):
            take=min(z, rem/w if w>0 else z)
            tot+=take; rem-=take*w
        return tot
    lo,hi=1e-6,1.0
    for _ in range(80):
        mid=(lo+hi)/2
        if summax(mid)>=1.0: hi=mid
        else: lo=mid
    return hi,summax
w_restricted=[min(4,max(0,q-13)) for q in range(1,18)]   # F=Σ_{36..39}(1−m)
w_bridge=[q-1 for q in range(1,18)]                      # F=Σ_{24..39}(1−m)
z,_=lp(w_restricted); print(f'  restricted(36..39): z*={z:.6f}  (1/14={1/14:.6f})')
z,_=lp(w_bridge);     print(f'  bridge-literal(24..39): z*={z:.6f} (≈1/2) optimum shape ε1=ε2=1/2 → NOT uniform-14')
# uniqueness check by explicit vertices for restricted
print('  restricted: proof: Σε ≤ 13z + Σ_{q≥14}ε_q ≤ 13z + Σ_{q≥14}wε (w≥1) ≤ 14z ⇒ z≥1/14; equality ⇒ ε1..13=1/14, ε14=1/14, ε15..17=0 — unique vertex = uniform-14 ✓')

# ---------- (3) family tables §2 ----------
print('\n== (3) §2 family values N\'=13..17 ==')
for Np in range(13,18):
    u=uni(Np); r=ramp(Np)
    fu=sum(max(0,1-u[j]) for j in range(36,40)); fr=sum(max(0,1-r[j]) for j in range(36,40))
    du=max(eps(u)[g] for g in range(23,40)); dr=max(eps(r)[g] for g in range(23,40))
    print(f'  N\'={Np}: uni δ={du:.4f} F={fu:.4f} max={max(du,fu):.4f} | ramp δ={dr:.4f}(2/(N\'+1)={2/(Np+1):.4f}) F={fr:.4f} max={max(dr,fr):.4f}  uniF_formula={ (min(4,Np-13)*(min(4,Np-13)+1)/2)/Np :.4f}')
# dual identity ramp: ε_{N'} == 1−m_{j*−1} == 2/(N'+1)
for Np in range(13,18):
    r=ramp(Np); js=23+Np
    print(f'  ramp{Np}: ε_last={r[23+Np]-r[23+Np-1]:.6f} 1−m[js−1]={1-r[js-1]:.6f} 2/(N\'+1)={2/(Np+1):.6f}')

# ---------- (4) §3 per-method vs G1 ----------
print('\n== (4) §3 table recomputed from G1 ==')
def feats(m,nu):
    F=sum(max(0.0,1-m[j]) for j in range(36,40))
    BL=sum(m[j] for j in range(24,29))
    rho=[nu[g]/nu[g+1] for g in range(63)]
    return F,BL,rho
for name,label in [('MrPro','MrPro'),('E1_s28_less','s28_less'),('LongBridgeSlower','LBS'),('LongBridgeFaster','LBF'),
                   ('FullLagP2_Transfer3B','P2'),('Smooth_MrBudget','Smooth'),('MrUni','MrUni'),
                   ('MrProBM','MrProBM'),('E7_local_projection','E7'),('YaRN_linear_official','YaRN'),
                   ('E1_pair28_29','pair'),('E2_tail_more','E2'),('StackFrontBack','Stack'),
                   ('MrProN16','N16'),('MrProN15','N15')]:
    e=M[name]; m=e['m_j']; nu=e['nu_j']
    F,BL,rho=feats(m,nu)
    arg=max(range(63),key=lambda g:rho[g])
    jstar=next(j for j in range(64) if m[j]>=1-1e-9)
    js99=next(j for j in range(64) if m[j]>=0.99)
    sm=sum(m); cen=sum((g-23)*max(0.0,m[g+1]-m[g]) for g in range(23,40))
    inband=max(rho[g] for g in range(23,40))
    outband=max(rho[g] for g in range(40,63))
    print(f'  {label:8s} j*={jstar}({js99}) F={F:.4f} BL={BL:.4f} maxhole={inband:.4f}@g{max(range(23,40),key=lambda g:rho[g])} Σm={sm:.2f} centroid={cen:.2f} outband_max={outband:.4f}')
print('  D4 doc says: MrPro 1.0458/.2288/1.448@g39/Σm? ; s28 1.0458/.1961; LBS .8189/.2288/1.493@g38; LBF 1.2538/.2288/1.619@g39;')
print('  P2 0/.1084/2.970@g29/Σm=34.18/cen=5.83; Smooth .6900/.0413/1.451@g35; MrUni .5882/.8824/1.346 flat; BM .3199/.5418/1.393@g31;')
print('  E7 .4306/.2547/1.552@g34; YaRN 1.0270/.5214/1.460@g39; pair 1.0458/.1961/1.461@g28; E2 1.0458/.2288 out-of-band; Stack .8189/.1961/1.493@g38;')
print('  N16 .6765/.2574/1.461@g38 j*=39; N15 .3667/.2917/1.476@g37 j*=38; LBF centroid 10.87, MrPro centroid 10.67; MrPro Σm=29.33')
print('  LBS holes ρ37/ρ38/ρ39:', [round(feats(M["LongBridgeSlower"]["m_j"],M["LongBridgeSlower"]["nu_j"])[2][g],4) for g in (37,38,39)])

# ---------- (5) §4 ceilings / R(s) scan ----------
print('\n== (5) §4: ramp band features + H,A,R, knots ==')
bank_cap,mid_cap,term_cap=0.1176,0.1130,0.1919
FEAT={}
for Np in range(13,18):
    r=ramp(Np); e_=eps(r)
    bankd=max(e_[23:29]); midd=max(e_[29:36]); termd=max(e_[36:40])
    A=sum(max(0,1-r[j]) for j in range(36,40))/1.04575
    H=max(bankd/bank_cap,midd/mid_cap,termd/term_cap)
    FEAT[Np]=(bankd,midd,termd,H,A)
    print(f'  N\'={Np} bankδ={bankd:.4f} midδ={midd:.4f} termδ={termd:.4f} H={H:.4f} A={A:.4f} R(.5)={max(.5*H,.5*A):.4f}')
print('  doc: mid .1429/.1238/.1083/.0956/.0850 bank .0659/.0571/.0500/.0441/.0392 R(.5)=.632/.548/.479/.423/.500')
# argmin sequence over s grid + exact knots (pairwise crossovers of R)
def Rv(Np,s): H,A=FEAT[Np][3],FEAT[Np][4]; return max(s*H,(1-s)*A)
seq=[]
for i in range(1,10):
    s=i/10; best=min(range(13,18),key=lambda N:Rv(N,s)); seq.append(best)
print('  argmin grid s=.1..0.9:',seq,' (doc 14,14,15,15,16,17,17,17,17)')
for a,b in [(14,15),(15,16),(16,17)]:
    lo=None
    n=20000
    prev=Rv(a,0.001)-Rv(b,0.001)
    for i in range(1,n):
        s=0.001+(0.998*i/n)
        cur=Rv(a,s)-Rv(b,s)
        if prev<0<=cur: lo=s; break
        prev=cur
    print(f'  knot {a}/{b} (R{a} crosses below R{b} upward): s*={lo if lo else float("nan"):.4f}')

# ---------- (6) §5 transport arithmetic ----------
print('\n== (6) §5 transport ==')
e17=1/9; e16_16=2/17
print(f'  17→16: m-extract={e17:.6f} nats={e17*ln4:.6f} (doc 0.15399) pct={e17*100:.2f}% λg39: 4^{{1/9}}={4**e17:.4f}(doc 1.1665) scale=306/272={306/272:.4f}')
print(f'  16→15: m-extract={e16_16:.6f} nats={e16_16*ln4:.6f} (doc 0.16306) λg38: 4^{{2/17}}={4**e16_16:.4f}(doc 1.1771) scale=272/240={272/240:.4f}')
tot=e17+e16_16
print(f'  cumulative: {tot:.6f} m (doc 0.22876) = {tot*ln4:.6f} nats (doc 0.31713) = {tot*100:.2f}% (doc 22.9)')
r17,r16,r15=ramp(17),ramp(16),ramp(15)
print(f'  slot dm: m39 17→16 {r16[39]-r17[39]:.4f}; m38 17→16 {r16[38]-r17[38]:.4f}; m38 16→15 {r15[38]-r16[38]:.4f}; m38 two-step {r15[38]-r17[38]:.4f} (doc +0.1111/+0.0980/+0.2157)')
print(f'  BL increment 17→16: {sum(r16[j] for j in range(24,29))-sum(r17[j] for j in range(24,29)):.4f} (doc +.0286)')
D38=math.exp(r16[38]*ln4)*W
print(f'  D38(ramp16) closed-form={D38:.0f} G1 stored={M["MrProN16"]["D_j"][38]:.0f} (doc 111,347)')
# notch & pair
m28p=r17[28]; m28s=M['E1_s28_less']['m_j'][28]; m27p=r17[27]
print(f'  notch: MrPro m28={m28p:.6f}(=20/306? {20/306:.6f}) s28 m28={m28s:.6f}(=m27? {m27p:.6f}) Δ={m28p-m28s:.6f}(doc .032680) nats={(m28p-m28s)*ln4:.6f}(doc .045304) %ln4={(m28p-m28s)*ln4/ln4*100:.2f}')
mp=M['E1_pair28_29']['m_j']
print(f'  pair: δ28={mp[29]-mp[28]:.6f}(doc .1176) δ27={mp[28]-mp[27]:.6f} 1.8× reading: δ28/{m28p-m28s:.6f}={(mp[29]-mp[28])/(m28p-m28s):.3f}×notch; 0.0453·1.8={(m28p-m28s)*1.8:.6f} m={0.045304*1.8/ln4:.6f} m-units vs δ28')
# ramp16 delivered table rows (line 129–146)
r=r16; nu_r=[1e6**(-j/64)*4**(-r[j]) for j in range(64)]
doc_m=[0,.007353,.022059,.044118,.073529,.110294,.154412,.205882,.264706,.330882,.404412,.485294,.573529,.669118,.772059,.882353,1.0,1.0]
ok=all(abs(r[23+i]-doc_m[i])<6e-6 for i in range(18)); print('  RAMP16 m-col vs doc rows:', 'MATCH' if ok else 'MISMATCH')
g1=M['MrProN16']
print('  ν39 doc 5.5168e−5 vs closed',nu_r[39], ' G1',g1['nu_j'][39])
print('  ρ38 doc 1.4608 vs',rnat*4**(32/272))
# r_j native windings (line 16)
print('  r_j native 36..40:',[round(W*1e6**(-j/64)/(2*math.pi),3) for j in range(36,41)],'(doc 2.199/1.772/1.428/1.151/0.927)')

# ---------- raw-gap identity check (T3 §3.0) ----------
print('\n== raw identity Σgap=17c+ln4 on endpoint-fixed G1 tables ==')
for name in ['MrPro','MrUni','YaRN_linear_official','MrProN16']:
    nu=M[name]['nu_j']
    raw=sum(math.log(nu[g]/nu[g+1]) for g in range(23,40))
    print(f'  {name}: Σ_raw={raw:.6f} 17c+ln4={17*c+ln4:.6f} diff={raw-(17*c+ln4):.2e}')
nu=M['FullLagP2_Transfer3B']['nu_j']
raw=sum(math.log(nu[g]/nu[g+1]) for g in range(23,40))
print(f'  FullLagP2: S_raw={raw:.6f} 17c+ln4={17*c+ln4:.6f} diff={raw-(17*c+ln4):.2e}; literal "17gap=ln4" would need raw=1.3863, OFF by {raw-ln4:.3f}')
g29=math.log(M['FullLagP2_Transfer3B']['T_j'][30]/M['FullLagP2_Transfer3B']['T_j'][29])
print(f'  P2 gap29 raw={g29:.6f}; under monotonicity every gap>=c={c:.4f}: 17c={17*c:.4f} > ln4={ln4:.4f} ⇒ literal identity infeasible on ANY table')
