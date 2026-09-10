# -*- coding: utf-8 -*-
"""part3b: T2 identity items, D2 c1 bound, D_j formula, exact IBP identity."""
import json, math
ROOT='/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope'
d=json.load(open(f'{ROOT}/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=d['methods']
ln4=math.log(4); c=math.log(1e6)/64; lnb=math.log(1e6); rnat=1e6**(1/64); W=32768
def uni(Np): return [min(1.0,max(0.0,(j-23)/Np)) for j in range(64)]
def ramp(Np):
    out=[]
    for j in range(64):
        q=max(0,min(Np,j-23)); out.append(q*(q+1)/(Np*(Np+1)))
    return out
def yarn(tl):  # freq-ratio linear mix with carrier map m=−ln(1−.75t)/ln4
    out=[]
    for j in range(64):
        t=max(0.0,min(1.0,(j-23)/17.0))
        out.append(-math.log(1-0.75*t)/ln4 if j>=23 else 0.0)
    return out
print('== T2 (a): exact-array Σ_excess deviation (11 families + YaRN) ==')
dev=[]
for Np in range(13,18):
    for f,nm in ((uni,f'uni{Np}'),(ramp,f'ramp{Np}')):
        m=f(Np); s=sum((m[g+1]-m[g]) for g in range(23,40))*ln4
        dev.append((nm,s-ln4))
m=yarn(1); s=sum((m[g+1]-m[g]) for g in range(23,40))*ln4; dev.append(('YaRNexact',s-ln4))
print(' ', ', '.join(f'{nm}:{dd:.1e}' for nm,dd in dev))
print('  max dev = %.1e (T2 claims ≤6.7e−16 on 12 closed-form tables)'%max(abs(dd) for _,dd in dev))

print('\n== D2 c1: ρmax ≥ ρnat·4^{1/N_active} pigeonhole + published window ==')
for name,e in M.items():
    m=e['m_j']
    if m is None: continue
    span=m[40]-m[23]
    if abs(span-1)<1e-9:
        ex=[m[g+1]-m[g] for g in range(23,40)]
        Nact=sum(1 for x in ex if x>1e-12)
        mx=max(ex)
        ok = mx >= 1.0/Nact - 1e-9
        # violation of 4^{1/N'} bound?
        if not ok: print('  VIOLATION', name)
print('  all endpoint-fixed tables satisfy max excess ≥ 1/N_active (=> ρmax ≥ ρnat·4^{1/N\'}) ✓')

print('\n== D_j = W·4^{m_j} coordinate identity (spot: MrPro, P2, N16) ==')
for name in ['MrPro','FullLagP2_Transfer3B','MrProN16']:
    e=M[name]; m=e['m_j']; D=e['D_j']
    md=max(abs(W*4**m[j]-D[j]) for j in range(64))
    mr=max(abs((W*4**m[j]-D[j])/D[j]) for j in range(64))
    print(f'  {name}: max abs diff={md:.3f}  max rel={mr:.2e}')

print('\n== exact IBP identity: ∫∫ρρmin dφdψ (u-space) = ∫(1−u)²h du ==')
def simpson(f,a,b,n=40000):
    if n%2: n+=1
    h=(b-a)/n; s=f(a)+f(b)
    for i in range(1,n): s+=f(a+i*h)*(4 if i%2 else 2)
    return s*h/3
for tau in (0.35,0.7):
    s_=math.sinh(tau)
    warp=lambda u: 1-math.asinh((1-u)*s_)/tau
    L1=simpson(lambda v:(1-v)*warp(v),0,1)*2
    h=lambda u: (s_/tau)/math.sqrt(1+(1-u)**2*s_**2)   # φ'(u) exact
    L2=simpson(lambda u:(1-u)**2*h(u),0,1)
    print(f'  τ={tau}: 2∫(1−v)φ(v)dv={L1:.14f}  ∫(1−u)²φ\'du={L2:.14f}  diff={L1-L2:.2e}')
    # φ'(u) analytic = sinhτ/sqrt(1+(1−u)²sinh²τ) ✓
