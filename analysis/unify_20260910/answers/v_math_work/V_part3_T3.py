# -*- coding: utf-8 -*-
"""V_math part3: T3 four limit theorems — numeric re-derivation (EVQ cosh chain,
KKT algebra Thm2a, YaRN Thm3 numbers, conversions, NLC ridge)."""
import json, math
ROOT='/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope'
d=json.load(open(f'{ROOT}/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=d['methods']
ln4=math.log(4); c=math.log(1e6)/64; lnb=math.log(1e6); rnat=1e6**(1/64); W=32768

def simpson(f,a,b,n=20000):
    if n%2: n+=1
    h=(b-a)/n; s=f(a)+f(b)
    for i in range(1,n): s+=f(a+i*h)*(4 if i%2 else 2)
    return s*h/3

# ---------- 1. conversions / ε1 / BL / D / rho ----------
print('== 1. T3 §3.0/§3.1 numbers ==')
for name in ['MrUni','MrPro','YaRN_linear_official']:
    m=M[name]['m_j']; e=[m[g+1]-m[g] for g in range(23,40)]
    prod=4.0**sum(e)
    print(f'  {name:22s} Σε={sum(e):.6f} ∏λ={prod:.6f}(=S=4) ε1={e[0]:.6f} ε17={e[-1]:.6f} BL={sum(m[j] for j in range(24,29)):.4f}')
mp=M['MrPro']['m_j']
print('  ramp17 m36..39 exact:',[round((q*(q+1))/306,6) for q in (13,14,15,16)])
print('  D36..39 closed:',[round(W*4**((q*(q+1))/306)) for q in (13,14,15,16)],'(doc 74,737/84,845/97,197/112,361)')
print('  D36..39 G1:',[M['MrPro']['D_j'][j] for j in range(36,40)])
print('  ρ34=ρnat·4^{12/153}=',rnat*4**(12/153),'(T3/D2 1.3835)')
# YaRN closed forms
eps1Y=-math.log(1-0.75/17)/ln4
mY=lambda t:-math.log(1-0.75*t)/ln4
print(f'  YaRN: ε1 formula={eps1Y:.6f} ε1·17={eps1Y*17:.6f} leading const 0.75/ln4={0.75/ln4:.6f}')
print(f'  YaRN m39=mY(16/17)={mY(16/17):.6f} (T3 0.8828) vs Pro 272/306={272/306:.6f} diff={272/306-mY(16/17):.6f}')
print(f'  YaRN ε17=1−mY(16/17)={1-mY(16/17):.6f} (T3 0.1172); Σε^Y exact=1 ✓ (telescope); G1 array above')
print(f'  Pro ε1=2/306={2/306:.6f} (T3 .006536); Uni ε1=1/17={1/17:.6f} (T3 .0588)')

# ---------- 2. Thm2a KKT algebra ----------
print('\n== 2. Thm 2a: KKT of  min (σ/2)Σε² − Σ bq·ε  s.t. Σε=1 ==')
import random
random.seed(1)
for trial in range(3):
    bs=2*random.random()+0.3; sg=random.random()+0.5; Np=17
    ell=bs*(Np+1)/2-sg/Np
    # check εq=(bq−ℓ)/σ sums to 1 and satisfies stationarity for interior-support case
    e=[(bs*q-ell)/sg for q in range(1,Np+1)]
    s=sum(e)
    # KKT: σε_q − b q + ℓ = 0
    res=max(abs(sg*eq-bs*q+ell) for q,eq in zip(range(1,Np+1),e))
    print(f'  b={bs:.4f} σ={sg:.4f}: Σ={s:.12f} stationarity max|res|={res:.2e}')
# zero-intercept condition
Np=17
for trial in range(3):
    bs=2*random.random()+0.3
    sg=bs*Np*(Np+1)/2
    ell=bs*(Np+1)/2-sg/Np
    e=[(bs*q-ell)/sg for q in range(1,Np+1)]
    ref=[2*q/(Np*(Np+1)) for q in range(1,Np+1)]
    print(f'  ℓ=0 ⟺ σ=b·N\'(N\'+1)/2: max|ε−Eq14|={max(abs(x-y) for x,y in zip(e,ref)):.2e}')
# positivity/support when ℓ>0: ε_q<0 for q<ℓ/b → clipped support (not claimed by Thm2a beyond ℓ=0); note

# ---------- 3. Thm1 EVQ cosh chain ----------
print('\n== 3. Thm 1 EVQ→cosh (numeric) ==')
for tau in (0.35,0.7):
    al=1.0; be=tau**2*al
    rho=lambda phi: tau*math.cosh(tau*(1-phi))/math.sinh(tau)
    I=simpson(rho,0,1);
    # ODE residual (finite diff, h small)
    hh=1e-5; maxres=0
    for k in range(1,200):
        phi=k/200
        d2=(rho(phi+hh)-2*rho(phi)+rho(phi-hh))/hh**2
        maxres=max(maxres,abs(d2-be/al*rho(phi)))
    # first integral: α ρ' + β ∫_φ¹ ρ  (μ_F=0) constant?
    vals=[]
    for k in range(0,201):
        phi=k/200
        dp=(rho(phi+hh)-rho(phi-hh))/(2*hh)
        tail=simpson(rho,phi,1,n=4000)
        vals.append(al*dp+be*tail)
    print(f'  τ={tau}: ∫ρ={I:.12f}  ODE res(finite-diff)={maxres:.1e}  first-integral spread={max(vals)-min(vals):.1e}  ρ(1)={rho(1):.6f}=τ/sinhτ={tau/math.sinh(tau):.6f}  ρ\'(1)≈{(rho(1)-rho(1-hh))/hh:.2e}(BC=0)')
    # CDF & warp
    F=lambda phi: 1-math.sinh(tau*(1-phi))/math.sinh(tau)
    warp=lambda u: 1-math.asinh((1-u)*math.sinh(tau))/tau
    rt=max(abs(warp(F(phi))-phi) for phi in [i/1000 for i in range(1001)])
    def wnum(u):
        e2=1e-6
        return (warp(min(1,u+e2))-warp(max(0,u-e2)))/((min(1,u+e2))-max(0,u-e2))
    dv2=max(abs(wnum(u)*rho(warp(u))-1) for u in [i/999 for i in range(1,999)])
    contr=max(u-warp(u) for u in [i/20000 for i in range(20001)])
    print(f'   CDF∫ρ−F spread={max(abs(simpson(rho,0,phi)-F(phi)) for phi in [.13,.5,.77]):.1e}; warp roundtrip={rt:.1e}; φ\'·ρ={dv2:.1e}; max(u−φ)= {contr:.6f}')
    taylor_est=tau**2/6*max(u*(1-u)*(2-u) for u in [i/10000 for i in range(10001)])
    print(f'   Taylor τ²/6·max[u(1−u)(2−u)]={taylor_est:.6f};  ln-units={contr*lnb:.5f}; %of ln4={contr*lnb/ln4*100:.2f}')
    # residual after τ² subtraction → O(τ⁴)
    rr=max(abs((u-warp(u))-tau**2*u*(1-u)*(2-u)/6) for u in [i/2000 for i in range(2001)])
    print(f'   |contr−τ²·term|_max={rr:.2e}  (ratio to τ⁴={rr/tau**4:.3f})')
    # particular solution identity: (P b^{-2φ})'' − τ² P b^{-2φ} = γ_F b^{-2φ}
    muF=1.0
    gam=muF*(2*lnb)**2/al; P=gam/(4*lnb**2-tau**2)
    for phi in (0.0,0.2,0.5):
        f=lambda x: P*1e6**(-2*x)
        hh=1e-4
        d2=(f(phi+hh)-2*f(phi)+f(phi-hh))/hh**2
        lhs=d2-tau**2*f(phi); rhs=gam*1e6**(-2*phi)
        print(f'   φ={phi}: ODE-particular lhs/rhs={lhs/rhs:.10f}  P/ρ_cosh(0.05-scale): P·b^0/ρ(0)={P/rho(0):.3f} P·b^{{-0.2}}/ρ(.05)={P*1e6**(-0.1)/rho(0.05):.3f}')
    # waterbed numeric (E=1/(cρb^{-2φ}), c=1): ∫lnE vs lnb
    lw=simpson(lambda phi: math.log(1/(rho(phi)*1e6**(-2*phi))),0,1)
    print(f'   waterbed: ∫lnE={lw:.6f} ≥ lnb−ln1={lnb:.6f}? diff={lw-lnb:.2e} (equality gap = −∫lnρ = {-simpson(lambda phi: math.log(rho(phi)),0,1):.6f} ≥0)')
    # h↔ρ equivalence: Jρ (μ_F=0) vs Jh with h=φ'
    hphi=lambda u: 1.0  # placeholder
    # Jρ = α/2∫ρ² + β/2∫∫ρρmin
    Jr=al/2*simpson(lambda p: rho(p)**2,0,1)
    # double integral via identity: ∫∫ = ∫(1−u)²h(u)du ; direct double sum as cross-check too
    # direct double (grid 400):
    n=400; hp=1/n; dd=0
    for i in range(n):
        phi=(i+0.5)*hp
        inner=0
        for j in range(n):
            psi=(j+0.5)*hp
            inner+=min(phi,psi)*rho(psi)*hp
        dd+=rho(phi)*inner*hp
    Jr2=al/2*simpson(lambda p: rho(p)**2,0,1)+be/2*dd
    # warp form:
    def hu(u):
        hh2=1e-7
        w=lambda x: 1-math.asinh((1-x)*math.sinh(tau))/tau
        return (w(u+hh2)-w(u-hh2))/(2*hh2)
    Jh=0.5*(al*simpson(lambda u: 1/hu(u),1e-6,1-1e-6)+be*simpson(lambda u:(1-u)**2*hu(u),0,1))
    dbl_ident=simpson(lambda u:(1-u)**2*hu(u),0,1)
    print(f'   Jρ(α/2∫ρ²+β/2∫∫)={Jr2:.12f}   Jh(½∫[α/h+β(1−u)²h])={Jh:.12f}  diff={Jr2-Jh:.2e}')
    print(f'   ∫∫ρρmin={dd:.10f} vs ∫(1−u)²h du={dbl_ident:.10f} (identity factor: β/2 both)')
    # KKT of h*: α/h² − β(1−u)² = const?
    ks=[al/hu(u)**2-be*(1-u)**2 for u in (0.05,0.3,0.55,0.9)]
    print(f'   α/h*²−β(1−u)² at u=.05/.3/.55/.9: {[round(x,6) for x in ks]} const? spread={max(ks)-min(ks):.2e} (expect 2·λ_h)')
    # h*(u)=sqrt(α/(β(1−u)²+ν)), ν=β/sinh²τ:
    nu=be/math.sinh(tau)**2
    md=max(abs(hu(u)-math.sqrt(al/(be*(1-u)**2+nu))) for u in [i/1000 for i in range(1,1000)])
    print(f'   max|h_warp − sqrt(α/(β(1−u)²+ν))|={md:.2e}  ∫h du={simpson(hu,0,1):.12f}')

# ---------- 4. NLC ridge multipliers ----------
print('\n== 4. NLC ridge ĥ_c(k)/A0 ==')
A0=math.pi**2/(12*lnb)
def hhat(k):
    return math.pi/(2*k)/math.tanh(math.pi*k/(2*lnb))-lnb/k**2
for k in (math.pi,2*math.pi,4*math.pi,32,64):
    print(f'  k={k:6.3f}: ĥ/A0={hhat(k)/A0:.6f}')
print('  doc claims 0.991596/0.967550/0.885888/0.599065/0.355619')

# ---------- 5. raw Σ and T3 fp32 quotes ----------
print('\n== 5. T3 §5.2/quoted fp32 ==')
mpd=M['MrPro']
print('  MrPro stored Σ23-39=%.10f (T3 "1.3862943856")  Σ0-63=%.10f (T2 1.3862943591)'%(mpd['sum_transition_gap_increments_gaps23_39'],mpd['sum_all_gap_increments_slots0_63']))
print('  gain field:',mpd['gain'],' expected 1.138629436111989')
