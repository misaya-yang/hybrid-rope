import numpy as np
from scipy.optimize import brentq
def zipf_prior(L,alpha,dmin=1):
    D=np.arange(dmin,L+1,dtype=float); w=D**(-alpha); return D,w/w.sum()
def AB(ts,D,w,CH=400):
    ts=np.atleast_1d(np.asarray(ts,float)); A=np.empty_like(ts)
    for i in range(0,len(ts),CH):
        A[i:i+CH]=np.cos(np.outer(ts[i:i+CH],D))@w
    return A
def q_mu_grid(D,w,n=6000,lo=1e-8,hi=3e3):
    g=np.geomspace(lo,hi,n); A1=AB(g,D,w); A2=AB(2*g,D,w)
    return g, 0.5+0.5*A2-A1**2
def omega0_from_grid(g,qq):
    lg=np.log(g)
    F=np.concatenate([[0.0],np.cumsum(0.5*(qq[1:]+qq[:-1])*np.diff(lg))])   # int q dlog x
    G=np.concatenate([[0.0],np.cumsum(0.5*((qq[1:]-.5)+(qq[:-1]-.5))*np.diff(lg))])
    tot=G[-1]
    # I(o0) = F(o0) + (tot - G(o0)) = 0
    I=F+(tot-G)
    k=np.where(np.sign(I[:-1])!=np.sign(I[1:]))[0]
    if len(k)==0: return np.nan
    i=k[0]; t=-I[i]/(I[i+1]-I[i]); return float(np.exp(lg[i]+t*(lg[i+1]-lg[i])))
L=4096; x0ref=2.0743
print("prior mu_alpha on [1,%d]:  omega_0 and the derived L_rng = x_0/omega_0"%L)
print("%-6s %12s %12s %10s %10s %10s %10s"%("alpha","omega_0","L_rng","L_rng/L","sd_mu","median","mean"))
res={}
for al in [0.0,0.5,1.0,1.5,2.0,2.5,3.0]:
    Dz,wz=zipf_prior(L,al); g,qq=q_mu_grid(Dz,wz); o0=omega0_from_grid(g,qq)
    m1=float(wz@Dz); sd=np.sqrt(float(wz@Dz**2)-m1**2)
    med=Dz[np.searchsorted(np.cumsum(wz),0.5)]
    res[al]=(o0,x0ref/o0)
    print("%-6.1f %12.5e %12.1f %10.4f %10.1f %10.0f %10.1f"%(al,o0,x0ref/o0,x0ref/o0/L,sd,med,m1))
print("\n  uniform check: omega_0*L = %.4f  (paper x_0 = 2.0743)"%(res[0.0][0]*L))
print("\n  => L_rng tracks the prior's own scale, NOT the window length.")
for al in [0.5,1.0,1.5,2.0,2.5,3.0]:
    Dz,wz=zipf_prior(L,al); m1=float(wz@Dz); sd=np.sqrt(float(wz@Dz**2)-m1**2)
    print("     alpha=%.1f : L_rng=%8.1f   sd_mu=%8.1f   L_rng/sd_mu=%.2f"%(al,res[al][1],sd,res[al][1]/sd))
import numpy as np
print("="*80); print("O2.  L_eff^J as a functional of mu (variance-ratio identity)"); print("="*80)
# L_eff^J = n * Var_{U_n}[G] / Var_mu[G];  linear-G limit G(r) ~ gamma r
print("  identity (linear-G limit):  L_eff^J = n * Var_{U_n}[Delta] / Var_mu[Delta]")
print("  audit: Geo/Geo kappa=4.6476e-4 -> L_eff=2151.6 ; probe query positions 63,127,255,383,511\n")
qpos=np.array([63,127,255,383,511]); ns=qpos+1
Leff=2151.6
for n in ns:
    varU=(n**2-1)/12.0
    varmu=n*varU/Leff
    print("   n=%3d : Var_U=%9.1f  => required Var_mu=%9.1f  => sd_mu=%6.1f tokens  (sd_U=%6.1f)"%(n,varU,varmu,np.sqrt(varmu),np.sqrt(varU)))
# ratio-of-means aggregates; use the ||Pg||^2-weighted mixture. G ~ gamma r => ||Pg||^2 ~ gamma^2 n Var_U
varU=(ns**2-1)/12.0; wts=ns*varU; wts=wts/wts.sum()
print("\n   ||Pg||^2-weighted aggregate (weights prop. to n*Var_U):", np.round(wts,3))
# kappa = E[Var_mu]/E[||Pg||^2] = sum_n w_n Var_mu(n) / sum_n w_n n Var_U(n)  -- solve for a common sd
from scipy.optimize import brentq
def kappa_of_sd(sd):
    num=np.sum(wts*np.minimum(sd**2,varU)); den=np.sum(wts*ns*varU)/np.sum(wts*ns*varU)  # normalised
    return np.sum(wts*np.minimum(sd**2,varU))/np.sum(wts*ns*varU/ (ns*varU).sum()*(ns*varU).sum())
# simpler: kappa = sum_n p_n Var_mu_n / sum_n p_n n Var_U_n  with p_n uniform over the 5 positions
p=np.ones(5)/5
f=lambda sd: (p@np.minimum(sd**2,varU))/(p@(ns*varU)) - 4.6476e-4
sd_star=brentq(f,1.0,2000.0)
print("   uniform-over-query-position aggregation => implied common sd_mu = %.1f tokens"%sd_star)
print("   (falsifiable: recompute sd of the attention-weighted distance distribution from the")
print("    probe's saved p tensors; the prediction is a few tens of tokens, not ~150 = sd of U[0,512])")

print("\n  contraction modulus of tau -> c d / sqrt(L_eff(tau)):")
tau=np.array([0.0,2.83]); Le=np.array([2151.6,1867.9]); slope=(Le[1]-Le[0])/(tau[1]-tau[0])
c,d=1.0,64.0; L0=2000.0
Tp=0.5*c*d*L0**-1.5*abs(slope)
print("   secant dL_eff/dtau = %.1f ; |T'| = (c d /2) L_eff^{-3/2} |dL_eff/dtau| = %.4f  << 1"%(slope,Tp))
print("   => Banach: unique fixed point, |tau_n - tau*| <= %.3f^n |tau_0 - tau*|"%Tp)

print("\n"+"="*80); print("O3.  closed-form local L-exponent gamma_eff, vs App. A.11's stiffness sweep"); print("="*80)
x0=2.0743; g=lambda p:p*(1-p)*(2-p); gp=lambda p:2-6*p+3*p**2
def gamma_eff(L,b):
    ph=np.log(L/x0)/np.log(b)
    if ph<=0 or ph>=1: return np.nan
    return 0.5 - gp(ph)/(2*g(ph)*np.log(b))
print("  gamma_eff(L) = 1/2 - g'(phi_*) / (2 g(phi_*) log b),   phi_* = log(L/x_0)/log b\n")
for b in [1e4,1e5,5e5]:
    Ls=np.array([128,256,512,1024,2048,4096]); ge=np.array([gamma_eff(L,b) for L in Ls])
    # effective single exponent over the sweep = -d log tau / d log L fitted by OLS
    ph=np.log(Ls/x0)/np.log(b); logtau=-0.5*np.log(Ls)+0.5*np.log(g(ph))
    fit=np.polyfit(np.log(Ls),logtau,1)[0]
    print("  b=%8.0e  gamma_eff over L=128..4096: %s   OLS single exponent = %.3f"%(b," ".join("%.3f"%v for v in ge),-fit))
print("\n  App. A.11 reports, forcing a pure power law: L2 0.626, geo-mean 0.561, p=0.80 0.498, chi^2 0.465.")
print("  The closed form says tau_*(L) is NOT a power law: gamma_eff drifts by ~0.1 across that very sweep,")
print("  so 'which stiffness matches gamma=0.500' is fitting a functional form the model does not predict.")
import numpy as np
from scipy.optimize import brentq
exec(open('verify_three_completions.py' if False else '/home/claude/work/out/verify_three_completions.py').read().split('CFG=[')[0])
x0=2.0743; g=lambda p:p*(1-p)*(2-p); gmax=2/(3*np.sqrt(3))
lam=12/(45*gmax)
Ls=np.array([128,256,512,1024,2048,4096]); d=64.0
print("exact (non-Taylor) solve of Psi(tau)=2 lambda d^2/L, fitted OLS exponent  tau ~ L^{-gamma}")
print("%-10s %s   %-10s %-10s"%("b"," ".join("%8d"%L for L in Ls),"gamma_exact","gamma_1storder"))
for b in [1e4,1e5,5e5]:
    te=[];tf=[]
    for L in Ls:
        te.append(brentq(lambda u: Psi(u,L,b)-2*lam*d**2/L,1e-4,30.0,xtol=1e-9))
        ph=min(1,np.log(L/x0)/np.log(b)); tf.append(np.sqrt(g(ph)/gmax)*d/np.sqrt(L))
    ge=-np.polyfit(np.log(Ls),np.log(te),1)[0]; gf=-np.polyfit(np.log(Ls),np.log(tf),1)[0]
    print("%-10.0e %s   %-10.3f %-10.3f"%(b," ".join("%8.3f"%v for v in te),ge,gf))
print("\nApp. A.11 reported chi^2 -> gamma=0.465 (target 0.500).")
print("With the closed-form Q1 the same chi^2 stiffness gives gamma_exact above; the residual")
print("difference is the small-tau truncation, not the choice of stiffness.")
import numpy as np
print("="*80); print("O4.  Forced allocation: rho'' - tau^2 rho = (lambda/alpha) q_mu''  ==> EVQ-Cosh-R"); print("="*80)
def rho_cosh(p,t): return t*np.cosh(t*(1-p))/np.sinh(t)
def rho_R(p,t,ps,k):
    sm=np.sinh(t*ps); sp=np.sinh(t*(1-ps))
    out=rho_cosh(p,t).copy()
    lo=p<ps
    out[lo]  += k*sp*np.cosh(t*p[lo])/np.sinh(t)
    out[~lo] -= k*sm*np.cosh(t*(1-p[~lo]))/np.sinh(t)
    return out
def F_R(p,t,ps,k):
    sm=np.sinh(t*ps); sp=np.sinh(t*(1-ps))
    F=1-np.sinh(t*(1-p))/np.sinh(t)
    lo=p<ps
    F=F.copy()
    F[lo]  += k*sp*np.sinh(t*p[lo])/(t*np.sinh(t))
    F[~lo] -= k*sm*(np.sinh(t*(1-ps))-np.sinh(t*(1-p[~lo])))/(t*np.sinh(t))
    F[~lo] += k*sp*np.sinh(t*ps)/(t*np.sinh(t))
    return F
grid=np.linspace(0,1,2000001)
print("  closed form:  rho(phi) = tau cosh(tau(1-phi))/sinh tau")
print("                         + (kappa/sinh tau) * { sinh(tau(1-phi_*)) cosh(tau phi),  phi<phi_*")
print("                                              {-sinh(tau phi_*)   cosh(tau(1-phi)), phi>phi_*")
print("  with kappa = lambda/(2 alpha) the jump height at the resolution threshold phi_*.\n")
print("  %-6s %-6s %-8s %12s %12s %12s %10s"%("tau","phi_*","kappa","int rho","jump","min rho","F(1)"))
for (t,ps,k) in [(2.0,0.578,0.0),(2.0,0.578,0.2),(2.0,0.578,0.6),(2.0,0.578,1.2),(4.0,0.42,0.5),(1.414,0.631,0.4)]:
    r=rho_R(grid,t,ps,k); I=np.trapezoid(r,grid)
    j=rho_R(np.array([ps+1e-9]),t,ps,k)[0]-rho_R(np.array([ps-1e-9]),t,ps,k)[0]
    print("  %-6.3f %-6.3f %-8.2f %12.9f %12.6f %12.6f %10.6f"%(t,ps,k,I,j,r.min(),F_R(np.array([1.0]),t,ps,k)[0]))
print("\n  => int rho = 1 holds for EVERY (tau, phi_*, kappa): the mass constraint is automatic,")
print("     exactly as in Thm. 'surrogate self-consistency'. Jump = -kappa. Positivity needs")
print("     kappa < tau / sinh(tau phi_*)  (sufficient; checked above).")
for (t,ps) in [(2.0,0.578),(4.0,0.42),(1.414,0.631)]:
    print("     tau=%.3f phi_*=%.3f  ->  kappa_max = %.3f"%(t,ps,t/np.sinh(t*ps)))

# inverse CDF in closed form (piecewise arcsinh) -- verify by inversion
print("\n  inverse CDF check (piecewise closed form vs numerical inversion):")
t,ps,k=2.0,0.578,0.6
def Phi_R(u,t,ps,k):
    sm=np.sinh(t*ps); sp=np.sinh(t*(1-ps)); S=np.sinh(t)
    uc=F_R(np.array([ps]),t,ps,k)[0]
    out=np.empty_like(u)
    lo=u<uc
    # branch 1: 1 - sinh(t(1-phi))/S + k sp sinh(t phi)/(t S) = u
    #   = A sinh(t phi) + B cosh(t phi) + 1 - ... ; solve as R sinh(t phi + delta)
    a=np.cosh(t)/S + k*sp/(t*S); b=-np.sinh(t)/S           # coeff of sinh(t phi), cosh(t phi)
    R=np.sqrt(abs(a**2-b**2)); dl=np.arctanh(b/a)
    out[lo]=(np.arcsinh((u[lo]-1)/R*1.0+np.sinh(dl)*0)-dl)/t if False else np.nan
    return out,uc
# numerical inversion is enough for the note; verify monotone + invertible
u=np.linspace(1e-6,1-1e-6,200001); Fg=F_R(grid,t,ps,k)
print("     F monotone increasing:", bool(np.all(np.diff(Fg)>=-1e-12)), " F(0)=%.2e F(1)=%.6f"%(Fg[0],Fg[-1]))
phi_of_u=np.interp(u,Fg,grid)
print("     max |F(Phi(u)) - u| = %.2e   (numerical inverse; the analytic branch is R sinh(tau phi + delta)=const)"%np.abs(np.interp(phi_of_u,grid,Fg)-u).max())
import numpy as np
def rho_cosh(p,t): return t*np.cosh(t*(1-p))/np.sinh(t)
def rho_R(p,t,ps,k):
    sm,sp=np.sinh(t*ps),np.sinh(t*(1-ps)); out=rho_cosh(p,t).copy(); lo=p<ps
    out[lo]+=k*sp*np.cosh(t*p[lo])/np.sinh(t); out[~lo]-=k*sm*np.cosh(t*(1-p[~lo]))/np.sinh(t); return out
def Phi_R(u,t,ps,k):
    """analytic piecewise inverse CDF of the forced (EVQ-Cosh-R) density"""
    sm,sp,S=np.sinh(t*ps),np.sinh(t*(1-ps)),np.sinh(t)
    P=1/np.tanh(t)+k*sp/(t*S); R=np.sqrt(P**2-1); dl=np.arctanh(1/P)
    ustar=1+R*np.sinh(t*ps-dl)
    C=(t-k*sm)/S
    out=np.where(u<=ustar,(dl+np.arcsinh((u-1)/R))/t,
                          1-np.arcsinh(np.clip(t*(1-u)/C,-1e12,1e12))/t)
    return out,ustar
# verify against numerical inversion and against the tau->cosh limit
g=np.linspace(0,1,2000001)
for (t,ps,k) in [(2.0,0.578,0.0),(2.0,0.578,0.6),(4.0,0.42,0.5),(1.414,0.631,0.4)]:
    F=np.concatenate([[0],np.cumsum(0.5*(rho_R(g,t,ps,k)[1:]+rho_R(g,t,ps,k)[:-1])*np.diff(g))])
    u=np.linspace(1e-5,1-1e-5,20001); num=np.interp(u,F,g); ana,us=Phi_R(u,t,ps,k)
    ref=1-np.arcsinh((1-u)*np.sinh(t))/t
    print("tau=%.3f phi_*=%.3f kappa=%.2f : max|Phi_ana - Phi_num| = %.2e   u_*=%.4f   (k=0 vs paper warp: %.1e)"
          %(t,ps,k,np.abs(ana-num).max(),us,np.abs(ana-ref).max() if k==0 else float('nan')))
import numpy as np
from scipy.optimize import minimize
def zipf_prior(L,alpha,dmin=1):
    D=np.arange(dmin,L+1,dtype=float); w=D**(-alpha); return D,w/w.sum()
def AB(ts,D,w,CH=800):
    ts=np.atleast_1d(np.asarray(ts,float)); A=np.empty_like(ts); B=np.empty_like(ts)
    for i in range(0,len(ts),CH):
        M=np.outer(ts[i:i+CH],D); A[i:i+CH]=np.cos(M)@w; B[i:i+CH]=np.sin(M)@w
    return A,B
def cbar(om,D,w):
    om=np.asarray(om,float); K=len(om); iu,ju=np.triu_indices(K,1)
    ts=np.concatenate([2*om,np.abs(om[iu]-om[ju]),om[iu]+om[ju]])
    A,B=AB(ts,D,w); a2,b2=A[:K],B[:K]; n=len(iu)
    ad,bd=A[K:K+n],B[K:K+n]*np.sign(om[iu]-om[ju]); asu,bs=A[K+n:],B[K+n:]
    S=np.stack([np.stack([1+a2,b2],-1),np.stack([b2,1-a2],-1)],-2)*0.5
    ev,V=np.linalg.eigh(0.5*(S+np.swapaxes(S,-1,-2)))
    Si=V@(np.maximum(ev,1e-300)[...,None]**-0.5*np.swapaxes(V,-1,-2))
    H=0.5*np.stack([np.stack([ad+asu,bs-bd],-1),np.stack([bs+bd,ad-asu],-1)],-2)
    Q=Si[iu]@H@Si[ju]; return float(np.mean(0.5*np.sum(Q*Q,axis=(-1,-2))))
r2=lambda cb,K:2*K/(1+(K-1)*cb)
NK=16                                   # knots for the density
def dens_from(theta):                   # piecewise-constant rho on NK cells, mass 1
    s=np.exp(theta-theta.max()); return s/s.mean()/NK*NK/ (s/s.mean()).mean() if False else s/ s.mean()
def phis_from(theta,K):
    rho=dens_from(theta); cell=1.0/NK
    F=np.concatenate([[0.0],np.cumsum(rho*cell)]); F=F/F[-1]
    e=np.linspace(0,1,NK+1); u=(np.arange(K)+0.5)/K
    return np.interp(u,F,e)
def Schi2(theta):
    rho=dens_from(theta); return float(np.mean((1-rho)**2/rho))

L,b,K=512,1e3,16; x0=2.0743
ps=np.log(L/x0)/np.log(b); tau=2.0
print("L=%d b=%.0e K=%d  phi_*=%.3f  tau=%.2f"%(L,b,K,ps,tau))
def cosh_phi(K,t):
    u=(np.arange(K)+0.5)/K; return 1-np.arcsinh((1-u)*np.sinh(t))/t
def R_phi(K,t,ps,k):
    u=(np.arange(K)+0.5)/K; sm,sp,S=np.sinh(t*ps),np.sinh(t*(1-ps)),np.sinh(t)
    P=1/np.tanh(t)+k*sp/(t*S); R=np.sqrt(P**2-1); dl=np.arctanh(1/P)
    us=1+R*np.sinh(t*ps-dl); C=(t-k*sm)/S
    return np.where(u<=us,(dl+np.arcsinh((u-1)/R))/t,1-np.arcsinh(t*(1-u)/C)/t)
# reference stiffness = that of the deployed cosh at tau
gg=np.linspace(0,1,20001); rc=tau*np.cosh(tau*(1-gg))/np.sinh(tau)
S0=float(np.trapezoid((1-rc)**2/rc,gg))
print("stiffness budget S0 = S_chi2[rho_tau] = %.5f\n"%S0)
for al in [0.0,1.0,2.0]:
    D,w=zipf_prior(L,al)
    pen=lambda th: cbar(b**(-phis_from(th,K)),D,w) + 50.0*max(0.0,Schi2(th)-S0)**2
    best=None
    for seed in range(3):
        th0=np.zeros(NK) if seed==0 else np.random.default_rng(seed).normal(0,.4,NK)
        r=minimize(pen,th0,method='Powell',options=dict(maxiter=4000,maxfev=4000,xtol=1e-3,ftol=1e-5))
        if best is None or r.fun<best.fun: best=r
    rho=dens_from(best.x); rho=rho/rho.mean()
    cb_opt=cbar(b**(-phis_from(best.x,K)),D,w)
    cb_cosh=cbar(b**(-cosh_phi(K,tau)),D,w)
    kbest=min([(cbar(b**(-R_phi(K,tau,ps,k)),D,w),k) for k in np.linspace(0,1.3,14)])
    print("alpha=%.1f  cbar: geo=%.4f  cosh(tau=2)=%.4f  coshR(k*=%.2f)=%.4f  free-opt=%.4f  [S=%.4f<=%.4f]"
          %(al,cbar(b**(-np.arange(K)/(K-1)),D,w),cb_cosh,kbest[1],kbest[0],cb_opt,Schi2(best.x),S0))
    print("   optimal rho on 16 cells: "+" ".join("%.2f"%v for v in rho))
    m=len(rho)//2
    print("   monotone decreasing? %s   U-shaped (both ends > middle)? %s   argmax cell=%d"
          %(bool(np.all(np.diff(rho)<0.02)), bool(rho[0]>1.15*rho[m] and rho[-1]>1.15*rho[m]), int(np.argmax(rho))))
