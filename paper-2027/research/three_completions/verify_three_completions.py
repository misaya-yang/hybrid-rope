#!/usr/bin/env python3
"""Reproduces every number in evq_three_completions.tex.  Pure NumPy/SciPy, ~90 s on CPU.

Sections:
  A  Part I   -- Q1 closed form, universal x0, exact self-consistent tau, constant-free rule
  B  Part III -- collapse coefficient Lambda(mu), verified against 19/12600
  C  Part III -- r2 under power-law priors, overload turnover, prior-side ceiling
  D  Part II  -- transplant mismatch spectra and LoRA rank floors
  E  Part II  -- lemma verification (block spectrum, rank-2r residual, frozen subspace, Weyl)
"""
import numpy as np
from scipy.optimize import brentq

# ------------------------------------------------------------------ shared ---
def q(x):
    x = np.asarray(x, float); out = np.empty_like(x); sm = np.abs(x) < 1e-6
    out[sm] = x[sm]**4/45.0; xs = x[~sm]
    out[~sm] = 0.5 + np.sin(2*xs)/(4*xs) - (np.sin(xs)/xs)**2
    return out
a_pert   = lambda p: (1-p)**2/2 - 1/6
rho_tau  = lambda p,t: np.ones_like(p) if t < 1e-8 else t*np.cosh(t*(1-p))/np.sinh(t)
def S_tilde(t):
    t=float(t)
    return t**4/45.0 if t < 1e-6 else np.sinh(t)*np.arctan(np.sinh(t))/t**2 - 1.0
def W(t,L,b,n=20001):
    p=np.linspace(0,1,n); return np.trapezoid(rho_tau(p,t)*q(L*b**(-p)), p)
dnum = lambda f,x,h=1e-4: (f(x+h)-f(x-h))/(2*h)
Psi  = lambda t,L,b: dnum(S_tilde,t)/dnum(lambda u: W(u,L,b), t)
def Q0Q1(L,b,n=200001):
    p=np.linspace(0,1,n); qq=q(L*b**(-p))
    return np.trapezoid(qq,p), np.trapezoid(a_pert(p)*qq,p)
def geo(K,b,mid=False):
    u=(np.arange(K)+0.5)/K if mid else np.arange(K)/K; return b**(-u)
def evq(K,b,tau,mid=True):
    u=(np.arange(K)+0.5)/K if mid else np.arange(K)/K
    return b**(-(1-np.arcsinh((1-u)*np.sinh(tau))/tau))
def zipf_prior(L,alpha,dmin=1):
    D=np.arange(dmin,L+1,dtype=float); w=D**(-alpha); return D, w/w.sum()
def charfun(ts,D,w,CH=200):
    ts=np.asarray(ts,float); A=np.empty_like(ts); B=np.empty_like(ts)
    for i in range(0,len(ts),CH):
        M=np.outer(ts[i:i+CH],D); A[i:i+CH]=np.cos(M)@w; B[i:i+CH]=np.sin(M)@w
    return A,B

CFG=[("50M diagnostic",64,512,5e5,2.83),("151.9M fixed-support",64,256,256.,4.00),
     ("432M MLA (d_rot=32)",32,8192,5e5,0.354),("750M continuation",64,4096,5e5,1.00),
     ("OLMo-2 1.485B",128,4096,5e5,2.00),("LLaMA-3-8B",128,8192,5e5,1.414)]

# =========================================================== A: Part I =======
print("="*84); print("A.  PART I -- constant-free tau"); print("="*84)
def I(x0):
    xs=np.geomspace(1e-8,x0,40000); v1=np.trapezoid(q(xs)/xs,xs)
    xs=np.geomspace(x0,1e7,4000000); v2=np.trapezoid((q(xs)-0.5)/xs,xs)
    return v1+v2
x0=brentq(I,0.5,8.0,xtol=1e-6)
print(f"universal resolution threshold  x0 = {x0:.4f}   (paper value 2.0743)")
g    = lambda p: p*(1-p)*(2-p)
gmax = 2/(3*np.sqrt(3)); phimax = 1-1/np.sqrt(3)
phistar = lambda L,b: min(1.0,max(0.0,np.log(L/x0)/np.log(b)))
print(f"g_max = 2/(3 sqrt3) = {gmax:.6f} at phi_max = 1-1/sqrt3 = {phimax:.6f}\n")
lam0 = 12/(45*gmax)                       # zero-constant convention (c=1 at phi_max)
lamC = 1.0/(45*Q0Q1(4096,5e5)[1])         # tex Table 3 convention (c=1 at L=4096,b=5e5)
print(f"lambda(zero-constant) = {lam0:.4f}   lambda(Table-3 calibration) = {lamC:.4f}\n")
print(f"{'config':22s} {'phi_*':>6s} {'Q1 exact':>9s} {'Q1 closed':>9s} {'err%':>6s} "
      f"{'c(Pi)':>7s} {'tau pred':>8s} {'tau dep':>8s} {'tau exact':>9s}")
for n,d,L,b,t in CFG:
    _,Q1=Q0Q1(L,b); ps=phistar(L,b); Q1c=g(ps)/12; c=np.sqrt(g(ps)/gmax)
    te=brentq(lambda u: Psi(u,L,b)-2*lamC*d**2/L, 1e-3, 20.0, xtol=1e-8)
    print(f"{n:22s} {ps:6.3f} {Q1:9.5f} {Q1c:9.5f} {100*(Q1c-Q1)/Q1:6.1f} "
          f"{c:7.4f} {c*d/np.sqrt(L):8.3f} {t:8.3f} {te:9.3f}")
print("\nkappa_att audit (research/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md), L_train=512, d_eff=64:")
for n,k in [("Geo/Geo",4.6476e-4),("Geo/EVQ",2.53131e-3),("EVQ/Geo",1.14364e-3),("EVQ/EVQ",5.3535e-4)]:
    Le=1/k; print(f"   {n:9s} kappa={k:.5e}  L_eff={Le:7.1f}  L_eff/L={Le/512:5.2f}  tau*=d/sqrt(L_eff)={64/np.sqrt(Le):.3f}")
print(f"   => contractive fixed point tau^sc ~ 1.43 = {1.43/2.83:.2f} x the deployed rule")

# ========================================================== B: Lambda(mu) ====
print("\n"+"="*84); print("B.  PART III -- collapse coefficient Lambda(mu)"); print("="*84)
def Lambda_of_prior(D,w):
    w=w/w.sum(); ip=lambda f,h: float(np.sum(w*f*h))
    B=[np.ones_like(D),D]; G0=np.array([[ip(B[i],B[j]) for j in range(2)] for i in range(2)])
    Gi=np.linalg.inv(G0)
    def res(f):
        be=Gi@np.array([ip(B[0],f),ip(B[1],f)]); return f-be[0]*B[0]-be[1]*B[1]
    U=[-0.5*res(D**2),-(1/6)*res(D**3)]
    M=np.array([[ip(U[i],U[j]) for j in range(2)] for i in range(2)])
    return float(np.trace(Gi@M))
n=400001; D=np.linspace(0,1,n); w=np.ones(n)
print(f"Lambda(Unif[0,1]) = {Lambda_of_prior(D,w):.10f}    19/12600 = {19/12600:.10f}")
print("power-law priors (L=4096), Lambda/L^4 relative to uniform:")
for al in [0.0,0.5,1.0,1.5,2.0,2.5,3.0]:
    Dz,wz=zipf_prior(4096,al)
    print(f"   alpha={al:.1f}  ratio={Lambda_of_prior(Dz/4096,wz)/(19/12600):7.3f}"
          f"   sqrt(E[D^2])/L={np.sqrt(float(np.sum(wz*(Dz/4096)**2))):.4f}")

# ========================================================== C: r2 / overload =
print("\n"+"="*84); print("C.  PART III -- r2 under power-law priors, overload turnover"); print("="*84)
def cbar_r2(om,D,w,subset=None):
    om=np.asarray(om,float); idx=om if subset is None else om[subset]; K=len(idx)
    pr=[(i,j) for i in range(K) for j in range(i+1,K)]
    ts=list(2*idx)+[v for (i,j) in pr for v in (abs(idx[i]-idx[j]), idx[i]+idx[j])]
    A,B=charfun(np.array(ts),D,w)
    def isq(M):
        ev,V=np.linalg.eigh((M+M.T)/2); return V@np.diag(np.maximum(ev,1e-300)**-.5)@V.T
    Si=[isq(0.5*np.array([[1+A[i],B[i]],[B[i],1-A[i]]])) for i in range(K)]
    cs=[]
    for n_,(i,j) in enumerate(pr):
        ad,bd=A[K+2*n_],B[K+2*n_]*(1.0 if idx[i]>=idx[j] else -1.0)
        as_,bs=A[K+2*n_+1],B[K+2*n_+1]
        Q=Si[i]@(0.5*np.array([[ad+as_,bs-bd],[bs+bd,ad-as_]]))@Si[j]
        cs.append(0.5*np.sum(Q*Q))
    cb=np.mean(cs); return cb, 2*K/(1+(K-1)*cb), np.array(cs)
L,b,K=4096,5e5,64
taus=[0.0,1,2,3,3.5,4,5,6,8,12]
print(f"{'alpha':>6s} "+"".join(f"{('tau='+str(t)):>8s}" for t in taus)+"   argmax")
for al in [0.0,0.5,1.0,1.5,2.0,2.5,3.0]:
    Dz,wz=zipf_prior(L,al)
    r=[cbar_r2(geo(K,b,True) if t==0 else evq(K,b,t),Dz,wz)[1] for t in taus]
    print(f"{al:6.1f} "+"".join(f"{v:8.2f}" for v in r)+f"   tau*={taus[int(np.argmax(r))]}")
print("\nslow subset (omega L <= 1), the paper's r2=2.00 claim:")
for al in [0.0,1.0,2.0]:
    Dz,wz=zipf_prior(L,al); om=geo(K,b,False); sub=np.where(om*L<=1.0)[0]
    cb,r2,_=cbar_r2(om,Dz,wz,subset=sub)
    print(f"   alpha={al:.1f}  n_slow={len(sub)}  cbar={cb:.5f}  r2={r2:.3f}")

# ========================================================== D: rank floors ===
print("\n"+"="*84); print("D.  PART II -- transplant mismatch and LoRA rank floors"); print("="*84)
Le=8192; Dz,wz=zipf_prior(Le,0.0); nat=geo(64,5e5,False)
def spec(omp):
    A,_=charfun(np.abs(nat-omp),Dz,wz); return np.sort(2*(1-A))[::-1]
def rneed(s,f):
    return int(np.ceil((np.searchsorted(np.cumsum(s)/s.sum(),f)+1)/2))
cases=[("Native -> EVQ(1.414)",evq(64,5e5,1.414)),("Native -> EVQ(4)",evq(64,5e5,4.0)),
       ("base 5e5 -> 5e6",geo(64,5e6,False)),("PI: all omega/4",nat/4.0),
       ("YaRN-style: slow(omega L<1)/4",np.where(nat*Le<1.0,nat/4.0,nat))]
print(f"{'intervention':32s} {'||D||_F^2':>10s} {'r(50%)':>7s} {'r(90%)':>7s} {'r(99%)':>7s} {'rigorous':>9s}")
for n_,omp in cases:
    s=spec(omp)
    print(f"{n_:32s} {2*s.sum():10.2f} {rneed(s,.5):7d} {rneed(s,.9):7d} {rneed(s,.99):7d} {s.sum()/8:9.2f}")
print("   (rigorous = (1-eta) sum_k s_k / 8 at eta=0; exact-repair floor is r >= K = 32 for d_rot=64,")
print("    r >= 64 for d_rot=128 -- the deployed LoRA rank 64 sits exactly on it)")

# ========================================================== E: lemma checks ==
print("\n"+"="*84); print("E.  PART II -- lemma verification"); print("="*84)
rng=np.random.default_rng(1); Rot=lambda t: np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
Rblk=lambda om,Dl: np.block([[Rot(o*Dl) if i==j else np.zeros((2,2)) for j in range(len(om))] for i,o in enumerate(om)])
a_,b_=rng.normal(size=2)*3
sv=np.linalg.svd(Rot(a_)-Rot(b_),compute_uv=False)
print(f"  block spectrum: sv={sv[0]:.12f},{sv[1]:.12f}  2|sin((a-b)/2)|={2*abs(np.sin((a_-b_)/2)):.12f}")
Kv=8; dd=2*Kv; om=np.sort(rng.uniform(.05,1,Kv))[::-1]; omp=om*(1+.1*rng.normal(size=Kv))
R,Rp=Rblk(om,1.7),Rblk(omp,1.7)
for r in [1,2,3]:
    dA=rng.normal(size=(dd,r))@rng.normal(size=(r,dd)); dB=rng.normal(size=(dd,r))@rng.normal(size=(r,dd))
    A_,B_=np.eye(dd)+dA,np.eye(dd)+dB; E=A_.T@Rp@B_-Rp
    N=np.linalg.svd(np.vstack([dA,dB]))[2][2*r:].T
    print(f"  r={r}: rank(A^T R' B - R')={np.linalg.matrix_rank(E,tol=1e-8)} (<=2r={2*r});"
          f"  dim N={N.shape[1]} (>=d-2r={dd-2*r});  frozen-subspace err={np.abs(N.T@(A_.T@Rp@B_)@N-N.T@Rp@N).max():.1e}")
    Nb=np.linalg.svd(rng.normal(size=(2*r,dd)))[2][2*r:].T; P=Nb@Nb.T
    s1,s0=np.linalg.svd(P@(R-Rp)@P,compute_uv=False),np.linalg.svd(R-Rp,compute_uv=False)
    print(f"       Weyl compression sigma_j(PDP)>=sigma_(j+4r)(D): "
          f"{all(s1[j]>=s0[j+4*r]-1e-9 for j in range(len(s0)-4*r))}")
print(f"  rank(R_Omega - R_Omega') at generic Delta = {np.linalg.matrix_rank(R-Rp,tol=1e-9)} = 2K = {2*Kv}  => r >= K")
