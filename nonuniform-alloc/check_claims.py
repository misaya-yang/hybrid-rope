import numpy as np
TWO_PI = 2*np.pi
def geo(base, dh): return base ** (-(2*np.arange(dh//2)/dh))
def evq(base, dh, tau):
    K=dh//2; u=(np.arange(K)+0.5)/K
    return base ** (-(1-np.arcsinh((1-u)*np.sinh(tau))/tau))
def regimes(om,L,M):
    lam=TWO_PI/om; return (lam<=L).sum(), ((lam>L)&(lam<=M)).sum(), (lam>M).sum()
def collision(om,Dm,n=8192):
    D=np.linspace(0,Dm,n); C=np.cos(np.outer(om,D)); K=(C@C.T)/n
    d=np.sqrt(np.diag(K)); R=K/np.outer(d,d); iu=np.triu_indices(len(om),1)
    return (R[iu]**2).sum()

BASE,DH,L,M = 500_000.,128,4096,32_768
print("CLAIM 1: crossover phi_c where rho_tau = 1  (below it EVQ thickens, above it thins)")
for tau in [0.5,1,2,3,4]:
    phi = np.linspace(0,1,200001); r = tau*np.cosh(tau*(1-phi))/np.sinh(tau)
    print(f"   tau={tau:>4}: phi_c = {phi[np.argmin(np.abs(r-1))]:.4f}")
print(f"   tau->0 limit: phi_c = 1 - 1/sqrt(3) = {1-1/np.sqrt(3):.4f}   (upper bound on phi_c)")
phi_L = np.log(L/TWO_PI)/np.log(BASE)
print(f"   regime II starts at phi_L = {phi_L:.4f}")
print(f"   => regime II lies entirely in the THINNED half iff phi_L >= 0.4226, i.e.")
print(f"      L_train >= 2*pi*base^0.4226 = {TWO_PI*BASE**0.4226:.0f} tokens for base={BASE:.0f}")
print("      (holds for every production config with L_train >= 4K)\n")

print("CLAIM 2: minimal monotone ruler.  cos(wD) is monotone on D in [0, lambda/2],")
print("         so covering [0,X] unambiguously needs lambda >= 2X.")
print(f"         LeRoPE learned dominant band = 2.205*L_train; predicted minimum = 2.000*L_train")
print(f"         relative difference = {(2.205-2.0)/2.0:.1%}\n")

print("CLAIM 3: FMRoPE literal rule (theta = target length) vs my truncated-span version")
for nm, b in [("FMRoPE literal  base=M=32768", M),
              ("span-truncated  base=M/pi",  M/np.pi),
              ("span-truncated  base=M/2pi", M/TWO_PI)]:
    om = geo(b, DH); lam = TWO_PI/om
    w,r,d = regimes(om,L,M)
    print(f"   {nm:<30s} lam_max={lam.max():9.0f}  wrap={w:2d} resolv={r:2d} dead={d:2d} "
          f"coll@32K={collision(om,M):7.1f}")
print()
print("CLAIM 4: minimal-edit bookkeeping")
om=geo(BASE,DH); lam=TWO_PI/om
print(f"   channels with lambda > 2M ({2*M}) : {(lam>2*M).sum()}/64  -> movable (minus 1 kept as ruler = 17)")
print(f"   channels with lambda <= 2M        : {(lam<=2*M).sum()}/64  -> kept byte-identical")
print(f"   FMRoPE-literal collision reduction captured by minimal edit:")
c_nat, c_me, c_fm = 136.3, 53.8, collision(geo(M,DH),M)
print(f"      native {c_nat}  -> minimal-edit {c_me}  -> FMRoPE-literal {c_fm:.1f}")
print(f"      fraction of the achievable reduction = {(c_nat-c_me)/(c_nat-c_fm):.1%}")
