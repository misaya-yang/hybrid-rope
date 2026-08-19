import numpy as np
TWO_PI=2*np.pi
def geo(b,dh): return b**(-(2*np.arange(dh//2)/dh))
def wceil(eps,L): return np.arccos(np.clip(1-eps,-1,1))/L
BASE,DH,L,M=500_000.,128,4096,32_768
om=geo(BASE,DH); lam=TWO_PI/om

print("### R6: does BPB eps=0.05 actually buy ruler slots?")
for eps in [0.02,0.05,0.10]:
    lf=TWO_PI/wceil(eps,L); o2=om.copy(); l2=TWO_PI/o2
    idx=np.where(l2>lf)[0]
    if len(idx):
        tgt=np.exp(np.linspace(np.log(lf),np.log(l2[idx].max()),len(idx))); o2[idx]=TWO_PI/tgt
    l2=TWO_PI/o2
    f=lambda x:( (x>=2*M).sum(), ((x>=2*M)&(x<=4*M)).sum(), (TWO_PI/x[x>=2*M]).max()*M if (x>=2*M).any() else 0)
    a,b,c=f(lam); d,e,g=f(l2)
    print(f"  eps={eps}: lam>=2M {a}->{d} | in [2M,4M] {b}->{e} | best ruler sweep@32K "
          f"{c:.3f}->{g:.3f} rad | min log-gap {np.diff(np.sort(np.log(om))).min():.4f}"
          f"->{np.diff(np.sort(np.log(o2))).min():.4f}")
print("  Native channels already in [2M,4M]:",
      [f"{x:.0f}(sweep {TWO_PI/x*M:.3f}r)" for x in np.sort(lam[(lam>=2*M)&(lam<=4*M)])])

print("\n### R12: the eps bound covers cos only; what does sin do?")
eps=0.05; lf=TWO_PI/wceil(eps,L); o2=om.copy(); l2=TWO_PI/o2
idx=np.where(l2>lf)[0]; tgt=np.exp(np.linspace(np.log(lf),np.log(l2[idx].max()),len(idx))); o2[idx]=TWO_PI/tgt
D=np.linspace(0,L,4096)
dc=np.abs(np.cos(np.outer(o2,D))-np.cos(np.outer(om,D))).max()
ds=np.abs(np.sin(np.outer(o2,D))-np.sin(np.outer(om,D))).max()
print(f"  worst |dcos| = {dc:.4f}   worst |dsin| = {ds:.4f}   ratio {ds/dc:.1f}x"
      f"   (analytic sin scale ~ w'L = sqrt(2*eps) = {np.sqrt(2*eps):.3f})")

print("\n### Ra: do 'dead' channels really do nothing positional?")
dead=lam>M
for Dm,nm in [(L,'in-window 4K'),(M,'at 32K')]:
    sw=(1-np.cos(np.outer(om[dead],np.array([Dm])))).sum()
    print(f"  sum_k (1-cos(w_k D)) over the {dead.sum()} dead channels at D={Dm:6d}: {sw:7.3f}"
          f"  -> /sqrt(d_head) = {sw/np.sqrt(DH):.3f} logit units")
print("  => they implement a smooth monotone recency kernel.  NOT invariant under softmax,")
print("     because the contribution varies with the KEY index, not just the query.")

print("\n### R19: the exact-range seed-42 experiment ran at base 256, L_train 256")
b2,dh2=256.,64
o3=geo(b2,dh2); l3=TWO_PI/o3
print(f"  base=256, K={dh2//2}: lambda in [{l3.min():.1f}, {l3.max():.1f}];  eval ceiling 2048")
print(f"  channels with lambda > 2048: {(l3>2048).sum()}/{dh2//2}  -> NO dead band exists there.")
print("  => 'retargeting harvested the dead band' cannot explain that experiment's reversal.")
