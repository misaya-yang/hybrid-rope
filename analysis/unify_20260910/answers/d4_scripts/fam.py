import math
LN4=math.log(4); W=32768; L=131072; RN=10**(6/64)  # native rho = b^{1/64}
RN=math.exp(math.log(1e6)/64)
print("native rho", RN)
def ramp(n):
    m=[0.0]*64
    for j in range(64):
        q=max(0,min(j-23,n)); m[j]=q*(q+1)/(n*(n+1))
    return m
def unif(n):
    m=[0.0]*64
    for j in range(64): m[j]=max(0,min(1,(j-23)/n))
    return m
def feats(m):
    dg=[m[g+1]-m[g] for g in range(63)]
    Farc=sum(max(0,1-m[j]) for j in range(36,40))
    bank=max(dg[23:29]); mid=max(dg[29:36]); term=dg[38]+dg[39]  # hole bands
    rmax=max(dg[23:40]); argmax=dg[23:40].index(rmax)+23
    m24=dg[23]; m28=sum(dg[23:28])
    return dict(Farc=Farc,bank=bank,mid=mid,rmax=rmax,argmax=argmax,m24=m24,BL=sum(m[24:29]))
print("\n== ramp family n=13..17 (all in m-units; 1 m-unit = ln4 nats) ==")
print("n  F_arc   eps_bankmax(g23-28) eps_midmax(g29-35) eps_g36-39 rho_max@pos  m24  BL")
for n in range(13,18):
    f=feats(ramp(n)); m=ramp(n); dg=[m[g+1]-m[g] for g in range(63)]
    right=sum(dg[36:40])
    print(f"{n} {f['Farc']:.4f}  {f['bank']:.4f}  {f['mid']:.4f}  {right:.4f}  {RN*4**f['rmax']:.3f}@g{f['argmax']}  {m[24]:.4f} {f['BL']:.4f}")
print("== uniform family ==")
for n in range(13,18):
    f=feats(unif(n)); m=unif(n); dg=[m[g+1]-m[g] for g in range(63)]
    right=sum(dg[36:40])
    print(f"{n} {f['Farc']:.4f}  {f['bank']:.4f}  {f['mid']:.4f}  {right:.4f}  {RN*4**f['rmax']:.3f}@g{f['argmax']}  {m[24]:.4f} {f['BL']:.4f}")

# (i) literal min-max, position-blind, m-units equal: F_hole = max bridge delta (1/n form), F_arc
print("\n== min-max position-blind: R(n)=max(F_hole, F_arc) ==")
for fam,fn in (('unif',unif),('ramp',ramp)):
    rows=[]
    for n in range(13,18):
        m=fn(n); dg=[m[g+1]-m[g] for g in range(63)]
        Fh=max(dg[23:40]); Fa=feats(m)['Farc']
        rows.append((n,Fh,Fa,max(Fh,Fa)))
    print(fam, [(n,round(a,4),round(b,4),round(c,4)) for n,a,b,c in rows])
# (ii) ceiling version: bands must not exceed tested-damage levels:
CEIL_BANK=0.1307   # pair g28 level
CEIL_MID =0.1130   # Smooth g35 level (mid-band damage)
CEIL_TERM=0.1919   # LBF g39 level
print("\n== ceiling version (ramp): violations vs tested-damage levels ==")
for n in range(13,18):
    m=ramp(n); dg=[m[g+1]-m[g] for g in range(63)]
    bank=max(dg[23:29]); mid=max(dg[29:36]); term=max(dg[36:40])
    viol=(bank>CEIL_BANK)+(mid>CEIL_MID)+(term>CEIL_TERM)
    Farc=feats(m)['Farc']
    print(f"n={n} bank{bank:.4f} mid{mid:.4f} term{term:.4f} viol={viol} Farc={Farc:.4f}")
# (iii) theta scan: R_theta = max(theta/(1-theta) * F_arc_norm, H_norm) with H_norm = max(bank/CEIL_BANK, mid/CEIL_MID), A_norm=Farc/Farc(MrPro)
A0=feats(ramp(17))['Farc']
print("\n== theta scan (ramp): R=min-max over n, s=short-weight ==")
print("s    "+" ".join(f"n={n}" for n in range(13,18))+"  argmin")
import numpy as np
for s in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    vals=[]
    for n in range(13,18):
        m=ramp(n); dg=[m[g+1]-m[g] for g in range(63)]
        H=max(max(dg[23:29])/CEIL_BANK, max(dg[29:36])/CEIL_MID)
        A=feats(m)['Farc']/A0
        vals.append(max(s*H,(1-s)*A))
    k=min(range(5),key=lambda i:vals[i])
    print(f"{s:.1f}  "+" ".join(f"{v:.3f}" for v in vals)+f"   n={13+k}")
# transport numbers n=17->16, 16->15
for a,b in ((17,16),(16,15),(17,15)):
    ma=ramp(a); mb_=ramp(b)
    dga=[ma[g+1]-ma[g] for g in range(63)]; dgb=[mb_[g+1]-mb_[g] for g in range(63)]
    moved=sum(max(0,dgb[g]-dga[g]) for g in range(23,40))  # mass gained by finer gaps
    print(f"\ntransport {a}->{b}: m-units redistributed={moved:.4f} = {moved*LN4:.4f} nats; new completion slot={23+b}")
# recommended table n=16 & 15: full m, lambda, D, rho
for n in (17,16,15):
    m=ramp(n); dg=[m[g+1]-m[g] for g in range(63)]
    print(f"\n== ramp N'={n}: q | m | eps | lam=4^eps | delta_gap(nats) | rho | D_j ==")
    for j in range(23,41):
        q=max(0,min(j-23,n)); mm=q*(q+1)/(n*(n+1))
        D=W*4**mm
        if j<40:
            e=dg[j]; lam=4**e
            print(f"j={j} m={mm:.6f} D={D:,.0f} | gap_j: eps={e:.6f} lam={lam:.6f} {e*LN4:.6f}nats rho={RN*lam:.4f}")
        else:
            print(f"j={j} m={mm:.6f} D={D:,.0f}")
