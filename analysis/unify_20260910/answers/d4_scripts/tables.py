import math
W,S,L=32768,4,131072
LN4=math.log(4); NB=math.log(1e6)/64
def om(j): return 10**(-3*j/32)
def ramp(n,j):
    q=max(0,min(n,j-23)); return q*(q+1)/(n*(n+1))
def uni(n,j): return min(1,max(0,(j-23)/n))
for fam,fn in [('RAMP',(lambda n:lambda j:ramp(n,j))),('UNIFORM',(lambda n:lambda j:uni(n,j)))]:
    for n in ([17,16,15] if fam=='RAMP' else [17,16]):
        f=fn(n)
        print(f"--- {fam} N'={n} ---")
        print(" j      m_j       nu_j          D_j        eps_next   lam_next   rho_j")
        for j in range(23,41):
            m=f(j); nu=om(j)*4**(-m); D=W*4**m
            e=f(j+1)-m if j<63 else 0.0
            lam=4**e; rho=math.exp(NB+LN4*e)
            print(f"{j:3d}  {m:.6f}  {nu:.6e}  {D:9.0f}  {e:.6f}  {lam:.6f}  {rho:.4f}")
# transport quantities
print("=== transport ===")
for n_new in (16,15):
    n_old=17
    old_last=ramp(n_old,n_old)-ramp(n_old,n_old-1)   # eps at gap g=22+n = 39 for n=17
    new_last=ramp(n_new,n_new)-ramp(n_new,n_new-1)
    # per-gap delta: gap g=23+q, eps_q = 2q/(n(n+1))
    tot=0.0
    for q in range(1,n_new+1):
        e_o=2*q/(n_old*(n_old+1)); e_n=2*q/(n_new*(n_new+1))
        m_o=q*(q+1)/(n_old*(n_old+1)); m_n=q*(q+1)/(n_new*(n_new+1))
        # budget at position j=23+q: m difference
    # simpler: total moved = m_{j*-1} difference (mass removed from last gap)
    # Actually compare m_j at fixed slots: shift of completion boundary.
    # per-slot mass diff over j=24..(j*-1):
    diffs=[f"j={23+q}: {ramp(n_new,23+q)-ramp(n_old,23+q):+.6f}" for q in range(1,18)]
    print(f"17->{n_new} per-slot dm:", diffs[:4], "...", diffs[-3:])
# uniform flat eps
for n in (13,14,15,16,17): print(f"uniform n={n}: eps=1/n={1/n:.6f} lam={4**(1/n):.6f}")
for n in (15,16,17): print(f"ramp n={n}: eps_q=q/{n*(n+1)/2:.0f}  eps_last=2/{n+1}={2/(n+1):.6f} lam_last={4**(2/(n+1)):.6f} rho_last={math.exp(NB+LN4*2/(n+1)):.4f}")
