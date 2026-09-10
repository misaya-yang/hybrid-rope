# Astra02 — Exact finite-window EVQ has an atomic equilibrium, not a corrected Cosh ODE

Status: mathematical derivation and modest CPU demonstration complete; full-text ingestion of all 108 assigned files is complete and the coverage receipt is authoritative. No GPU/model execution or deployment table was created. This report does not assert downstream superiority.

## 1. The missing finite-window assumption

The exact collision objective in `docs/theory/EVQ_COSH_THEORY.tex:81-105` is a bounded analytic Gram kernel. Its replacement by a delta ridge plus the min kernel (`:109-125`) changes the admissible optimizer qualitatively. The delta term is not merely a harmless approximation to a smooth density optimum: it adds an L2 anti-concentration cost that the exact finite-window objective does not possess.

The correction note correctly retains the nonlocal logarithmic ridge only away from its finite-width diagonal and boundaries (`docs/research/EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md:8-114`). Restoring just that singular ridge is still different from restoring the exact kernel: the latter is finite on the diagonal. This report solves the variational structure of the exact problem.

Let I=[a,b] with 0<a<b<infinity, 1<L<infinity, p(t)=1/(t log L) on [1,L], and q>=0. For a Borel probability measure mu on I define

    f_mu(t) = integral_I cos(omega t) dmu(omega)
    K(omega,nu) = integral_1^L p(t) cos(omega t) cos(nu t) dt
    E_q(mu) = (1/2) integral_1^L p(t) f_mu(t)^2 dt - q integral_I omega^2 dmu(omega).

The standard EVQ support is a=1/base, b=1. Transforming omega=base^(-phi) is a continuous bijection and does not change the measure statements below. Pure collision has q=0; q is the Fisher coefficient when retained.

### Theorem: unique finite atomic equilibrium

For the stated continuous distance prior and compact positive nondegenerate frequency interval, E_q has a unique minimizer mu*. Its support is finite. In particular, no positive continuous density, including Cosh, minimizes the exact objective.

**Existence.** Probability measures on compact I are weakly compact. K and omega^2 are continuous and bounded on I and I^2, so E_q is weakly continuous. A minimizer exists.

**Strict convexity.** For any nonzero finite signed measure eta on I,

    integral integral K d_eta d_eta = integral_1^L p(t) f_eta(t)^2 dt > 0.

To prove strict positivity, if the right side vanishes, continuity and p>0 imply f_eta=0 throughout [1,L]. Since eta has compact support, f_eta(z) is an entire function of complex z. The identity theorem gives f_eta=0 everywhere. Let eta_even be the measure assigning half eta to positive frequencies and half its reflection to negative frequencies. Its Fourier transform is f_eta. Fourier-transform uniqueness for finite measures gives eta_even=0. Because I and -I are disjoint, eta=0, a contradiction. This is why positive frequency support is stated explicitly: allowing both signs would introduce the cosine omega/-omega degeneracy. Including zero alone can be handled separately, but is unnecessary for RoPE and is not assumed here. The linear Fisher term does not alter strict convexity. Consequently the minimizer is unique.

**KKT potential.** Define

    V_mu(omega) = integral K(omega,nu) dmu(nu) - q omega^2,
    c_mu = integral V_mu dmu.

The directional derivative toward any point mass delta_omega is V_mu(omega)-c_mu. At the optimum,

    V_mu*(omega) >= c_mu* for all omega in I,
    V_mu*(omega) = c_mu* on supp(mu*).

The second assertion first holds mu*-almost everywhere; continuity promotes it to the topological support. Conversely these inequalities suffice by convexity, since E_q(nu)-E_q(mu)>=integral V_mu d(nu-mu)>=0.

**Finiteness.** V_mu(z) is entire in complex frequency z because the t interval is bounded. If supp(mu*) were infinite, compactness would give an accumulation point in I, including possibly an endpoint; endpoints are interior points of the complex analyticity domain. The identity theorem would force V_mu*(z)=c_mu* for every z.

Write V_mu(omega)=F_mu(omega)-q omega^2 with F_mu(omega)=integral_1^L p(t)f_mu(t)cos(omega t)dt. Riemann-Lebesgue gives F_mu(omega)->0 as real omega->infinity. If q>0, V_mu cannot be constant. If q=0, the constant must be zero. The zero cosine transform of the integrable compactly supported function p f_mu then implies p f_mu=0 a.e.; analyticity of f_mu gives f_mu(0)=0, contradicting f_mu(0)=mu(I)=1. Thus V_mu-c_mu is a nonzero entire function, has only finitely many zeros on I, and supp(mu*) is a finite subset of those zeros. QED.

This is exact mathematics of the declared collision model, not a discovery that a language model ought to collapse its channel table.

### There is no universal bound on the atom count as L changes

The theorem is for each finite L and does not bound support size by 2, 3, K, or another universal number. For q=0 and fixed 0<a<b, one can show that the minimum necessary atom count is unbounded as L increases.

Put A_L(z)=integral_1^L cos(zt)/t dt. For z in [0,b-a], A_L(z) has a lower bound independent of L: for zL<=1 the integrand stays positive; otherwise split the cosine integral at 1, whose oscillatory tail is bounded below. For z in [2a,2b], A_L(z) is uniformly bounded. It follows from the product identity that K_L(omega,nu)>=-C/log L uniformly, and K_L(omega,omega)=1/2+O(1/log L) uniformly for omega in I. Any measure with <=M atoms therefore has

    E_0(mu) >= 1/(4M) - C'/log L,

using sum weights^2>=1/M. A fixed set of M+1 distinct frequencies with equal masses has energy approaching 1/[4(M+1)], because all off-diagonal entries tend to zero and diagonals tend to 1/2. For sufficiently large L it beats every <=M-atom measure. Thus no L-independent atom-count bound exists. The constants can depend on the fixed interval; this is not a joint a->0 limit.

## 2. Actual constructive finite-window solution and certificate

The measure problem admits support exchange, without a regularizer or a candidate-frequency grid as a modeling assumption:

1. Maintain finitely many frequencies and nonnegative masses summing to one.
2. Solve the restricted strictly convex weight QP exactly (or to a stated numerical tolerance).
3. Find the global minimum omega_new of the continuous potential V_mu on I.
4. If g(mu)=c_mu-min_I V_mu is small, stop. Convexity proves

       0 <= E_q(mu)-E_q(mu*) <= g(mu).

5. Otherwise add omega_new, reoptimize masses, and optionally refine existing support positions. Every exact exchange step has a descent direction.

For fixed support x_i and active positive masses, solve the linear KKT system

    [ K_AA  -1 ] [ p_A ] = [ q x_A^2 ]
    [ 1^T    0 ] [  c  ]   [     1     ],

then verify all masses and excluded reduced gradients. Position refinement satisfies V'(x_i)=0 at interior atoms and one-sided inequalities at endpoint atoms. These finite equations and the continuous dual inequality replace the Cosh ODE. Endpoint atoms are allowed; there is no Neumann condition on a nonexistent smooth density.

### CPU exhibit (one modest interval, not a deployment search)

Declared before solving: I=[0.1,1], base=10, L=8, q=0. Numpy-only implementation used 96-point Gauss-Legendre integration over [1,8]. It began at the single atom 0.55, refined support coordinates and weights, and added the most violating potential point. Restricted weight QPs were solved by enumerating their tiny active subsets, not by scanning allocation candidates. The sequence was:

| Support size | E | Numerical dual gap |
|---:|---:|---:|
| 1 | .183907229150896 | .458261951556866 |
| 2 | .058949724094609 | .004252824362431 |
| 3 | .058639327401539 | 4.2e-17 |

Final frequencies: [0.1, 0.4066617223165344, 1.0].
Final masses: [0.0987464879705222, 0.4002375621647298, 0.5010159498647481].

For the global potential check, a 10001-node uniform frequency mesh was used only as a numerical certificate. Since |V''|<=E_p[t^2]=(L^2-1)/(2 log L), piecewise-linear interpolation bounds the continuous minimum below by min_mesh V - E[t^2] h^2/8. Hence the computed global energy-gap upper bound is 1.53377e-8 plus quadrature/floating error. This is an analytic interpolation certificate with numerically evaluated nodes, not an interval-arithmetic proof of every floating operation.

Repeating at 192 quadrature nodes changed the optimum energy by 4.7e-15. This is a quadrature-convergence check, not a substitute for the theorem. Geometric log-uniform density gives approximately .13323829017761; Cosh tau=2 gives .09518226679009 (both 4096 midpoint-quantile approximations). The atomic equilibrium is qualitatively different from both smooth allocations. No relevance to Qwen task ranking is inferred from those lower energies.

### Finite K and equal channel weights are different constraints

A K-channel table represents mu_K=(1/K) sum_j delta_(omega_j), not an arbitrary probability measure. Atomic masses must be integer multiples of 1/K when coincident frequencies are allowed. The preceding unrestricted optimum need not lie in that feasible class. For K=8, midpoint quantization of its CDF assigns multiplicities (1,3,4); its energy is .05892945173956, above the measure optimum .05863932740154. This is one feasible construction, not a globally solved K=8 optimum.

The ordered closed domain a<=omega_1<=...<=omega_K<=b has an exact finite-K minimizer by continuity. Requiring all frequencies to be strictly distinct makes the domain open and can remove attainment; an explicit minimum spacing makes it compact again but introduces a new architectural constraint. Those alternatives must be stated rather than silently smuggled in through an L2 term.

The measure solution is a lower bound and a candidate initializer for the equal-mass finite-K problem, not its solution. A finite-K solver can refine actual frequencies directly using exact K derivatives, as derived by Sol02. Its local stationarity does not supply the unrestricted global dual certificate unless that certificate is separately checked.

## 3. Why source anchoring cannot by itself derive a universal middle band

E_q depends on an unlabeled frequency measure. A frozen checkpoint depends on the association between frequency slots and signed Q/K content, on values, and on subsequent layers. Swapping slot associations leaves E_q unchanged but can alter learned computations. Adding source distance weights to K leaves this invariance intact. No universal slot-specific middle-band rule follows from that information alone.

There is nevertheless a precise two-clock special case. Consider finitely many learned rotary functions

    f_r(d)=sum_j [A_rj cos(omega_j d)+B_rj sin(omega_j d)].

If a local family must be retained exactly, f_r^dep(d)=f_r(d) on an open interval; a long family must be retimed exactly, f_r^dep(Sd)=f_r(d). Linear independence of distinct exponential frequencies makes the active spectral components match their target frequencies and coefficient signatures. Under distinct identifiable slot signatures this forces native omega_j on exclusively local components and omega_j/S on exclusively retimed components. Components shared by incompatible exact clocks make the joint demand infeasible. Degenerate coefficient signatures permit permutations and must be excluded from the slot-by-slot conclusion.

Thus a native block plus common-PI block has a mathematical justification **when useful computations separate that way**. The middle is the set of conflicting or coupled active computations, not automatically an interval determined by one rotation count. Coherent slow cancellations explain why uniform compression can help; they do not tell us how much task margin to sacrifice in an overlapping middle band.

## 4. Constructive frozen rule with an actual task consequence

A defensible next allocation rule should optimize a declared useful contrast, not another undirected distance. Here is a finite-window construction with a genuine, limited consequence for conditional pointer attention.

Freeze conditional Q/K coefficients for one attention row and declare a useful key k*. Let M_r(x)=ell_(k*)(x)-ell_k(x) be each target-versus-competitor margin, where nu_j=omega_j exp(-x_j). Scores retain their actual signed cosine/sine coefficients and the actual two key lags; gain is fixed. Around x^r, write a candidate displacement v. Each slot term is C cos(z)+D sin(z), z=nu_j^r exp(-v_j)d, so

    derivative wrt v_j = -z[-C sin(z)+D cos(z)],
    |second derivative| <= sqrt(C^2+D^2) (|z|+z^2).

On |v_j|<=r_j, use |z|<=nu_j^r exp(r_j)|d|. Summing the bounds for the useful and competing key gives explicit M_rj>=0. Taylor's theorem gives a rigorous lower margin bound for frozen conditional content:

    margin_r(x^r+v) >= margin_r(x^r) + g_r^T v - (1/2) sum_j M_rj v_j^2.

Maximize the minimum certified long-margin gain eta, subject to each already-useful source margin retaining its declared threshold, the corresponding long-margin inequalities, box bounds, and the actual log-frequency ordering/endpoints. This is a convex QCQP: each lower-bound constraint is a superlevel set of a concave quadratic, and the objective eta is linear. It contains no arbitrary adjacent smoothness penalty, no independently invented channel utility, and no new norm used as a capability selector. For the separated exact-clock case, it admits the native/PI blocks; in the overlap, its active margin constraints determine the bridge.

**Actual consequence.** If all N-1 competing margins remain >=m>0, the useful key remains the unique argmax and its attention probability is >=1/[1+(N-1)exp(-m)]. For a declared pointer task whose output is that argmax, correctness follows. The mass bound also shows why a positive pairwise margin alone need not suffice with many distractors. A full language model additionally changes upstream states and performs value mixing/readout, so this conditional theorem cannot certify its generation. For that deployment, use full-model contrasts/derivatives and check the actual finite change; the conditional QCQP is a mechanistic allocation proposal with an explicit missing assumption, not a whole-model theorem.

This rule can prefer a sharper transition when smoothing would reduce a useful signed margin, even when smoothing improves every audited geometry norm. It is therefore compatible with, rather than contradicted by, the existing Smooth failure. The rule is not yet computed for Qwen: the required content-conditioned useful contrasts are not supplied by geometry or raw projection norms.

## 5. Counterexamples and evidence boundaries

Fully read `docs/research/ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md` records Smooth MrBudget below MrPro on remote unresolved energy (.0494352 vs .235928), C26 (.000159185 vs .000240986), C32768 (35.3793 vs 42.5180), all four Q/K-weighted local costs, and weighted unresolved response across all 36 layers and three cutoffs; its existing 128K development score is worse. Neither correcting the EVQ kernel, nor weighting the same norms, reverses that logical obstruction. The present atomic optimum must not become another proposed frozen winner on those grounds.

The assigned `docs/exp/2026-07/2026-07-14_lora_retrieval_conversion_probe.md` adds an independent warning: EVQ's source deletion has a real +1.5055 answer-NLL effect, but target first-token median rank remains about 2043 and exact retrieval remains zero. Even causal source use and improved ranking need not yield readout success. A margin rule needs the actual requested output contrast at the relevant stage.

The older `docs/exp/2026-02/2026-02-27_evq_tau_sweep_results.md` says tau=1.5 improves every 50M length, but its own 4K table gives 6.667 versus geometric 6.183; that universal sentence is false on its face. Its PPL observation also cannot refute the theoretical logarithmic Fisher-proxy waterbed bound. Do not import the historical narrative as a theorem.

Scratch training may use exact finite-window allocation as a representation design, then coadapt weights and evaluate the requested task. Frozen deployment must retain useful source associations. The common mathematical variable is a finite labeled table or its log-gaps; the continuous Cosh law, exact measure equilibrium, and frozen margin problem are different objective/constraint choices over that allocation question.

## 6. Next decision

The finite-window theorem closes the request for a universally smooth exact-EVQ replacement: without additional channel-density/spacing assumptions that replacement does not exist. Use the exact measure certificate to audit the geometry approximation and as a lower bound for a declared finite-K problem. For frozen Qwen, derive one signed useful-margin allocation on the evidence-supported feasible face and compare its direction with the matched mirror plus MrPro/P2/E1 on the actual held-out long task. Do not optimize the atomic collision table and call it a unified deployment rule.

## 7. A well-defined finite-resolution projection and its scaling

The phrase “Hilbert-Schmidt projection” in `EVQ_COSH_THEORY.tex:120` requires a qualification. On infinite-dimensional L2[0,1], I is not Hilbert-Schmidt: its squared HS norm is sum_n 1=infinity. The exact continuous K_L and min kernels are HS. Therefore ||K_L-alpha I-beta G||_HS is infinite for every alpha!=0; there is no literal continuous HS least-squares projection with a nonzero delta coefficient.

A constructive repair is to declare the resolution first. Divide [0,1] into n equal bins C_i and take the orthonormal basis e_i=sqrt(n)1_(C_i). Define

    A_ij = n integral_(C_i) integral_(C_j) K_L(phi,psi) dphi dpsi,
    B_ij = n integral_(C_i) integral_(C_j) min(phi,psi) dphi dpsi.

The Galerkin fit min_(alpha,beta) ||A-alpha I_n-beta B||_F^2 is finite and unambiguous. Its unconstrained normal equations are

    n alpha + beta tr(B) = tr(A),
    alpha tr(B) + beta ||B||_F^2 = <A,B>_F.

For alpha,beta>=0, use the corresponding two-variable nonnegative least-squares/KKT solution; an unconstrained negative alpha must not be used in a Cosh functional. For a binwise constant density with bin masses p_i, its basis coefficients are sqrt(n)p_i. Thus the fitted energy is

    (alpha n/2) sum_i p_i^2 + (beta n/2) p^T B p.

The factor n matters: a discrete mass-space diagonal coefficient d corresponds to alpha=d/n, not alpha=d. B has exact trace 1/2-1/(6n). For fixed L and base as n->infinity,

    tr(A) -> integral_0^1 K_L(phi,phi)dphi <= 1,
    tr(B) -> 1/2,
    ||B||_F^2 -> integral integral min(phi,psi)^2 = 1/6,
    <A,B>_F -> <K_L,G>_HS.

Hence beta tends to 6<K_L,G>_HS, whereas alpha=[tr(A)-beta tr(B)]/n=O(1/n) when the interior fit is positive. If its leading numerator is positive, tau=sqrt(beta/alpha) scales like sqrt(n) at fixed L, not automatically n/sqrt(L). If positivity binds, alpha can be zero and the Cosh coercivity assumption fails instead. Projection resolution n and actual channel count K are separate choices; setting n=K is another declared modeling decision.

This does not disprove an empirical tau proportional to K/sqrt(L) under a particular joint K,L scaling. It shows precisely what a derivation must supply: the joint-regime behavior of the projected coefficients (or a different finite-channel risk model) and a justified relation between projection resolution and channel count. Fixed-L refinement of the exact kernel supplies no such law. The theory note already labels the scaling conjectural (`:330-350`); finite-resolution normalization explains why that qualification is substantive.

## Reproducer for the CPU exchange exhibit

Execute the following Python with NumPy. The fixed validation mesh is a certificate device; it is not an allocation grid. The final dual certificate validates the found solution despite the local coordinate-refinement heuristic.

```python
import numpy as np
from numpy.polynomial.legendre import leggauss
L=8.; lo=.1; hi=1.; z,w=leggauss(96); t=1+(z+1)*(L-1)/2; wt=w*(L-1)/2/(t*np.log(L))
def design(x):return np.cos(np.outer(t,np.atleast_1d(x)))
def weights(x):
 K=design(x).T@(wt[:,None]*design(x)); best=(1e99,None)
 for mask in range(1,1<<len(x)):
  ix=np.array([i for i in range(len(x)) if mask>>i&1]); A=K[np.ix_(ix,ix)]
  v=np.linalg.lstsq(A,np.ones(len(ix)),rcond=1e-13)[0]; v/=v.sum()
  if min(v)<-1e-10:continue
  full=np.zeros(len(x));full[ix]=v; E=.5*full@K@full
  if E<best[0]:best=(E,full)
 return best
def golden(f,a,b):
 r=(np.sqrt(5)-1)/2; c=b-r*(b-a);d=a+r*(b-a);fc=f(c);fd=f(d)
 for _ in range(70):
  if fc<fd:b,d,fd=d,c,fc;c=b-r*(b-a);fc=f(c)
  else:a,c,fc=c,d,fd;d=a+r*(b-a);fd=f(d)
 return (a+b)/2
x=np.array([.55]); hist=[]
for it in range(20):
 E,p=weights(x);keep=p>1e-9;x=x[keep];p=p[keep]
 for sweep in range(10):
  before=E
  for j in range(len(x)):
   a=lo if j==0 else x[j-1];b=hi if j==len(x)-1 else x[j+1]
   def f(v):
    xx=x.copy();xx[j]=v;return weights(xx)[0]
   v=golden(f,a,b); cand=[(f(a),a),(f(b),b),(f(v),v),(E,x[j])]; E,x[j]=min(cand)
  E,p=weights(x)
  if before-E<1e-14:break
 grid=np.linspace(lo,hi,10001); pot=design(grid).T@(wt*(design(x)@p)); idx=pot.argmin();v=golden(lambda y:(design(y).T@(wt*(design(x)@p))).item(),grid[max(idx-1,0)],grid[min(idx+1,len(grid)-1)])
 pmin=min(pot.min(),(design(v).T@(wt*(design(x)@p))).item());gap=2*E-pmin
 hist.append((it,len(x),E,gap))
 if gap<1e-9:break
 x=np.sort(np.r_[x,v])
print('history',hist);print('atoms',x.tolist());print('mass',p.tolist());print('energy',E);print('numerical_gap',gap); print('mesh_gap_bound',2*E-pot.min()+(L*L-1)/(2*np.log(L))*(.9/10000)**2/8)
for tau in (0.,2.):
 u=(np.arange(4096)+.5)/4096;phi=u if tau==0 else 1-np.arcsinh((1-u)*np.sinh(tau))/tau
 om=10**(-phi);f=design(om).mean(axis=1);print('continuous_quantile_4096_tau',tau,'E',.5*np.sum(wt*f*f))
```


### Finite-resolution arithmetic check

For the same base=10,L=8 problem, 96-point distance quadrature and 16-point quadrature inside each exponent bin give:

| n | alpha | beta | n alpha | tau |
|---:|---:|---:|---:|---:|
|16|.00709041638147|.858718721249|.113446662104|11.004988569|
|32|.00329366373151|.870290110287|.105397239408|16.255202923|
|64|.00157961736196|.875477427278|.101095511166|23.542171940|

This is numerical refinement of one declared projection, not a search of task candidates. It illustrates why a fixed continuous kernel cannot supply a resolution-independent nonzero alpha through HS projection. It also differs from `scripts/analysis/tau_static_vs_dynamic_experiment.py:71-83`, whose fit uses mean(diagonal)*mean_spacing for alpha without subtracting the fitted min-kernel diagonal and whose finite log-distance quadrature is not an exact Ci evaluation. The old script's final dichotomy between a static exponent and training dynamics is broader than what that numerical experiment can prove.
