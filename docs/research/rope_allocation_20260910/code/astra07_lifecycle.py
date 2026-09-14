"""CPU-only lifecycle identities; synthetic probability models, no LM claim."""
from __future__ import annotations
import json
import numpy as np


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def kl(p, q):
    return np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1))


def moment_fit(features, offset, p):
    u = np.zeros(features.shape[-1])
    for _ in range(100):
        q = softmax(offset + features @ u)
        means = np.einsum('nk,nkd->nd', q, features)
        grad = np.einsum('nk,nkd->d', q-p, features) / len(p)
        hess = (np.einsum('nk,nkd,nke->de', q, features, features)
                - means.T @ means) / len(p)
        if np.linalg.norm(grad) < 1e-12:
            break
        direction = np.linalg.solve(hess, grad)
        old = kl(p, q)
        step = 1.
        while kl(p, softmax(offset + features @ (u-step*direction))) > old:
            step *= .5
            if step < 1e-12:
                raise RuntimeError('Newton line search failed')
        u -= step*direction
    return u, softmax(offset + features @ u)


def adapt(mu, covariance, a0, allowed, steps, eta):
    """Exact finite GD on Gaussian log E exp(-margin), with no ridge penalty."""
    H = allowed.T @ covariance @ allowed
    d = allowed.T @ (mu - covariance @ a0)
    eig, Q = np.linalg.eigh(H)
    filt = np.empty_like(eig)
    for j, lam in enumerate(eig):
        filt[j] = eta*steps if abs(lam) < 1e-14 else (1-(1-eta*lam)**steps)/lam
    u = Q @ (filt * (Q.T @ d))
    return a0 + allowed @ u


def cumulant(mu, C, a):
    return float(.5*a@C@a-mu@a)


def main():
    rng = np.random.default_rng(707)
    f = rng.normal(size=(4, 5, 2)); b = rng.normal(size=(4, 5))
    p = softmax(rng.normal(size=(4, 5)))
    u, qstar = moment_fit(f,b,p)
    q0 = softmax(offset + features @ u)
    residual = kl(p,q0)-kl(p,qstar)-kl(qstar,q0)
    assert abs(residual) < 1e-10
    row_residuals = np.sum(p*(np.log(p)-np.log(q0)),axis=1)-np.sum(p*(np.log(p)-np.log(qstar)),axis=1)-np.sum(qstar*(np.log(qstar)-np.log(q0)),axis=1)
    assert np.max(np.abs(row_residuals)) > 1e-3

    # Constrained affine family: u >= 0, unconstrained optimum is -0.7.
    ff = np.array([1.,-1.,0.])
    pp = softmax(-.7*ff); qs = softmax(0*ff); qq = softmax(.6*ff)
    slack = kl(pp,qq)-kl(pp,qs)-kl(qs,qq)
    expected_slack = .6*((qs-pp)@ff)
    assert slack > 0 and abs(slack-expected_slack) < 1e-12

    # A curved logit family eta(u)=(u,u^2,0), with stationary local optimum 0.
    pn = np.array([1/3,.38,1-1/3-.38]); qn0 = np.ones(3)/3
    qn = softmax(np.array([.5,.25,0.]))
    curved_residual = kl(pn,qn)-kl(pn,qn0)-kl(qn0,qn)
    assert abs(curved_residual) > .01
    assert 2/9+2*(1/3-pn[1]) > 0

    A = rng.normal(size=(7,7)); C=A.T@A; mu=rng.normal(size=7)
    a0=rng.normal(size=7); U=np.linalg.qr(rng.normal(size=(7,3)))[0]
    eta=.5/np.linalg.eigvalsh(U.T@C@U).max(); steps=17
    explicit=a0.copy()
    for _ in range(steps): explicit -= eta*U@(U.T@(C@explicit-mu))
    closed=adapt(mu,C,a0,U,steps,eta)
    dynamic_error=float(np.max(np.abs(explicit-closed)))
    assert dynamic_error < 1e-12

    # Better fully fitted representation B is initially worse on inherited loading.
    a0=np.array([1.,0.]); U=np.eye(2)
    ma=np.array([1.,0.]); ca=np.eye(2)
    mb=np.array([0.,1.]); cb=.25*np.eye(2)
    crossover=[]
    for n in (0,1,2,4,16):
        aa=adapt(ma,ca,a0,U,n,.5); ab=adapt(mb,cb,a0,U,n,.5)
        ka=cumulant(ma,ca,aa); kb=cumulant(mb,cb,ab)
        assert abs(kb-(-2+2.125*.875**(2*n)))<1e-12
        crossover.append({'steps':n,'A_log_exp_error_bound':ka,'B_log_exp_error_bound':kb})
    assert crossover[0]['A_log_exp_error_bound']<crossover[0]['B_log_exp_error_bound']
    assert crossover[2]['B_log_exp_error_bound']<crossover[2]['A_log_exp_error_bound']

    # More allowed adaptation can hurt a different deployment population.
    train_mu=np.array([1.,1.]); deploy_mu=np.array([1.,-1.]); eye=np.eye(2)
    full=adapt(train_mu,eye,a0,eye,4,.5)
    restricted=adapt(train_mu,eye,a0,eye[:,:1],4,.5)
    bad_full=cumulant(deploy_mu,eye,full); good_restricted=cumulant(deploy_mu,eye,restricted)
    assert bad_full > good_restricted

    # Density/loading gauge and independent-channel Neyman count allocation.
    rho1=np.array([.1,.2,.3,.4]); rho2=np.array([.4,.3,.2,.1])
    loading=np.array([1.,-2.,3.,.5]); Z=rng.normal(size=(13,4))
    gauge_error=float(np.max(np.abs(Z@(rho1*loading)-Z@(rho2*(rho1/rho2*loading)))))
    assert gauge_error < 1e-12
    w=np.array([1.,-2.,.5,3.]); sigma=np.array([1.,2.,.5,1.5]); K=64
    optimal=np.abs(w)*sigma; optimal/=optimal.sum()
    vstar=float(np.sum(sigma**2*w**2/optimal)/K)
    vuniform=float(np.sum(sigma**2*w**2/.25)/K)
    assert abs(vstar-(np.abs(w)*sigma).sum()**2/K)<1e-12
    assert vstar<vuniform
    out={'scope':'synthetic CPU identities only; no model results',
         'softmax_pythagorean_residual':float(residual),
         'per_context_identity_does_not_hold_max_residual':float(np.max(np.abs(row_residuals))),
         'constrained_pythagorean_slack':float(slack),
         'curved_family_nonzero_residual':float(curved_residual),
         'finite_update_max_error':dynamic_error,'crossover':crossover,
         'shifted_deployment_full_update_risk':bad_full,
         'shifted_deployment_restricted_update_risk':good_restricted,
         'density_loading_gauge_max_error':gauge_error,
         'neyman_count_density':optimal.tolist(),
         'independent_noise_variance_optimal':vstar,
         'independent_noise_variance_uniform':vuniform}
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
