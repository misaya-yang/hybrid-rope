"""Exact finite-support frozen-frequency reference, CPU NumPy only.

A record is {a,b: [keys,pairs], distance: [keys], support: bool[keys]}.
Optional bias: [keys] is the native additive logit bias, unchanged.
For split-half Qwen RoPE with distance=key_pos-query_pos:
a=(qx*kx+qy*ky)/sqrt(head_dim); b=(qy*kx-qx*ky)/sqrt(head_dim).
Causal/ineligible keys MUST be removed before constructing each record.
Optional primary objective: output_cotangent_dot_value: [keys], where each
entry is baseline downstream answer-loss cotangent dotted with that head's V.
This automatically accounts for head roles; cotangents stay frozen. The support
objective is diagnostic and requires separately retrieval-qualified heads.
Every channel coefficient and label is retained. No amplitude optimization.
"""
import numpy as np


def _lse_prob(s):
    z = np.max(s)
    e = np.exp(s-z)
    return float(z+np.log(e.sum())), e/e.sum()


def evaluate(theta, records, target_length, need_gradient=True):
    loss = 0.0
    gradient = np.zeros_like(theta, dtype=np.float64)
    lipschitz = 0.0
    for row in records:
        a, b = np.asarray(row['a']), np.asarray(row['b'])
        distance = np.asarray(row['distance'])
        utility = row.get('output_cotangent_dot_value')
        support = np.asarray(row.get('support', []), dtype=bool)
        if a.shape != b.shape or a.shape != (len(distance), len(theta)):
            raise ValueError('coefficient dimensions differ')
        if utility is None and (not support.any() or support.all()):
            raise ValueError('need nonempty support and distractor sets')
        if np.max(np.abs(distance)) > target_length:
            raise ValueError('target_length must bound all absolute distances')
        phase = distance[:,None]*theta[None,:]
        sine, cosine = np.sin(phase), np.cos(phase)
        score = np.sum(a*cosine+b*sine, axis=-1)
        bias = np.asarray(row.get('bias', np.zeros(len(distance))),dtype=np.float64)
        if bias.shape != score.shape or not np.isfinite(bias).all():
            raise ValueError('bias must be one finite number per key')
        score = score+bias
        la, pa = _lse_prob(score)
        if utility is None:
            ls, ps = _lse_prob(score[support])
            loss += la-ls
            residual = pa.copy()
            residual[support] -= ps
        else:
            utility = np.asarray(utility,dtype=np.float64)
            if utility.shape != score.shape or not np.isfinite(utility).all():
                raise ValueError('output utility must be one finite number per key')
            expected = pa@utility
            loss += expected
            residual = np.zeros_like(pa) if np.ptp(utility)==0 else pa*(utility-expected)
        if need_gradient:
            delta = distance[:,None]/target_length
            jac = delta*(-a*sine+b*cosine)
            gradient += residual@jac
            amplitude = np.hypot(a,b)
            # Global bound for x=T*(theta-theta0), independent of x.
            g2 = np.max(np.sum((delta*amplitude)**2,axis=-1))
            h = np.max(delta**2*amplitude)
            lipschitz += (2 if utility is None else np.ptp(utility))*(g2+h)
    n = len(records)
    if n == 0:
        raise ValueError('empty calibration set')
    return float(loss/n), gradient/n, float(lipschitz/n)


def generate(theta0, records, target_length, max_target_phase=1.0):
    """One deterministic projected-gradient step; returns frequency table + audit.

    max_target_phase is an explicitly chosen trust budget in radians, not a
    fitted coefficient. Gap boxes preserve original order and pair labels.
    Captured loss descent is guaranteed up to numeric precision; not task gain.
    """
    theta0 = np.asarray(theta0,dtype=np.float64)
    if np.any(theta0<=0) or np.any(np.diff(theta0)>=0):
        raise ValueError('expected positive strictly descending frequency table')
    if max_target_phase <= 0 or target_length <= 0:
        raise ValueError('positive trust budget and length required')
    before, g, lip = evaluate(theta0,records,target_length)
    gap = theta0[:-1]-theta0[1:]
    nearest = np.minimum(np.r_[np.inf,gap],np.r_[gap,np.inf])
    radius = np.minimum(max_target_phase,
        target_length*np.minimum(.49*nearest,.49*theta0))
    step = np.clip(-g/max(lip,np.finfo(float).tiny),-radius,radius)
    theta = theta0+step/target_length
    after, _, _ = evaluate(theta,records,target_length,False)
    bound = float(g@step+.5*lip*(step@step))
    if after > before+1e-10:
        raise AssertionError('exact objective unexpectedly increased')
    return theta, {'loss_before':before,'loss_after':after,
        'quadratic_change_upper_bound':bound,'actual_change':after-before,
        'max_target_phase_change':float(np.max(np.abs(step))),
        'lipschitz_bound':float(lip),'gradient_norm':float(np.linalg.norm(g)),
        'changed_pairs':np.flatnonzero(theta!=theta0).tolist(),
        'scope':'captured-activation calibration; output cotangents held fixed where supplied'}


def self_test():
    # Full covariance gives wrong block ranking; exact finite log-MGF does not.
    rare=np.r_[10.,np.zeros(63)]; constant=np.ones(64)
    gaussian=lambda x: np.log(len(x))+x.mean()+.5*x.var()
    assert gaussian(rare)<gaussian(constant)
    assert _lse_prob(rare)[0]>_lse_prob(constant)[0]
    rng=np.random.default_rng(9)
    theta=np.array([.02,.005,.001])
    rows=[{'a':rng.normal(size=(17,3)), 'b':rng.normal(size=(17,3)),
           'distance':np.arange(-17,0), 'support':np.arange(17)==2}]
    loss,g,_=evaluate(theta,rows,20)
    for j in range(3):
        e=np.zeros(3);e[j]=1e-5/20
        numeric=(evaluate(theta+e,rows,20,False)[0]-
                 evaluate(theta-e,rows,20,False)[0])/(2e-5)
        assert np.isclose(g[j],numeric,rtol=1e-6,atol=1e-8),(g[j],numeric)
    out,audit=generate(theta,rows,20)
    assert np.all(np.diff(out)<0)
    assert audit['actual_change']<=audit['quadratic_change_upper_bound']+1e-10
    # A pair whose coefficients vanish must retain its original frequency.
    rows[0]['a'][:,1]=0;rows[0]['b'][:,1]=0
    out,_=generate(theta,rows,20)
    assert out[1]==theta[1]
    print(audit)
    rows[0]['output_cotangent_dot_value']=rng.normal(size=17)
    _,g,_=evaluate(theta,rows,20)
    for j in range(3):
        e=np.zeros(3);e[j]=1e-5/20
        numeric=(evaluate(theta+e,rows,20,False)[0]-evaluate(theta-e,rows,20,False)[0])/(2e-5)
        assert np.isclose(g[j],numeric,rtol=1e-6,atol=1e-8),(g[j],numeric)
    _,audit=generate(theta,rows,20)
    assert audit['actual_change']<=audit['quadratic_change_upper_bound']+1e-10
    print(audit)
    rows[0]['output_cotangent_dot_value']=np.ones(17)
    out,_=generate(theta,rows,20)
    assert np.array_equal(theta,out)

if __name__=='__main__':
    self_test()
