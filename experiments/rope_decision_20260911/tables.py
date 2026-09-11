"""Explicit table definitions; no claim of performance from geometry alone."""
import math
import numpy as np


def band(theta, window, k=64, alpha=1., beta=32.):
    return (math.floor(k*math.log(window/(2*math.pi*beta))/math.log(theta)),
            math.ceil(k*math.log(window/(2*math.pi*alpha))/math.log(theta)))


def beta_exponents(b, lo, hi, k=64):
    n = hi-lo
    z = np.arange(1,n+1,dtype=float)
    w = z*(n+1-z)**b
    m = np.zeros(k)
    m[lo+1:hi+1] = np.cumsum(w/w.sum())
    m[hi+1:] = 1.
    return m


def evq_phi(tau, k=64, midpoint=False):
    u = (np.arange(k,dtype=float)+(.5 if midpoint else 0.))/k
    if abs(tau)<1e-8:
        return u
    return 1-np.arcsinh((1-u)*np.sinh(tau))/tau


def build_tables(theta,window,k=64,scale=4.):
    native = theta**(-np.arange(k,dtype=float)/k)
    lo,hi = band(theta,window,k)
    ramp = np.clip((np.arange(k)-lo)/(hi-lo),0.,1.)
    gain = 1+.1*math.log(scale)
    ms = {'mrpro': beta_exponents(0.,lo,hi,k),
          'bm': beta_exponents(1.,lo,hi,k)}
    wide_lo,wide_hi = band(theta,window,k,beta=64.)
    ms['b4wide'] = beta_exponents(4.,wide_lo,wide_hi,k)
    freq = {n:native*scale**(-m) for n,m in ms.items()}
    # Official YaRN code ramps in channel index. This is NOT a linear ramp
    # in the rotation count used by the MrRoPE appendix's regressive proof.
    freq['yarn_index'] = native*(1-(1-1/scale)*ramp)
    turns = window*native/(2*math.pi)
    retain = np.clip((turns-1)/(32-1),0.,1.)
    freq['yarn_turns_paper'] = native*(retain+(1-retain)/scale)
    # Literal EVQ grids, distinguished from the old midpoint delta installed
    # on an endpoint-native grid (which shifts every frequency by theta^.5/K).
    freq['evq_endpoint_t1'] = theta**(-evq_phi(1.,k,midpoint=False))
    freq['evq_midpoint_t1'] = theta**(-evq_phi(1.,k,midpoint=True))
    freq['native'] = native
    tables = {name:dict(values_float32=np.asarray(v,np.float32).tolist(),
                       gain=1. if name=='native' else gain)
              for name,v in freq.items()}
    return tables


def evq_midband_direction(nu,lo,hi,tau=1.):
    """EVQ companding of the initializer's actual log-frequency coordinates.

    Positive alpha moves u toward the EVQ quantile, keeping band endpoints and
    all outside frequencies fixed. This is an EVQ-derived residual, not a
    literal canonical-grid replacement or an unconstrained 64-slot fit.
    """
    nu=np.asarray(nu,float)
    if not (0<=lo<hi<len(nu)) or np.any(np.diff(nu)>=0):
        raise ValueError('ordered initializer and valid band required')
    span=math.log(nu[lo]/nu[hi])
    u=np.log(nu[lo]/nu[lo:hi+1])/span
    phi=u if abs(tau)<1e-8 else 1-np.arcsinh((1-u)*np.sinh(tau))/tau
    direction=np.zeros(len(nu))
    direction[lo:hi+1]=span*(phi-u)/math.log(4)
    direction[lo]=direction[hi]=0.
    return direction


def formula_audit():
    out = {}
    for name,theta,window in [('olmo',500000.,4096),('qwen',1e6,32768)]:
        tables = build_tables(theta,window)
        native = theta**(-np.arange(64)/64)
        lo,hi = band(theta,window)
        row = {}
        for arm in ('yarn_index','yarn_turns_paper','mrpro'):
            m = -np.log(np.asarray(tables[arm]['values_float32'])/native)/np.log(4)
            eps = np.diff(m[lo:hi+1])
            row[arm] = dict(sum_m=float(m.sum()), first_increment=float(eps[0]),
                           last_increment=float(eps[-1]),
                           interior_increments_increasing=bool(np.all(np.diff(eps[1:-1])>0)))
        midpoint = (np.arange(64)+.5)/64
        phi = evq_phi(1.,midpoint=True)
        old_m = (phi-midpoint)*np.log(theta)/np.log(4)
        old_freq = native*4**(-old_m)
        canonical = theta**(-phi)
        row['old_evq_vs_canonical'] = dict(min_ratio=float((old_freq/canonical).min()),
                                         max_ratio=float((old_freq/canonical).max()),
                                         expected_ratio=theta**(1/128))
        out[name]=row
    return out


if __name__=='__main__':
    import json
    print(json.dumps(formula_audit(),indent=2))
