"""Model-independent TailSpline constructor and checks of the paper's design identities.

The input frequency array is the public native RoPE table, not model weights.
Only NumPy is needed. This file runs no model and fits no observations.
"""
from __future__ import annotations
import math
import numpy as np


def tailspline(native_inv_freq, reference_length: int, scale: float):
    """Return FP32 frequencies, cosine/sine gain and the chosen canonical band.

    Install after model dtype conversion, before prefill; keep the table and gain
    fixed for the request. The 32/1-turn rule assumes a nonempty transition.
    For scale=1, returns exact input FP32 frequencies and unit gain.
    """
    native=np.asarray(native_inv_freq,dtype=np.float32)
    if (native.ndim!=1 or len(native)<2 or not np.isfinite(native).all()
        or not (native>0).all() or not (np.diff(native)<0).all()
        or reference_length<=0 or not math.isfinite(scale) or scale<1):
        raise ValueError('Expected positive ordered native frequencies, length and scale >= 1')
    if scale==1:return native.copy(),1.0,None
    turns=native.astype(np.float64)*reference_length/(2*math.pi)
    fast=np.flatnonzero(turns>32);slow=np.flatnonzero(turns<1)
    if not len(fast) or not len(slow) or slow[0]<=fast[-1]:
        raise ValueError('Native grid has no 32/1-turn transition')
    low,high=int(fast[-1]),int(slow[0]);n=high-low
    q=np.clip(np.arange(len(native),dtype=np.float64)-low,0,n)
    m=q*(3*n*n+3*n+1-q*q)/(n*(n+1)*(2*n+1))
    result=(native.astype(np.float64)*np.power(scale,-m)).astype(np.float32)
    if not np.isfinite(result).all() or not (result>0).all() or not (np.diff(result)<0).all():
        raise ValueError('Installed table is not strictly ordered in FP32')
    return result,1+0.1*math.log(scale),(low,high)


def verify():
    from fractions import Fraction as F
    for n in [1,2,3,17,18,32,64]:
        q=np.arange(1,n+1,dtype=float)
        epsilon=3*(n+q)*(n-q+1)/(n*(n+1)*(2*n+1))
        diff=np.eye(n-1,n,k=1)-np.eye(n-1,n)
        h=diff.T@diff;h[-1,-1]+=1
        optimum=np.linalg.solve(h,np.ones(n));optimum/=optimum.sum()
        assert np.allclose(optimum,epsilon,atol=1e-13)
        ratio=F(6,(n+1)*(2*n+1))/F(2,n+1)
        assert ratio==F(3,2*n+1)
        c,logs=.2,math.log(4)
        gaps=c+epsilon*logs
        geometric_energy=np.square(np.diff(gaps)).sum()+(c-gaps[-1])**2
        assert abs(geometric_energy-logs**2*(np.square(np.diff(epsilon)).sum()+epsilon[-1]**2))<1e-12
    nodes,w=np.polynomial.legendre.leggauss(192);u=(nodes+1)/2;w=w/2
    for tau in [.2,math.sqrt(2),2.,4.]:
        rho=tau*np.cosh(tau*(1-u))/np.sinh(tau)
        tail=np.sinh(tau*(1-u))/np.sinh(tau)
        density=.5*np.dot(w,rho*rho+tau*tau*tail*tail)
        gaps=np.sinh(tau)/(tau*np.sqrt(1+(1-u)**2*np.sinh(tau)**2))
        interval=.5*np.dot(w,1/gaps+tau*tau*(1-u)**2*gaps)
        assert abs(density-interval)<1e-10 and abs(np.dot(w,gaps)-1)<1e-10
    for length,band in [(8192,(18,35)),(4096,(14,32))]:
        native=np.exp(-math.log(500000)*np.arange(64)/64).astype(np.float32)
        values,g,found=tailspline(native,length,4)
        assert found==band and g==1+0.1*math.log(4)
        l,h=band
        assert np.array_equal(values[:l+1],native[:l+1])
        assert np.array_equal(values[h:],native[h:]/4)
        assert np.array_equal(tailspline(native,length,1)[0],native)
    # Reproduce the historical static-rank convention with discrete lags.
    u=(np.arange(32)+.5)/32
    omega=np.power(500000.,-u)
    lag=np.arange(512,dtype=float)
    phase=lag[:,None]*omega
    features=np.stack([np.cos(phase),np.sin(phase)],axis=2).reshape(512,64)
    gram=features.T@features/512
    whitener=np.zeros_like(gram)
    for pair in range(32):
        sl=slice(2*pair,2*pair+2);e,v=np.linalg.eigh(gram[sl,sl]);whitener[sl,sl]=(v/np.sqrt(e))@v.T
    corr=whitener@gram@whitener
    rank=np.trace(corr)**2/np.square(corr).sum()
    assert abs(rank-4.569127263)<1e-7
    print('PASS: finite-grid KKT, exact log-gap objective, terminal ratio, density/gap equivalence, installation endpoints.')

if __name__=='__main__':verify()
