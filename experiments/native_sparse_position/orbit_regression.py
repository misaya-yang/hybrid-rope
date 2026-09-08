"""Independent candidate: centered rotary-orbit regression of the key measure.

Two sufficient arrays (mean and complex orbit coefficient) reconstruct virtual
selector keys. Actual reader K/V never change. This is not a proven LM method.
"""
import numpy as np

def basis(omega,block=64):
    omega=np.asarray(omega,dtype=np.float64)
    t=np.arange(block,dtype=np.float64)-(block-1)/2
    # expm1 avoids subtracting two numbers near 1 for the slow native frequencies.
    z=np.expm1(1j*t[:,None]*omega[None,:])
    z-=z.mean(0,keepdims=True)
    sigma=np.sqrt(np.mean(np.abs(z)**2,axis=0))
    active=omega!=0
    if np.any(sigma[active]==0):raise ValueError('Degenerate nonzero orbit')
    phi=np.zeros_like(z);phi[:,active]=z[:,active]/sigma[None,active]
    return phi

def fit(keys,omega):
    """keys [..., B, D], native split-half RoPE in its first 2K dimensions."""
    keys=np.asarray(keys,dtype=np.float64);K=len(omega);B=keys.shape[-2]
    if 2*K>keys.shape[-1]:raise ValueError('Rotary width exceeds key width')
    phi=basis(omega,B);mean=keys.mean(-2)
    yc=keys[...,:K]+1j*keys[...,K:2*K]
    beta=np.mean(np.conj(phi)*yc,axis=-2)
    return mean,beta

def reconstruct(mean,beta,omega,block=64):
    K=len(omega);phi=basis(omega,block)
    out=np.broadcast_to(mean[...,None,:],(*mean.shape[:-1],block,mean.shape[-1])).copy()
    wave=phi*beta[...,None,:]
    out[...,:K]+=wave.real;out[...,K:2*K]+=wave.imag
    return out

def logmass(queries,mean,beta,omega,block=64):
    # Explicit virtual reconstruction is only the transparent CPU reference.
    # A deployment must fuse coefficient contraction and phase basis evaluation.
    virtual=reconstruct(mean,beta,omega,block)
    scores=np.einsum('qd,btd->qbt',queries,virtual,optimize=False)
    return np.logaddexp.reduce(scores,axis=-1)
