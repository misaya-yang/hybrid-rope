import numpy as np
from experiments.native_sparse_position.orbit_regression import basis,fit,reconstruct,logmass

def rotate(x,angles):
    K=angles.shape[-1];z=x[...,:K]+1j*x[...,K:2*K]
    z=z*np.exp(1j*angles)
    return np.concatenate([z.real,z.imag,x[...,2*K:]],axis=-1)

def test_exact_constant_native_and_constant_pre_rotation_classes():
    rng=np.random.default_rng(1);w=np.array([.7,.13,.003,1e-9]);u=rng.normal(size=(3,1,8))
    for keys in [np.broadcast_to(u,(3,64,8)).copy(),rotate(np.broadcast_to(u,(3,64,8)),np.arange(64)[:,None]*w)]:
        mean,beta=fit(keys,w);np.testing.assert_allclose(reconstruct(mean,beta,w),keys,atol=2e-12,rtol=2e-12)
        q=rng.normal(size=(5,8));exact=np.logaddexp.reduce(np.einsum('qd,btd->qbt',q,keys),axis=-1)
        np.testing.assert_allclose(logmass(q,mean,beta,w),exact,atol=2e-12,rtol=2e-12)

def test_pythagorean_identity_partial_rotation_and_zero_frequency():
    rng=np.random.default_rng(2);w=np.array([1.,.17,1e-9,0.]);keys=rng.normal(size=(5,64,12))
    mean,beta=fit(keys,w);pred=reconstruct(mean,beta,w)
    original=np.mean(np.sum((keys-mean[:,None])**2,-1),-1)
    residual=np.mean(np.sum((keys-pred)**2,-1),-1)
    np.testing.assert_allclose(original-residual,np.sum(np.abs(beta)**2,-1),atol=1e-12)
    assert np.all(residual<=original+1e-12)
    assert np.all(beta[:,-1]==0)
    np.testing.assert_allclose(basis(w).mean(0),0,atol=1e-14)

def test_common_translation_preserves_selector_response():
    rng=np.random.default_rng(3);w=np.array([.9,.03,1e-5]);keys=rng.normal(size=(2,64,8));q=rng.normal(size=(3,8))
    m,b=fit(keys,w);f=logmass(q,m,b,w)
    angle=9876*w;m2,b2=fit(rotate(keys,angle),w)
    f2=logmass(rotate(q,angle),m2,b2,w)
    np.testing.assert_allclose(f,f2,atol=1e-12,rtol=1e-12)
