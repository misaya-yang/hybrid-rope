import numpy as np
def logsumexp(x):
    return np.logaddexp.reduce(x)
from experiments.native_sparse_position.groups import phase_groups,controls

def test_single_frequency_phase_classes_preserve_response():
    omega=np.array([np.pi/4]);labels=phase_groups(omega,64,8)
    theta=np.arange(64)*omega[0];keys=np.stack([np.cos(theta),np.sin(theta)],axis=-1)
    rng=np.random.default_rng(17)
    for q in rng.normal(size=(13,2))*3:
        exact=logsumexp(keys@q)
        means=np.array([keys[labels==r].mean(0) for r in range(8)])
        estimated=logsumexp(np.log(np.bincount(labels))+means@q)
        np.testing.assert_allclose(estimated,exact,atol=1e-12)

def test_controls_match_counts_and_jensen_bound():
    omega=10_000_000.**(-np.arange(32)/32)
    lab=phase_groups(omega);cs=controls(lab)
    for v in cs.values():np.testing.assert_array_equal(np.bincount(v),np.bincount(lab))
    keys=np.random.default_rng(4).normal(size=(64,7));q=np.arange(7)/7
    for v in cs.values():
        means=np.array([keys[v==r].mean(0) for r in range(4)])
        assert logsumexp(np.log(np.bincount(v))+means@q)<=logsumexp(keys@q)+1e-12
    np.testing.assert_array_equal(phase_groups(np.zeros(32)),np.repeat(np.arange(4),16))
