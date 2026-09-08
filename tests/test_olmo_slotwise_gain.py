from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from scripts.experiments.olmo_fast_screen.prepare_gain import allocations
from scripts.experiments.olmo_fast_screen.runtime import install, verify
from scripts.experiments.cross_audit.tables import tensor_sha


def test_pairwise_bilinear_gain_and_rotate_half_layout():
    native = [1., .4, .1]
    frequencies = [1., .2, .025]
    gains, rms, exponent = allocations(native, frequencies, 4, 1.2)
    assert exponent == pytest.approx([0, .5, 1])
    assert gains[0] == 1 and gains[-1] == pytest.approx(1.2)
    assert rms*rms == pytest.approx(sum(a*a for a in gains)/3)
    q = torch.tensor([.7,-.3,1.1,.2,.9,-.8],dtype=torch.float64)
    k = torch.tensor([-.4,.6,.5,1.2,-.7,.1],dtype=torch.float64)
    freq = torch.tensor(frequencies,dtype=torch.float64)
    gain = torch.tensor(gains*2,dtype=torch.float64)
    def rotate(x, position):
        theta = (freq*position).repeat(2)
        half = torch.cat((-x[3:],x[:3]))
        return (x*theta.cos()+half*theta.sin())*gain
    actual = rotate(q,7) @ rotate(k,2)
    expected = 0.
    for j in range(3):
        delta = frequencies[j]*5
        expected += gains[j]**2*((q[j]*k[j]+q[j+3]*k[j+3])*math.cos(delta)
                   +(q[j]*k[j+3]-q[j+3]*k[j])*math.sin(delta))
    assert actual.item() == pytest.approx(float(expected),abs=1e-12)


def test_scalar_vector_scalar_install_and_drift_detection():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(hidden_size=6,num_attention_heads=1)
            self.model = SimpleNamespace(rotary_emb=SimpleNamespace(rope_type='default'))
    model = Toy()
    values = np.array([1.,.2,.025],dtype=np.float32)
    table = dict(values_float32=values.tolist(),tensor_sha256=tensor_sha(values),gain=1.2)
    install(model,table);verify(model,table)
    vector = dict(table,gain_by_slot=[1.,1.1,1.2])
    install(model,vector);verify(model,vector)
    assert model.model.rotary_emb.attention_scaling.tolist() == pytest.approx([1,1.1,1.2,1,1.1,1.2])
    model.model.rotary_emb.attention_scaling[0] = .9
    with pytest.raises(RuntimeError,match='amplitude'):verify(model,vector)
    install(model,table);verify(model,table)
    assert isinstance(model.model.rotary_emb.attention_scaling,float)


def test_reject_invalid_exponent_and_gain_vector():
    with pytest.raises(ValueError):allocations([1],[2],4,1.2)
    from scripts.experiments.olmo_fast_screen.runtime import gain_vector
    with pytest.raises(ValueError):gain_vector(dict(values_float32=[1, .1],gain_by_slot=[1]),'cpu')
