import numpy as np
from experiments.rope_decision_20260911.tables import build_tables,evq_phi,evq_midband_direction


def test_endpoint_evq_zero_recovers_native_coordinates():
    np.testing.assert_array_equal(evq_phi(0.),np.arange(64)/64)
    assert not np.array_equal(evq_phi(0.,midpoint=True),np.arange(64)/64)


def test_yarn_mrpro_have_identical_unchanged_plateaus():
    t=build_tables(500000.,4096)
    a,b=[np.array(t[n]['values_float32']) for n in ('yarn_index','mrpro')]
    np.testing.assert_array_equal(a[:15],b[:15])
    np.testing.assert_array_equal(a[32:],b[32:])
    assert not np.array_equal(a[15:32],b[15:32])


def test_evq_residual_keeps_endpoints_and_slot_order():
    nu=np.array(build_tables(500000.,4096)['b4wide']['values_float32'])
    d=evq_midband_direction(nu,11,32)
    np.testing.assert_array_equal(d[:12],np.zeros(12))
    np.testing.assert_array_equal(d[32:],np.zeros(32))
    assert np.any(d[12:32]<0)
    for alpha in (0.,.1,1.):
        out=nu*4**(-alpha*d)
        assert np.all(np.diff(out)<0)
        assert out[11]==nu[11] and out[32]==nu[32]
