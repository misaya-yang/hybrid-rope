import numpy as np
from experiments.rope_decision_20260911.solve import propose


def test_minimum_move_keeps_frequency_order():
    # Unconstrained repair would swap this close pair. The solver must retain
    # strict order and expose the incompatible task margin as slack.
    nu=np.array([1.,.99])
    step,slack=propose(np.array([-1.]),np.array([[1.,0.,0.]]),
                       np.array([1.,0.,0.]),nu,np.array([0.]))
    changed=nu*np.exp(-np.log(4)*step[:2])
    assert changed[0]>changed[1]
    assert slack[0]>.9


def test_feasible_margin_is_met_without_slack():
    step,slack=propose(np.array([-.2]),np.array([[0.,0.,1.]]),
                       np.array([.1,.1,1.]),np.array([1.,.5]),np.array([0.]))
    assert abs(slack[0])<1e-8
    assert abs(step[2]-.2)<1e-5
    assert np.max(abs(step[:2]))<1e-6


def test_restricted_basis_and_nonnegative_amount():
    basis=np.array([[.1,0.],[.2,0.],[0.,1.]])
    step,_=propose(np.array([-.05]),np.array([[1.,1.,0.]]),basis,
                   np.array([1.,.5]),np.array([0.]),bounds=[(0.,1.),(-1.,1.)])
    assert step[0]>=0 and step[1]==2*step[0]
    assert abs(step[2])<1e-6
