"""Local margin constraints with exact log-frequency ordering constraints."""
import math
import numpy as np
from scipy.optimize import linprog, minimize


def propose(gamma,jac,radii,nu,desired_margin,bounds=None):
    gamma,jac,radii,nu,desired_margin=map(np.asarray,(gamma,jac,radii,nu,desired_margin))
    n=len(nu)
    basis=np.diag(radii) if radii.ndim==1 else radii
    if jac.shape!=(len(gamma),n+1) or basis.shape[0]!=n+1:
        raise ValueError('constraint dimensions do not agree')
    parameters=basis.shape[1]
    A=jac@basis
    # log(nu_j/nu_{j+1}) - ln(4)*(d_j-d_{j+1}) >= eps.
    C=math.log(4)*(basis[1:n]-basis[:n-1])
    order_rhs=1e-6-np.log(nu[:-1]/nu[1:])
    count=len(gamma)
    bounds=bounds or [(-1,1)]*parameters
    lp=linprog(np.r_[np.zeros(parameters),np.ones(count)],
        A_ub=np.r_[np.c_[-A,-np.eye(count)],np.c_[-C,np.zeros((n-1,count))]],
        b_ub=np.r_[gamma-desired_margin,-order_rhs],
        bounds=bounds+[(0,None)]*count,method='highs')
    if not lp.success:raise RuntimeError(lp.message)
    slack=lp.x[parameters:]
    rhs=desired_margin-gamma-slack-1e-6
    constraints=[dict(type='ineq',fun=lambda z:A@z-rhs,jac=lambda z:A),
                 dict(type='ineq',fun=lambda z:C@z-order_rhs,jac=lambda z:C)]
    qp=minimize(lambda z:.5*z@z,lp.x[:parameters],jac=lambda z:z,
                bounds=bounds,constraints=constraints,
                method='SLSQP',options={'ftol':1e-9,'maxiter':200})
    z=qp.x if qp.success else lp.x[:parameters]
    return basis@z,slack
