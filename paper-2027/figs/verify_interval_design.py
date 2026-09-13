#!/usr/bin/env python3
"""Independent scalar checks for the conditional interval propositions; no models."""
import json
import math
from pathlib import Path


def simpson(f, lo, hi, n=4096):
    h = (hi-lo)/n
    return h/3*(f(lo)+f(hi)+sum((4 if i%2 else 2)*f(lo+i*h) for i in range(1,n)))


def objective(a,s,f):
    return simpson(lambda t:f(a*math.exp(t)-1),0,math.log(s))/math.log(s)


def main():
    errors=[]
    for s in (2.,4.,8.):
        a=2/(s+1)
        for f in (lambda z:z*z,lambda z:z**4):
            j=objective(a,s,f)
            assert all(j < objective(a+d,s,f) for d in (-.01,.01))
            for x in (1/s+.02,a, .9):
                eps=1e-6
                numeric=(objective(x+eps,s,f)-objective(x-eps,s,f))/(2*eps)
                exact=(f(x*s-1)-f(x-1))/(x*math.log(s))
                errors.append(abs(numeric-exact))
        er=(s-1)/math.log(s);er2=(s*s-1)/(2*math.log(s))
        for lam,eta in ((0,0),(1,0),(0,1),(2,3)):
            x=(er+lam+eta*s)/(er2+lam+eta*s*s)
            assert 1/s<=x<=1
            deriv=2*(x*er2-er+lam*(x-1)+eta*s*(x*s-1))
            assert abs(deriv)<1e-12
        for k in (.1,1,10):
            f=lambda z:k*z*z if z>=0 else z*z
            x=(1+math.sqrt(k))/(1+s*math.sqrt(k))
            assert abs(f(x*s-1)-f(x-1))<1e-12
    a=.4;s=4;f=lambda z:1-math.cos(2*math.pi*z)
    eps=1e-4
    numeric=(objective(a+eps,s,f)-2*objective(a,s,f)+objective(a-eps,s,f))/eps**2
    exact=25*math.pi*math.sin(6*math.pi/5)/math.log(4)
    assert numeric<0 and abs(numeric-exact)<1e-4
    assert max(errors)<1e-6
    inputs=json.loads(Path(__file__).with_name('interval_development_inputs.json').read_text())
    auc_errors=[]
    for panel in inputs['panels']:
        lengths=panel['lengths']
        for arm in panel['arms']:
            y=arm['scores_percent']
            auc=sum((y[i]+y[i+1])/2*math.log(lengths[i+1]/lengths[i]) for i in range(len(y)-1))/math.log(lengths[-1]/lengths[0])
            auc_errors.append(abs(auc-arm['reported_auc']))
    assert max(auc_errors)<.01
    print(json.dumps({'scalar_checks':'passed','max_derivative_error':max(errors),'circular_second_derivative':{'numeric':numeric,'analytic':exact},'rounded_summary_auc_max_difference_pp':max(auc_errors),'evidence_scope':'summary arithmetic only; no bootstrap or raw-row revalidation','model_execution':False},indent=2))

if __name__=='__main__':main()
