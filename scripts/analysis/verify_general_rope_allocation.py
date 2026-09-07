"""Standard-library mathematical checks; no model, dataset download, or GPU.

This checks invariance, a two-clock operator objective, exact rotary derivatives,
and a classical minimum-norm common-descent solver. It does not evaluate an LM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def sigmoid(x):
    return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))


def common_direction(gradients,tolerance=1e-12,max_iterations=20000):
    """Classical convex-hull minimum-norm direction, with a numeric certificate."""
    n=len(gradients)
    assert n and all(len(g)==len(gradients[0]) for g in gradients)
    first=min(range(n),key=lambda i:dot(gradients[i],gradients[i]))
    weights=[float(i==first) for i in range(n)];u=list(gradients[first])
    for iteration in range(max_iterations):
        inner=[dot(g,u) for g in gradients]
        vertex=min(range(n),key=lambda i:inner[i]);gap=max(0.,dot(u,u)-inner[vertex])
        if gap<=tolerance:break
        d=[g-x for g,x in zip(gradients[vertex],u)];denom=dot(d,d)
        gamma=min(1.,max(0.,-dot(u,d)/denom)) if denom else 0.
        weights=[(1-gamma)*w+gamma*(i==vertex) for i,w in enumerate(weights)]
        u=[x+gamma*y for x,y in zip(u,d)]
    v=[-x for x in u];slopes=[dot(g,v) for g in gradients]
    gap=max(0.,dot(u,u)-min(dot(g,u) for g in gradients))
    return dict(direction=v,weights=weights,norm_squared=dot(u,u),dual_gap=gap,
                slopes=slopes,descent_certified=max(slopes)<-1e-12,
                solver_converged=gap<=tolerance,iterations=iteration+1)


def rotary_binary_loss(x,records):
    """Exact finite sin/cos toy likelihood, used only for differentiation checks."""
    loss=0.;gradient=[0.]*len(x)
    for delta,c,d,label in records:
        angles=[math.exp(-a)*delta for a in x]
        score=sum(ci*math.cos(t)+di*math.sin(t) for ci,di,t in zip(c,d,angles))
        loss+=max(score,0.)+math.log1p(math.exp(-abs(score)))-label*score
        adjoint=sigmoid(score)-label
        for j,(ci,di,t) in enumerate(zip(c,d,angles)):
            gradient[j]+=adjoint*t*(ci*math.sin(t)-di*math.cos(t))
    return loss/len(records),[g/len(records) for g in gradient]


def sinc(x):
    return math.sin(x)/x if abs(x)>1e-7 else 1-x*x/6+x**4/120


def two_clock_minimum(q,scale,grid=2000):
    def risk(y):return 2-sinc(y-q)-sinc(scale*y-q)
    values=[q/scale+(q-q/scale)*i/grid for i in range(grid+1)]
    y=min(values,key=risk)
    return y,risk(y)


def checks():
    # Finite uniform-circle shifts exactly permute independent key samples.
    n=32
    distributions=[]
    for shift1,shift2 in [(0,0),(3,11),(9,21)]:
        probabilities=[]
        for i in range(n):
            for j in range(n):
                z1=math.cos(2*math.pi*(i+shift1)/n)
                z2=math.cos(2*math.pi*(j+shift2)/n)
                probabilities.append(sigmoid(z1-z2))
        distributions.append(sorted(probabilities))
    invariant_error=max(abs(a-b) for row in distributions[1:] for a,b in zip(row,distributions[0]))
    assert invariant_error<1e-12
    # Removing independence/rotational symmetry restores a frequency effect.
    correlated_probabilities=[sigmoid(math.cos(t)) for t in [0.,math.pi]]
    assert correlated_probabilities[0]>correlated_probabilities[1]+.4

    scale=4.;clock_rows=[]
    for q in [.001,.01,.1,1.,4.,8.,16.,32.,64.,128.]:
        y,risk=two_clock_minimum(q,scale)
        clock_rows.append(dict(native_phase=q,frequency_ratio=y/q,risk=risk))
    low_limit=(scale+1)/(scale*scale+1)
    assert abs(clock_rows[1]['frequency_ratio']-low_limit)<.001
    # These are geometry-objective optima, never language-model selections.
    qs=[4096*500000**(-j/64) for j in range(64)]
    ys=[two_clock_minimum(q,scale)[0] for q in qs]
    crossings=[j for j in range(63) if ys[j]<=ys[j+1]]

    rng=random.Random(20260907);x=[-math.log(w) for w in [.35,.17,.065,.021]]
    records=[]
    for maximum in [8,32]:
        pack=[]
        for _ in range(96):
            delta=rng.uniform(.1,maximum)
            c=[rng.gauss(0,1) for _ in x];d=[rng.gauss(0,1) for _ in x]
            label=rng.randrange(2)
            pack.append((delta,c,d,label))
        records.append(pack)
    losses_gradients=[rotary_binary_loss(x,pack) for pack in records]
    derivative_error=0.;h=1e-6
    for pack,(_,g) in zip(records,losses_gradients):
        for j in range(len(x)):
            plus=x.copy();minus=x.copy();plus[j]+=h;minus[j]-=h
            fd=(rotary_binary_loss(plus,pack)[0]-rotary_binary_loss(minus,pack)[0])/(2*h)
            derivative_error=max(derivative_error,abs(fd-g[j]))
    assert derivative_error<1e-7
    direction=common_direction([g for _,g in losses_gradients])
    assert direction['descent_certified']
    before=[loss for loss,_ in losses_gradients];alpha=1.
    for backtracks in range(60):
        trial=[a+alpha*v for a,v in zip(x,direction['direction'])]
        after=[rotary_binary_loss(trial,pack)[0] for pack in records]
        ordered=all(trial[j]<trial[j+1] for j in range(len(trial)-1))
        if ordered and all(b<=a+1e-4*alpha*s for a,b,s in zip(before,after,direction['slopes'])):break
        alpha*=.5
    else:raise AssertionError('no checked finite descent step')
    assert all(b<a for a,b in zip(before,after))
    conflict=common_direction([[1.,0.],[-1.,0.]])
    assert conflict['norm_squared']==0. and not conflict['descent_certified']
    compatible=common_direction([[1.,0.],[0.,1.]])
    assert all(abs(a-b)<1e-12 for a,b in zip(compatible['direction'],[-.5,-.5]))
    return dict(status='MATHEMATICAL_CHECKS_ONLY',
        isotropic_distribution_max_error=invariant_error,
        correlated_key_probabilities=correlated_probabilities,
        two_clock=dict(scale=scale,low_phase_limit=low_limit,rows=clock_rows,
                       olmo_geometry_order_crossing_indices=crossings,
                       limits='Grid minima, not continuous global-optimum certificates or LM predictions.'),
        exact_rotary_gradient_max_error=derivative_error,
        common_descent=dict(certificate=direction,losses_before=before,losses_after=after,
                            step_size=alpha,backtracks=backtracks,
                            frequency_ratios=[math.exp(-alpha*v) for v in direction['direction']]),
        opposing_gradients=conflict,
        limits='Synthetic numeric checks only. No LM, tokenizer, benchmark, learned checkpoint or GPU was used. The solver is classical MGDA, not claimed novel.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    args=parser.parse_args();result=checks()
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
