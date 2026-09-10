"""Nonnegative fixed-budget minimum-roughness allocation with KKT certificate."""
import argparse
import json
from pathlib import Path
import numpy as np


def solve(n,budget):
    lap=2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1)
    a=np.stack([np.ones(n),np.arange(n-1,-1,-1)]);b=np.array([1.,budget])
    # The two requested budgets are interior. Check leading/trailing active
    # faces and accept only a full primal/dual KKT certificate: convexity then
    # proves global optimality, independent of the enumeration order.
    faces=[np.arange(k,n) for k in range(n-1)]+[np.arange(n-k) for k in range(1,n-1)]
    for free in faces:
        kkt=np.block([[2*lap[np.ix_(free,free)],a[:,free].T],[a[:,free],np.zeros((2,2))]])
        answer=np.linalg.solve(kkt,np.r_[np.zeros(len(free)),b])
        eps=np.zeros(n);eps[free]=answer[:len(free)];dual=2*lap@eps+a.T@answer[len(free):]
        active=np.setdiff1d(np.arange(n),free)
        if (eps.min()>=-1e-10 and np.max(np.abs(a@eps-b))<1e-9
            and np.max(np.abs(dual[free]))<1e-9 and (not len(active) or dual[active].min()>=-1e-9)):
            eps[np.abs(eps)<1e-14]=0.
            return eps,dict(budget=float(budget),budget_actual=float((a@eps)[1]),total=float(eps.sum()),
                roughness=float(eps@lap@eps),minimum_increment=float(eps.min()),
                active_zero_indices_1based=(active+1).tolist(),dual_multipliers=dual.tolist(),
                equality_residual=float(np.max(np.abs(a@eps-b))),stationarity_free_residual=float(np.max(np.abs(dual[free]))))
    raise ValueError('No certified solution found for the requested budget')


def construct(tables,lo,hi):
    n=hi-lo;eps,certificate=solve(n,(n-1)/3)
    bm,bm_certificate=solve(n,(n-1)/2)
    i=np.arange(1,n+1);expected=6*i*(n+1-i)/(n*(n+1)*(n+2))
    if not np.allclose(bm,expected,rtol=0,atol=1e-11):raise ValueError('BM recovery check failed')
    native=np.array(tables['Native']['values_float32']);base=np.array(tables['MrPro']['values_float32'])
    m=np.r_[0,np.cumsum(eps)];smooth=base.copy();uniform=base.copy()
    for j in range(lo+1,hi):
        smooth[j]=native[j]*4**(-m[j-lo]);uniform[j]=native[j]*4**(-(j-lo)/n)
    for f in (smooth,uniform):
        if not (np.isfinite(f).all() and (f>0).all() and (np.diff(f)<0).all()):raise ValueError('invalid ordered table')
        if not (np.array_equal(f[:lo+1],base[:lo+1]) and np.array_equal(f[hi:],base[hi:])):raise ValueError('endpoint or exterior drift')
    gain=tables['MrPro']['gain']
    return dict(Smooth_MrBudget={'values_float32':smooth.astype(np.float32).tolist(),'gain':gain},
        MrUni={'values_float32':uniform.astype(np.float32).tolist(),'gain':gain},
        construction=dict(bounds=[lo,hi],increments=eps.tolist(),cumulative_exponents=m.tolist(),
            kkt=certificate,bm_recovery_kkt=bm_certificate,
            scope='Unique nonnegative minimum roughness at fixed cumulative budget; not a task-optimality result'))


def prepare(root,qwen_tables,olmo_tables):
    root=Path(root);out=root/'planned_controls/smooth_budget';out.mkdir(parents=True,exist_ok=True)
    for target,path,bounds in [('qwen3b',qwen_tables,(23,40)),('olmo1b',olmo_tables,(14,32))]:
        result=construct(json.loads(Path(path).read_text()),*bounds)
        (out/(target+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(target,result['construction']['kkt'])
        if target=='qwen3b':
            for suffix,name in [('0','Smooth_MrBudget'),('1','MrUni')]:
                job=dict(id=name,spec=dict(operator='static',table=result[name]),panel='full',nll_docs=16,nll_lengths=[8192,16384,32768])
                (root/'queue'/f'044{suffix}_{name}.json').write_text(json.dumps(job,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--qwen-tables',required=True);p.add_argument('--olmo-tables',required=True)
    a=p.parse_args();prepare(a.root,a.qwen_tables,a.olmo_tables)
