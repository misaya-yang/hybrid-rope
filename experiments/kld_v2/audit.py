"""Independent algebra checks. These do not establish trained-model utility."""
import json
from pathlib import Path
import torch

torch.set_default_dtype(torch.float64)

def recurrence(k,v,beta,decay,z=None,s0=None,h0=None):
    d=k.shape[1];s=torch.zeros(d,v.shape[1]) if s0 is None else s0.clone();h=torch.zeros_like(s) if h0 is None else h0.clone()
    if z is not None:s=s+z*h
    for kt,vt,bt,dt in zip(k,v,beta,decay):
        sb=dt[:,None]*s;hb=dt[:,None]*h
        read=kt@sb
        if z is None:
            s=sb+bt*torch.outer(kt,vt-read)
            h=hb+bt*torch.outer(kt,read-kt@hb)
        else:s=sb+bt*torch.outer(kt,vt-(1-z)*read)
    return s,h

def chunk(k,v,beta,decay,s0=None,h0=None):
    n,d=k.shape;L=torch.zeros(n,n);X=torch.zeros(n,d)
    s0=torch.zeros(d,v.shape[1]) if s0 is None else s0
    h0=torch.zeros_like(s0) if h0 is None else h0
    for t in range(n):
        X[t]=k[t]*decay[:t+1].prod(0)
        for s in range(t):L[t,s]=beta[s]*torch.sum(k[t]*decay[s+1:t+1].prod(0)*k[s])
    A=torch.eye(n)+L
    es=torch.linalg.solve_triangular(A,v-X@s0,upper=False)
    eh=torch.linalg.solve_triangular(A,X@(s0-h0)+L@es,upper=False)
    s=decay.prod(0)[:,None]*s0;h=decay.prod(0)[:,None]*h0
    for j in range(n):
        write=beta[j]*decay[j+1:].prod(0)*k[j]
        s+=torch.outer(write,es[j]);h+=torch.outer(write,eh[j])
    return s,h

def main():
    torch.manual_seed(42);n,d=13,7
    k=torch.nn.functional.normalize(torch.randn(n,d),dim=-1)
    v=torch.randn(n,3);beta=torch.rand(n);decay=.8+.2*torch.rand(n,d)
    s,h=recurrence(k,v,beta,decay)
    deriv=torch.autograd.functional.jacobian(lambda z:recurrence(k,v,beta,decay,z)[0],torch.tensor(0.))
    cs,ch=chunk(k,v,beta,decay)
    errors={'derivative_max_abs':float((deriv-h).abs().max()),'chunk_s_max_abs':float((cs-s).abs().max()),'chunk_h_max_abs':float((ch-h).abs().max())}
    s0=torch.randn(d,3);h0=torch.randn(d,3)
    ns,nh=recurrence(k,v,beta,decay,s0=s0,h0=h0)
    ncs,nch=chunk(k,v,beta,decay,s0=s0,h0=h0)
    nd=torch.autograd.functional.jacobian(lambda z:recurrence(k,v,beta,decay,z,s0,h0)[0],torch.tensor(0.))
    errors.update(nonzero_chunk_s=float((ns-ncs).abs().max()),nonzero_chunk_h=float((nh-nch).abs().max()),nonzero_derivative=float((nd-nh).abs().max()))
    assert max(errors.values())<1e-12
    # Equal extra state can approximate the derivative: O(epsilon) convergence.
    fd=[]
    for eps in (.01,.001,.0001):
        x=recurrence(k,v,beta,decay,torch.tensor(eps))[0]
        fd.append(float(((x-s)/eps-h).norm()))
    assert fd[2]<fd[1]<fd[0]
    # The recoverability identity includes all competing sources.
    C=torch.randn(4,20);sigma=.1*torch.eye(4)
    capacity=torch.trace(torch.linalg.pinv(C@C.T+sigma)@(C@C.T))
    assert 0<=capacity<=4
    result={'status':'CPU_ALGEBRA_ONLY','errors':errors,'finite_difference_errors':fd,
      'linear_capacity':float(capacity),'state_budget_bytes_per_sequence_fp32':16*128*128*4,
      'random_interference':{str(n):{'S':(127/128)**n,'H':n/128*(127/128)**(n-1)} for n in (128,512)},
      'limitation':'No trained checkpoint, full-network generation, or GPU-kernel performance tested.'}
    Path(__file__).with_name('audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
