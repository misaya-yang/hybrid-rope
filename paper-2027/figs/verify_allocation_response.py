"""NumPy-only reproduction of Appendix K mathematics; no model execution."""
from pathlib import Path
import json
import math
import numpy as np


def profiles(n):
    q=np.arange(n+1,dtype=float)
    p=q*(q+1)/(n*(n+1))
    t=q*(3*n*n+3*n+1-q*q)/(n*(n+1)*(2*n+1))
    w=3*n/(2*(2*n+1))
    return t,p,(1-w)*q/n+w*(2*q/n-p)


def tables():
    native=np.exp(-np.log(500000.)*np.arange(64)/64)
    return {name:native*4**(-np.r_[np.zeros(18),m,np.ones(28)])
            for name,m in zip(('T','P','C'),profiles(17))}


def cross(w,v,length):
    d,s=(w-v)*length,(w+v)*length
    sinc=lambda x:np.sinc(x/np.pi)
    odd=lambda x:0. if x==0 else 2*np.sin(x/2)**2/x
    return .5*np.array([[sinc(d)+sinc(s),odd(s)-odd(d)],
                        [odd(s)+odd(d),sinc(d)-sinc(s)]])


def rank_origin(omega,length):
    own=[cross(w,w,length) for w in omega]
    frame=2*len(omega)
    for i in range(len(omega)):
        for j in range(i):
            h=cross(omega[i],omega[j],length)
            frame+=2*np.trace(np.linalg.solve(own[i],h)@np.linalg.solve(own[j],h.T))
    return (2*len(omega))**2/frame


def rank_midpoint(omega,length):
    d=(omega[:,None]-omega[None,:])*length
    s=(omega[:,None]+omega[None,:])*length
    cc=(np.sinc(d/(2*np.pi))+np.sinc(s/(2*np.pi)))/2
    ss=(np.sinc(d/(2*np.pi))-np.sinc(s/(2*np.pi)))/2
    cc/=np.sqrt(np.diag(cc)[:,None]*np.diag(cc)[None,:])
    ss/=np.sqrt(np.diag(ss)[:,None]*np.diag(ss)[None,:])
    return (2*len(omega))**2/(np.square(cc).sum()+np.square(ss).sum())


def rho(delta,length):
    return 2-2*np.sinc(length*delta/(2*np.pi))/np.sinc(delta/(2*np.pi))*np.cos((length-1)*delta/2)


def main():
    saved=json.loads(Path(__file__).with_name('allocation_response_inputs.json').read_text())
    omega=tables(); rank_rows={}; max_error=0.
    for length in (8192,16384,32768):
        row={}
        for name,values in omega.items():
            a,b=rank_origin(values,length),rank_midpoint(values,length)
            max_error=max(max_error,abs(a-b))
            assert abs(a-saved['ranks'][str(length)][name]['continuous_uniform'])<1e-8
            row[name]=a
        assert row['T']<row['C']<row['P']
        rank_rows[str(length)]=row
    assert max_error<1e-8
    for n in range(1,65):
        t,p,c=profiles(n); q=np.arange(n+1)
        expected=q*(n-q)*(3*n+q+1)/(n*(n+1)*(2*n+1))
        assert np.max(abs(t-p-expected))<1e-13
        assert abs(sum(t-c))<1e-12
    t,p,c=profiles(17); largest_ratio={}
    for scale in (2,4,8,16):
        largest_ratio[str(scale)]=float(np.max(scale**(t-p)))
        assert abs(largest_ratio[str(scale)]-saved['largest_wavelength_ratios'][str(scale)])<1e-12
    phase={}
    for name in ('P','C'):
        delta=omega['T']-omega[name]
        phase[name]={'max_unwrapped':float(32767*np.max(abs(delta))),
                     'slot_rms':float(np.sqrt(np.mean(rho(delta,32768))))}
        assert abs(phase[name]['max_unwrapped']-saved['phase32k'][name]['max_unwrapped'])<1e-9
        assert abs(phase[name]['slot_rms']-saved['phase32k'][name]['slot_rms'])<1e-8
    rng=np.random.default_rng(20260915)
    for _ in range(200):
        pairs=int(rng.integers(1,10)); length=float(rng.uniform(1,10000))
        w=rng.uniform(.001,.49,pairs)/length; eta=length*w.max()
        distance=float(rng.uniform(-length,length)); R=np.zeros((2*pairs,2*pairs))
        for j,v in enumerate(w):
            co,si=np.cos(v*distance),np.sin(v*distance)
            R[2*j:2*j+2,2*j:2*j+2]=[[co,-si],[si,co]]
        q,k=rng.normal(size=(2,2*pairs))
        assert abs(q@(R-np.eye(2*pairs))@k)<=eta*np.linalg.norm(q)*np.linalg.norm(k)+1e-12
        for j in range(2*pairs):
            assert R[j,j]-max(np.delete(R[j],j))>=1-2*eta-1e-12
    max_derivative_error=0.
    for _ in range(300):
        u=float(rng.uniform(.001,.999)); s=float(rng.uniform(1.01,32)); h=1e-5
        f=lambda z:-math.log(1-u+u/z)
        numeric=(f(s*math.exp(h))-f(s*math.exp(-h)))/(2*h)
        exact=u/(s*(1-u)+u)
        max_derivative_error=max(max_derivative_error,abs(numeric-exact))
        assert 1/(1-u+u/s)<1/(1-u)
    assert max_derivative_error<1e-9
    print(json.dumps({'status':'PASS','scope':'Public-parameter math; no new model answers.',
        'ranks':rank_rows,'max_independent_rank_error':max_error,
        'largest_wavelength_ratios':largest_ratio,'phase32k':phase,
        'slow_content_cases':200,'finite_profile_cases':64,'scale_derivative_cases':300,
        'max_scale_derivative_error':max_derivative_error},indent=2))


if __name__=='__main__':main()
