"""Reimplemented from DESIGN_SOURCE: finite integer-prior QR checks, not LM evidence."""
import json
from pathlib import Path
import numpy as np
from build_tables import active_table, build_table, ARMS

def geometry(freq, length, prior):
    t=np.arange(length,dtype=float)/length
    mu=np.ones(length) if prior=='uniform' else length-np.arange(length,dtype=float)
    mu/=mu.sum()
    root=np.sqrt(mu)[:,None]
    q0=np.linalg.qr(root*np.column_stack([np.ones(length),t]))[0]
    qs=[]; errors=[]
    for w in freq:
        x=float(w)*length
        v=np.column_stack([np.cos(x*t),t*np.sinc(x*t/np.pi)])
        q=np.linalg.qr(root*v)[0]
        qs.append(q)
        errors.append(float(np.square(q-np.einsum('ij,jk->ik',q0,np.einsum('ji,jk->ik',q0,q))).sum()))
    a=np.concatenate(qs,axis=1)
    spectrum=np.linalg.eigvalsh(np.einsum('ji,jk->ik',a,a))[::-1]
    slow=np.asarray(freq)*length<=.25
    m=int(slow.sum()); r0=2*(len(freq)-m+1) if m else 2*len(freq)
    eta=float(np.asarray(errors)[slow].sum())
    tail=float(spectrum[r0:].sum())
    assert tail<=eta+1e-10
    assert abs(spectrum.sum()-2*len(freq))<1e-9
    # T1 with the registered epsilon and exact discrete moments.
    v0=root*np.column_stack([np.ones(length),t])
    sigma=np.linalg.svd(v0,compute_uv=False)[-1]
    h=np.sqrt(np.sum(mu*(t**4/4+t**6/36)))
    for w,e in zip(freq,errors):
        x=float(w)*length
        if x<=.25:
            bound=x**4*h*h/(sigma-.25**2*h)**2
            assert e<=bound+1e-12
    return {'K':len(freq),'length':length,'prior':prior,'slow_pairs':m,'r0':r0,
            'eta':eta,'tail':tail,'spectrum':spectrum.tolist(),
            'threshold_counts':{str(z):int((spectrum>=z).sum()) for z in (.001,.01,.1)},
            'pair_residuals':errors}

if __name__=='__main__':
    records=[]
    for length in (2048,4096):
        for prior in ('causal','uniform'):
            for arm in ARMS:
                f=build_table(arm); f=f[f>0]
                records.append({'arm':arm,**geometry(f,length,prior)})
            for evq in (False,True):
                records.append({'arm':'E8' if evq else 'G8',**geometry(active_table(8,evq),length,prior)})
    Path(__file__).with_name('theory_checks.json').write_text(json.dumps({'status':'CPU_GEOMETRY_ONLY','records':records},indent=2)+'\n')
    print('32 finite-grid configurations: T1/T2 checked; no neural-model claim')
