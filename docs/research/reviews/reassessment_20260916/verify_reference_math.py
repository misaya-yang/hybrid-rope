"""CPU-only checks for the reassessment; no checkpoint or model execution."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name('cpu_checks.json')

def lse(x):
    m = np.max(x)
    return float(m + np.log(np.exp(x-m).sum()))

rng = np.random.default_rng(20260916)
error = 0.0
for _ in range(1000):
    logits = rng.normal(size=12)*3
    eta = rng.normal(size=12)*2
    def odds(x): return lse(x[:4])-lse(x[4:])
    direct = odds(logits+eta)-odds(logits)
    rhs = lse(logits[:4]-lse(logits[:4])+eta[:4])-lse(logits[4:]-lse(logits[4:])+eta[4:])
    error = max(error, abs(direct-rhs))
assert error < 1e-12
nodes,weights = np.polynomial.legendre.leggauss(128)
t=(nodes+1)/2; weights=weights/2
cosh=[]
for tau in [.05,.5,1.414,4.,10.]:
    rho=tau*np.cosh(tau*(1-t))/np.sinh(tau)
    survival=np.sinh(tau*(1-t))/np.sinh(tau)
    value=.5*np.dot(weights,rho*rho+tau*tau*survival*survival)
    exact=.5*tau/np.tanh(tau)
    assert abs(value-exact)<1e-10
    assert abs(np.dot(weights,rho)-1)<1e-12
    cosh.append({'tau':tau,'objective':float(value),'closed_form':float(exact)})
waterbed=[]
for distance in [1,3]:
    old=float(np.logaddexp(0,-np.cos(distance*.2)))
    new=float(np.logaddexp(0,-np.cos(distance*.1)))
    assert new<old
    waterbed.append({'distance':distance,'original_loss':old,'reallocated_loss':new})
data=json.loads((ROOT/'paper-2027/figs/completed_evidence_inputs.json').read_text())['score_rows']
means={}
for group,arm in [('ncp','ncp'),('native','native')]:
    tasks=defaultdict(list)
    for _,task,length,score in data[group][arm]:
        assert length==4096
        tasks[task].append(score)
    assert len(tasks)==13 and all(len(x)==60 for x in tasks.values())
    means[arm]={k:float(np.mean(v)) for k,v in tasks.items()}
delta={k:means['ncp'][k]-means['native'][k] for k in means['ncp']}
report={'status':'PASS','model_execution':False,'softmax_log_odds_identity':{'cases':1000,'max_error':error,'scope':'Fixed local logits and finite shifts; not an end-to-end mediation result'},'cosh_reference_objective':cosh,'fixed_budget_not_task_zero_sum':waterbed,'ncp_existing_score_reaggregation':{'rows_per_arm':780,'macro_delta_pp':100*float(np.mean(list(delta.values()))),'vt_delta_pp':100*delta['vt'],'qa_family_delta_pp':100*float(np.mean([delta['qa_1'],delta['qa_2']])),'leave_one_task_out_delta_pp':{k:100*float(np.mean([v for q,v in delta.items() if q!=k])) for k in delta},'primary_endpoint_unchanged':'Full-13 task-equal; LOTO is descriptive only'}}
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if __name__=='__main__': pass
