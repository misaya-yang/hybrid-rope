"""Validate and package recovered result assets, without running a model."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OWNERS={
 'eos':'rebuttal/rebuttal_0723/theory_results/evq_query_gap_realized_eos32_20260728/FINAL_METRICS_AND_LINEAGE.json',
 'qk':'rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/metrics.json',
 'c2':'paper-2027/research/attention-aware-retrofit/evidence/LOW_DIM_COUPLING_GPU_RECEIPT_20260901.json',
 'full_lag':'docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json',
 'controls':'docs/research/ROPE_OLMO_BM_RESULT_20260908.json',
}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reconstruct():
    values={k:json.loads((ROOT/v).read_text()) for k,v in OWNERS.items()}
    # C42: use the exact recorded construction functions; do not execute the module.
    code=ROOT/'ds_workspace/recon_20260910/code/coverage_theory_20260911.py'
    tree=ast.parse(code.read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['m_C42','m_C42V24']]
    scope={'np':np,'K':64}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(code),'exec'),scope)
    a,b=scope['m_C42'](),scope['m_C42V24']()
    assert abs(a.sum()-42)<1e-12 and abs(b.sum()-42)<1e-12
    assert np.allclose(a[33:],1) and np.allclose(b[33:],1)
    assert a[0]==b[0]==0 and a[-1]==b[-1]==1
    epsa,epsb=np.diff(a),np.diff(b)
    assert np.all(epsa>=-1e-14) and np.all(epsb>=-1e-14)
    assert abs(epsa@np.arange(1,64)-epsb@np.arange(1,64))<1e-12
    rawpaths=[ROOT/f'ds_workspace/recon_20260910/work/jsonl/olmo_c42/{name}.jsonl' for name in ['ctl_C42','ctl_C42V24']]
    raw=[{r['row_id']:r for r in map(json.loads,p.read_text().splitlines())} for p in rawpaths]
    assert set(raw[0])==set(raw[1]) and len(raw[0])==350
    scores=np.array([[raw[j][i]['correct'] for i in sorted(raw[0])] for j in range(2)])
    diff=scores[1]-scores[0]
    assert np.allclose(scores.mean(1),[.4367142857142857,.544],atol=1e-14)
    contrast={'means':scores.mean(1).tolist(),'difference':float(diff.mean()),'paired_t':float(diff.mean()/(diff.std(ddof=1)/np.sqrt(350))),
      'wins':int(np.count_nonzero(diff>0)),'losses':int(np.count_nonzero(diff<0)),'ties':int(np.count_nonzero(diff==0)),
      'sum_m':[float(a.sum()),float(b.sum())],'m':[a.tolist(),b.tolist()],
      'scores':scores.tolist(),'tasks':[raw[0][i]['task'] for i in sorted(raw[0])],
      'raw_sha256':[digest(p) for p in rawpaths],'construction_sha256':digest(code)}
    values['c42']=contrast
    return {'sources':{k:{'path':v,'sha256':digest(ROOT/v)} for k,v in OWNERS.items()},'data':values}

def verify(packet):
    d=packet['data'];e=d['eos']['strict_autoregressive_full_string_plus_eos']
    assert [e['evq'][str(L)]['correct'] for L in [4096,8192,16384]]==[100,98,60]
    assert [e['native'][str(L)]['correct'] for L in [4096,8192,16384]]==[95,18,0]
    assert all(e[a]['terminal_eos_rate_all_cells']==1 for a in ['evq','native'])
    q=d['qk']['qa_2wiki']['qk_phase_adapted'];assert abs(q['evq']['8192']['token_f1']*100-21.481024531024534)<1e-9
    for family in ['summary64','summary128']:
        for row in d['full_lag'][family]:
            pairs=row['paired'] if family=='summary64' else {'mrpro':row['paired']}
            from fractions import Fraction
            for arm,cmp in pairs.items():
                delta=np.mean([float(Fraction(v)) for v in cmp['differences_fraction']])*100
                assert abs(delta-(row['scores']['fulllagp2']-row['scores'][arm]))<1e-10
    c=d['controls']['experiments']['existing_controls']['arms']
    for r in c.values():
        assert r['table_identity']['gain']==1.138629436111989
        for v in r['summary']['by_length'].values(): assert abs(np.mean(list(v['task_accuracy'].values()))-v['macro_accuracy'])<1e-12
    scores=np.array(d['c42']['scores']);diff=scores[1]-scores[0]
    assert abs(diff.mean()-d['c42']['difference'])<1e-12
    assert (d['c42']['wins'],d['c42']['losses'],d['c42']['ties'])==(75,26,249)
    c2=d['c2']['qwen_post_gate_diagnostic'];assert (c2['c2_64k_macro'],c2['c2_128k_macro'])==(.6775,.5725)
    print('Verified complete-answer/EOS and QK metrics, C2 receipt, FullLag paired fractions, four-method controls, and C42 raw-score/shape invariants.')

if __name__=='__main__':
    out=OUT/'recovered_asset_inputs.json'
    if all((ROOT/v).is_file() for v in OWNERS.values()) and (ROOT/'ds_workspace/recon_20260910/code/coverage_theory_20260911.py').is_file():
        packet=reconstruct();out.write_text(json.dumps(packet,indent=2)+'\n')
    else:packet=json.loads(out.read_text())
    verify(packet)
