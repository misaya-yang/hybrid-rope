"""CPU mechanical contracts; fixtures are NOT qualified natural evidence."""
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.lib.rope.generation_contract import kl_argmax_radius,projection_parameter_count
from scripts.analysis.export_single_table_controls import same_support_geometric
from scripts.analysis.review_native_constrained_transfer import decide,task_pairs

SPEC=importlib.util.spec_from_file_location('constrained',Path(__file__).parents[1]/'scripts/train/train_single_table_native_constrained.py')
M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def test_kl_boundary_is_attained_and_top2_is_nearest():
    p=[.6,.3,.1]; q=[.45,.45,.1]
    radius=kl_argmax_radius(p)
    assert radius==pytest.approx(sum(a*math.log(a/b) for a,b in zip(p,q)))
    farther=.6*math.log(1.2/.7)+.1*math.log(.2/.7)
    assert radius < farther
    assert kl_argmax_radius([.5,.5])==0
    assert kl_argmax_radius([1.,0.])==pytest.approx(math.log(2))
    with pytest.raises(ValueError): kl_argmax_radius([.6,.6])


def test_all_linear_and_attention_only_match_actual_olmo_budget():
    c={'hidden_size':2048,'intermediate_size':8192,'num_hidden_layers':16,
       'num_attention_heads':16,'num_key_value_heads':16}
    assert projection_parameter_count(c,M.MODULES,16)==12058624
    assert projection_parameter_count(c,M.MODULES[:4],46)==12058624


def test_g_control_pins_sampled_support_and_normalizes_interiors():
    table=np.geomspace(1.,1e-5,64).astype(np.float32)
    table[1:-1] *= np.linspace(.9,1.,62).astype(np.float32)
    g=same_support_geometric(table)
    np.testing.assert_array_equal(g[[0,-1]],table[[0,-1]])
    z=(-np.log(g)-(-np.log(g[0])))/(-np.log(g[-1])+np.log(g[0]))
    np.testing.assert_allclose(z,np.linspace(0.,1.,64),atol=1e-7)


def test_joint_pair_relabel_requires_qk_norm_permutation():
    rng=np.random.default_rng(5); k=8
    q,key=rng.normal(size=(2,2*k)); gamma_q,gamma_k=rng.uniform(.3,2.,size=(2,2*k))
    perm=rng.permutation(k); idx=np.r_[perm,perm+k]
    omega=np.geomspace(1.,1e-4,k)
    def norm(x,gamma): return x/np.sqrt(np.mean(x*x))*gamma
    def rot(w):
        c=np.diag(np.cos(w*37)); s=np.diag(np.sin(w*37))
        return np.block([[c,-s],[s,c]])
    original=norm(q,gamma_q)@rot(omega)@norm(key,gamma_k)
    joint=norm(q[idx],gamma_q[idx])@rot(omega[perm])@norm(key[idx],gamma_k[idx])
    wrong=norm(q[idx],gamma_q)@rot(omega[perm])@norm(key[idx],gamma_k)
    assert joint==pytest.approx(original,abs=1e-12)
    assert abs(wrong-original) > .01


def test_native_pool_rejects_cross_split_sources(tmp_path):
    data=[{'id':'a','source_id':'same','split':'train','group':'text','input_ids':[1,2], 'prediction_positions':[0]},
          {'id':'b','source_id':'same','split':'validation','group':'text','input_ids':[1,2], 'prediction_positions':[0]}]
    path=tmp_path/'pool.jsonl'; path.write_text('\n'.join(json.dumps(r) for r in data))
    manifest=tmp_path/'manifest.json'
    M.write_json(manifest,{'status':'NATIVE_REPLAY_POOL_V1','rows_path':path.name,'rows_sha256':M.sha(path)})
    with pytest.raises(ValueError,match='leakage'): M.read_native_pool(manifest)


def mechanical_task_fixture(root):
    qualification=[]; views=[]
    for i in range(128):
        family='single_evidence' if i<64 else 'double_evidence' if i<96 else 'binding'
        for world in (0,1):
            gold=[1000+2*i+world,0]
            qualification.append({'semantic_id':str(i),'world':world,'truth_verified':True,
                                  'generated_ids':gold,'target_ids':gold,'compact_prompt_ids':[3,4], 'gold_margins':[.5,2.]})
            for length in (2048,8192,16384):
                views.append({'semantic_id':str(i),'world':world,'source_id':f'doc{i}',
                              'template_lineage':f'template{i}','split':'train','family':family,
                              'layout':'compact' if length==2048 else 'far','length_cap':length,
                              'prompt_ids':[3,4] if length==2048 else [8]*(length-2),'target_ids':gold})
    qpath=root/'qualification.json'; vpath=root/'views.jsonl'
    candidate_pool=root/'candidates.jsonl'; candidate_pool.write_text('mechanical fixture, not natural truth\n')
    M.write_json(qpath,{'teacher_weight_sha256':M.WEIGHT_SHA,'teacher_table':'native','teacher_gain':1.,'rows':qualification,
                       'selection_rule':'fixed_order_native_compact_both_worlds','screened_candidates':128,
                       'candidate_pool_path':candidate_pool.name,'candidate_pool_sha256':M.sha(candidate_pool),'rejections':[]})
    with vpath.open('w') as f:
        for row in views: f.write(json.dumps(row)+'\n')
    manifest=root/'manifest.json'
    M.write_json(manifest,{'status':'QUALIFIED_NATURAL_TRANSPORT_V1','eos_token_id':0,
        'views_path':vpath.name,'views_sha256':M.sha(vpath),
        'qualification_path':qpath.name,'qualification_sha256':M.sha(qpath),
        'evaluation_splits':{'validation':{'groups':64,'lengths':[2048,16384]},
                             'test':{'groups':256,'lengths':[2048,16384,32768,65536]}}})
    return manifest


def test_task_contract_and_exposure_schedule(tmp_path,monkeypatch):
    manifest=mechanical_task_fixture(tmp_path)
    # This fixture has only training views. Production must reject it as incomplete.
    with pytest.raises(ValueError,match='missing evaluation'): M.read_tasks(manifest)
    # Isolate training scheduling after independently tested evaluation validation.
    monkeypatch.setattr(M,'evaluation_rows',lambda *args: [])
    task,_=M.read_tasks(manifest)
    assert len(task)==768
    order=M.task_order(task,42)
    assert [r['length_cap'] for r in order[:6]]==[2048,8192,16384]*2
    assert sum(r['length_cap'] for r in order)==6815744
    assert {r['semantic_id'] for r in order}=={str(i) for i in range(128)}
    assert task[0]['teacher_margin_targets']==[.5,1.]
    qpath=tmp_path/'qualification.json'; q=json.loads(qpath.read_text())
    q['rows'][0]['generated_ids']=[999,0]
    M.write_json(qpath,q)
    m=json.loads(manifest.read_text()); m['qualification_sha256']=M.sha(qpath); M.write_json(manifest,m)
    with pytest.raises(ValueError,match='qualification failed'): M.read_tasks(manifest)


def test_missing_assets_are_explicit_blockers(tmp_path):
    with pytest.raises(ValueError,match='BLOCKED_DATA_QUALIFICATION'): M.read_tasks(tmp_path/'missing.json')
    with pytest.raises(ValueError,match='BLOCKED_NATIVE_DATA'): M.read_native_pool(tmp_path/'missing.json')


def test_native_replay_covers_448_unique_rows_with_fixed_strata():
    rows=[{'id':f'{g}:{i}','group':g,'split':'train'} for g in M.GROUPS for i in range(128)]
    pools=M.native_replay_order(rows,42)
    chosen=[r['id'] for group in M.GROUPS for r in pools[group][:112]]
    assert len(chosen)==len(set(chosen))==448
    assert pools==M.native_replay_order(rows,42)
    assert pools!=M.native_replay_order(rows,43)


@pytest.mark.parametrize('count',[1,32,96])
def test_smoke_and_full_schedule_really_update(count):
    rates=[M.learning_rate_factor(i,count) for i in range(count)]
    assert all(0<r<=1 for r in rates)
    assert rates[0]>0 and max(rates)==1


def test_evaluation_matrix_rejects_missing_or_duplicate_before_cuda():
    data=[{'semantic_id':sid,'split':'validation','world':world,'layout':layout,'length_cap':length}
          for sid in ('a','b') for world in (0,1)
          for layout,length in (('compact',2048),('near',16384),('far',16384))]
    assert len(M.evaluation_rows(data,'validation',[2048,16384],2))==12
    for broken in (data[:-1],data+[data[0]],data[:6]):
        with pytest.raises(ValueError,match='missing/duplicate|missing evaluation'):
            M.evaluation_rows(broken,'validation',[2048,16384],2)


def test_worlds_must_differ_in_complete_answer_not_just_token_ids():
    tok=SimpleNamespace(decode=lambda ids,**kw: ''.join({1:'a',2:'b',3:'ab',4:'c'}[i] for i in ids))
    base={'semantic_id':'one','split':'validation'}
    data=[{**base,'world':0,'target_ids':[1,2,0]},{**base,'world':1,'target_ids':[4,0]}]
    M.validate_answer_worlds(data,tok)
    data[1]['target_ids']=[3,0]
    with pytest.raises(ValueError,match='share a complete'): M.validate_answer_worlds(data,tok)


@pytest.mark.parametrize('native,confirmed,compact,near,far,groups,step,expected',[
    (False,False,1,1,1,16,64,'STOP_NATIVE_DAMAGE'),
    (True,True,1,1,0,4,64,'UNRESOLVED_CONTROLS'),
    (True,True,1,.2,0,16,64,'UNRESOLVED_LOCAL_OR_BACKGROUND'),
    (True,True,1,1,0,16,64,'STOP_ZERO_LONG_GENERATION'),
    (True,False,1,1,.5,16,64,'RETENTION_INTERVAL_UNRESOLVED'),
    (True,True,1,1,.5,16,64,'REVIEW_THEN_RESUME_FIXED_RECIPE'),
    (True,True,1,1,.5,16,128,'VALIDATION_CANDIDATE_NOT_FINAL_CLAIM'),
])
def test_next_action_cannot_promote_proxy_or_unresolved_controls(native,confirmed,compact,near,far,groups,step,expected):
    cells={'one':{'native_compact_groups':groups,'compact':compact,'near':near,'far':far}}
    assert decide(native,confirmed,cells,step)[0]==expected


def test_natural_review_rescores_raw_full_output_and_eos():
    tok=SimpleNamespace(decode=lambda ids,**kw: ''.join({1:'correct',2:' extra',0:'<EOS>'}[i] for i in ids))
    base={'semantic_id':'a','family':'one','layout':'far','length_cap':16384,'eos_token_id':0,
          'accepted_full_answers':['correct'],'full_exact_eos':True,'unmodified_output_text':'correct','generated_ids':[1,0]}
    assert list(task_pairs([{**base,'world':w} for w in (0,1)],tok).values())==[True]
    for ids,text in (([1],'correct'),([1,2,0],'correct extra')):
        invalid=[{**base,'world':w,'generated_ids':ids,'unmodified_output_text':text} for w in (0,1)]
        with pytest.raises(ValueError,match='re-score'): task_pairs(invalid,tok)


def test_teacher_cache_rejects_duplicate_or_corrupted_entries(tmp_path):
    checkpoint=tmp_path/'model'; checkpoint.mkdir()
    M.write_json(checkpoint/'config.json',{'vocab_size':2})
    M.write_json(checkpoint/'tokenizer.json',{'fixture':True})
    pool=tmp_path/'pool.json'; M.write_json(pool,{'fixture':True})
    cache=tmp_path/'cache'; cache.mkdir()
    row={'id':'a','split':'train','input_ids':[1,2],'prediction_positions':[0]}
    array=cache/'a.npy'; np.save(array,np.zeros((1,2),dtype=np.float32))
    entry={'id':'a','path':'a.npy','sha256':M.sha(array),'row_sha256':M.canonical(row),'shape':[1,2],'dtype':'float32'}
    manifest={'status':'ORIGINAL_NATIVE_FULL_VOCAB_CACHE_V1','pool_sha256':M.sha(pool),'entries':[entry],
              'teacher':{'table_is_native':True,'gain':1.,'adapter_sha256':None,'checkpoint_sha256':M.WEIGHT_SHA,
                         'tokenizer_files':M.tokenizer_identity(checkpoint)}}
    M.write_json(cache/'manifest.json',manifest)
    args=SimpleNamespace(teacher_cache=cache,native_pool=pool,checkpoint=checkpoint)
    assert M.validate_teacher_cache(args,[row])['a']==entry
    manifest['entries'].append(entry); M.write_json(cache/'manifest.json',manifest)
    with pytest.raises(ValueError,match='duplicate'): M.validate_teacher_cache(args,[row])
    manifest['entries']=[entry]; M.write_json(cache/'manifest.json',manifest)
    np.save(array,np.ones((1,2),dtype=np.float32))
    with pytest.raises(ValueError,match='bytes'): M.validate_teacher_cache(args,[row])


def test_ffn_weight_derivative_matches_finite_difference():
    rng=np.random.default_rng(9); h=rng.normal(size=3)
    gate,up=rng.normal(size=(2,5,3)); down=rng.normal(size=(3,5))
    dg,du=rng.normal(size=(2,5,3)); dd=rng.normal(size=(3,5))
    sigmoid=lambda x: 1/(1+np.exp(-x))
    silu=lambda x: x*sigmoid(x)
    f=lambda g,u,d: d@(silu(g@h)*(u@h))
    g=gate@h; u=up@h; derivative=sigmoid(g)+g*sigmoid(g)*(1-sigmoid(g))
    predicted=dd@(silu(g)*u)+down@((derivative*(dg@h))*u+silu(g)*(du@h))
    eps=1e-6
    observed=(f(gate+eps*dg,up+eps*du,down+eps*dd)-f(gate-eps*dg,up-eps*du,down-eps*dd))/(2*eps)
    np.testing.assert_allclose(predicted,observed,atol=1e-8)


def test_e2_stage_driver_dispatches_one_stage_and_preserves_resume_args(tmp_path):
    fake=tmp_path/'capture.py'
    fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o700)
    capture=tmp_path/'argv.json'
    env={**os.environ,'EVQ_PYTHON':str(fake),'EVQ_CHECKPOINT':str(tmp_path/'model'),
         'EVQ_WORK_DIR':str(tmp_path/'output'),'TASK_MANIFEST':str(tmp_path/'tasks'),
         'NATIVE_POOL':str(tmp_path/'native'),'E2_ARM':'N','E2_LABEL':'fixture',
         'CAPTURE':str(capture),'TEACHER_CACHE':str(tmp_path/'cache')}
    env.pop('SINGLE_TABLE_GPU_AUTHORIZED',None)
    driver=Path(__file__).parents[1]/'scripts/train/run_native_constrained_transfer.sh'
    blocked=subprocess.run(['bash',str(driver),'train'],env=env,capture_output=True,text=True)
    assert blocked.returncode==2 and not capture.exists()
    env.update(SINGLE_TABLE_GPU_AUTHORIZED='YES',E2_RESUME=str(tmp_path/'step_064'),E2_STOP_STEP='96')
    subprocess.run(['bash',str(driver),'resume'],env=env,check=True)
    argv=json.loads(capture.read_text())
    assert argv[:2]==['scripts/train/train_single_table_native_constrained.py','train']
    assert argv[argv.index('--resume')+1]==str(tmp_path/'step_064')
    assert argv[argv.index('--stop-after-step')+1]=='96'
    subprocess.run(['bash',str(driver),'native'],env=env,check=True)
    argv=json.loads(capture.read_text())
    assert argv[1]=='native-evaluate' and '--retention-manifest' not in argv
