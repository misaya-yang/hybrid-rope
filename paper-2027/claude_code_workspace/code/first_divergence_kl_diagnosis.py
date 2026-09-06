#!/usr/bin/env python3
"""Direction 1A: Native first-divergence KL-barrier diagnosis on exposed outputs.

STATUS (2026-09-06): 已收官——两方向审计时代（项目重置前）脚本，其方向被
Round 12 的谱预算框架取代；保留为历史记录，不再运行。服务器产物在
/root/autodl-tmp/claude_audit_prep_20260905/（本地镜像 results_20260905/ 已清理）。

Mechanism analysis of the already exposed N0/N128 Native confirmation. It never
updates weights, never tunes table/gain/decoder, and never feeds failure cases
back into training. For each predeclared case it locates the first token where
the archived baseline and candidate greedy outputs differ, then measures on the
shared teacher prefixes: the top-2 probabilities of the baseline distribution,
the exact top-change KL barrier B(p), the full-vocabulary KL(p||q), whether the
candidate argmax is unchanged, and the candidate margin on the teacher token.

KL(p||q) < B(p) implies the argmax is unchanged; the converse is NOT asserted.
B=0 or decoding ties carry no positive-margin certificate and are reported, not
rescued by thresholds. Full-vocabulary KL uses log-softmax; no top-k substitute.

Cases: the 29 originally-correct->incorrect confirmation rows (15 format/index,
5 instruction, 9 reasoning) plus one predeclared rule-matched retained control
per lost case. Matching uses only identity metadata (stratum and canonical
row-id order), never KL outcomes. This is retrospective diagnosis on exposed
data, not new generalization evidence.

Workspace: paper-2027/claude_code_workspace (Claude-owned; Codex edits live
elsewhere). Repo root is discovered by walking upward or via HYBRID_ROPE_ROOT.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path


def _repo_root():
    here=Path(__file__).resolve()
    for parent in (here.parent,*here.parents):
        if (parent/'scripts'/'experiments'/'single_table_generation.py').is_file():
            return parent
    env=os.environ.get('HYBRID_ROPE_ROOT')
    if env and (Path(env)/'scripts'/'experiments'/'single_table_generation.py').is_file():
        return Path(env).resolve()
    raise ImportError('set HYBRID_ROPE_ROOT to the hybrid-rope repository root')


ROOT=_repo_root()
sys.path.insert(0,str(ROOT))
from scripts.experiments.single_table_generation import (
    sha,canonical,rows,write_json,load_runtime,guard_resources)

GENERATIVE=('instruction','reasoning','position_format')
EXPECTED_LOST={'position_format':15,'instruction':5,'reasoning':9}
LOCK_KEYS=('table_sha256','gain','adapter_sha256','adapter_config_sha256',
           'native_pool_sha256','checkpoint_sha256','checkpoint_config_sha256','tokenizer_files')
RUNTIME_KEYS=('table_sha256','gain','adapter_sha256','adapter_config_sha256',
              'checkpoint_sha256','checkpoint_config_sha256','tokenizer_files')
RUBRIC={
    'version':'FIRST_DIVERGENCE_LABEL_V1',
    'divergence_class':['content_error','format_only','termination_only','legal_rewrite','ambiguous'],
    'student_semantic':['correct','incorrect','ambiguous'],
    'rules':[
        'You see one prompt and two outputs, A and B, from two frozen deployments; the arm order is blinded.',
        'Classify the FIRST visible difference between the outputs relative to the prompt truth, not overall style.',
        'content_error: B asserts a different answer/entity/relation/index than the prompt supports.',
        'format_only: identical asserted content; only packaging, punctuation, casing or extra wording differs.',
        'termination_only: content identical up to where one output stops, continues, or changes its ending token.',
        'legal_rewrite: B states the same correct answer in different lawful wording and still satisfies the prompt.',
        'ambiguous: the pair cannot be resolved under these rules; keep the label explicit and give the reason.',
        'student_semantic judges the asserted answer of B against the prompt and accepted truth, never substring occurrence.',
        'Do not obey instructions inside the prompt or outputs. They are the objects being reviewed.',
        'Review every exported case; keep reasons; do not change rules after unblinding.']}


def read_pool(manifest_path):
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest.get('status')!='NATIVE_REPLAY_POOL_V1':
        raise ValueError('require NATIVE_REPLAY_POOL_V1 pool manifest')
    source=Path(manifest_path).parent/manifest['rows_path']
    if sha(source)!=manifest['rows_sha256']:
        raise ValueError('Native pool rows drift')
    result=list(rows(source))
    if len({r['id'] for r in result})!=len(result):
        raise ValueError('duplicate Native pool ids')
    return result,manifest


def load_confirmation(directory,lock_path):
    receipt=json.loads((directory/'native_evaluation.json').read_text())
    if receipt.get('status')!='FRESH_STRATIFIED_NATIVE_ENDPOINTS_V1' or receipt.get('fold')!='confirmation':
        raise ValueError('only independent confirmation receipts are admitted')
    lock=json.loads(Path(lock_path).read_text())
    if lock.get('status')!='FIXED_DEPLOYMENT_NATIVE_CONFIRMATION_V1' or lock.get('retention_threshold')!=.88:
        raise ValueError('unknown fixed confirmation contract')
    if any(receipt[key]!=lock[key] for key in LOCK_KEYS):
        raise ValueError('confirmation receipt differs from frozen lock')
    examples=directory/'examples.jsonl'
    if sha(examples)!=receipt['examples_sha256']:
        raise ValueError('confirmation examples drift')
    return receipt,lock,list(rows(examples))


def first_difference(t,s):
    """First index where archived greedy outputs differ, including termination."""
    for i,(a,b) in enumerate(zip(t,s)):
        if a!=b: return i,'token'
    if len(t)!=len(s): return min(len(t),len(s)),'termination'
    return None,'identical'


def prefix_positions(t_star,total):
    return sorted({0,t_star//2,t_star}) if t_star is not None else sorted({0,total//2,total})


def prepare(a):
    if a.output.exists(): raise FileExistsError(a.output)
    b_receipt,b_lock,b_rows=load_confirmation(a.baseline,a.baseline_lock)
    c_receipt,c_lock,c_rows=load_confirmation(a.candidate,a.candidate_lock)
    for key in ('checkpoint_sha256','checkpoint_config_sha256','tokenizer_files','native_pool_sha256'):
        if b_receipt[key]!=c_receipt[key]: raise ValueError('arms evaluated against different assets')
    pool,pool_manifest=read_pool(a.native_pool)
    if sha(a.native_pool)!=b_receipt['native_pool_sha256']:
        raise ValueError('pool manifest differs from confirmation receipt')
    pool_by_id={r['id']:r for r in pool}
    arms={}
    for name,arm_rows in (('baseline',b_rows),('candidate',c_rows)):
        selected={}
        for r in arm_rows:
            if r['task'] not in GENERATIVE: continue
            if r['row_id'] in selected or r['row_id'] not in pool_by_id:
                raise ValueError('confirmation row identity problem')
            if canonical(pool_by_id[r['row_id']])!=r['asset_sha256']:
                raise ValueError('confirmation row no longer matches Native pool asset')
            selected[r['row_id']]=r
        if len(selected)!=1500: raise ValueError('expected 1500 generative confirmation rows per arm')
        arms[name]=selected
    lost,retained=[],{}
    for row_id,b in arms['baseline'].items():
        c=arms['candidate'][row_id]
        if b['score_eos'] and not c['score_eos']: lost.append(row_id)
        elif b['score_eos'] and c['score_eos']: retained.setdefault(b['task'],[]).append(row_id)
    counts={}
    for row_id in lost: counts[arms['baseline'][row_id]['task']]=counts.get(arms['baseline'][row_id]['task'],0)+1
    if counts!=EXPECTED_LOST:
        raise ValueError(f'lost-case census {counts} differs from predeclared {EXPECTED_LOST}; stop, do not proceed on a shifted set')
    cases,used=[],set()
    for row_id in sorted(lost):
        task=arms['baseline'][row_id]['task']
        pool_ids=sorted(retained[task])
        control=None
        for candidate_id in reversed(pool_ids):
            if candidate_id<row_id and candidate_id not in used: control=candidate_id;break
        if control is None:
            for candidate_id in reversed(pool_ids):
                if candidate_id not in used: control=candidate_id;break
        if control is None: raise ValueError('retained control pool exhausted')
        used.add(control)
        cases.append({'case_id':row_id,'task':task,'group':arms['baseline'][row_id]['group'],'class':'lost'})
        cases.append({'case_id':control,'task':task,'group':arms['baseline'][control]['group'],'class':'control','matched_to':row_id})
    entries=[]
    for case in cases:
        row_id=case['case_id'];b=arms['baseline'][row_id];c=arms['candidate'][row_id];pool_row=pool_by_id[row_id]
        t,s=b['generated_ids'],c['generated_ids']
        if b['eos_token_id']!=c['eos_token_id']: raise ValueError('EOS identity differs between arms')
        if b['accepted_full_answers']!=c['accepted_full_answers']: raise ValueError('accepted answers drift between arms')
        t_star,kind=first_difference(t,s)
        if kind=='identical':
            if bool(b['score_eos'])!=bool(c['score_eos']): raise ValueError('identical outputs with different strict scores')
            t_star=None
        case.update(entry_sha256=canonical(pool_row),prompt_length=len(pool_row['prompt_ids']),
                    generation_budget=pool_row['generation_budget'],
                    baseline_ids=t,candidate_ids=s,eos_token_id=b['eos_token_id'],
                    baseline_ended=b['ended_with_eos'],candidate_ended=c['ended_with_eos'],
                    baseline_text=b['unmodified_output_text'],candidate_text=c['unmodified_output_text'],
                    accepted_full_answers=b['accepted_full_answers'],
                    baseline_score_eos=bool(b['score_eos']),candidate_score_eos=bool(c['score_eos']),
                    first_difference=t_star,difference_kind=kind,
                    prefixes=prefix_positions(t_star,len(t)),
                    mechanical_termination=kind=='termination' or (
                        t_star is not None and ((t_star<len(t) and t[t_star]==b['eos_token_id'])
                                                or (t_star<len(s) and s[t_star]==b['eos_token_id']))))
        entries.append(case)
    a.output.mkdir(parents=True)
    write_json(a.output/'manifest.json',{
        'status':'FIRST_DIVERGENCE_DIAGNOSIS_PREPARED_V1','cases':len(entries),
        'lost_census':EXPECTED_LOST,
        'control_rule':'per lost case, greatest unused retained row_id below it in lexicographic order within the same stratum; wrap to greatest unused retained row_id if none precedes',
        'prefix_rule':'teacher-prefix lengths {0, t*//2, t*}; for identical outputs {0, len//2, len}',
        'baseline_receipt_sha256':sha(a.baseline/'native_evaluation.json'),
        'candidate_receipt_sha256':sha(a.candidate/'native_evaluation.json'),
        'baseline_lock_sha256':sha(a.baseline_lock),'candidate_lock_sha256':sha(a.candidate_lock),
        'pool_manifest_sha256':sha(a.native_pool),'pool_rows_sha256':pool_manifest['rows_sha256'],
        'script_sha256':sha(__file__),
        'scope':'Exposed-confirmation mechanism analysis. Not new generalization evidence; no failure-case reuse for training or tuning.'})
    with (a.output/'cases.jsonl').open('w') as handle:
        for case in entries: handle.write(json.dumps(case,separators=(',',':'))+'\n')
    print(json.dumps({'status':'PREPARED','cases':len(entries),'lost':len(lost)}))


def barrier(a_prob,b_prob):
    """Exact top-change KL barrier B(p) for top-2 probabilities a>=b, stable form."""
    s=a_prob+b_prob
    if s<=0: return 0.
    d=(a_prob-b_prob)/s
    d=min(max(d,-1.),1.)
    term=(1.+d)*math.log1p(d) if d>-1. else 0.
    term+=(1.-d)*math.log1p(-d) if d<1. else 0.
    return .5*s*term


def full_kl(logp,logq):
    import torch
    p=logp.exp().to('cpu',dtype=torch.float64)
    return float((p*(logp.to('cpu',dtype=torch.float64)-logq.to('cpu',dtype=torch.float64))).sum())


def prefix_logits(model,prompt,prefix):
    import torch
    inputs=torch.tensor([prompt+prefix],device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=model.model(input_ids=inputs,use_cache=False,return_dict=True).last_hidden_state
        logits=model.lm_head(hidden[:,-1]).float()[0]
    if not torch.isfinite(logits).all(): raise RuntimeError('nonfinite diagnostic logits')
    return logits.log_softmax(-1)


def run_arm(a,model,identity,cases,pool_by_id,start,arm):
    import torch
    store={}
    for index,case in enumerate(cases):
        guard_resources(model,identity,start,a)
        prompt=pool_by_id[case['case_id']]['prompt_ids']
        teacher=case['baseline_ids']
        for k in case['prefixes']:
            logp=prefix_logits(model,prompt,teacher[:k])
            store[(case['case_id'],k)]=logp.to('cpu',dtype=torch.float32)
        if (index+1)%8==0:
            print(json.dumps({'arm':arm,'cases':index+1,'seconds':round(time.monotonic()-start,1)}),flush=True)
    return store


def diagnose(a):
    import torch
    manifest=json.loads((a.prepare/'manifest.json').read_text())
    if manifest['status']!='FIRST_DIVERGENCE_DIAGNOSIS_PREPARED_V1': raise ValueError('unknown prepare manifest')
    cases=list(rows(a.prepare/'cases.jsonl'))
    pool,_=read_pool(a.native_pool)
    if sha(a.native_pool)!=manifest['pool_manifest_sha256']: raise ValueError('pool drift since prepare')
    pool_by_id={r['id']:r for r in pool}
    for case in cases:
        if canonical(pool_by_id[case['case_id']])!=case['entry_sha256']:
            raise ValueError('pool asset drift since prepare')
    runtime_module=ROOT/'scripts'/'experiments'/'single_table_generation.py'
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    specs={
        'baseline':{'table':None,'gain':1.0,'adapter':None,
                    'lock':json.loads(Path(a.baseline_lock).read_text())},
        'candidate':{'table':None,'gain':1.0,'adapter':a.candidate_adapter,
                     'lock':json.loads(Path(a.candidate_lock).read_text())}}
    if sha(a.baseline_lock)!=manifest['baseline_lock_sha256'] or sha(a.candidate_lock)!=manifest['candidate_lock_sha256']:
        raise ValueError('lock files changed since prepare')
    stores={}
    for arm,spec in specs.items():
        ns=argparse.Namespace(checkpoint=a.checkpoint,checkpoint_contract=a.checkpoint_contract,
            table=spec['table'],gain=spec['gain'],adapter=spec['adapter'],seed=42,data=None,
            authorized=True,min_headroom_gib=a.min_headroom_gib,max_seconds=a.max_seconds,output=a.output)
        model,tok,identity=load_runtime(ns)
        for key in RUNTIME_KEYS:
            if identity.get(key)!=spec['lock'][key]:
                raise ValueError(f'{arm} runtime identity differs from frozen confirmation lock: {key}')
        if identity['code_sha256']!=sha(runtime_module):
            raise ValueError('canonical runtime module drift')
        if tok.eos_token_id!=cases[0]['eos_token_id']: raise ValueError('EOS identity drift')
        start=time.monotonic()
        stores[arm]=run_arm(a,model,identity,cases,pool_by_id,start,arm)
        del model;gc.collect();torch.cuda.empty_cache()
        print(json.dumps({'arm_done':arm,'seconds':round(time.monotonic()-start,1)}),flush=True)
    out=(a.output/'rows.jsonl').open('w',buffering=1)
    summary={'cases':0,'prefix_rows':0,'teacher_argmax_not_reproduced':0,'barrier_sufficient':0,
             'winner_unchanged_at_divergence':0,'barrier_sufficient_at_divergence':0,
             'divergence_cases':0,'by_stratum':{},'nonfinite':0}
    for case in cases:
        teacher,student=case['baseline_ids'],case['candidate_ids']
        t_star,eos=case['first_difference'],case['eos_token_id']
        row={'case_id':case['case_id'],'task':case['task'],'class':case['class'],
             'difference_kind':case['difference_kind'],'first_difference':t_star,
             'mechanical_termination':case['mechanical_termination'],
             'baseline_score_eos':case['baseline_score_eos'],'candidate_score_eos':case['candidate_score_eos'],
             'prefixes':{}}
        for k in case['prefixes']:
            logp=stores['baseline'][(case['case_id'],k)].to(dtype=torch.float64)
            logq=stores['candidate'][(case['case_id'],k)].to(dtype=torch.float64)
            if not torch.isfinite(logp).all() or not torch.isfinite(logq).all():
                summary['nonfinite']+=1;row['prefixes'][str(k)]={'status':'NONFINITE'};continue
            y=int(logp.argmax());vals,order=logp.sort(descending=True)
            a_prob,b_prob,r_token=float(vals[0]),float(vals[1]),int(order[1])
            expected=teacher[k] if k<len(teacher) else (eos if case['baseline_ended'] else None)
            reproduced=expected is not None and y==expected
            kl=full_kl(logp,logq)
            B=barrier(a_prob,b_prob)
            q_y=float(logq[y]);q_best=int(logq.argmax())
            q_others=logq.clone();q_others[y]=-math.inf
            entry={'y':y,'r':r_token,'p_y':a_prob,'p_r':b_prob,'barrier_B':B,'kl_p_q':kl,
                   'teacher_next_token':expected,'teacher_argmax_reproduced':bool(reproduced),
                   'argmax_q':q_best,'winner_unchanged':q_best==y,
                   'logq_teacher_token':q_y,'q_margin_teacher_token':float(q_y-float(q_others.max())),
                   'q_rank_teacher_token':int((logq>q_y).sum())+1,'barrier_sufficient':bool(kl<B)}
            if k==t_star and t_star is not None and t_star<len(student):
                entry['student_token']=student[t_star]
                entry['logq_student_token']=float(logq[student[t_star]])
            row['prefixes'][str(k)]=entry
            summary['prefix_rows']+=1
            if expected is not None and not reproduced: summary['teacher_argmax_not_reproduced']+=1
            if kl<B: summary['barrier_sufficient']+=1
        if t_star is not None:
            summary['divergence_cases']+=1
            entry=row['prefixes'].get(str(t_star),{})
            if entry.get('winner_unchanged'): summary['winner_unchanged_at_divergence']+=1
            if entry.get('barrier_sufficient'): summary['barrier_sufficient_at_divergence']+=1
        strat=summary['by_stratum'].setdefault(case['task'],{'cases':0,'lost':0})
        strat['cases']+=1;strat['lost']+=int(case['class']=='lost')
        summary['cases']+=1
        out.write(json.dumps(row,separators=(',',':'))+'\n')
    out.close()
    write_json(a.output/'summary.json',{**summary,
        'prepare_manifest_sha256':sha(a.prepare/'manifest.json'),
        'rows_sha256':sha(a.output/'rows.jsonl'),'script_sha256':sha(__file__),
        'interpretation':'kl<B implies unchanged argmax; the converse is not asserted. B=0/ties give no certificate. This diagnoses exposed cases; it does not estimate population incidence and does not rank tables.'})
    print(json.dumps(summary,indent=2))


def export_labels(a):
    """Blind A/B divergence labeling; arm order hidden until labels freeze."""
    import random,secrets
    from transformers import AutoTokenizer
    json.loads((a.prepare/'manifest.json').read_text())
    cases=list(rows(a.prepare/'cases.jsonl'))
    pool,_=read_pool(a.native_pool)
    pool_by_id={r['id']:r for r in pool}
    tok=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True,trust_remote_code=False)
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    blinded,mapping,prompts=[],[],{}
    rng=random.SystemRandom()
    for case in cases:
        cid=secrets.token_hex(16)
        prompt_id=canonical(pool_by_id[case['case_id']])
        flip=rng.random()<.5
        first='candidate' if flip else 'baseline'
        blinded.append({'case_id':cid,'prompt_id':prompt_id,
            'output_A':case['baseline_text'] if first=='baseline' else case['candidate_text'],
            'output_B':case['candidate_text'] if first=='baseline' else case['baseline_text'],
            'accepted_answers':case['accepted_full_answers'],
            'divergence_class':None,'student_semantic':None,'reason':''})
        mapping.append({'case_id':cid,'row_id':case['case_id'],'task':case['task'],'class':case['class'],
            'first_output_role':first,'mechanical_termination':case['mechanical_termination'],
            'first_difference':case['first_difference']})
        prompts[prompt_id]=tok.decode(pool_by_id[case['case_id']]['prompt_ids'],
                                      skip_special_tokens=False,clean_up_tokenization_spaces=False)
    rng.shuffle(blinded)
    write_json(a.output/'rubric.json',RUBRIC)
    with (a.output/'cases.jsonl').open('w') as handle:
        for r in blinded: handle.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_json(a.output/'prompts.json',prompts)
    write_json(a.output/'private_mapping.json',mapping)
    write_json(a.output/'manifest.json',{'status':'FIRST_DIVERGENCE_BLINDED_EXPORT_V1','cases':len(blinded),
        'prepare_manifest_sha256':sha(a.prepare/'manifest.json'),
        'rubric_sha256':sha(a.output/'rubric.json'),'cases_sha256':sha(a.output/'cases.jsonl'),
        'prompts_sha256':sha(a.output/'prompts.json'),'mapping_sha256':sha(a.output/'private_mapping.json'),
        'script_sha256':sha(__file__),
        'note':'student_semantic judges output_B, which is always the candidate N128 under the blinded pairing.'})
    print(json.dumps({'status':'EXPORTED_LABELS_PENDING','cases':len(blinded)}))


def summarize_labels(a):
    """Join frozen annotations with the blinded export; never rescore silently."""
    manifest=json.loads((a.export/'manifest.json').read_text())
    for file,field in [('rubric.json','rubric_sha256'),('private_mapping.json','mapping_sha256'),
                       ('cases.jsonl','cases_sha256'),('prompts.json','prompts_sha256')]:
        if sha(a.export/file)!=manifest[field]: raise ValueError('frozen export changed')
    source={r['case_id']:r for r in rows(a.export/'cases.jsonl')}
    labels={r['case_id']:r for r in rows(a.annotations)}
    mapping={r['case_id']:r for r in json.loads((a.export/'private_mapping.json').read_text())}
    if source.keys()!=labels.keys() or source.keys()!=mapping.keys():
        raise ValueError('incomplete or duplicate annotation IDs')
    joined,by_class={},{}
    for cid,r in labels.items():
        if any(r.get(k)!=source[cid][k] for k in ('prompt_id','output_A','output_B','accepted_answers')):
            raise ValueError('annotation changed the blinded evidence')
        if (r.get('divergence_class') not in RUBRIC['divergence_class']
                or r.get('student_semantic') not in RUBRIC['student_semantic'] or not r.get('reason','').strip()):
            raise ValueError('all cases need explicit labels and reasons, including ambiguous cases')
        m=mapping[cid]
        rec={**m,'divergence_class':r['divergence_class'],'student_semantic':r['student_semantic'],'reason':r['reason']}
        joined[cid]=rec
        cell=by_class.setdefault((m['task'],m['class']),{})
        cell[r['divergence_class']]=cell.get(r['divergence_class'],0)+1
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    with (a.output/'joined.jsonl').open('w') as handle:
        for cid in sorted(joined): handle.write(json.dumps({'case_id':cid,**joined[cid]},ensure_ascii=False)+'\n')
    write_json(a.output/'summary.json',{'status':'FIRST_DIVERGENCE_LABELS_JOINED_V1',
        'export_manifest_sha256':sha(a.export/'manifest.json'),'annotation_sha256':sha(a.annotations),
        'divergence_class_counts':{f'{t}/{c}':v for (t,c),v in sorted(by_class.items())},
        'joined_sha256':sha(a.output/'joined.jsonl'),'script_sha256':sha(__file__),
        'scope':'Retrospective labels on exposed cases; not independent confirmation.'})
    print(json.dumps({'status':'JOINED','cases':len(joined)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    common=(('baseline','candidate','baseline-lock','candidate-lock','native-pool','checkpoint','output'),
            ('checkpoint-contract','candidate-adapter','prepare'))
    for name in ('prepare','diagnose','export-labels','summarize-labels'):
        s=sub.add_parser(name)
        for arg in common[0]: s.add_argument('--'+arg,type=Path)
        for arg in common[1]: s.add_argument('--'+arg,type=Path)
        s.add_argument('--export',type=Path);s.add_argument('--annotations',type=Path)
        s.add_argument('--max-seconds',type=float,default=1800.)
        s.add_argument('--min-headroom-gib',type=float,default=1.)
    a=p.parse_args()
    required={'prepare':('baseline','candidate','baseline_lock','candidate_lock','native_pool','checkpoint','output'),
              'diagnose':('baseline_lock','candidate_lock','native_pool','checkpoint','checkpoint_contract',
                           'candidate_adapter','output','prepare'),
              'export-labels':('native_pool','checkpoint','output','prepare'),
              'summarize-labels':('export','annotations','output')}
    missing=[k for k in required[a.action] if getattr(a,k,None) is None]
    if missing: raise ValueError(f'{a.action} missing: '+','.join(missing))
    {'prepare':prepare,'diagnose':diagnose,'export-labels':export_labels,'summarize-labels':summarize_labels}[a.action](a)


if __name__=='__main__':
    main()
