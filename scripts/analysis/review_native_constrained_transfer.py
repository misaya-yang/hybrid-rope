#!/usr/bin/env python3
"""Read checkpoint validation receipts and write the next bounded research action.

CPU only. No launch, parameter tuning, final-test selection or automatic claim
promotion. Requires fresh stratified Native endpoints, not the legacy pack.
"""
from __future__ import annotations
import argparse
import ast
import json
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.single_table_generation import sha,canonical,rows,write_json,tokenizer_identity
from scripts.lib.rope.generation_contract import paired_retention_intervals,retention_verdict


def equivalent_evaluation_sources(left,right):
    """An unrelated trainer fix need not force recomputing identical baselines."""
    names={'token_ids','evaluation_rows','validate_answer_worlds','read_native_pool','read_tasks',
           'check_assets','evaluate_tasks','evaluate_native'}
    def extract(path):
        tree=ast.parse(path.read_text());result={}
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
                result[node.name]=ast.dump(node,include_attributes=False)
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GROUPS' for t in node.targets):
                result['GROUPS']=ast.dump(node.value,include_attributes=False)
        if set(result)!=names|{'GROUPS'}: raise ValueError('evaluation equivalence source is incomplete')
        return result
    if extract(left)!=extract(right): raise ValueError('Native/task evaluation source changed semantically')
    return {'baseline_engine_sha256':sha(left),'current_engine_sha256':sha(right),
            'check':'exact AST equality of evaluation/asset-validation functions and strata; shared runtime bytes checked separately'}


def load_receipt(directory,name):
    receipt=json.loads((directory/name).read_text())
    raw=directory/'examples.jsonl'
    if sha(raw)!=receipt['examples_sha256']: raise ValueError('raw validation receipt hash drift')
    return receipt,list(rows(raw))


def native_metrics(data):
    tasks=('instruction','reasoning','position_format')
    cells={task:[r for r in data if r['task']==task] for task in ('text',*tasks)}
    if any(not v for v in cells.values()): raise ValueError('incomplete Native validation')
    return (sum(r['nll'] for r in cells['text'])/len(cells['text']),
            *[sum(sum(r[field] for r in cells[t])/len(cells[t]) for t in tasks)/len(tasks)
              for field in ('score','score_eos')])


def rescore_native(data,tokenizer):
    for row in data:
        if row['task']=='text': continue
        ids=row['generated_ids']; eos=row['eos_token_id']
        ended=bool(ids and ids[-1]==eos and eos not in ids[:-1])
        text=tokenizer.decode(ids[:-1] if ended else ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        correct=text in row['accepted_full_answers']
        if text!=row['unmodified_output_text'] or float(correct)!=row['score'] or float(correct and ended)!=row['score_eos']:
            raise ValueError('Native full-output re-score differs from saved score')


def task_pairs(data,tokenizer):
    result={}
    for row in data:
        ids=row['generated_ids']; eos=row['eos_token_id']
        ended=bool(ids and ids[-1]==eos and eos not in ids[:-1])
        text=tokenizer.decode(ids[:-1] if ended else ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        score=ended and text in row['accepted_full_answers']
        if score!=row['full_exact_eos'] or text!=row['unmodified_output_text']:
            raise ValueError('natural full-output re-score differs from saved score')
        key=(row['family'],row['semantic_id'],row['layout'],row['length_cap'])
        pair=result.setdefault(key,{})
        if row['world'] in pair: raise ValueError('duplicate natural pair')
        pair[row['world']]=score
    if any(set(pair)!={0,1} for pair in result.values()): raise ValueError('missing natural world')
    return {key:all(value.values()) for key,value in result.items()}


def decide(retention_pass,retention_confirmed,cells,step):
    """Operational stopping rules; no proxy can override the generated endpoint."""
    if not retention_pass:
        return 'STOP_NATIVE_DAMAGE','Close this checkpoint/candidate under the current budget; retain earlier feasible checkpoints. Do not run longer cells or extra seeds.'
    if not cells or any(c['native_compact_groups']<8 for c in cells.values()):
        return 'UNRESOLVED_CONTROLS','Insufficient Native-compact-correct validation groups; repair assay resolving power without selecting rows by candidate outcomes.'
    if any(min(c['compact'],c['near'])<.8 for c in cells.values()):
        return 'UNRESOLVED_LOCAL_OR_BACKGROUND','Inspect compact/near errors, EOS and FFN/task versus Native gradients. No remote-only mechanism verdict or longer matrix.'
    if any(c['far']==0 for c in cells.values()):
        return 'STOP_ZERO_LONG_GENERATION','Controls resolve but a primary long family is zero. Close this candidate/protocol; preserve raw outputs and do not extend lengths.'
    if not retention_confirmed:
        return 'RETENTION_INTERVAL_UNRESOLVED','Point retention passes but paired interval does not establish the budget. Use the predeclared independent confirmation pool; no non-inferiority claim or longer matrix.'
    if step<128:
        return 'REVIEW_THEN_RESUME_FIXED_RECIPE','Valid nonzero working checkpoint: inspect raw cases and resume the same frozen recipe, optimizer and order to the next saved step; do not retune rank/loss/table.'
    return 'VALIDATION_CANDIDATE_NOT_FINAL_CLAIM','Compare only predeclared feasible saved steps using 16K far macro, then calibration KL and earlier step. Seal before blind test; author promotion still required.'


def decide_v4(retention_pass, retention_confirmed, cells, step):
    """Single-evidence system comparison; no retroactive change to legacy decide."""
    if step != 128:
        return 'DIAGNOSTIC_CHECKPOINT_ONLY', 'Only the frozen final step128 can enter this comparison; no checkpoint reselection or resume.'
    if not retention_pass:
        return 'STOP_NATIVE_DAMAGE', 'Close this candidate at the fixed budget. Keep its completed small-matrix outputs; no farther confirmation.'
    primary = cells.get('single_evidence')
    if not primary or primary['native_compact_groups'] < 8:
        return 'UNRESOLVED_PRIMARY_FAMILY', 'Primary family resolving power is insufficient; retain other families and do not tune on their outcomes.'
    if primary['far'] == 0:
        return 'STOP_ZERO_PRIMARY_GENERATION', 'The qualified primary far score is zero; close this candidate and its farther matrix.'
    if not retention_confirmed:
        return 'POINT_FEASIBLE_CONFIRMATION_PENDING', 'Freeze the candidate and a fresh independent Native confirmation. The old N128 confirmation is exposed; no automatic test launch.'
    return 'VALIDATION_FEASIBLE_CONFIRMATION_PENDING', 'Validation is encouraging, not semantic mechanism or blind transfer proof. Use a new frozen confirmation protocol; do not select earlier steps.'


def validate_task_rows(data, tasks, tokenizer):
    """Bind all raw validation rows to the declared prompts and truth aliases."""
    manifest = json.loads(tasks.read_text())
    source = tasks.parent / manifest['views_path']
    if sha(source) != manifest['views_sha256']:
        raise ValueError('task source bytes drift')
    def key(r):
        return (r['family'], r['semantic_id'], r['world'], r['layout'], r['length_cap'])
    expected_rows = [r for r in rows(source) if r['split']=='validation' and r['length_cap'] in (2048,16384)]
    expected = {key(r):r for r in expected_rows}
    actual = {key(r):r for r in data}
    if len(expected)!=len(expected_rows) or len(actual)!=len(data) or actual.keys()!=expected.keys():
        raise ValueError('missing/duplicate/unregistered validation row')
    if len({r['semantic_id'] for r in expected_rows})!=64 or len(expected_rows)!=384:
        raise ValueError('require the entire original 64-group/384-row validation matrix')
    sources = {}
    for k, row in actual.items():
        original = expected[k]
        aliases = original.get('accepted_full_answers') or [tokenizer.decode(
            original['target_ids'][:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)]
        if (row['prompt_sha256']!=canonical(original['prompt_ids'])
                or row['accepted_full_answers']!=aliases or row['eos_token_id']!=manifest['eos_token_id']
                or len(row['generated_ids'])>original['generation_budget']):
            raise ValueError('prompt/truth/EOS differs from frozen validation asset')
        sid = row['semantic_id']
        if sid in sources and sources[sid]!=original['source_id']:
            raise ValueError('semantic group source identity changed')
        sources[sid] = original['source_id']
    return sources


def family_comparisons(left, right, sources, resamples=1000):
    """Paired source-cluster descriptive intervals, not training-seed uncertainty."""
    result = {}
    for family in sorted({k[0] for k in left}):
        ids = sorted({k[1] for k in left if k[0]==family})
        def score(table, sid, layout):
            return int(table[(family,sid,layout,2048 if layout=='compact' else 16384)])
        def summary(chosen):
            delta = sum(score(right,i,'far')-score(left,i,'far') for i in chosen)/len(chosen)
            interaction = sum((score(right,i,'far')-score(right,i,'near'))-
                              (score(left,i,'far')-score(left,i,'near')) for i in chosen)/len(chosen)
            return delta, interaction
        groups = {}
        for sid in ids: groups.setdefault(sources[sid],[]).append(sid)
        rng = random.Random(20260905); draws = [[],[]]; clusters = sorted(groups)
        for _ in range(resamples):
            sample = [sid for _ in clusters for sid in groups[rng.choice(clusters)]]
            for target,value in zip(draws,summary(sample)): target.append(value)
        result[family] = {'semantic_groups':len(ids),'source_clusters':len(clusters),
            'all_groups':{layout:{'baseline_correct':sum(score(left,i,layout) for i in ids),
                'candidate_correct':sum(score(right,i,layout) for i in ids),
                'lost':sum(score(left,i,layout) and not score(right,i,layout) for i in ids),
                'gained':sum(not score(left,i,layout) and score(right,i,layout) for i in ids)}
                for layout in ('compact','near','far')},
            'far_delta':summary(ids)[0], 'near_far_interaction':summary(ids)[1],
            'ci95':{name:[sorted(values)[int(.025*(resamples-1))],sorted(values)[int(.975*(resamples-1))]]
                    for name,values in zip(('far_delta','near_far_interaction'),draws)}}
        cohort = [i for i in ids if score(left,i,'compact') and score(left,i,'near')]
        result[family]['Native_compact_and_near_cohort'] = {'n':len(cohort),
            'candidate_far_correct':sum(score(right,i,'far') for i in cohort)}
    return result


def native_transitions(left, right):
    a = {(r['task'],r['asset_sha256']):r for r in left}
    b = {(r['task'],r['asset_sha256']):r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or a.keys()!=b.keys():
        raise ValueError('unpaired Native transition rows')
    result = {}
    for task in ('instruction','reasoning','position_format'):
        keys = [k for k in a if k[0]==task]
        before = sum(a[k]['score_eos'] for k in keys)
        after = sum(b[k]['score_eos'] for k in keys)
        lost = [a[k]['row_id'] for k in keys if a[k]['score_eos'] and not b[k]['score_eos']]
        gained = [a[k]['row_id'] for k in keys if not a[k]['score_eos'] and b[k]['score_eos']]
        result[task] = {'n':len(keys),'baseline_correct':before,'candidate_correct':after,
            'score_retention':after/before if before else None,
            'original_correct_preserved':(before-len(lost))/before if before else None,
            'lost_ids':lost,'gained_ids':gained}
    return result


def review(args):
    from transformers import AutoTokenizer
    checkpoint=json.loads((args.adapter/'checkpoint.json').read_text())
    if (checkpoint['adapter_sha256']!=sha(args.adapter/'adapter_model.safetensors')
            or checkpoint['adapter_config_sha256']!=sha(args.adapter/'adapter_config.json')):
        raise ValueError('adapter changed since checkpoint receipt')
    bn,raw_bn=load_receipt(args.native_baseline,'native_evaluation.json')
    cn,raw_cn=load_receipt(args.native_candidate,'native_evaluation.json')
    bt,raw_bt=load_receipt(args.task_baseline,'evaluation.json')
    ct,raw_ct=load_receipt(args.task_candidate,'evaluation.json')
    for baseline in (bn,bt):
        if baseline['adapter_sha256'] is not None or not baseline['table_is_native'] or baseline['gain']!=1:
            raise ValueError('baseline must be original Native')
    for field in ('checkpoint_sha256','checkpoint_config_sha256','tokenizer_files','code_sha256'):
        if len({json.dumps(r[field],sort_keys=True) for r in (bn,cn,bt,ct)})!=1:
            raise ValueError('checkpoint/tokenizer/runtime implementation mismatch')
    for field in ('table_sha256','gain','adapter_sha256','adapter_config_sha256'):
        if cn[field]!=ct[field]: raise ValueError('Native/task evaluations used different deployment functions')
    for field,value in checkpoint['deployment'].items():
        if cn[field]!=value: raise ValueError('evaluated deployment differs from saved checkpoint')
    if (ct['adapter_sha256']!=checkpoint['adapter_sha256'] or ct['adapter_config_sha256']!=checkpoint['adapter_config_sha256']
            or ct['task_manifest_sha256']!=checkpoint['recipe']['tasks_sha256']):
        raise ValueError('checkpoint/task selection identity drift')
    if (bn['data_sha256']!=cn['data_sha256'] or bn['fold']!=cn['fold'] or cn['fold']!='selection'
            or bt['task_manifest_sha256']!=ct['task_manifest_sha256'] or bt['split']!='validation' or ct['split']!='validation'):
        raise ValueError('unpaired data or attempted final-test selection')
    evaluation_equivalence=None
    engine_hashes={r['evaluation_engine_sha256'] for r in (bn,cn,bt,ct)}
    if len(engine_hashes)!=1:
        current=getattr(args,'candidate_engine_source',None) or ROOT/'scripts/train/train_single_table_native_constrained.py'
        if not getattr(args,'baseline_engine_source',None): raise ValueError('natural evaluator changed; supply exact baseline source for CPU equivalence audit')
        if engine_hashes!={sha(current),sha(args.baseline_engine_source)}: raise ValueError('unexpected evaluation code identity')
        evaluation_equivalence=equivalent_evaluation_sources(args.baseline_engine_source,current)
    if bn['status']!='FRESH_STRATIFIED_NATIVE_ENDPOINTS_V1' or cn['status']!=bn['status']:
        raise ValueError('fresh independent Native endpoints required')
    if cn['native_pool_sha256']!=checkpoint['recipe']['native_pool_sha256'] or bn['native_pool_sha256']!=cn['native_pool_sha256']:
        raise ValueError('Native replay/validation manifest drift')
    if tokenizer_identity(args.checkpoint)!=ct['tokenizer_files']: raise ValueError('local tokenizer drift')
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True,trust_remote_code=False)
    rescore_native(raw_bn,tokenizer); rescore_native(raw_cn,tokenizer)
    a,b=native_metrics(raw_bn),native_metrics(raw_cn)
    official=retention_verdict(a[0],b[0],a[1],b[1]); eos=retention_verdict(a[0],b[0],a[2],b[2])
    ci=paired_retention_intervals(raw_bn,raw_cn,nll_task='text')
    left,right=task_pairs(raw_bt,tokenizer),task_pairs(raw_ct,tokenizer)
    if left.keys()!=right.keys(): raise ValueError('unpaired natural validation cells')
    if {k[3] for k in left}!={2048,16384}: raise ValueError('selection must use frozen compact/16K near/far matrix')
    families=sorted({k[0] for k in left}); cells={}
    for family in families:
        cohort=sorted(k[1] for k,v in left.items() if k[0]==family and k[2:] == ('compact',2048) and v)
        values={}
        for layout,length in (('compact',2048),('near',16384),('far',16384)):
            keys=[(family,sid,layout,length) for sid in cohort]
            if any(key not in right for key in keys): raise ValueError('missing qualified-cohort layout')
            values[layout]=sum(right[key] for key in keys)/len(keys) if keys else 0.
        cells[family]={'native_compact_groups':len(cohort),**values}
    passed=official['strict_088_pass'] and eos['strict_088_pass']
    protocol=getattr(args,'protocol','legacy_v3')
    extra={}
    if protocol=='single_evidence_v4':
        if not getattr(args,'tasks',None) or sha(args.tasks)!=ct['task_manifest_sha256']:
            raise ValueError('v4 requires the exact task manifest for row/provenance verification')
        if (checkpoint['recipe']['seed']!=42 or checkpoint['recipe']['placement']!='all_linear'
                or checkpoint['recipe'].get('prefix_lm',False)):
            raise ValueError('v4 is the fixed seed42 all-linear prefix-off comparison')
        sources=validate_task_rows(raw_bt,args.tasks,tokenizer)
        if validate_task_rows(raw_ct,args.tasks,tokenizer)!=sources:
            raise ValueError('source mapping drift')
        extra={'protocol':protocol,'family_comparisons':family_comparisons(left,right,sources),
               'Native_stratum_transitions':native_transitions(raw_bn,raw_cn),
               'family_scope':{f:('QUALIFIED_COMPACT_COHORT' if c['native_compact_groups']>=8 else
                                  'UNDERQUALIFIED_MECHANISM_COHORT_REPORT_ALL_GROUPS') for f,c in cells.items()},
               'semantic_status':'NOT_MEASURED_BY_STRICT_SCORER; see separate blinded retrospective annotation',
               'legacy_joint_verdict_unchanged':decide(passed,ci.get('all_lower_bounds_ge_088',False),cells,checkpoint['step'])[0]}
        status,action=decide_v4(passed,ci.get('all_lower_bounds_ge_088',False),cells,checkpoint['step'])
    else:
        status,action=decide(passed,ci.get('all_lower_bounds_ge_088',False),cells,checkpoint['step'])
    result={**extra,'status':status,'next_action':action,'checkpoint_step':checkpoint['step'],
            'evaluation_source_equivalence':evaluation_equivalence,
            'official_gate':official,'EOS_gate':eos,'paired_retention_uncertainty':ci,
            'Native_compact_qualified_cohort':cells,
            'unfiltered_16K_far_macro':sum(sum(v for k,v in right.items() if k[0]==f and k[2:] == ('far',16384))/sum(k[0]==f and k[2:] == ('far',16384) for k in right) for f in families)/len(families),
            'input_receipts':{name:sha(path) for name,path in (
                ('adapter',args.adapter/'checkpoint.json'),('native_baseline',args.native_baseline/'native_evaluation.json'),
                ('native_candidate',args.native_candidate/'native_evaluation.json'),('task_baseline',args.task_baseline/'evaluation.json'),
                ('task_candidate',args.task_candidate/'evaluation.json'))},
            'limits':'Validation is not blind test or seed uncertainty. Native-source truth and counterfactual construction need owner-backed qualification. This file neither certifies FFN necessity nor authorizes compute.'}
    if args.output.exists(): raise FileExistsError(args.output)
    write_json(args.output,result); print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','adapter','native-baseline','native-candidate','task-baseline','task-candidate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--baseline-engine-source',type=Path)
    p.add_argument('--candidate-engine-source',type=Path)
    p.add_argument('--tasks',type=Path)
    p.add_argument('--protocol',choices=('legacy_v3','single_evidence_v4'),default='legacy_v3')
    args=p.parse_args()
    try: review(args)
    except (ValueError,KeyError,FileNotFoundError) as error:
        result={'status':'UNRESOLVED_RECEIPTS_OR_CONTROLS','reason':str(error),
                'next_action':'Recover or validate the exact missing control/receipt; no candidate verdict, longer run or retuning.'}
        if not args.output.exists(): write_json(args.output,result)
        print(json.dumps(result),file=sys.stderr); sys.exit(2)
