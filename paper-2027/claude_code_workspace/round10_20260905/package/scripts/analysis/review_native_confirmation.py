#!/usr/bin/env python3
"""One fixed deployment's Native confirmation; never select a checkpoint here."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.single_table_generation import sha,write_json,tokenizer_identity
from scripts.analysis.review_native_constrained_transfer import load_receipt,native_metrics,rescore_native
from scripts.lib.rope.generation_contract import retention_verdict,paired_retention_intervals

EXPECTED={'text':256,'instruction':500,'reasoning':500,'position_format':500}
DEPLOYMENT=('table_sha256','gain','adapter_sha256','adapter_config_sha256','native_pool_sha256',
            'checkpoint_sha256','checkpoint_config_sha256','tokenizer_files')


def validate_identity(baseline,candidate,locks,data):
    for receipt,lock,raw in zip((baseline,candidate),locks,data):
        if receipt['status']!='FRESH_STRATIFIED_NATIVE_ENDPOINTS_V1' or receipt['fold']!='confirmation':
            raise ValueError('only complete independent confirmation receipts are admitted')
        if lock['status']!='FIXED_DEPLOYMENT_NATIVE_CONFIRMATION_V1' or lock['retention_threshold']!=.88:
            raise ValueError('unknown fixed confirmation contract')
        if lock['expected_rows']!=EXPECTED or Counter(r['task'] for r in raw)!=Counter(EXPECTED):
            raise ValueError('incomplete or changed confirmation strata')
        if any(receipt[key]!=lock[key] for key in DEPLOYMENT):
            raise ValueError('deployment differs from frozen confirmation lock')
        if len({r['row_id'] for r in raw})!=len(raw):raise ValueError('duplicate confirmation rows')
    if not baseline['table_is_native'] or baseline['gain']!=1 or baseline['adapter_sha256'] is not None:
        raise ValueError('baseline is not original Native')
    for key in ('checkpoint_sha256','checkpoint_config_sha256','tokenizer_files','code_sha256',
                'evaluation_engine_sha256','native_pool_sha256','data_sha256'):
        if baseline[key]!=candidate[key]:raise ValueError('unmatched confirmation '+key)
    if [(r['row_id'],r['asset_sha256'],r['task'],r['group']) for r in data[0]]!=[(r['row_id'],r['asset_sha256'],r['task'],r['group']) for r in data[1]]:
        raise ValueError('confirmation rows/order/source groups differ')


def main(a):
    from transformers import AutoTokenizer
    if a.output.exists():raise FileExistsError(a.output)
    b,br=load_receipt(a.baseline,'native_evaluation.json')
    c,cr=load_receipt(a.candidate,'native_evaluation.json')
    locks=[json.loads(p.read_text()) for p in (a.baseline_lock,a.candidate_lock)]
    validate_identity(b,c,locks,(br,cr))
    if tokenizer_identity(a.checkpoint)!=b['tokenizer_files']:raise ValueError('tokenizer bytes drift')
    tok=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True)
    rescore_native(br,tok);rescore_native(cr,tok)
    x,y=native_metrics(br),native_metrics(cr)
    official=retention_verdict(x[0],y[0],x[1],y[1]);eos=retention_verdict(x[0],y[0],x[2],y[2])
    interval=paired_retention_intervals(br,cr,nll_task='text')
    point=official['strict_088_pass'] and eos['strict_088_pass']
    status='STOP_NATIVE_CONFIRMATION_DAMAGE' if not point else ('NATIVE_CONFIRMATION_PASS_AT_DECLARED_SCOPE' if interval['all_lower_bounds_ge_088'] else 'NATIVE_CONFIRMATION_INTERVAL_UNRESOLVED')
    groups={}
    for task in ('instruction','reasoning','position_format'):
        paired=[(u,v) for u,v in zip(br,cr) if u['task']==task]
        n=sum(u['score_eos'] for u,v in paired);m=sum(v['score_eos'] for u,v in paired)
        groups[task]={'rows':len(paired),'source_groups':len({u['group'] for u,v in paired}),
            'native_correct_EOS':n,'candidate_correct_EOS':m,'retention':m/n if n else None,
            'lost':sum(bool(u['score_eos']) and not bool(v['score_eos']) for u,v in paired),
            'gained':sum(not bool(u['score_eos']) and bool(v['score_eos']) for u,v in paired),
            'native_EOS':sum(u['ended_with_eos'] for u,v in paired),'candidate_EOS':sum(v['ended_with_eos'] for u,v in paired)}
    result={'status':status,'baseline_metrics':{'nll':x[0],'task_macro':x[1],'EOS_task_macro':x[2]},
        'candidate_metrics':{'nll':y[0],'task_macro':y[1],'EOS_task_macro':y[2]},
        'official_gate':official,'EOS_gate':eos,'paired_uncertainty':interval,'strata':groups,
        'input_receipts':{name:sha(path) for name,path in (
            ('baseline',a.baseline/'native_evaluation.json'),('candidate',a.candidate/'native_evaluation.json'),
            ('baseline_lock',a.baseline_lock),('candidate_lock',a.candidate_lock))},'reviewer_sha256':sha(__file__),
        'scope':'One fixed checkpoint/deployment on the predeclared Native confirmation pool. Source-group uncertainty, not seed uncertainty or universal ability preservation. No checkpoint reselection, task-test qualification or automatic new training.'}
    write_json(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','baseline','candidate','baseline-lock','candidate-lock','output'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
