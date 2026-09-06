#!/usr/bin/env python3
"""Development guard: source-only worlds, exact block swaps and source deletion.

This is a companion to the frozen natural validation set, not a replacement
training set or a new checkpoint-selection dataset. Deleted paired success is
structurally zero for disjoint answers; it is not proof of a reasoning mechanism.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.single_table_generation import (
    canonical,sha,rows,write_json,tokenizer_identity,load_runtime,greedy,guard_resources)


def matched_views(far0,far1,near0,near1,pad):
    """Keep physical positions and make world differences source-local exactly."""
    old=(far0,far1); starts=[r['source_block'][0] for r in old]
    widths=[r['source_block'][1] for r in old]
    if starts[0]!=starts[1]: raise ValueError('far anchors differ across worlds')
    start=starts[0];width=max(widths)
    ends=[r['source_block'][0]+r['source_block'][1] for r in (near0,near1)]
    if ends[0]!=ends[1]: raise ValueError('near source ends differ across worlds')
    tail=ends[0]-width
    x,y=[r['prompt_ids'] for r in old]
    if len(x)!=len(y) or x[:start]!=y[:start] or x[start+width:]!=y[start+width:]:
        raise ValueError('original far world differs outside source union')
    if not 0<=start<start+width<=tail<tail+width<=len(x):
        raise ValueError('invalid or overlapping source positions')
    result={}
    for world,row in enumerate(old):
        far=list(row['prompt_ids'])
        block=far[start:start+widths[world]]+[pad]*(width-widths[world])
        far[start:start+width]=block
        near=list(far)
        near[start:start+width],near[tail:tail+width]=far[tail:tail+width],block
        if Counter(near)!=Counter(far): raise AssertionError('swap changed token multiset')
        result[world]={'far':far,'near':near}
    for layout,position in (('far',start),('near',tail)):
        a,b=[result[w][layout] for w in (0,1)]
        if a[:position]!=b[:position] or a[position+width:]!=b[position+width:]:
            raise AssertionError('world difference escaped source')
        deleted=[]
        for values in (a,b):
            values=list(values);values[position:position+width]=[pad]*width;deleted.append(values)
        if deleted[0]!=deleted[1]: raise AssertionError('deleted worlds differ')
        result['deleted_'+layout]=deleted[0]
    return result,{'far':[start,width],'near':[tail,width],'original_widths':widths}


def prepare(a):
    from transformers import AutoTokenizer
    m=json.loads(a.tasks.read_text());source=a.tasks.parent/m['views_path']
    if sha(source)!=m['views_sha256']: raise ValueError('task views drift')
    tok=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True)
    if tokenizer_identity(a.checkpoint)!=m['tokenizer_files']: raise ValueError('tokenizer drift')
    pad_ids=tok.encode('\n',add_special_tokens=False)
    if len(pad_ids)!=1: raise ValueError('require a single newline padding token')
    groups={}
    for row in rows(source):
        if row['split']=='validation':
            key=(row['world'],row['layout'])
            group=groups.setdefault(row['semantic_id'],{})
            if key in group: raise ValueError('duplicate validation cell')
            group[key]=row
    if len(groups)!=64: raise ValueError('require all64 original validation groups')
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True);count=0
    with (a.output/'rows.jsonl').open('w') as handle:
        for sid,group in sorted(groups.items()):
            worlds=[group[(w,'far')] for w in (0,1)]
            if set(worlds[0]['accepted_full_answers'])&set(worlds[1]['accepted_full_answers']):
                raise ValueError('world truth aliases overlap')
            views,placement=matched_views(*worlds,group[(0,'near')],group[(1,'near')],pad_ids[0])
            common={'semantic_id':sid,'family':worlds[0]['family'],'length_cap':16384,
                    'generation_budget':64,'source_placement':placement,
                    'answers_by_world':[r['accepted_full_answers'] for r in worlds],
                    'original_far_prompt_sha256':[canonical(r['prompt_ids']) for r in worlds]}
            for layout in ('near','far'):
                for w in (0,1):
                    r={**common,'layout':layout,'world':w,'prompt_ids':views[w][layout]}
                    if len(r['prompt_ids'])+64!=16384: raise ValueError('physical length drift')
                    handle.write(json.dumps(r,separators=(',',':'))+'\n');count+=1
                r={**common,'layout':'deleted_'+layout,'world':None,'prompt_ids':views['deleted_'+layout]}
                handle.write(json.dumps(r,separators=(',',':'))+'\n');count+=1
    write_json(a.output/'manifest.json',{'status':'SOURCE_ONLY_DEVELOPMENT_GUARD_V1','groups':64,'rows':count,
        'task_manifest_sha256':sha(a.tasks),'rows_sha256':sha(a.output/'rows.jsonl'),
        'tokenizer_files':m['tokenizer_files'],'padding_token_id':pad_ids[0],'script_sha256':sha(__file__),
        'checks':['exact source-only world difference','within-world token-multiset-preserving swap',
                  'bitwise identical deleted worlds','disjoint complete-answer aliases','physical16K including64-token reserve'],
        'scope':'All original validation groups; no candidate-outcome selection. Padding is a declared assay change. Deleted pair zero is structural, not an independent reasoning test.'})


def evaluate(a):
    manifest=json.loads((a.guard_data/'manifest.json').read_text())
    if sha(a.guard_data/'rows.jsonl')!=manifest['rows_sha256']: raise ValueError('guard rows drift')
    if manifest['tokenizer_files']!=tokenizer_identity(a.checkpoint): raise ValueError('tokenizer drift')
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True);model,tok,identity=load_runtime(a);start=time.monotonic();records=[]
    with (a.output/'examples.jsonl').open('w',buffering=1) as handle:
        for row in rows(a.guard_data/'rows.jsonl'):
            guard_resources(model,identity,start,a)
            ids=greedy(model,row['prompt_ids'],tok.eos_token_id,row['generation_budget'])
            ended=bool(ids and ids[-1]==tok.eos_token_id and tok.eos_token_id not in ids[:-1])
            text=tok.decode(ids[:-1] if ended else ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            scores=[bool(ended and text in answers) for answers in row['answers_by_world']]
            record={k:row[k] for k in ('semantic_id','family','layout','world','answers_by_world')}
            record.update(prompt_sha256=canonical(row['prompt_ids']),generated_ids=ids,raw_text=text,
                          ended_with_eos=ended,eos_token_id=tok.eos_token_id,correct_by_world=scores)
            if row['world'] is not None:record['full_exact_eos']=scores[row['world']]
            records.append(record);handle.write(json.dumps(record)+'\n')
            if len(records)%32==0:print(json.dumps({'rows':len(records),'seconds':time.monotonic()-start}),flush=True)
    if len(records)!=manifest['rows']: raise ValueError('incomplete guard')
    result={**identity,'status':'SOURCE_ONLY_DEVELOPMENT_GUARD_COMPLETE','rows':len(records),
        'guard_manifest_sha256':sha(a.guard_data/'manifest.json'),'examples_sha256':sha(a.output/'examples.jsonl'),
        'script_sha256':sha(__file__),'seconds':time.monotonic()-start,'scope':manifest['scope']}
    write_json(a.output/'guard.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','evaluate'))
    for name in ('checkpoint','checkpoint-contract','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('tasks','guard-data','table','adapter'):p.add_argument('--'+name,type=Path)
    p.add_argument('--gain',type=float,default=1.);p.add_argument('--max-seconds',type=float,default=1800.)
    p.add_argument('--authorized',action='store_true');a=p.parse_args();a.seed=42;a.data=None;a.min_headroom_gib=1.
    prepare(a) if a.action=='prepare' else evaluate(a)
