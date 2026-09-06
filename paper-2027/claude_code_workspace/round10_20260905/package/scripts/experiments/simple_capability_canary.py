#!/usr/bin/env python3
"""Matched compact skill diagnostics; not natural benchmark or general retention."""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.single_table_generation import canonical,sha,rows,write_json,load_runtime,guard_resources,greedy


def make_cases():
    rng=random.Random(20260906);result=[]
    for i in range(16):
        values=list(map(str,rng.sample(range(1000000,10000000),4)))
        keys=['amber','birch','cedar','dahlia']
        words=rng.sample(['red','blue','green','white','black','gold','pink','gray'],3)
        a,b,c=rng.sample(range(10,50),3)
        for world in (0,1):
            assigned=values if world==0 else values[1:]+values[:1]
            table='\n'.join(k+' = '+v for k,v in zip(keys,assigned))
            question='Look up the value assigned to cedar. Output only its seven digits, without punctuation or explanation.\n\n'+table
            result.append({'semantic_id':f'binding_{i:02d}','family':'key_value_binding','world':world,'prompt':question,'answer':assigned[2]})
            table='A = B\nB = '+assigned[0]+'\nC = '+assigned[1]+'\nD = '+assigned[2]
            question='Resolve A by following the assignments until reaching a number. Output only that number, without punctuation or explanation.\n\n'+table
            result.append({'semantic_id':f'tracing_{i:02d}','family':'two_hop_tracing','world':world,'prompt':question,'answer':assigned[0]})
            sequence=words if world==0 else [words[1],words[0],words[2]]
            question='Reverse the order of the words. Separate the output words with | and no spaces. Output only the reversed sequence. Example: red blue becomes blue|red.\n\nInput: '+' '.join(sequence)
            result.append({'semantic_id':f'reverse_{i:02d}','family':'reverse_three_words','world':world,'prompt':question,'answer':'|'.join(sequence[::-1])})
            operand=b if world==0 else c
            question=f'Calculate {a} + {operand}. Output only the integer answer, without punctuation or explanation.'
            result.append({'semantic_id':f'addition_{i:02d}','family':'two_digit_addition','world':world,'prompt':question,'answer':str(a+operand)})
    for sid in {r['semantic_id'] for r in result}:
        pair=[r for r in result if r['semantic_id']==sid]
        if len(pair)!=2 or pair[0]['answer']==pair[1]['answer']:raise AssertionError('non-resolving worlds')
    return result


def prepare(a):
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    with (a.output/'cases.jsonl').open('w') as handle:
        for r in make_cases():handle.write(json.dumps(r)+'\n')
    write_json(a.output/'manifest.json',{'status':'SIMPLE_CAPABILITY_CANARY_V1','rows':128,'groups':64,
        'cases_sha256':sha(a.output/'cases.jsonl'),'seed':20260906,'script_sha256':sha(__file__),
        'scope':'Four declared synthetic compact skills,16 paired groups each. Deterministic source rules; no candidate-outcome selection. Same semantic prompts across tokenizers; not same token count, natural benchmark or global retention.'})


def evaluate(a):
    m=json.loads((a.cases/'manifest.json').read_text())
    if sha(a.cases/'cases.jsonl')!=m['cases_sha256']:raise ValueError('simple cases drift')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);model,tok,identity=load_runtime(a);start=time.monotonic();records=[]
    with (a.output/'examples.jsonl').open('w',buffering=1) as handle:
        for row in rows(a.cases/'cases.jsonl'):
            guard_resources(model,identity,start,a)
            prompt=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True,return_dict=False)
            if len(prompt)+64>2048:raise ValueError('compact prompt too long')
            ids=greedy(model,prompt,tok.eos_token_id,64)
            ended=bool(ids and ids[-1]==tok.eos_token_id and tok.eos_token_id not in ids[:-1])
            text=tok.decode(ids[:-1] if ended else ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            record={k:row[k] for k in ('semantic_id','family','world','answer')}
            record.update(prompt_sha256=canonical(prompt),input_tokens=len(prompt),generated_ids=ids,
                          eos_token_id=tok.eos_token_id,raw_text=text,ended_with_eos=ended,
                          full_exact_eos=bool(ended and text==row['answer']))
            records.append(record);handle.write(json.dumps(record)+'\n')
    pairs={}
    for row in records:pairs.setdefault((row['family'],row['semantic_id']),[]).append(row['full_exact_eos'])
    cells={}
    for (family,sid),pair in pairs.items():
        if len(pair)!=2:raise ValueError('missing world')
        cells.setdefault(family,[]).append(all(pair))
    write_json(a.output/'canary.json',{**identity,'status':'SIMPLE_CAPABILITY_CANARY_COMPLETE',
        'cases_manifest_sha256':sha(a.cases/'manifest.json'),'examples_sha256':sha(a.output/'examples.jsonl'),
        'script_sha256':sha(__file__),'seconds':time.monotonic()-start,
        'cells':{k:{'groups':len(v),'both_worlds_exact_eos':sum(v)/len(v)} for k,v in cells.items()},'scope':m['scope']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','evaluate'))
    p.add_argument('--output',type=Path,required=True)
    for k in ('checkpoint','checkpoint-contract','cases','table','adapter'):p.add_argument('--'+k,type=Path)
    p.add_argument('--gain',type=float,default=1.);p.add_argument('--authorized',action='store_true')
    p.add_argument('--max-seconds',type=float,default=600.);a=p.parse_args();a.seed=42;a.data=None;a.min_headroom_gib=1.
    prepare(a) if a.action=='prepare' else evaluate(a)
