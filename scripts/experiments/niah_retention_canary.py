#!/usr/bin/env python3
"""Small paired passkey canary for retrieval retention, not general forgetting.

Eight random keys, two worlds, compact plus physical4K/16K/32K at10/50/90%.
Natural-text haystacks; no training, substring credit or checkpoint selection.
This is a declared NIAH variant, not an exact reproduction of RULER.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.single_table_generation import (
    canonical,sha,write_json,rows,tokenizer_identity,load_runtime,greedy,guard_resources,checkpoint_contract)


def prepare(a):
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    tok=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True)
    if any(length>checkpoint_contract(a)['native_context_length'] for length in a.lengths):
        raise ValueError('retention canary lengths exceed this checkpoint Native window')
    pool=json.loads(a.native_pool.read_text())
    excluded={r['source_id'] for r in rows(a.native_pool.parent/pool['rows_path'])}
    backgrounds=[];tokens=[];sources=[]
    for batch in pq.ParquetFile(a.fineweb).iter_batches(batch_size=64,columns=['text','url']):
        for row in batch.to_pylist():
            source='fineweb:'+canonical(row['url'])
            if source in excluded: continue
            ids=tok.encode('\n\n'+row['text'],add_special_tokens=False)
            if len(ids)<128: continue
            tokens.extend(ids);sources.append(source)
            if len(tokens)>=32768:
                backgrounds.append((tokens[:32768],sources));tokens=[];sources=[]
            if len(backgrounds)==8:break
        if len(backgrounds)==8:break
    if len(backgrounds)!=8:raise ValueError('insufficient independent natural haystacks')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);rng=random.Random(20260905);count=0
    with (a.output/'rows.jsonl').open('w') as handle:
        for i,(bank,sources) in enumerate(backgrounds):
            answers=rng.sample(range(1000000,10000000),2);answers=list(map(str,answers))
            if any(answer in tok.decode(bank) for answer in answers):raise ValueError('accidental background answer')
            name=f'RECORD_{i:02d}'
            instruction='Find the passcode stated in the text. Return ONLY its seven digits, with no words, spaces, or punctuation.\n\n'
            question='\n\nWhat is the passcode for '+name+'?'
            marked=tok.apply_chat_template([{'role':'user','content':instruction+'TEXT_MARKER'+question}],tokenize=False,add_generation_prompt=True)
            left,right=marked.split('TEXT_MARKER');prefix=tok.encode(left,add_special_tokens=False);suffix=tok.encode(right,add_special_tokens=False)
            needles=[tok.encode('\nThe passcode for '+name+' is '+v+'.\n',add_special_tokens=False) for v in answers]
            if len(needles[0])!=len(needles[1]):raise ValueError('needle token widths differ')
            for world,needle in enumerate(needles):
                cells=[(2048,None)]+[(length,depth) for length in a.lengths for depth in (.1,.5,.9)]
                for length,depth in cells:
                    if depth is None:ids=prefix+needle+suffix;position=len(prefix)
                    else:
                        size=length-32-len(prefix)-len(suffix);body=list(bank[:size])
                        position=round(depth*(size-len(needle)));body[position:position+len(needle)]=needle
                        ids=prefix+body+suffix;position+=len(prefix)
                        if len(ids)+32!=length:raise ValueError('physical length drift')
                    record={'semantic_id':f'passkey_{i:02d}','world':world,'length_cap':length,'depth':depth,
                            'layout':'compact' if depth is None else 'needle','prompt_ids':ids,'generation_budget':32,
                            'answer':answers[world],'answers_by_world':answers,'source_block':[position,len(needle)],
                            'background_source_ids':sources}
                    handle.write(json.dumps(record,separators=(',',':'))+'\n');count+=1
    write_json(a.output/'manifest.json',{'status':'PAIRED_NIAH_RETENTION_CANARY_V1','groups':8,'rows':count,
        'rows_sha256':sha(a.output/'rows.jsonl'),'tokenizer_files':tokenizer_identity(a.checkpoint),
        'fineweb_sha256':sha(a.fineweb),'native_pool_sha256':sha(a.native_pool),'script_sha256':sha(__file__),
        'seed':20260905,'lengths':[2048,*a.lengths],'depths':[.1,.5,.9],
        'scope':'Development retrieval/format canary. Eight semantic groups give limited uncertainty; same keys across lengths/depths are not independent samples. Not broad Native retention or general capability transfer.'})


def evaluate(a):
    import torch
    from transformers import GenerationConfig
    manifest=json.loads((a.guard_data/'manifest.json').read_text())
    if sha(a.guard_data/'rows.jsonl')!=manifest['rows_sha256']:raise ValueError('NIAH data drift')
    if manifest['tokenizer_files']!=tokenizer_identity(a.checkpoint):raise ValueError('NIAH tokenizer drift')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);model,tok,identity=load_runtime(a);start=time.monotonic();parity=[];records=[]
    write_json(a.output/'native_generation_config.json',model.generation_config.to_dict())
    compact_parity=0
    with (a.output/'examples.jsonl').open('w',buffering=1) as handle:
        for row in rows(a.guard_data/'rows.jsonl'):
            guard_resources(model,identity,start,a)
            ids=greedy(model,row['prompt_ids'],tok.eos_token_id,32)
            short_probe=row['layout']=='compact' and compact_parity<4
            long_probe=row['semantic_id']=='passkey_00' and row['world']==0 and row['depth']==.1 and row['length_cap'] in (4096,32768)
            if short_probe or long_probe:
                inputs=torch.tensor([row['prompt_ids']],device='cuda')
                probe_budget=32 if short_probe else 8
                generation=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=probe_budget,
                    eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,
                    repetition_penalty=1.,no_repeat_ngram_size=0,use_cache=True)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    official=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                        generation_config=generation)[0,inputs.shape[1]:].tolist()
                custom=ids[:probe_budget];compact_parity+=int(short_probe)
                parity.append({'semantic_id':row['semantic_id'],'world':row['world'],'length_cap':row['length_cap'],
                               'exact':custom==official,'custom_ids':custom,'HF_ids':official,
                               'generation_config':generation.to_dict()})
                write_json(a.output/'HF_greedy_parity.json',{'rows':parity,'scope':'Four compact outputs and4K/32K eight-token boundaries when present; fresh explicit greedy GenerationConfig, not default sampling or all-length equivalence'})
                if custom!=official:raise RuntimeError('custom versus HF greedy differs; do not interpret canary')
            ended=bool(ids and ids[-1]==tok.eos_token_id and tok.eos_token_id not in ids[:-1])
            text=tok.decode(ids[:-1] if ended else ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            record={k:row[k] for k in ('semantic_id','world','length_cap','depth','layout','answer','source_block')}
            record.update(prompt_sha256=canonical(row['prompt_ids']),generated_ids=ids,eos_token_id=tok.eos_token_id,
                          raw_text=text,ended_with_eos=ended,complete_answer_exact=text==row['answer'],
                          full_exact_eos=bool(ended and text==row['answer']))
            records.append(record);handle.write(json.dumps(record)+'\n')
            if len(records)%20==0:print(json.dumps({'rows':len(records),'seconds':time.monotonic()-start}),flush=True)
    if len(records)!=manifest['rows']:raise ValueError('incomplete NIAH matrix')
    pairs={};cells={}
    for row in records:pairs.setdefault((row['length_cap'],row['depth'],row['semantic_id']),[]).append(row['full_exact_eos'])
    for (length,depth,sid),pair in pairs.items():
        if len(pair)!=2:raise ValueError('missing paired world')
        cells.setdefault(f'{length}:{depth}',[]).append(all(pair))
    write_json(a.output/'canary.json',{**identity,'status':'PAIRED_NIAH_RETENTION_CANARY_COMPLETE',
        'manifest_sha256':sha(a.guard_data/'manifest.json'),'examples_sha256':sha(a.output/'examples.jsonl'),
        'HF_parity_sha256':sha(a.output/'HF_greedy_parity.json'),'script_sha256':sha(__file__),
        'native_generation_config_sha256':sha(a.output/'native_generation_config.json'),
        'cells':{key:{'groups':len(v),'both_worlds_exact_eos':sum(v)/len(v)} for key,v in cells.items()},
        'seconds':time.monotonic()-start,'scope':manifest['scope']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','evaluate'))
    for k in ('checkpoint','checkpoint-contract','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('native-pool','fineweb','guard-data','table','adapter'):p.add_argument('--'+k,type=Path)
    p.add_argument('--gain',type=float,default=1.);p.add_argument('--max-seconds',type=float,default=900.)
    p.add_argument('--lengths',type=int,nargs='+',choices=(4096,16384,32768),default=[4096,16384,32768])
    p.add_argument('--authorized',action='store_true');a=p.parse_args();a.seed=42;a.data=None;a.min_headroom_gib=1.
    prepare(a) if a.action=='prepare' else evaluate(a)
