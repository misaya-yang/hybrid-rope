#!/usr/bin/env python3
"""Continuous-report Native-window retention guard for a declared checkpoint.

Complements short Native QA. Summaries use source references and full generated
outputs with the checkpoint's real termination token; no prefix-only credit.
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.single_table_generation import sha,canonical,write_json,rows,load_runtime,guard_resources,greedy,checkpoint_contract,tokenizer_identity
from scripts.lib.rope.generation_contract import retention_verdict


def prepare(a):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True)
    limit=checkpoint_contract(a)['native_context_length']
    data=[json.loads(line) for line in zipfile.ZipFile(a.source_zip).open('data/gov_report_e.jsonl')]
    cells={0:[],1:[],2:[]}
    for row in sorted(data,key=lambda r:canonical(r['_id'])):
        text_ids=tok.encode(row['context'],add_special_tokens=False)
        prompt=tok.apply_chat_template([{'role':'user','content':'Summarize the following report in at most 150 words. Output only the summary.\n\n'+row['context']}],tokenize=True,add_generation_prompt=True,return_dict=False)
        if len(prompt)+512>limit: continue
        bucket=0 if len(prompt)<=4096 else 1 if len(prompt)<=16384 else 2
        if len(cells[bucket])>=8: continue
        cells[bucket].append({'id':row['_id'],'bucket':bucket,'prompt_ids':prompt,'text_ids':text_ids,
                             'references':row['answers'],'generation_budget':512,'source_row_sha256':canonical(row)})
        if all(len(v)==8 for v in cells.values()): break
    if any(len(v)!=8 for v in cells.values()): raise ValueError('incomplete declared Native-window bins')
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    path=a.output/'rows.jsonl'
    with path.open('w') as handle:
        for bucket in cells:
            for row in cells[bucket]:handle.write(json.dumps(row,separators=(',',':'))+'\n')
    write_json(a.output/'manifest.json',{'status':'NATIVE_WINDOW_REPORT_GUARD_V1','source_zip_sha256':sha(a.source_zip),
               'rows_sha256':sha(path),'tokenizer_files':tokenizer_identity(a.checkpoint),'native_context_length':limit,
               'scope':'development retention guard on24 real complete reports, eight per length bin; not final blind-test evidence'})


def evaluate(a):
    import torch
    import torch.nn.functional as F
    from scripts.eval.longbench_metrics import score_prediction
    manifest=json.loads((a.guard_data/'manifest.json').read_text())
    if (manifest['rows_sha256']!=sha(a.guard_data/'rows.jsonl') or manifest['tokenizer_files']!=tokenizer_identity(a.checkpoint)
            or manifest['native_context_length']!=checkpoint_contract(a)['native_context_length']):
        raise ValueError('Native-window guard identity drift')
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True);model,tok,identity=load_runtime(a);start=time.monotonic();records=[]
    with (a.output/'examples.jsonl').open('w',buffering=1) as handle:
        for row in rows(a.guard_data/'rows.jsonl'):
            guard_resources(model,identity,start,a)
            inputs=torch.tensor([row['text_ids']],device='cuda');count=min(512,inputs.shape[1]-1)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                hidden=model.model(input_ids=inputs,use_cache=False,return_dict=True).last_hidden_state
                logits=model.lm_head(hidden[:,-count-1:-1]).float()
                nll=float(F.cross_entropy(logits.flatten(0,1),inputs[:,-count:].flatten()))
            output=greedy(model,row['prompt_ids'],tok.eos_token_id,512)
            ended=bool(output and output[-1]==tok.eos_token_id)
            text=tok.decode(output[:-1] if ended else output,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            score=score_prediction('gov_report','rouge_l_f1',text,row['references'],[])
            record={'id':row['id'],'bucket':row['bucket'],'source_row_sha256':row['source_row_sha256'],
                    'input_tokens':len(row['prompt_ids']),'nll':nll,'generated_ids':output,'raw_text':text,
                    'ended_with_eos':ended,'score':score,'score_eos':score*ended}
            records.append(record);handle.write(json.dumps(record)+'\n')
    mean=lambda values,key:sum(r[key] for r in values)/len(values)
    result={**identity,'guard_manifest_sha256':sha(a.guard_data/'manifest.json'),'examples_sha256':sha(a.output/'examples.jsonl'),
            'nll':mean(records,'nll'),'task_macro':mean(records,'score'),'task_eos_macro':mean(records,'score_eos'),
            'EOS_rate':mean(records,'ended_with_eos'),'bins':{str(b):{'nll':mean(v,'nll'),'score':mean(v,'score'),
            'EOS_rate':mean(v,'ended_with_eos')} for b in range(3) for v in [[r for r in records if r['bucket']==b]]},
            'scope':manifest['scope'],'seconds':time.monotonic()-start,'script_sha256':sha(__file__)}
    if a.baseline:
        base=json.loads((a.baseline/'guard.json').read_text())
        if base['guard_manifest_sha256']!=result['guard_manifest_sha256'] or not base['table_is_native'] or base['adapter_sha256'] is not None:
            raise ValueError('unpaired or modified Native-window baseline')
        result['official_gate']=retention_verdict(base['nll'],result['nll'],base['task_macro'],result['task_macro'])
        result['EOS_gate']=retention_verdict(base['nll'],result['nll'],base['task_eos_macro'],result['task_eos_macro'])
    write_json(a.output/'guard.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','evaluate'))
    for key in ('checkpoint','checkpoint-contract','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('source-zip','guard-data','table','adapter','baseline'):p.add_argument('--'+key,type=Path)
    p.add_argument('--gain',type=float,default=1.);p.add_argument('--authorized',action='store_true')
    p.add_argument('--max-seconds',type=float,default=1800.)
    a=p.parse_args();a.data=None;a.seed=42;a.min_headroom_gib=1.
    prepare(a) if a.action=='prepare' else evaluate(a)
