"""Bounded frozen/adapted evaluation on previously frozen complete groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import GenerationConfig
from .contracts import read_rows, score, sha_file, sha_json, select_groups, summarize, write_json
from .runtime import cuda_runtime, load_model, verify_prepared, versions
from .tables import verify_static


def evaluate(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    prepared=Path(args.prepared)
    assets=verify_prepared(prepared)
    rows=list(read_rows(prepared/'eval_rows.jsonl'))
    if args.layouts: rows=[r for r in rows if r['layout'] in args.layouts]
    rows=select_groups(rows,args.groups_per_cell)
    hardware=cuda_runtime()
    model,tok,values,entry=load_model(assets['model'],prepared,args.arm,checkpoint=args.checkpoint)
    model.eval()
    eos=tok.eos_token_id
    decoding=GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
                              eos_token_id=eos,pad_token_id=tok.pad_token_id or eos)
    start=time.monotonic(); records=[]
    with (out/'examples.jsonl').open('x') as f, torch.inference_mode():
        for row in rows:
            ids=torch.tensor([row['prompt_ids']],device='cuda')
            if len(row['prompt_ids'])+row['generation_budget']>row['length_cap']:
                raise ValueError('prompt/reserve overflow')
            torch.cuda.reset_peak_memory_stats()
            before=time.monotonic()
            generated=model.generate(ids,attention_mask=torch.ones_like(ids),
                generation_config=decoding,max_new_tokens=row['generation_budget'])
            new=generated[0,ids.shape[1]:].tolist()
            ended=bool(new and new[-1]==eos)
            text=tok.decode(new[:-1] if ended else new,skip_special_tokens=False)
            verify_static(model,values,entry['amplitude'])
            record={k:row[k] for k in ('row_id','group_id','family','layout','length_cap','world','source_lineages','prompt_sha256')}
            record.update(score(row,text,ended),generated_ids=new,output_text=text,
                stop_reason='eos' if ended else 'budget',input_tokens=ids.shape[1],
                elapsed_seconds=time.monotonic()-before,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                arm=args.arm,table_sha256=entry['tensor_sha256'])
            f.write(json.dumps(record)+'\n');f.flush();records.append(record)
            print(json.dumps({k:record[k] for k in ('row_id','family','layout','full_answer_exact_eos','elapsed_seconds')}),flush=True)
    manifest=dict(status='COMPLETE',scope='development apparatus; controls and Native review required',
        arm=args.arm,arm_identity=entry,checkpoint=args.checkpoint,hardware=hardware,software=versions(),
        prepared_sha256=sha_file(prepared/'assets_and_missing.json'),
        selected_row_ids=[r['row_id'] for r in rows],selected_prompts_sha256=sha_json([r['prompt_ids'] for r in rows]),
        summary=summarize(records,rows),wall_seconds=time.monotonic()-start,
        examples_sha256=sha_file(out/'examples.jsonl'),decoding=decoding.to_dict())
    if args.qualify_native:
        if args.arm!='Native' or args.checkpoint:raise ValueError('qualification requires original Native')
        qualified={}
        for family in sorted({r['family'] for r in rows}):
            compact=[r for r in records if r['family']==family and r['layout']=='compact']
            deleted=[r for r in records if r['family']==family and r['layout']=='deleted']
            groups=manifest['summary'].get(f'{family}:compact:2048',{})
            qualified[family]=dict(positive_group_successes=groups.get('complete_group_successes',0),
                deleted_original_answer_hits=sum(r['full_answer_exact_eos'] for r in deleted),deleted_rows=len(deleted),
                resolved=bool(compact and deleted and groups.get('complete_group_successes',0)>0
                    and sum(r['full_answer_exact_eos'] for r in deleted)/len(deleted)<=.05))
        manifest['control_qualification']=qualified
        if not qualified or not all(q['resolved'] for q in qualified.values()):manifest['status']='UNRESOLVED_CONTROLS'
    write_json(out/'manifest.json',manifest)
    if manifest['status']!='COMPLETE':raise SystemExit(4)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('prepared','arm','out'):p.add_argument('--'+a,required=True)
    p.add_argument('--checkpoint');p.add_argument('--groups-per-cell',type=int,default=32)
    p.add_argument('--layouts',nargs='+',choices=('compact','near','far','deleted'))
    p.add_argument('--qualify-native',action='store_true')
    evaluate(p.parse_args())


if __name__=='__main__':main()
