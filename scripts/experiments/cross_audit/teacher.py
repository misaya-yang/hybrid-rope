"""Explicit original-Native teacher cache and per-stratum Native evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from .contracts import read_rows, score, sha_file, sha_json, write_json
from .runtime import cuda_runtime, load_model, verify_prepared, versions
from .tables import verify_static


def pool_rows(manifest_path,model_dir,split):
    path=Path(manifest_path);m=json.loads(path.read_text());p=path.parent/m['rows_path']
    if sha_file(p)!=m['rows_sha256']:raise ValueError('Native pool hash mismatch')
    for name,expected in m['tokenizer_files'].items():
        if sha_file(Path(model_dir)/name)!=expected:raise ValueError('Native pool tokenizer mismatch')
    rows=[r for r in read_rows(p) if r['split']==split]
    if not rows:raise ValueError('empty Native pool split')
    return rows,m


def prediction_positions(row,limit=2048,require_target=False):
    # Verified against prepare_source_grounded_transfer.build_native: these are
    # hidden indices, starting at len(prompt)-1. The final hidden position is
    # valid for teacher KL (including the next/EOS distribution); only labeled
    # NLL excludes it because input_ids contains no next-token target there.
    ids=row['input_ids'][:limit]
    end=len(ids)-1 if require_target else len(ids)
    positions=[int(p) for p in row['prediction_positions'] if 0<=int(p)<end]
    if not positions:raise ValueError('Native row has no surviving prediction positions')
    return ids,positions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('cache','evaluate'))
    for a in ('prepared','pool','out'):p.add_argument('--'+a,required=True)
    p.add_argument('--arm',default='Native');p.add_argument('--checkpoint')
    p.add_argument('--rows-per-stratum',type=int,default=64)
    a=p.parse_args();root=Path(a.prepared);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    assets=verify_prepared(root);split='train' if a.mode=='cache' else 'validation'
    rows,pool=pool_rows(a.pool,assets['model'],split)
    if a.mode=='cache' and (a.arm!='Native' or a.checkpoint):raise ValueError('teacher must be original Native')
    hardware=cuda_runtime();model,tok,values,entry=load_model(assets['model'],root,a.arm,checkpoint=a.checkpoint)
    model.eval();start=time.monotonic();counts=defaultdict(int);records=[]
    decoding=GenerationConfig(do_sample=False,num_beams=1,use_cache=True,eos_token_id=tok.eos_token_id,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
    with (out/'rows.jsonl').open('x') as f,torch.inference_mode():
        for row in rows:
            if a.mode=='evaluate' and counts[row['group']]>=a.rows_per_stratum:continue
            counts[row['group']]+=1
            ids,positions=prediction_positions(row,require_target=a.mode=='evaluate')
            if a.mode=='evaluate' and row['group']=='text':positions=list(range(len(ids)-1))
            x=torch.tensor([ids],device='cuda');pos=torch.tensor(positions,device='cuda')
            h=model.model(input_ids=x,use_cache=False).last_hidden_state[0,pos]
            logits=F.linear(h,model.lm_head.weight).float()
            r=dict(id=row['id'],group=row['group'],source_id=row['source_id'],row_sha256=sha_json(row),
                   input_ids=ids,positions=positions,position_convention='hidden i predicts input i+1',
                   text_domain=row.get('text_domain'),nll_scope='all next-token positions of <=2048-token text' if row['group']=='text' else 'declared answer prediction positions')
            if a.mode=='cache':
                path=f'{row["id"]}.npy';probs=logits.softmax(-1).cpu().numpy()
                np.save(out/path,probs)
                r.update(path=path,sha256=sha_file(out/path),dtype='float32',shape=list(probs.shape))
            else:
                targets=x[0,pos+1];nll=F.cross_entropy(logits,targets,reduction='none')
                r.update(nll_sum=float(nll.sum()),prediction_tokens=len(positions),nll=float(nll.mean()))
                if row['group']!='text':
                    prompt=row['prompt_ids'];xgen=torch.tensor([prompt],device='cuda')
                    if len(prompt)+row['generation_budget']>4096:raise ValueError('Native prompt overflow')
                    output=model.generate(xgen,attention_mask=torch.ones_like(xgen),generation_config=decoding,
                                          max_new_tokens=row['generation_budget'])
                    tokens=output[0,len(prompt):].tolist();eos=bool(tokens and tokens[-1]==tok.eos_token_id)
                    text=tok.decode(tokens[:-1] if eos else tokens,skip_special_tokens=False)
                    r.update(score({**row,'family':row['group']},text,eos),output_text=text,generated_ids=tokens)
            verify_static(model,values,entry['amplitude'])
            # Preserve raw input ids privately for exact cache and prediction replay.
            f.write(json.dumps(r)+'\n');f.flush();records.append(r)
            print(json.dumps(dict(id=row['id'],stratum=row['group'],mode=a.mode)),flush=True)
    write_json(out/'manifest.json',dict(status='COMPLETE',mode=a.mode,arm=a.arm,arm_identity=entry,
        model_config_sha256=sha_file(Path(assets['model'])/'config.json'),
        model_weights={k:v for k,v in assets['assets'].items() if k.endswith('.safetensors')},
        pool_sha256=pool['rows_sha256'],split=split,counts=dict(counts),rows_sha256=sha_file(out/'rows.jsonl'),
        hardware=hardware,software=versions(),wall_seconds=time.monotonic()-start,
        limits='source-truth teacher prefixes; sampled KL is not Native retention; Native strata stay separate'))


if __name__=='__main__':main()
