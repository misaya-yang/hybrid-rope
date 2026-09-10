"""Full generated answers, short retention and real-text NLL remain separate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from .acquire import file_hash
from .data import JsonlIndex,qa_scores,write_json
from .runtime import load
from scripts.experiments.cross_audit.training import causal_loss


def select_rows(root,split,panel):
    groups=defaultdict(list)
    for filename,kind in [(f'qa_{split}.jsonl','qa'),(f'ruler_{split}.jsonl','ruler'),(f'native_{split}.jsonl','native')]:
        path=root/'data'/filename
        if not path.exists():raise FileNotFoundError(path)
        index=JsonlIndex(path)
        for row in (index[i] for i in range(len(index))):
            if kind=='native' and row['task']=='text':continue
            row={**row,'suite':kind}
            groups[kind,row['task'],row.get('length_bucket',4096)].append(row)
    rows=[]
    for key,values in sorted(groups.items()):
        values=sorted(values,key=lambda r:r['id'])
        limit=(16 if key[0]=='native' else 8) if panel=='core' else len(values)
        rows.extend(values[:limit])
    return rows


def score_output(row,text):
    scores=qa_scores(text,row['references'])
    if row['suite']=='ruler':
        # The selected official NIAH/VT tasks use lowercased answer-item containment recall.
        scores['official_recall']=sum(ref.lower() in text.lower() for ref in row['references'])/len(row['references'])
        scores['all_answers_found']=float(scores['official_recall']==1.)
        if len(row['references'])>1:
            # These are jointly required answer items, not alternative whole answers.
            scores['exact']=None
    return scores


def language_scores(model,ids,chunk_size=128):
    hidden=model.model(input_ids=ids[:,:-1],use_cache=False).last_hidden_state[0]
    targets=ids[0,1:];count=len(targets);total=tail=0.
    tail_start=max(0,count-1024)
    for start in range(0,count,chunk_size):
        logits=torch.nn.functional.linear(hidden[start:start+chunk_size],model.lm_head.weight).float()
        losses=torch.nn.functional.cross_entropy(logits,targets[start:start+chunk_size],reduction='none')
        total+=float(losses.sum())
        offset=max(0,tail_start-start)
        if offset<len(losses):tail+=float(losses[offset:].sum())
    return total/count,tail/(count-tail_start),count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--regime',choices=['lora','full'],default='lora')
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--panel',choices=['core','full'],default='core')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();root=a.root.resolve();plan=json.loads((root/'plan.json').read_text())
    rows=select_rows(root,a.split,a.panel)
    label=f'{a.arm}_{a.regime}_{a.checkpoint.name if a.checkpoint else "untrained"}_{a.split}_{a.panel}'
    print(json.dumps(dict(label=label,generation_rows=len(rows),execute=a.execute)),flush=True)
    if not a.execute:return
    lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/'evaluation'/label
    if (out/'summary.json').exists():
        print(f'already complete: {out}');return
    out.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.enable_math_sdp(False)
    model,_,tokenizer,table=load(plan,root,a.arm,a.checkpoint,False,a.regime)
    eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else eos
    contract=dict(arm=a.arm,checkpoint=str(a.checkpoint) if a.checkpoint else None,
                  table=table,plan_sha256=file_hash(root/'plan.json'),
                  data_manifest_sha256=file_hash(root/'data/data_manifest.json'),
                  split=a.split,panel=a.panel,eos_token_ids=eos,
                  decoding='native chat template; greedy; full decoded response; no answer extraction or substring primary QA scoring')
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=contract:
        raise ValueError('cannot resume evaluations under changed inputs')
    write_json(out/'contract.json',contract)
    completed={}
    output=out/'generations.jsonl'
    if output.exists():
        for line in output.read_text().splitlines():
            row=json.loads(line);completed[row['id']]=row
    with output.open('a') as f,torch.no_grad():
        for row in rows:
            if row['id'] in completed:continue
            ids=torch.tensor([row['prompt_ids']],device='cuda')
            started=time.monotonic()
            answer=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                num_beams=1,max_new_tokens=row['generation_budget'],eos_token_id=eos,
                pad_token_id=tokenizer.pad_token_id,use_cache=True,logits_to_keep=1)[0,ids.shape[1]:].tolist()
            text=tokenizer.decode(answer,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            result={k:v for k,v in row.items() if k not in ['prompt_ids','input_ids']}
            result.update(prediction=text,generated_ids=answer,eos_terminated=bool(answer and answer[-1] in eos),
                          seconds=time.monotonic()-started,**score_output(row,text))
            f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();completed[row['id']]=result
            if len(completed)%16==0:print(json.dumps({'generated':len(completed),'total':len(rows)}),flush=True)
    groups=defaultdict(list)
    for r in completed.values():groups[r['suite'],r['task'],r.get('length_bucket',4096)].append(r)
    scores={str(k):dict(rows=len(v),f1=float(np.mean([r['f1'] for r in v])),
                       exact=float(np.mean([r['exact'] for r in v if r['exact'] is not None])) if any(r['exact'] is not None for r in v) else None,
                       exact_rows=sum(r['exact'] is not None for r in v),
                       eos_fraction=float(np.mean([r['eos_terminated'] for r in v])),
                       **({'official_recall':float(np.mean([r['official_recall'] for r in v]))} if k[0]=='ruler' else {}))
            for k,v in groups.items()}
    lm_file=root/'data'/f'lm_{"validation" if a.split=="dev" else "test"}.npy'
    text=np.load(lm_file,mmap_mode='r')
    lm=[]
    with torch.no_grad():
        for i in range(min(4,len(text)) if a.panel=='core' else len(text)):
            for length in (4096,8192,16384,32768):
                ids=torch.tensor([text[i,-length-1:].tolist()],device='cuda')
                full,tail,n=language_scores(model,ids)
                lm.append(dict(book_index=i,length=length,nll=full,tail_nll=tail,prediction_tokens=n))
    write_json(out/'lm_rows.json',lm)
    write_json(out/'summary.json',dict(status='COMPLETE',contract=contract,scores=scores,
        lm_by_length={str(n):dict(nll=float(np.mean([r['nll'] for r in lm if r['length']==n])),
                                 tail_nll=float(np.mean([r['tail_nll'] for r in lm if r['length']==n])))
                      for n in (4096,8192,16384,32768)},
        limits='Core is development. RULER recall, full-response QA F1/exact, EOS and LM NLL are separate outcomes.'))


if __name__=='__main__':main()
