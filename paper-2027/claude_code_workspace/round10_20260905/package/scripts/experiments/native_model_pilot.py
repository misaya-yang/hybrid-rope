#!/usr/bin/env python3
"""Matched short natural-QA and format diagnostics across available checkpoints.

Unmodified Native models only. Never labels within-native scores extrapolation.
"""
import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.single_table_generation import sha,canonical,write_json,greedy,render_case,condition_prompt,target_score,tokenizer_identity


def native_greedy(model,ids,stop_ids,budget):
    import torch
    cache=None;out=[];current=torch.tensor([ids],device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(budget):
            result=model.model(input_ids=current,past_key_values=cache,use_cache=True,return_dict=True)
            logits=model.lm_head(result.last_hidden_state[:,-1:]).float()
            if not torch.isfinite(logits).all(): raise RuntimeError('nonfinite Native pilot logits')
            token=int(logits[0,-1].argmax());out.append(token);cache=result.past_key_values
            if token in stop_ids: break
            current=torch.tensor([[token]],device='cuda')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--squad',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--authorized',action='store_true');p.add_argument('--max-seconds',type=float,default=1800)
    a=p.parse_args()
    if not a.authorized: p.error('GPU execution must be authorized')
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision('high')
    articles=sorted(json.loads(a.squad.read_text())['data'],key=lambda r:canonical(r['title']))
    selected=[]
    for article in articles:
        found=None
        for para in article['paragraphs']:
            if len(para['context'])>3200: continue
            qa=para['qas'][0]
            if not qa['answers']: continue
            found={'source':article['title'],'question':qa['question'],'context':para['context'],
                   'answers':sorted({r['text'] for r in qa['answers']}),'id':qa['id']};break
        if found: selected.append(found)
        if len(selected)==48: break
    if len(selected)!=48: raise ValueError('fixed short-pilot quota missing')
    write_json(a.output/'frozen_cases.json',{'squad_sha256':sha(a.squad),'rows':selected,
                'scope':'first48 source articles in fixed hash order, one short paragraph each; development diagnostic'})
    start=time.monotonic();summaries=[]
    nonces=random.Random(20260904).sample(range(100000,1000000),80)
    for checkpoint in a.checkpoints:
        if time.monotonic()-start>a.max_seconds: raise RuntimeError('pilot time budget exhausted')
        out=a.output/checkpoint.name;out.mkdir()
        tok=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False,
                   dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        eos=model.generation_config.eos_token_id
        stop_ids=set(eos if isinstance(eos,list) else [eos if eos is not None else tok.eos_token_id])
        identity={'model_type':model.config.model_type,'native_max_position_embeddings':model.config.max_position_embeddings,
                  'config_sha256':sha(checkpoint/'config.json'),'weight_files':{p.name:sha(p) for p in checkpoint.glob('*.safetensors')},
                  'tokenizer_files':tokenizer_identity(checkpoint),'transformers':transformers.__version__,
                  'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'native_only':True,'script_sha256':sha(__file__),
                  'native_generation_eos_ids':sorted(stop_ids)}
        write_json(out/'run.json',identity);records=[]
        with (out/'examples.jsonl').open('w',buffering=1) as handle:
            for row in selected:
                if time.monotonic()-start>a.max_seconds: raise RuntimeError('pilot time budget exhausted')
                content='Answer the question using the paragraph. Give only the answer text, without explanation.\n\n'+row['context']+'\n\nQuestion: '+row['question']
                ids=tok.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True,return_dict=False)
                if len(ids)+64>2048: raise ValueError('shared natural pilot does not fit compact cap')
                output=native_greedy(model,ids,stop_ids,64);ended=bool(output and output[-1] in stop_ids)
                text=tok.decode(output[:-1] if ended else output,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                accepted=set(row['answers'])|{s+'.' for s in row['answers'] if not s.endswith('.')}
                record={'task':'squad_compact','source_id':row['id'],'generated_ids':output,'raw_text':text,
                        'accepted_full_answers':sorted(accepted),'exact_eos':ended and text in accepted,'EOS':ended,
                        'input_length':len(ids),'prompt_sha256':canonical(ids)}
                records.append(record);handle.write(json.dumps(record)+'\n')
            for i in range(16):
                task=('lookup','chain')[i%2]
                for world in (0,1):
                    row=render_case(tok,group=str(i),task=task,length=4096,nonce=nonces[i*5:i*5+5],variant=world,explicit_format=True)
                    ids=condition_prompt(row,'compact');output=native_greedy(model,ids,stop_ids,64)
                    terminal=output[-1] if output and output[-1] in stop_ids else tok.eos_token_id
                    score=target_score(output,row['answer_ids'],terminal,lambda x:tok.decode(x,skip_special_tokens=False,clean_up_tokenization_spaces=False))
                    record={'task':task+'_compact','group':i,'world':world,'generated_ids':output,
                            'raw_text':tok.decode(output),'answer':row['answer'],**score}
                    records.append(record);handle.write(json.dumps(record)+'\n')
        summary={**identity,'cells':{task:{'rows':len(cell),'exact_eos':sum(r['exact_eos'] for r in cell)/len(cell)}
                     for task in sorted({r['task'] for r in records}) for cell in [[r for r in records if r['task']==task]]},
                 'examples_sha256':sha(out/'examples.jsonl'),'scope':'short Native capability/format comparison only; no long-context result'}
        write_json(out/'summary.json',summary);summaries.append({'model':checkpoint.name,'cells':summary['cells']})
        print(json.dumps(summaries[-1]),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(a.output/'summary.json',{'models':summaries,'seconds':time.monotonic()-start,
                'next_action':'Prefer natural-task resolving power and cost; preserve each model actual Native reference length.'})

if __name__=='__main__': main()
