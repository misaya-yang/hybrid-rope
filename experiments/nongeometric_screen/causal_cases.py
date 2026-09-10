"""Retrospective positive-case prefill/read-table crossover, using original K."""
import json
import time
import torch
from .worker import read_rows, save, sha


def trim_prefix(cache,length):
    for layer in cache.layers:
        if type(layer).__name__!='DynamicLayer':raise ValueError('requires full-attention dynamic cache')
        layer.keys=layer.keys[...,:length,:].contiguous()
        layer.values=layer.values[...,:length,:].contiguous()
    if cache.get_seq_length()!=length:raise ValueError('cache length mismatch')


def rotated_key(raw,cos,sin):
    from transformers.models.qwen2.modeling_qwen2 import rotate_half
    return (raw*cos.unsqueeze(1)+rotate_half(raw)*sin.unsqueeze(1)).to(raw.dtype)


def run(worker,job):
    from scripts.experiments.olmo_fast_screen.diagnose import greedy
    from scripts.experiments.olmo_fast_screen.ruler_bench import score
    rows={r['row_id']:r for r in worker.screen}
    folder=worker.root/'causal_cases';folder.mkdir(exist_ok=True)
    summaries=[]
    with torch.inference_mode():
        for case in job['cases']:
            name,key=case['method'],case['row'];out=folder/(name+'__'+key+'.json')
            if out.exists():summaries.append(json.loads(out.read_text()));continue
            row=rows[key];ids=row['prompt_ids'];n=len(ids)
            specs={'MrPro':{'table':worker.tables['MrPro']},name:json.loads((worker.root/'results'/name/'contract.json').read_text())['spec']}
            references={'MrPro':worker.baseline[key],name:next(r for r in read_rows(worker.root/'results'/name/'ruler.jsonl') if r['row_id']==key)}
            result=dict(row=key,method=name,records=[],qualification=[],source_sha256=sha(__file__),
                scope='Retrospective mechanism of fixed observed gains; not held-out method evaluation')
            def execute(source,read,mode,tokens,positions,cache=None):
                worker.apply(specs[read]);start=time.monotonic()
                data,cache=greedy(worker.model,tokens,positions,max_new_tokens=row['max_new_tokens'],
                    eos_token_id=worker.decoding.eos_token_id,cache=cache)
                text=worker.tokenizer.decode(data['generated_ids'][:-1] if data['ended_eos'] else data['generated_ids'],skip_special_tokens=False)
                data.update(source=source,read=read,mode=mode,output_text=text,correct=score(row,text),elapsed_seconds=time.monotonic()-start)
                result['records'].append(data)
                save(worker.root/'live.json',dict(job=job['id'],phase='positive_case_crossover',method=name,row=key,source=source,read=read,mode=mode,correct=data['correct']))
                return data,cache
            for source in specs:
                raw={};handles=[]
                def capture(index,head_dim):
                    def hook(module,args,output):
                        if output.shape[1]==n:raw[index]=output.view(1,n,-1,head_dim).transpose(1,2)
                    return hook
                for j,layer in enumerate(worker.model.model.layers):
                    handles.append(layer.self_attn.k_proj.register_forward_hook(capture(j,layer.self_attn.head_dim)))
                try:original,cache=execute(source,source,'full',ids,list(range(n)))
                finally:
                    for handle in handles:handle.remove()
                if original['generated_ids']!=references[source]['generated_ids']:raise ValueError('positive case original differs from archived output')
                trim_prefix(cache,n-1);positions=torch.arange(n,device='cuda')[None]
                cos,sin=worker.model.model.rotary_emb(raw[0],positions)
                for j,layer in enumerate(cache.layers):
                    if not torch.equal(rotated_key(raw[j],cos,sin)[...,:n-1,:],layer.keys):raise ValueError('same-table K reconstruction differs')
                diagonal,cache=execute(source,source,'cached',ids[-1:],[n-1],cache)
                trim_prefix(cache,n-1)
                for j,layer in enumerate(cache.layers):layer.keys=rotated_key(raw[j],cos,sin)[...,:n-1,:].contiguous()
                rebuilt,cache=execute(source,source,'rebuilt_cached',ids[-1:],[n-1],cache)
                if rebuilt['generated_ids']!=diagonal['generated_ids']:raise ValueError('rebuilt cached reference differs')
                result['qualification'].append(dict(source=source,original_matches_archived=True,rebuilt_matches_cached=True,
                    cached_matches_full=diagonal['generated_ids']==original['generated_ids'],full_score=original['correct'],cached_score=diagonal['correct']))
                trim_prefix(cache,n-1);other=next(x for x in specs if x!=source);worker.apply(specs[other])
                cos,sin=worker.model.model.rotary_emb(raw[0],positions)
                for j,layer in enumerate(cache.layers):layer.keys=rotated_key(raw[j],cos,sin)[...,:n-1,:].contiguous()
                del raw,cos,sin,positions
                crossed,cache=execute(source,other,'crossed_cached',ids[-1:],[n-1],cache)
                del cache
            result['status']='COMPLETE';save(out,result);summaries.append(result)
    worker.apply({'table':worker.tables['MrPro']})
    return dict(status='COMPLETE',cases=[{k:r[k] for k in ('row','method','qualification')} for r in summaries])
