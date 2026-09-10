"""Swap the two competing values while preserving token counts and positions."""
from collections import Counter
import json
import torch
from .worker import read_rows,save,sha,digest


def occurrences(ids,pattern):
    return [i for i in range(len(ids)-len(pattern)+1) if ids[i:i+len(pattern)]==pattern]


def digit_margin(record,a,b):
    if record['generated_ids'][:1]!=[220]:return None
    step=record['token_trace'][1];scores=dict(zip(step['top_ids'],step['top_logits']))
    return scores[a]-scores[b] if a in scores and b in scores else None


def run(worker,job):
    from scripts.experiments.olmo_fast_screen.diagnose import greedy
    from scripts.experiments.olmo_fast_screen.ruler_bench import score
    key='niah_multikey_2_131072_2';name='E1_s28_less';a,b='6683176','9424151'
    row=next(r for r in worker.screen if r['row_id']==key);original=row['prompt_ids']
    ta,tb=[worker.tokenizer.encode(x,add_special_tokens=False) for x in (a,b)]
    pa,pb=occurrences(original,ta),occurrences(original,tb)
    if len(pa)!=1 or len(pb)!=1 or len(ta)!=len(tb):raise ValueError('nonunique or unequal-length swap')
    ids=list(original);ids[pa[0]:pa[0]+len(ta)]=tb;ids[pb[0]:pb[0]+len(tb)]=ta
    if Counter(ids)!=Counter(original):raise ValueError('token multiset changed')
    new={**row,'row_id':key+'__binding_swap','prompt_ids':ids,'prompt_sha256':digest(ids),'references':[b]}
    folder=worker.root/'counterfactual_bindings';folder.mkdir(exist_ok=True)
    save(folder/'input.json',dict(row=new,original_prompt_sha256=row['prompt_sha256'],value_a=a,value_b=b,
        value_token_positions=[pa[0],pb[0]],source_sha256=sha(__file__),scope='Retrospective binding-sensitivity diagnostic, not calibration or confirmation data'))
    causal=json.loads((worker.root/'causal_cases'/(name+'__'+key+'.json')).read_text());results=[]
    with torch.inference_mode():
        for arm in ('MrPro',name):
            out=folder/(arm+'.json')
            if out.exists():results.append(json.loads(out.read_text()));continue
            spec={'table':worker.tables['MrPro']} if arm=='MrPro' else json.loads((worker.root/'results'/name/'contract.json').read_text())['spec']
            worker.apply(spec)
            data,cache=greedy(worker.model,ids,list(range(len(ids))),max_new_tokens=row['max_new_tokens'],eos_token_id=worker.decoding.eos_token_id)
            text=worker.tokenizer.decode(data['generated_ids'][:-1] if data['ended_eos'] else data['generated_ids'],skip_special_tokens=False)
            old=next(r for r in causal['records'] if r['source']==arm and r['read']==arm and r['mode']=='full')
            plus,minus=digit_margin(old,ta[0],tb[0]),digit_margin(data,ta[0],tb[0])
            r=dict(status='COMPLETE',method=arm,original_score=old['correct'],swapped_score=score(new,text),output_text=text,**data,
                first_digit_D_plus=plus,first_digit_D_minus=minus,
                first_digit_evidence_response=(plus-minus)/2 if plus is not None and minus is not None else None,
                first_digit_shared_preference=(plus+minus)/2 if plus is not None and minus is not None else None,
                scope='First differing digit at the same preceding space when present; not complete answer likelihood. Full free-generation scores are separate.')
            save(out,r);results.append(r);del cache
            save(worker.root/'live.json',dict(job=job['id'],phase='binding_swap',method=arm,original_score=r['original_score'],swapped_score=r['swapped_score']))
    worker.apply({'table':worker.tables['MrPro']})
    return dict(status='COMPLETE',results=[{k:r[k] for k in ('method','original_score','swapped_score','first_digit_D_plus','first_digit_D_minus')} for r in results])
