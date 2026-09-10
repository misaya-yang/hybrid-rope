"""Global position translation is an exact static-RoPE symmetry before rounding."""
import json
import torch
from .worker import read_rows,save,sha


def run(worker,job):
    from scripts.experiments.olmo_fast_screen.diagnose import greedy
    from scripts.experiments.olmo_fast_screen.ruler_bench import score
    folder=worker.root/'origin_shift';folder.mkdir(exist_ok=True)
    by_id={r['row_id']:r for r in worker.screen};results=[]
    with torch.inference_mode():
        for case in job['cases']:
            name,key=case['method'],case['row'];row=by_id[key]
            causal=json.loads((worker.root/'causal_cases'/(name+'__'+key+'.json')).read_text())
            if not all(q['original_matches_archived'] for q in causal['qualification']):raise ValueError('unqualified full baseline')
            for arm in ('MrPro',name):
                out=folder/(arm+'__'+key+'.json')
                if out.exists():results.append(json.loads(out.read_text()));continue
                spec={'table':worker.tables['MrPro']} if arm=='MrPro' else json.loads((worker.root/'results'/name/'contract.json').read_text())['spec']
                if spec.get('operator','static')!='static':raise ValueError('requires static RoPE')
                old=worker.baseline[key] if arm=='MrPro' else next(r for r in read_rows(worker.root/'results'/name/'ruler.jsonl') if r['row_id']==key)
                worker.apply(spec);offset=1
                data,cache=greedy(worker.model,row['prompt_ids'],list(range(offset,offset+len(row['prompt_ids']))),
                    max_new_tokens=row['max_new_tokens'],eos_token_id=worker.decoding.eos_token_id)
                text=worker.tokenizer.decode(data['generated_ids'][:-1] if data['ended_eos'] else data['generated_ids'],skip_special_tokens=False)
                r=dict(status='COMPLETE',method=arm,row_id=key,offset=offset,original_score=old['correct'],
                    shifted_score=score(row,text),output_text=text,tokens_equal=old['generated_ids']==data['generated_ids'],
                    **data,source_sha256=sha(__file__),scope='Retrospective finite-precision symmetry stress; same tokens and relative distances, no task input change')
                save(out,r);results.append(r);del cache
                save(worker.root/'live.json',dict(job=job['id'],phase='origin_shift',method=arm,row=key,
                    old_score=r['original_score'],shifted_score=r['shifted_score']))
    worker.apply({'table':worker.tables['MrPro']})
    return dict(status='COMPLETE',rows=[{k:r[k] for k in ('method','row_id','original_score','shifted_score','tokens_equal')} for r in results])
