"""Prefix-only sentence-unit allocation with unchanged native pre-RoPE KeyDiff.

Greedily rank units by mean score of their unprotected tokens, admit whole units
that fit, then fill remaining slots by the original token score. A deterministic
permutation of unit lengths provides a boundary-alignment control. No query,
answer, record metadata, reader mask, or modified model state enters selection.
This is a mechanism baseline, not a novelty or optimal-knapsack claim.
"""
import argparse
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch
from .adapter import AdapterConfig, _sync
from .canonical_keydiff import CanonicalKeyDiffSession, ROW_IDS
from .run import digest, model_identity, records, write_json
from experiments.broad_position_eval.scoring import score


def sentence_units(text, offsets):
    starts = [s for s, e in offsets]
    cuts = {0, len(offsets)}
    for m in re.finditer(r'[.!?](?=\s)|\n+', text):
        cuts.add(bisect_left(starts, m.end()))
    cuts = sorted(cuts)
    return [(a,b) for a,b in zip(cuts,cuts[1:]) if b>a]


def permuted_units(units):
    lengths = [b-a for a,b in units]
    random.Random(20260910).shuffle(lengths)
    result, start = [], 0
    for length in lengths:
        result.append((start,start+length)); start += length
    return result


@torch.no_grad()
def allocate_units(scores, units, budget, *, sink_tokens=4, recent_tokens=256):
    h, t = scores.shape
    if not units or units[0][0] != 0 or units[-1][1] != t or any(a>=b for a,b in units) or any(x[1]!=y[0] for x,y in zip(units,units[1:])):
        raise ValueError('units must partition original token positions')
    protected = set(range(min(sink_tokens,t))) | set(range(max(0,t-recent_tokens),t))
    if not len(protected)<=budget<=t or not torch.isfinite(scores).all():
        raise ValueError('invalid fixed budget or nonfinite score')
    optional = [[i for i in range(a,b) if i not in protected] for a,b in units]
    output, receipts = [], []
    for values in scores.float().cpu().tolist():
        ranking = sorted((i for i,ids in enumerate(optional) if ids),
            key=lambda i:(-sum(values[j] for j in optional[i])/len(optional[i]),i))
        selected, admitted = set(protected), []
        for i in ranking:
            ids=optional[i]
            if len(selected)+len(ids)<=budget:
                selected.update(ids); admitted.append(i)
        remaining=budget-len(selected)
        if remaining:
            order=sorted((i for i in range(t) if i not in selected),key=lambda i:(-values[i],i))
            selected.update(order[:remaining])
        output.append(sorted(selected))
        receipts.append({'whole_units_admitted':len(admitted),'remainder_token_slots':remaining})
    return torch.tensor(output,device=scores.device,dtype=torch.long), receipts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','model','data','reuse-from','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(); root,old,out=Path(a.root),Path(a.reuse_from),Path(a.output)
    original=json.loads((old/'contract.json').read_text())
    identity=model_identity(a.model)
    for k in ('config_sha256','tokenizer_sha256','weights'):
        if identity[k]!=original['model'][k]: raise ValueError('model identity '+k)
    config=AdapterConfig(**original['config'])
    lookup={r['row_id']:r for r in records(a.data)}
    rows=[lookup[rid] for rid in ROW_IDS]
    controls={r['row_id']:r for r in records(old/'per_example.jsonl') if r['arm']=='K_pre'}
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    all_units={}
    for row in rows:
        rid=row['row_id']
        if row['split']!='dev' or digest(row['prompt_ids'])!=original['input_sha256'][rid]:
            raise ValueError('fixed DEV input identity')
        if row['prefix_ids']+row['suffix_ids']!=row['prompt_ids']:
            raise ValueError('prefix boundary')
        enc=tokenizer(row['prefix_text'],add_special_tokens=False,return_offsets_mapping=True)
        if enc['input_ids']!=row['prefix_ids']: raise ValueError('prefix retokenization')
        units=sentence_units(row['prefix_text'],enc['offset_mapping'])
        all_units[rid]={'sentence':units,'permuted':permuted_units(units)}
    contract={'probe':'coherent_prefix_sentence_keydiff_v1','model':identity,'config':original['config'],
        'dtype':original['dtype'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'row_ids':list(ROW_IDS),'inputs':original['input_sha256'],'units':all_units,
        'new_generations':48,'reuse_from':str(old),'control':'same length multiset, seed20260910 permuted boundaries',
        'scientific_change':'only token allocation to whole prefix units; same K_pre scores, cache, position, reader',
        'prediction':'sentence boundaries improve complete evidence retention and exact+EOS versus token and permuted units',
        'scope':'24 existing DEV; mechanism baseline, no novelty or independent TEST claim'}
    if not a.execute: print(json.dumps(contract)); return
    if (root/'STOP').exists() or out.exists() or not torch.cuda.is_available():
        raise RuntimeError('STOP, existing output, or CUDA unavailable')
    torch.set_num_threads(4); write_json(out/'contract.json',contract)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(out/'status.json',{'status':'LOADING'})
    started,completed=time.perf_counter(),[]
    try:
        model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
            torch_dtype=getattr(torch,original['dtype']),attn_implementation='sdpa').cuda().eval()
        eos=model.generation_config.eos_token_id; eos=set(eos if isinstance(eos,list) else [eos])
        with (out/'per_example.jsonl').open('w') as stream:
            for row in rows:
                if (root/'STOP').exists(): break
                rid=row['row_id']; row_started=time.perf_counter()
                write_json(out/'status.json',{'status':'RUNNING','row_id':rid,'completed_rows':completed})
                session=CanonicalKeyDiffSession(model,row['prefix_ids'],config).prefill()
                original_sets=session.keep_indices('K_pre')
                original_hash=digest([x.cpu().tolist() for x in original_sets])
                if original_hash!=controls[rid]['keep_indices_sha256']:
                    raise ValueError('same-prefix K_pre did not reproduce frozen support')
                stream.write(json.dumps({**controls[rid],'reused_cell':True})+'\n'); stream.flush()
                sets,allocation_receipts={},{}
                _sync(session.device); allocation_start=time.perf_counter()
                for arm in ('sentence','permuted'):
                    selected=[allocate_units(s,all_units[rid][arm],session.total_budget,
                        sink_tokens=config.sink_tokens,recent_tokens=config.recent_tokens) for s in session.scores['K_pre']]
                    sets[arm]=[x[0] for x in selected]; allocation_receipts[arm]=[x[1] for x in selected]
                _sync(session.device); allocation_seconds=time.perf_counter()-allocation_start
                torch.save({arm:[x.cpu() for x in indices] for arm,indices in sets.items()},out/(rid+'.keep_sets.pt'))
                write_json(out/(rid+'.selection.json'),{'prefix_sha256':digest(row['prefix_ids']),
                    'baseline_keep_sha256':original_hash,'budget_per_head':session.total_budget,
                    'keep_hashes':{arm:digest([x.cpu().tolist() for x in ids]) for arm,ids in sets.items()},
                    'allocation_receipts':allocation_receipts,'paired_allocation_seconds':allocation_seconds,
                    'scoring_and_prefill_timings':session.timings,'all_sets_frozen_before_question':True})
                for arm,indices in sets.items():
                    branch=session.branch(arm,indices).consume(row['suffix_ids'])
                    generated=branch.generate(row['max_new_tokens'],eos)
                    result={'row_id':rid,'task':row['task'],'arm':'K_pre_'+arm,'reused_cell':False,
                        'generated_token_ids':generated['generated_ids'],'generation':generated,
                        'keep_indices_sha256':digest([x.cpu().tolist() for x in indices]),
                        **score(row,generated['generated_ids'],tokenizer,eos)}
                    stream.write(json.dumps(result)+'\n');stream.flush();del branch
                completed.append(rid)
                write_json(out/'status.json',{'status':'RUNNING','completed_rows':completed,
                    'last_row_seconds':time.perf_counter()-row_started})
                del session,sets
        write_json(out/'status.json',{'status':'COMPLETE' if len(completed)==len(rows) else 'STOPPED',
            'completed_rows':completed,'elapsed_seconds':time.perf_counter()-started})
    except BaseException as exc:
        write_json(out/'status.json',{'status':'FAILED','error':str(exc)});raise

if __name__=='__main__': main()
