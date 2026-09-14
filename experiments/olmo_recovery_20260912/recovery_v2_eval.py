#!/usr/bin/env python3
"""Evaluate v2 adapters on complete generation and held-out long LM, without SHA gates."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

ARMS=('Native','Cosh_tau_sqrt2','Cosh_tau1','Cosh_tau2','C42V24_g4','CoshDeploy_tau1_g4','BM_g4',
      'BM_g8','MrPro_g4','MrPro_g8','MrUni_g4','BetaSym_gamma1p5_g4','BetaSym_gamma3_g4',
      'BetaSym_gamma3_g8','RangeBridge50_g8','BM_g8_RangeGain')
CHECKPOINT_ARMS=ARMS+('Cosh_tau_sqrt2_gradual_install',)
LENGTHS=(4096,8192,16384,32768)


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.incomplete')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def collect_rows(args,manifest):
    panels=[]
    if not args.only_extra_panels:
        panels.extend((Path(p),'v2_synthetic') for p in manifest.get('evaluation_panels',{}).get(args.split,[]))
        if args.regression_data:
            panels.extend((args.regression_data/f'{kind}_{args.split}.jsonl','regression_'+kind) for kind in ('qa','native'))
    panels.extend((p,'extra_'+p.parent.name) for p in args.extra_panel)
    selected=[];seen=set();cells=defaultdict(int)
    for path,suite in panels:
        if not path.is_file():raise FileNotFoundError(path)
        for index,row in enumerate(read_rows(path)):
            if args.row_split and row.get('split')!=args.row_split:continue
            if row.get('task')=='text':continue
            prompt=row.get('prompt_ids')
            if not prompt and 'target_start' in row:prompt=row['input_ids'][:row['target_start']]
            refs=row.get('references') or ([row['answer']] if row.get('answer') else [])
            if not prompt or not refs:continue
            task=row.get('task',row.get('family','qa'))
            budget=int(row.get('max_new_tokens',row.get('generation_budget',256)))
            cap=int(row.get('length_cap',row.get('length_bucket',next((x for x in LENGTHS if len(prompt)+budget<=x),32768))))
            if budget<=0 or len(prompt)+budget>cap:raise ValueError('prompt/generation reserve exceeds recorded physical length')
            cell=(suite,task,cap)
            if args.limit_per_cell and cells[cell]>=args.limit_per_cell:continue
            identifier=str(row.get('row_id',row.get('id',index)));eval_id=f'{suite}:{identifier}'
            if eval_id in seen:raise ValueError('duplicate evaluation row ID')
            seen.add(eval_id);cells[cell]+=1
            selected.append({**row,'eval_id':eval_id,'suite':suite,'task':task,'prompt_ids':prompt,
                             'references':refs,'max_new_tokens':budget,'length_cap':cap})
    return selected


def normalized(text):
    return text.strip().lower().strip(' .,!;:\"\'`\n\t')


def greedy_tokens(model, ids, *, max_new_tokens, eos_ids, pad_token_id, prefill_chunk_size=0):
    if not prefill_chunk_size or ids.shape[1] <= prefill_chunk_size:
        return model.generate(ids, attention_mask=ids.new_ones(ids.shape), do_sample=False, num_beams=1,
                              repetition_penalty=1., no_repeat_ngram_size=0, max_new_tokens=max_new_tokens,
                              eos_token_id=list(eos_ids), pad_token_id=pad_token_id,
                              use_cache=True)[0, ids.shape[1]:].tolist()
    from .runtime import register_blackwell_chunked_attention
    previous_attention = model.config._attn_implementation
    register_blackwell_chunked_attention(model)
    try:
        return chunked_greedy_tokens(model, ids, max_new_tokens=max_new_tokens,
                                     eos_ids=eos_ids, prefill_chunk_size=prefill_chunk_size)
    finally:
        model.config._attn_implementation = previous_attention


def batched_greedy_tokens(
    model, prompt_ids, *, max_new_tokens, eos_ids, pad_token_id, left_pad=False,
):
    """Generate a batch, optionally left-padding only at the attention-mask layer."""
    import torch

    lengths = [len(values) for values in prompt_ids]
    if not left_pad and len(set(lengths)) != 1:
        raise ValueError("Flash-only batching requires equal prompt lengths")
    maximum = max(lengths)
    if left_pad:
        if pad_token_id is None:
            raise ValueError("left-padded batching requires a tokenizer pad token")
        ids = torch.full(
            (len(prompt_ids), maximum), int(pad_token_id), device="cuda", dtype=torch.long,
        )
        mask = torch.zeros_like(ids)
        for index, values in enumerate(prompt_ids):
            length = len(values)
            ids[index, maximum - length:] = torch.tensor(values, device="cuda", dtype=torch.long)
            mask[index, maximum - length:] = 1
    else:
        ids = torch.tensor(prompt_ids, device="cuda", dtype=torch.long)
        mask = torch.ones_like(ids)
    output = model.generate(
        ids, attention_mask=mask, do_sample=False, num_beams=1,
        repetition_penalty=1., no_repeat_ngram_size=0,
        max_new_tokens=max_new_tokens, eos_token_id=list(eos_ids),
        pad_token_id=pad_token_id, use_cache=True,
    )[:, maximum:]
    results = []
    for sequence in output.tolist():
        trimmed = []
        for token in sequence:
            trimmed.append(token)
            if token in eos_ids:
                break
        results.append(trimmed)
    return results


def chunked_greedy_tokens(model, ids, *, max_new_tokens, eos_ids, prefill_chunk_size):
    import torch
    from transformers import DynamicCache
    cache = DynamicCache()
    total = 0
    logits = None
    for start in range(0, ids.shape[1], prefill_chunk_size):
        piece = ids[:, start:start + prefill_chunk_size]
        stop = start + piece.shape[1]
        output = model(input_ids=piece, attention_mask=ids.new_ones((1, stop)),
                       past_key_values=cache, use_cache=True,
                       cache_position=torch.arange(start, stop, device=ids.device),
                       logits_to_keep=1, return_dict=True)
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
        total = stop
        del output
    generated = []
    for _ in range(max_new_tokens):
        token = logits.argmax(dim=-1)
        value = int(token.item()); generated.append(value)
        if value in eos_ids:
            break
        output = model(input_ids=token[:, None], attention_mask=ids.new_ones((1, total + 1)),
                       past_key_values=cache, use_cache=True,
                       cache_position=torch.tensor([total], device=ids.device),
                       logits_to_keep=1, return_dict=True)
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
        total += 1
        del output
    return generated


def lm_loss_rows(model, token_ids, *, tail=128, chunk_size=128):
    """Compute exact next-token NLL without importing unrelated evaluation data code."""
    import torch.nn.functional as F
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] < 2:
        raise ValueError('LM window must have shape [1,L+1]')
    hidden=model.model(input_ids=token_ids[:,:-1],use_cache=False).last_hidden_state[0]
    targets=token_ids[0,1:];count=int(targets.numel());tail_count=min(int(tail),count)
    whole_sum=tail_sum=0.
    for start in range(0,count,chunk_size):
        stop=min(start+chunk_size,count)
        logits=F.linear(hidden[start:stop],model.lm_head.weight).float()
        losses=F.cross_entropy(logits,targets[start:stop],reduction='none')
        whole_sum+=float(losses.sum().detach());overlap=max(start,count-tail_count)
        if overlap<stop:tail_sum+=float(losses[overlap-start:].sum().detach())
    return {'whole_loss_sum':whole_sum,'whole_target_count':count,
            'tail128_loss_sum':tail_sum,'tail128_target_count':tail_count}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True,help='v2 data manifest')
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--checkpoint-arm',choices=CHECKPOINT_ARMS)
    parser.add_argument('--split',choices=['dev','test'],default='dev')
    parser.add_argument('--row-split',choices=['fit','select','internal_confirm'],
                        help='for explicit panels, retain only rows carrying this prepared split')
    parser.add_argument('--regression-data',type=Path);parser.add_argument('--extra-panel',type=Path,action='append',default=[])
    parser.add_argument('--only-extra-panels',action='store_true',help='evaluate only explicitly supplied panels')
    parser.add_argument('--skip-lm',action='store_true',help='skip the held-out LM panel for generation-only downstream runs')
    parser.add_argument('--lm-limit-documents',type=int,default=0,help='0 keeps every held-out LM document')
    parser.add_argument('--lm-length-cap',type=int,action='append',default=[],
                        help='restrict LM evaluation to these prepared lengths; default is the legacy full grid')
    parser.add_argument('--length-cap',type=int,action='append',default=[],help='keep only these physical length caps')
    parser.add_argument('--task',action='append',default=[],help='keep only these generation tasks')
    parser.add_argument('--limit-per-cell',type=int,default=0,help='0 keeps every row; a positive limit is explicitly reported as a subset')
    parser.add_argument('--prefill-chunk-size',type=int,default=0,
                        help='0 uses stock generate; positive values enable exact-context chunked KV prefill')
    parser.add_argument('--batch-size',type=int,default=1,
                        help='contiguous exact-length generation batch using Flash SDPA')
    parser.add_argument('--left-pad-batches',action='store_true',
                        help='batch variable prompt lengths using masked left pad tokens; prompt content and position ids are unchanged')
    parser.add_argument('--static-table-json',type=Path,
                        help='install the table object (or result.table) from this frozen solver receipt')
    parser.add_argument('--table-label',help='result label for --static-table-json; does not alter the table')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();manifest=json.loads(args.data.read_text())
    if not args.execute:
        print(json.dumps({'status':'PLAN_ONLY','arm':args.arm,'checkpoint':str(args.checkpoint) if args.checkpoint else None,
                          'split':args.split,'row_split':args.row_split,'lengths':LENGTHS,
                          'static_table_json':str(args.static_table_json) if args.static_table_json else None,
                          'asset_identity_policy':'user_attested_clone/no_sha_validation'}));return
    lm_lengths=tuple(args.lm_length_cap or LENGTHS)
    if (args.limit_per_cell<0 or args.lm_limit_documents<0 or args.prefill_chunk_size<0 or args.batch_size<1
            or len(set(lm_lengths))!=len(lm_lengths) or any(length not in LENGTHS for length in lm_lengths)):
        raise ValueError('limits and prefill chunk size must be nonnegative')
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from .recovery_v2_runtime import load_model
    from .runtime import validate_cuda
    from scripts.eval.longbench_metrics import qa_f1_score
    validate_cuda();rows=collect_rows(args,manifest)
    if args.length_cap:
        allowed=set(args.length_cap);rows=[row for row in rows if row['length_cap'] in allowed]
    if args.task:
        allowed_tasks=set(args.task);rows=[row for row in rows if row['task'] in allowed_tasks]
    if args.batch_size > 1:
        rows=sorted(rows,key=lambda row:(row['length_cap'],row['max_new_tokens'],len(row['prompt_ids']),row['task'],row['eval_id']))
    lm_path=None if args.skip_lm else manifest.get('lm_evaluation',{}).get(args.split)
    if not rows and not lm_path:raise ValueError('no evaluation material supplied')
    state=json.loads((args.checkpoint/'state.json').read_text()) if args.checkpoint else {}
    checkpoint_arm=args.checkpoint_arm or args.arm
    if args.checkpoint is None and args.checkpoint_arm:raise ValueError('--checkpoint-arm requires --checkpoint')
    if args.static_table_json and args.checkpoint:
        raise ValueError('--static-table-json is a frozen-model evaluation and cannot use a checkpoint')
    if args.table_label and not args.static_table_json:
        raise ValueError('--table-label requires --static-table-json')
    static_table=None
    if args.static_table_json:
        payload=json.loads(args.static_table_json.read_text())
        static_table=payload.get('table',payload)
        values=np.asarray(static_table.get('values_float32'),dtype=np.float32)
        gain=float(static_table.get('gain'))
        if values.shape!=(64,) or not np.isfinite(values).all() or not np.all(values[:-1]>values[1:]) or not np.isfinite(gain) or gain<=0:
            raise ValueError('invalid frozen solver table')
        static_table={'values_float32':values.tolist(),'gain':gain,
                      'construction':static_table.get('construction',{})}
    result_arm=args.table_label or args.arm
    if state and state.get('arm')!=checkpoint_arm:raise ValueError('checkpoint belongs to another arm')
    identity={'arm':result_arm,'base_arm':args.arm,'checkpoint_arm':checkpoint_arm if args.checkpoint else None,
              'split':args.split,'seed':state.get('seed'),'input_tokens':state.get('input_tokens'),
              'unadapted':args.checkpoint is None,'row_ids':[r['eval_id'] for r in rows],
              'lengths':list(LENGTHS),'generation_length_caps':sorted({r['length_cap'] for r in rows}),
              'lm_enabled':bool(lm_path),'lm_limit_documents':args.lm_limit_documents,
              'lm_lengths':list(lm_lengths),
              'limit_per_cell':args.limit_per_cell,'prefill_chunk_size':args.prefill_chunk_size,
              'batch_size':args.batch_size,
              **({'left_pad_batches':True} if args.left_pad_batches else {}),
              'row_split':args.row_split,'static_table':static_table}
    args.out.mkdir(parents=True,exist_ok=True);contract=args.out/'contract.json'
    if contract.exists() and json.loads(contract.read_text())!=identity:raise ValueError('output contains a different evaluation')
    write(contract,identity)
    model,wrapper,table=load_model(args.model,args.arm,checkpoint=args.checkpoint,training=False)
    if static_table:
        from scripts.experiments.cross_audit.tables import install_static
        install_static(model,np.asarray(static_table['values_float32'],dtype=np.float32),static_table['gain'])
        actual=model.model.rotary_emb.inv_freq.detach().cpu().float().numpy()
        if not np.array_equal(actual,np.asarray(static_table['values_float32'],dtype=np.float32)):
            raise RuntimeError('installed solver table differs from the frozen receipt')
        table=static_table
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    path=args.out/'generations.jsonl';saved=read_rows(path) if path.exists() else []
    if len(saved)>len(rows) or any(row['eval_id']!=rows[i]['eval_id'] for i,row in enumerate(saved)):
        raise ValueError('saved generations are not the expected prefix')
    with path.open('a') as stream,torch.inference_mode():
        cursor=len(saved)
        while cursor < len(rows):
            first=rows[cursor]
            batch=[first]
            while (len(batch)<args.batch_size and cursor+len(batch)<len(rows)
                   and rows[cursor+len(batch)]['max_new_tokens']==first['max_new_tokens']
                   and rows[cursor+len(batch)]['length_cap']==first['length_cap']
                   and (args.left_pad_batches
                        or len(rows[cursor+len(batch)]['prompt_ids'])==len(first['prompt_ids']))):
                batch.append(rows[cursor+len(batch)])
            if len(batch)==1:
                ids=torch.tensor([first['prompt_ids']],device='cuda',dtype=torch.long)
                token_batches=[greedy_tokens(
                    model,ids,max_new_tokens=first['max_new_tokens'],eos_ids=eos,
                    pad_token_id=tokenizer.pad_token_id,prefill_chunk_size=args.prefill_chunk_size,
                )]
            else:
                token_batches=batched_greedy_tokens(
                    model,[row['prompt_ids'] for row in batch],
                    max_new_tokens=first['max_new_tokens'],eos_ids=eos,
                    pad_token_id=tokenizer.pad_token_id,
                    left_pad=args.left_pad_batches,
                )
            for row,tokens in zip(batch,token_batches):
                ended=bool(tokens and tokens[-1] in eos)
                text=tokenizer.decode(tokens[:-1] if ended else tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                literal=any(text.strip()==ref.strip() for ref in row['references'])
                exact=any(normalized(text)==normalized(ref) for ref in row['references'])
                record={key:row.get(key) for key in (
                    'eval_id','row_id','suite','task','length_cap','input_tokens','prompt_sha256',
                    'document_cluster_id','group_id','source_seed','world','references',
                    'source_document_id','semantic_group_id','generator_seed','depth_profile',
                    'depth_target','depth_error_mean_abs','evidence_positions')}
                record.update(arm=result_arm,generated_ids=tokens,output_text=text,whole_response_f1=qa_f1_score(text,row['references']),
                              literal_exact=literal,literal_exact_plus_eos=literal and ended,normalized_exact=exact,exact_plus_eos=exact and ended,
                              ended_eos=ended,empty=not text.strip(),hit_cap=len(tokens)==row['max_new_tokens'] and not ended)
                if row['task'].startswith('niah_') or row['task'] in ('vt','cwe','fwe','qa_1','qa_2'):
                    from scripts.experiments.olmo_fast_screen.ruler_bench import score
                    record['ruler_official_score']=score(row,text)
                stream.write(json.dumps(record)+'\n');stream.flush();saved.append(record)
            cursor+=len(batch)
            write(args.out/'live.json',{'phase':'generation','completed':len(saved),'total':len(rows)})
    lm_file=args.out/'lm_rows.jsonl';lm_saved=read_rows(lm_file) if lm_file.exists() else []
    if lm_path:
        values=np.load(lm_path,mmap_mode='r',allow_pickle=False)
        if values.ndim!=2 or values.shape[1]<max(lm_lengths)+1:raise ValueError('held-out LM is shorter than the requested grid')
        if args.lm_limit_documents:values=values[:args.lm_limit_documents]
        expected=[(i,length) for i in range(len(values)) for length in lm_lengths]
        if len(lm_saved)>len(expected) or any((row['document'],row['length'])!=expected[i] for i,row in enumerate(lm_saved)):
            raise ValueError('LM output prefix differs')
        with lm_file.open('a') as stream,torch.inference_mode():
            for document,length in expected[len(lm_saved):]:
                tokens=torch.tensor(values[document,:length+1].copy(),device='cuda',dtype=torch.long).unsqueeze(0)
                with torch.autocast('cuda',dtype=torch.bfloat16):record=lm_loss_rows(model,tokens)
                record.update(document=document,length=length)
                stream.write(json.dumps(record)+'\n');stream.flush();lm_saved.append(record)
    cells=defaultdict(list)
    for row in saved:cells[(row['suite'],row['task'],row['length_cap'])].append(row)
    metrics={}
    for cell,items in cells.items():
        entry={'rows':len(items)}
        for key in ('whole_response_f1','literal_exact','literal_exact_plus_eos','normalized_exact','exact_plus_eos','ended_eos','empty','hit_cap'):
            entry[key]=sum(float(r[key]) for r in items)/len(items)
        official=[r['ruler_official_score'] for r in items if 'ruler_official_score' in r]
        if official:entry['ruler_official_score']=sum(official)/len(official)
        metrics['/'.join(map(str,cell))]=entry
    lm_summary={}
    for length in lm_lengths:
        items=[r for r in lm_saved if r['length']==length]
        if items:lm_summary[str(length)]={'documents':len(items),
            'whole_nll':sum(r['whole_loss_sum'] for r in items)/sum(r['whole_target_count'] for r in items),
            'tail128_nll':sum(r['tail128_loss_sum'] for r in items)/sum(r['tail128_target_count'] for r in items)}
    pairs=defaultdict(list)
    for row in saved:
        if row.get('group_id') and row.get('world') is not None:pairs[(row['suite'],row['group_id'])].append(row)
    paired=[all(r['exact_plus_eos'] for r in items) and normalized(items[0]['output_text'])!=normalized(items[1]['output_text'])
            for items in pairs.values() if len(items)==2 and {r['world'] for r in items}=={0,1}]
    summary={'status':'COMPLETE','identity':identity,'generation_metrics':metrics,'lm_metrics':lm_summary,
             'paired_source_follow':{'groups':len(paired),'accuracy':sum(paired)/len(paired) if paired else None},
             'table':table,
             'asset_identity_policy':'user_attested_clone/no_sha_validation',
             'scope':'development/regression or explicit supplied panels; no automatic stability gate or method win'}
    write(args.out/'summary.json',summary);write(args.out/'status.json',{'status':'COMPLETE','rows':len(saved),'lm_rows':len(lm_saved)})


if __name__=='__main__':main()
