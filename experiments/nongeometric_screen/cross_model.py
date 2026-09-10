"""Frozen relative-band E1 transfer; reuse each target model's own baseline."""
from contextlib import nullcontext
import gc
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace,MethodType
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from .worker import Worker,read_rows,save,sha,digest,install_table


def transferred_table(tables):
    # Freeze the rule selected on Qwen3B: 5/17 into the transition and one
    # predecessor exponent. No target-model labels or scores enter this map.
    native=np.asarray(tables['Native']['values_float32'],dtype=np.float64)
    base=np.asarray(tables['MrPro']['values_float32'],dtype=np.float64)
    scale=native[-1]/base[-1]
    exponents=np.log(native/base)/np.log(scale)
    middle=np.flatnonzero((exponents>1e-5)&(exponents<1-1e-5))
    if not len(middle):raise ValueError('missing MrPro transition')
    lo,hi=int(middle[0]-1),int(middle[-1]+1)
    if tables['MrPro'].get('construction'):
        construction=tables['MrPro']['construction']
        if [lo,hi]!=[construction['low'],construction['high']]:raise ValueError('table bounds do not match construction')
    j=lo+int(np.floor((hi-lo)*5/17+.5))
    frequency=base.copy();frequency[j]=native[j]*(base[j-1]/native[j-1])
    return dict(values_float32=frequency.astype(np.float32).tolist(),gain=tables['MrPro']['gain']),dict(
        source_slot=28,source_bounds=[23,40],transition_fraction='5/17',target_slot=j,target_bounds=[lo,hi],
        rule='Replace only m[j] by baseline m[j-1]; preserve target native frequencies, all other slots, and gain')


def run(worker,job):
    from scripts.experiments.olmo_fast_screen.prepare import verify_weight_stats
    from scripts.experiments.olmo_fast_screen.ruler_bench import summarize,score
    prepared,baseline=Path(job['prepared']),Path(job['baseline'])
    manifest=json.loads((prepared/'manifest.json').read_text());tables=json.loads((prepared/'tables.json').read_text())
    verify_weight_stats(manifest)
    for name in ('screen.jsonl','tables.json','generation_config.json'):
        if sha(prepared/name)!=manifest['prepared_files'][name]:raise ValueError('cross-model input drift')
    for name,h in manifest['model_files_sha256'].items():
        if sha(Path(manifest['model_path'])/name)!=h:raise ValueError('cross-model metadata drift')
    decoding=json.loads((prepared/'generation_config.json').read_text())
    if decoding['repetition_penalty']!=1.:raise ValueError('requires unchanged unit repetition penalty')
    saved={r['row_id']:r for r in read_rows(baseline/'MrPro.jsonl')}
    rows=[r for r in read_rows(prepared/'screen.jsonl') if r['row_id'] in saved]
    for r in rows:
        if digest(r['prompt_ids'])!=r['prompt_sha256']:raise ValueError('input identity invalid')
        if any(r[k]!=saved[r['row_id']][k] for k in ('prompt_sha256','references','task','length_cap')):raise ValueError('target baseline identity mismatch')
        if score(r,saved[r['row_id']]['output_text'])!=saved[r['row_id']]['correct']:raise ValueError('target baseline scorer differs')
    table,mapping=transferred_table(tables)
    folder=worker.root/'transfer'/job['target'];folder.mkdir(parents=True,exist_ok=True)
    contract=dict(target_model=manifest['model_id'],revision=manifest['revision'],table=table,mapping=mapping,
        prepared_sha256=sha(prepared/'manifest.json'),baseline_sha256=sha(baseline/'MrPro.jsonl'),
        decoding_sha256=sha(prepared/'generation_config.json'),mlp_chunk_size=job.get('mlp_chunk_size',0),source_sha256=sha(__file__))
    if (folder/'contract.json').exists() and json.loads((folder/'contract.json').read_text())!=contract:raise ValueError('transfer contract changed')
    save(folder/'contract.json',contract)
    # Keep the resident model in CPU RAM; only one model occupies the GPU.
    worker.apply({'table':worker.tables['MrPro']});worker.model.to('cpu');gc.collect();torch.cuda.empty_cache()
    context=None;chunk_context=None
    try:
        model=AutoModelForCausalLM.from_pretrained(manifest['model_path'],local_files_only=True,
            dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
        model.requires_grad_(False)
        if sum(p.numel() for p in model.parameters())!=manifest['actual_parameters']:raise ValueError('target parameter count differs')
        context=SimpleNamespace(model=model,tokenizer=AutoTokenizer.from_pretrained(manifest['model_path'],local_files_only=True),
            decoding=GenerationConfig.from_dict(decoding))
        context.generate=MethodType(Worker.generate,context)
        chunk_context=nullcontext()
        if job.get('mlp_chunk_size'):
            path=Path(job['chunk_source']);spec=importlib.util.spec_from_file_location('qualified_target_chunk',path)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            chunk_context=mod.tokenwise_mlp_chunks(model,job['mlp_chunk_size'])
        with chunk_context:
            if not (folder/'qualification.json').exists():
                install_table(model,tables['MrPro']);got=context.generate(rows[0]);exact=got['generated_ids']==saved[rows[0]['row_id']]['generated_ids']
                save(folder/'qualification.json',dict(status='PASS' if exact else 'FAIL',generated=got,original=saved[rows[0]['row_id']],
                    scope='Same target model, table, decoder and inference path; one archived-row reproduction'))
                if not exact:raise ValueError('target execution does not reproduce archived baseline')
            elif json.loads((folder/'qualification.json').read_text())['status']!='PASS':raise ValueError('target path qualification unresolved')
            install_table(model,table);raw=folder/'E1_band_transfer.jsonl';existing={r['row_id']:r for r in read_rows(raw)}
            for row in rows:
                if row['row_id'] in existing:continue
                result=context.generate(row);result['baseline_correct']=saved[row['row_id']]['correct']
                with raw.open('a') as stream:stream.write(json.dumps(result)+'\n')
                existing[row['row_id']]=result
                save(worker.root/'live.json',dict(job=job['id'],phase='cross_model',target=job['target'],row=row['row_id'],
                    completed=len(existing),requested=len(rows),correct=result['correct'],baseline=result['baseline_correct']))
            nll=[]
            if job.get('nll_prepared'):
                inp=Path(job['nll_prepared']);nmanifest=json.loads((inp/'manifest.json').read_text())
                nbase={(r['doc'],r['length']):r for r in read_rows(Path(job['nll_baseline'])/'MrPro.jsonl')}
                nll=read_rows(folder/'nll.jsonl');done={(r['doc'],r['length']) for r in nll}
                for length in sorted({length for doc,length in nbase}):
                    for doc in nmanifest['docs'][:4]:
                        if (doc['file'],length) in done:continue
                        if sha(inp/doc['file'])!=doc['sha256']:raise ValueError('target natural input drift')
                        data=np.load(inp/doc['file']);ids=torch.tensor(data[:length].astype(np.int64),device='cuda')[None]
                        target=torch.tensor(data[length-511:length+1].astype(np.int64),device='cuda')
                        with torch.inference_mode():
                            logits=model(ids,use_cache=False,logits_to_keep=512).logits[0].float()
                            losses=torch.nn.functional.cross_entropy(logits,target,reduction='none')
                        r=dict(doc=doc['file'],length=length,nll=losses.mean().item(),baseline_nll=nbase[(doc['file'],length)]['nll'],
                            input_sha256=doc['sha256'],token_nll=losses.tolist())
                        with (folder/'nll.jsonl').open('a') as stream:stream.write(json.dumps(r)+'\n')
                        nll.append(r);del ids,target,logits,losses
                        save(worker.root/'live.json',dict(job=job['id'],phase='cross_model_nll',target=job['target'],doc=doc['file'],length=length))
            records=list(existing.values());summary=dict(status='COMPLETE',candidate=summarize(records),baseline=summarize([saved[r['row_id']] for r in records]),
                wins=sum(r['correct']>r['baseline_correct'] for r in records),losses=sum(r['correct']<r['baseline_correct'] for r in records),
                nll_by_length={str(length):dict(n=sum(r['length']==length for r in nll),delta=float(np.mean([r['nll']-r['baseline_nll'] for r in nll if r['length']==length]))) for length in sorted({r['length'] for r in nll})},
                scope='Frozen source-selected band rule on another checkpoint; existing development inputs, no target outcome tuning')
            save(folder/'summary.json',summary)
    finally:
        if context is not None:context.model=None;context=None
        if 'model' in locals():del model
        chunk_context=None;gc.collect();torch.cuda.empty_cache();worker.model.to('cuda')
        worker.apply({'table':worker.tables['MrPro']})
    return summary
