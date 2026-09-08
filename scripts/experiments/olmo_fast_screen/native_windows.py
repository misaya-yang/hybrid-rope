"""Independent native-length token windows, followed by globally rephased reading."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from .cross_cache import rotated_key
from .diagnose import greedy
from .prepare import sha_file, verify_weight_stats
from .run import atomic
from .runtime import install, verify
from .ruler_bench import score
from scripts.experiments.scale_transport.position_visibility import check_decoder


def native_bank(model,ids,native_table,window):
    """Encode every prefix token once; no previous-window KV enters a forward."""
    if not ids or window<1 or native_table['gain']!=1.:
        raise ValueError('requires nonempty prefix and unit-gain Native formation')
    point='k_norm' if model.config.model_type=='olmo2' else 'k_proj'
    bank=[[] for _ in model.model.layers];boundaries=[]
    install(model,native_table);verify(model,native_table)
    for start in range(0,len(ids),window):
        tokens=ids[start:start+window];n=len(tokens);raw={};handles=[]
        def capture(index,dim):
            def hook(module,args,output):
                if output.shape[:2]!=(1,n):raise ValueError('unexpected native window projection shape')
                raw[index]=output.view(1,n,-1,dim).transpose(1,2)
            return hook
        for i,layer in enumerate(model.model.layers):
            handles.append(getattr(layer.self_attn,point).register_forward_hook(capture(i,layer.self_attn.head_dim)))
        positions=torch.arange(n,device=model.device)[None]
        try:
            output=model(input_ids=torch.tensor([tokens],device=model.device),position_ids=positions,
                attention_mask=torch.ones(1,n,dtype=torch.long,device=model.device),
                past_key_values=None,use_cache=True,logits_to_keep=1)
        finally:
            for h in handles:h.remove()
        cache=output.past_key_values
        if len(raw)!=len(bank) or cache.get_seq_length()!=n:
            raise ValueError('native window did not start with an empty cache')
        cos,sin=model.model.rotary_emb(raw[0],positions)
        for i,layer in enumerate(cache.layers):
            if type(layer).__name__!='DynamicLayer' or not torch.equal(rotated_key(raw[i],cos,sin),layer.keys):
                raise ValueError('native window key reconstruction differs')
            bank[i].append((raw[i],layer.values))
        boundaries.append([start,start+n])
        del output,cache,raw,cos,sin,positions
    joined=[]
    for pieces in bank:
        joined.append((torch.cat([x[0] for x in pieces],dim=-2),torch.cat([x[1] for x in pieces],dim=-2)))
        pieces.clear()
    return joined,boundaries


def rephase_bank(model,bank,table):
    install(model,table);verify(model,table)
    n=bank[0][0].shape[-2]
    positions=torch.arange(n,device=model.device)[None]
    cos,sin=model.model.rotary_emb(bank[0][0],positions)
    cache=DynamicCache(config=model.config)
    for i,(raw,values) in enumerate(bank):
        cache.update(rotated_key(raw,cos,sin),values,i)
    if cache.get_seq_length()!=n:raise ValueError('global cache length differs')
    return cache


def work(a):
    started=time.monotonic();m=json.loads((a.prepared/'manifest.json').read_text())
    if m.get('benchmark')!='ruler_mixed_v1':raise ValueError('this runner currently scores RULER only')
    verify_weight_stats(m)
    for name,sha in m['prepared_files'].items():
        if sha_file(a.prepared/name)!=sha:raise ValueError('prepared inputs changed')
    for name,sha in m['model_files_sha256'].items():
        if sha_file(Path(m['model_path'])/name)!=sha:raise ValueError('model metadata changed')
    by_id={r['row_id']:r for r in map(json.loads,(a.prepared/'screen.jsonl').read_text().splitlines())}
    rows=[by_id[key] for key in a.cases]
    if len(set(a.cases))!=len(a.cases):raise ValueError('duplicate cases')
    params=json.loads((a.prepared/'generation_config.json').read_text());checked=dict(params)
    for key,value in dict(encoder_no_repeat_ngram_size=0,encoder_repetition_penalty=1.,remove_invalid_values=False).items():
        if checked.get(key) is None:checked[key]=value
    check_decoder(checked)
    if params['repetition_penalty']!=1.:raise ValueError('requires ordinary greedy decoding')
    tables=json.loads((a.prepared/'tables.json').read_text())
    reused={};reused_prefills={};reused_sha=None
    if a.reuse_run:
        old=json.loads((a.reuse_run/'runtime.json').read_text())
        completion=json.loads((a.reuse_run/'status.json').read_text())
        reused_sha=sha_file(a.reuse_run/'results.json')
        if completion['status']!='COMPLETE' or completion['raw_sha256']!=reused_sha:
            raise ValueError('reused native-window run is incomplete or changed')
        expected=dict(model_id=m['model_id'],revision=m['revision'],native_window=m['native_length'],
            parent_manifest_sha256=sha_file(a.prepared/'manifest.json'),generation_config=params,
            tables=tables)
        if not set(a.readers)<=set(old['readers']):raise ValueError('requested reader was not previously measured')
        for key,value in expected.items():
            if old[key]!=value:raise ValueError('native-window reuse contract differs: '+key)
        eos=params['eos_token_id'];eos={eos} if isinstance(eos,int) else set(eos)
        for record in json.loads((a.reuse_run/'results.json').read_text()):
            row=by_id[record['row_id']];key=(record['row_id'],record['read'])
            if key in reused:raise ValueError('duplicate reused result')
            for field in ('prompt_sha256','input_tokens','task','references','length_cap','max_new_tokens'):
                if record[field]!=row[field]:raise ValueError('reused row differs: '+field)
            if (abs(score(row,record['output_text'])-record['correct'])>1e-12
                or record['ended_eos']!=(record['generated_ids'][-1] in eos)
                or len(record['generated_ids'])>row['max_new_tokens']):
                raise ValueError('reused score or termination differs')
            reused[key]=dict(record,reused_from=str(a.reuse_run))
        reused_prefills={x['row_id']:dict(x,reused_from=str(a.reuse_run))
            for x in json.loads((a.reuse_run/'prefills.json').read_text())}
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if active.strip():raise RuntimeError('GPU already busy')
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    model=AutoModelForCausalLM.from_pretrained(m['model_path'],local_files_only=True,dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa').eval()
    if model.config.model_type not in ('olmo2','qwen2') or getattr(model.config,'use_sliding_window',False):
        raise ValueError('requires reviewed full-attention model')
    if sum(p.numel() for p in model.parameters())!=m['actual_parameters']:raise ValueError('parameter count differs')
    if model.config.max_position_embeddings!=m['native_length']:raise ValueError('native context declaration differs')
    tokenizer=AutoTokenizer.from_pretrained(m['model_path'],local_files_only=True)
    root=Path(__file__).resolve().parents[3]
    deps=set(m['code_files'])|{'scripts/experiments/olmo_fast_screen/native_windows.py',
        'scripts/experiments/olmo_fast_screen/cross_cache.py','scripts/experiments/olmo_fast_screen/diagnose.py',
        'scripts/experiments/scale_transport/position_visibility.py'}
    atomic(a.out/'runtime.json',dict(model_id=m['model_id'],revision=m['revision'],
        parent_manifest_sha256=sha_file(a.prepared/'manifest.json'),code_files={p:sha_file(root/p) for p in deps},
        native_window=m['native_length'],generation_config=params,tables=tables,cases=a.cases,
        row_ids=a.cases,reused_results_sha256=reused_sha,
        readers=a.readers,torch=torch.__version__,backend='Flash SDPA only',
        definition='Every prefix token occurs once in independent native-length windows; raw keys globally rephased. Final prompt token and decode use the reader table. No copied header or discarded history token.'))
    records=[];prefills=[];reused_count=0
    with torch.inference_mode():
        for row in rows:
            if (a.out/'STOP').exists():raise RuntimeError('operator stop')
            if all((row['row_id'],reader) in reused for reader in a.readers):
                records.extend(reused[(row['row_id'],reader)] for reader in a.readers)
                prefills.append(reused_prefills[row['row_id']]);reused_count+=len(a.readers)
                atomic(a.out/'results.json',records);atomic(a.out/'prefills.json',prefills)
                continue
            atomic(a.out/'live.json',dict(row=row['row_id'],phase='NATIVE_WINDOWS'))
            begin=time.monotonic();ids=row['prompt_ids']
            bank,bounds=native_bank(model,ids[:-1],tables['Native'],m['native_length'])
            torch.cuda.synchronize();prefill=time.monotonic()-begin
            prefills.append(dict(row_id=row['row_id'],windows=bounds,seconds=prefill,source_keys_bitwise_rebuilt=True))
            atomic(a.out/'prefills.json',prefills)
            for reader in a.readers:
                atomic(a.out/'live.json',dict(row=row['row_id'],phase='READ',reader=reader))
                begin=time.monotonic();cache=rephase_bank(model,bank,tables[reader])
                data,cache=greedy(model,ids[-1:],[len(ids)-1],max_new_tokens=row['max_new_tokens'],
                    eos_token_id=params['eos_token_id'],cache=cache)
                tokens=data['generated_ids'];text=tokenizer.decode(tokens[:-1] if data['ended_eos'] else tokens,skip_special_tokens=False)
                data.update(method='NativeWindow'+reader,row_id=row['row_id'],read=reader,task=row['task'],
                    prompt_sha256=row['prompt_sha256'],input_tokens=len(ids),length_cap=row['length_cap'],
                    references=row['references'],max_new_tokens=row['max_new_tokens'],output_text=text,
                    correct=score(row,text),prefill_seconds=prefill,read_seconds=time.monotonic()-begin)
                records.append(data);atomic(a.out/'results.json',records)
                print(json.dumps({k:data[k] for k in ('row_id','read','correct','output_text','ended_eos')}),flush=True)
                del cache
            del bank
    for reader in a.readers:
        arm='NativeWindow'+reader;selected=[r for r in records if r['read']==reader]
        raw=a.out/(arm+'.jsonl');raw.write_text(''.join(json.dumps(r)+'\n' for r in selected))
        atomic(a.out/(arm+'.json'),dict(status='COMPLETE',row_ids=[r['row_id'] for r in selected],
            raw_sha256=sha_file(raw)))
    atomic(a.out/'status.json',dict(status='COMPLETE',generations=len(records)-reused_count,
        reused_generations=reused_count,
        independent_window_prefills=sum(len(p['windows']) for p in prefills if 'reused_from' not in p),
        elapsed_seconds=time.monotonic()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),
        raw_sha256=sha_file(a.out/'results.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--readers',nargs='+',choices=['MrPro','MrProBM'],default=['MrPro','MrProBM'])
    p.add_argument('--reuse-run',type=Path)
    a=p.parse_args();a.out.mkdir(exist_ok=False)
    atomic(a.out/'status.json',dict(status='RUNNING',pid=os.getpid()))
    try:work(a)
    except BaseException as e:
        atomic(a.out/'status.json',dict(status='FAILED',error=str(e),traceback=traceback.format_exc()));raise
