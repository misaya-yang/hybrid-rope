"""Pinned MiniCPM4.1 native dense/sparse full-prompt generation, no PE table edits."""
import argparse,hashlib,importlib,json,os,sys,time
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
try:
    from longbench_metrics import qa_f1_score,normalize_text
except ImportError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from scripts.eval.longbench_metrics import qa_f1_score,normalize_text

SPARSE={'kernel_size':32,'kernel_stride':16,'init_blocks':1,'block_size':64,
        'window_size':2048,'topk':64,'use_nope':False,'dense_len':8192}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['Dense','NativeRoPE','NativeNoPE'],required=True)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    receipt=json.loads((a.model.parent/'download_status.json').read_text())
    if receipt['status']!='COMPLETE':raise ValueError('Model not verified complete')
    a.output.mkdir(parents=True);torch.manual_seed(20260908)
    config=AutoConfig.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    config.sparse_config=None if a.mode=='Dense' else {**SPARSE,'use_nope':a.mode=='NativeNoPE'}
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model,loading=AutoModelForCausalLM.from_pretrained(a.model,config=config,torch_dtype=torch.bfloat16,
      attn_implementation='flash_attention_2',trust_remote_code=True,local_files_only=True,output_loading_info=True)
    if loading['missing_keys'] or loading['unexpected_keys'] or loading['mismatched_keys']:raise ValueError(f'Weight identity mismatch: {loading}')
    model=model.cuda().eval();module=importlib.import_module(model.__class__.__module__)
    attentions=[layer.self_attn for layer in model.model.layers]
    expected='MiniCPMFlashAttention2' if a.mode=='Dense' else 'MiniCPMInfLLMv2Attention'
    if any(m.__class__.__name__!=expected for m in attentions):raise ValueError('Attention implementation differs')
    if any(float(m.rotary_emb.scaling_factor)!=1. for m in attentions):raise ValueError('Native unit-amplitude contract failed')
    rows=[json.loads(x) for x in a.inputs.read_text().splitlines()]
    if a.limit:rows=rows[:a.limit]
    eos=config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    calls={'dense':0,'sparse':0};originals=[]
    for m in attentions:
        for name,kind in ([('_flash_attention_forward','dense')] if a.mode=='Dense' else [('_flash_attention_forward_dense','dense'),('_sparse_attention_forward','sparse')]):
            old=getattr(m,name)
            def wrapped(*args,_old=old,_kind=kind,**kwargs):
                calls[_kind]+=1;return _old(*args,**kwargs)
            originals.append((m,name,old));setattr(m,name,wrapped)
    meta={'status':'RUNNING','pid':os.getpid(),'method':a.mode,'questions':len(rows),'sparse_config':config.sparse_config,
      'model_revision':receipt['revision'],'verified_weight_files':{k:v for k,v in receipt['files'].items() if k.endswith('.safetensors')},
      'actual_parameters':sum(p.numel() for p in model.parameters()),'max_new_tokens':384,'input_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'modeling_source_sha256':hashlib.sha256((a.model/'modeling_minicpm.py').read_bytes()).hexdigest(),
      'boundary':'Native full prompt prefill and autoregressive decoding; original unit-amplitude reader and frequency tables',
      'selection_budget_note':'Official config topk64 adds32 local blocks internally; actual native masks determine the read set, unlike prior per-head oracles'}
    (a.output/'status.json').write_text(json.dumps(meta,indent=2));started=time.monotonic();records=[]
    try:
        with torch.inference_mode(),(a.output/'outputs.jsonl').open('x') as stream:
            for row in rows:
                text=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok(text,return_tensors='pt')['input_ids'].cuda();N=ids.shape[1]
                if N+384>config.max_position_embeddings:raise ValueError('Untruncated prompt exceeds native context')
                cache=module.InfLLMv2Cache(config=config,num_hidden_layers=config.num_hidden_layers)
                before=calls.copy();tick=time.monotonic()
                o=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(N,device='cuda')[None],past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=o.past_key_values;prefill=time.monotonic()-tick;generated=[]
                if cache.get_seq_length()!=N:raise ValueError('Prefill cache length mismatch')
                for _ in range(384):
                    if not bool(torch.isfinite(o.logits).all()):raise ValueError('Nonfinite logits')
                    token=int(o.logits[0,-1].argmax());generated.append(token)
                    if token in eos:break
                    pos=cache.get_seq_length()
                    o=model(input_ids=torch.tensor([[token]],device='cuda'),attention_mask=torch.ones((1,pos+1),device='cuda',dtype=torch.long),
                      position_ids=torch.tensor([[pos]],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1)
                    cache=o.past_key_values
                    if cache.get_seq_length()!=pos+1:raise ValueError('Decode cache length mismatch')
                answer=tok.decode(generated,skip_special_tokens=True)
                record={**row,'prompt':None,'method':a.mode,'input_tokens':N,'input_ids_sha256':hashlib.sha256(ids.cpu().numpy().astype('<i8').tobytes()).hexdigest(),
                  'generated_ids':generated,'output_text':answer,'ended_eos':generated[-1] in eos,'trimmed_exact':any(answer.strip()==x for x in row['references']),
                  'normalized_exact':any(normalize_text(answer)==normalize_text(x) for x in row['references']),'qa_f1':qa_f1_score(answer,row['references']),
                  'prefill_seconds':prefill,'seconds':time.monotonic()-tick,'attention_calls':{k:calls[k]-before[k] for k in calls}}
                if a.mode!='Dense' and N>=SPARSE['dense_len'] and not record['attention_calls']['sparse']:raise ValueError('Sparse branch never executed')
                stream.write(json.dumps(record)+'\n');stream.flush();records.append(record)
                print(json.dumps({k:record[k] for k in ('row_id','method','input_tokens','output_text','ended_eos','qa_f1','seconds','attention_calls')}),flush=True)
                del cache,o
    finally:
        for m,name,old in originals:setattr(m,name,old)
    meta.update(status='COMPLETE',outputs=len(records),seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),attention_calls=calls)
    (a.output/'status.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
