"""Natural long QA: query-blind prefix, native reader, exact block-ranking oracles.
F1 scores the entire completion; no first-line or substring extraction.
"""
import argparse,copy,hashlib,json,os,time,sys,re
from pathlib import Path
import torch
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,AutoModelForImageTextToText
from evidence_native import Oracle
try:
    from longbench_metrics import qa_f1_score,normalize_text
except ImportError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from scripts.eval.longbench_metrics import qa_f1_score,normalize_text
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=['Dense','RoPEMean'],default=['Dense','RoPEMean'])
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    tok=AutoTokenizer.from_pretrained(a.model);config=AutoConfig.from_pretrained(a.model)
    loader=AutoModelForImageTextToText if config.model_type=='qwen3_5' else AutoModelForCausalLM
    if config.model_type not in ('qwen2','qwen3_5'):raise ValueError('Unsupported model')
    model=loader.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    oracle=Oracle(model,topk=16);eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    rows=[json.loads(x) for x in a.inputs.read_text().splitlines()]
    assert rows and all(r['input_tokens']+128<=32768 for r in rows)
    for row in rows:
        text=tok.decode(row['prompt_ids'],skip_special_tokens=False);enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        if enc['input_ids']!=row['prompt_ids']:raise ValueError('Frozen input identity drift')
        cutchar=text.rfind('\nWhat are all the special magic')
        if cutchar<0:raise ValueError('Question boundary absent')
        row['prefix_tokens']=next(i for i,(_,end) in enumerate(enc['offset_mapping']) if end>cutchar)
    # All-keys gather qualification on real prefix tokens: exact Dense parity.
    with torch.inference_mode():
        qual_ids=torch.tensor([rows[0]['prompt_ids'][:129]],device='cuda')
        oracle.mode='prefix'
        qual_base=model(input_ids=qual_ids[:,:128],use_cache=True,logits_to_keep=1).past_key_values
        qual_outputs={}
        for method in ('Dense','RoPEMean','NoPEMean'):
            oracle.reset(method,128)
            o=model(input_ids=qual_ids[:,128:],past_key_values=copy.deepcopy(qual_base),position_ids=torch.tensor([[128]],device='cuda'),use_cache=True,logits_to_keep=1)
            qual_outputs[method]=o.logits.detach().clone()
        parity={m:bool(torch.equal(qual_outputs[m],qual_outputs['Dense'])) for m in ('RoPEMean','NoPEMean')}
        if not all(parity.values()):raise RuntimeError(f'All-key gather logit parity failed: {parity}')
        del qual_outputs,qual_base,o
    print(json.dumps({'all_key_gather_bitwise_logit_parity':parity}),flush=True)
    frozen_code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('evidence_native.py')]}
    meta={'status':'RUNNING','pid':os.getpid(),'questions':len(rows),'methods':a.methods,
          'all_key_gather_bitwise_logit_parity':parity,'block':64,'local':2048,'remote_topk_per_query_head':16,'max_answer_tokens':128,'model':str(a.model),'model_type':config.model_type,'sparse_layer_indices':sorted(oracle.layers),'input_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),'code_sha256':frozen_code,
          'boundary':'All original full-attention layers sparse from first question token; native RoPE reader unchanged',
          'cost_boundary':'Fixed m16 remote budget; new independently generated multiquery families with full four-number space-separated output+EOS; baseline quality, no new method'}
    (a.output/'status.json').write_text(json.dumps(meta,indent=2));start=time.monotonic();current_group=None;records=[]
    try:
        with torch.inference_mode(),open(a.output/'outputs.jsonl','x') as stream:
            for row in rows:
                ids=torch.tensor([row['prompt_ids']],device='cuda');cut=row['prefix_tokens']
                prefix=ids[:,:cut];tail=ids[:,cut:];group=row['row_id']
                if current_group!=group:
                    oracle.mode='prefix';oracle.prefix_raw.clear();oracle.q.clear()
                    tick=time.monotonic();base=model(input_ids=prefix,use_cache=True,logits_to_keep=1).past_key_values
                    prefix_seconds=time.monotonic()-tick;base_ids=prefix.clone();current_group=group
                elif not torch.equal(prefix,base_ids):raise ValueError('Questions do not share identical token prefix')
                if tail.shape[1]+128>=1024:raise ValueError('Continuation buffer exceeds qualification')
                for method in meta['methods']:
                    oracle.reset('Dense' if method=='Dense' else 'RoPEMean',cut);oracle.forced_blocks=row['support_blocks'] if method=='SupportRepair' else row['wrong_blocks'] if method=='WrongRepair' else [];oracle.forced_insertions=0;cache=copy.deepcopy(base);tick=time.monotonic();last=None
                    def step(token,cache):
                        oracle.pos=cache.get_seq_length()
                        return model(input_ids=token,past_key_values=cache,position_ids=torch.tensor([[oracle.pos]],device='cuda'),
                          attention_mask=torch.ones((1,oracle.pos+1),device='cuda',dtype=torch.long),use_cache=True,logits_to_keep=1)
                    for j in range(tail.shape[1]):
                        last=step(tail[:,j:j+1],cache);cache=last.past_key_values
                    generated=[]
                    for j in range(128):
                        token=int(last.logits[0,-1].argmax());generated.append(token)
                        if token in eos:break
                        last=step(torch.tensor([[token]],device='cuda'),cache);cache=last.past_key_values
                    answer=tok.decode(generated[:-1] if generated[-1] in eos else generated,skip_special_tokens=False)
                    record={**row,'prompt_ids':None,'method':method,'input_tokens':ids.shape[1],'prefix_tokens':cut,'question_ingest_tokens':tail.shape[1],
                      'prefix_sha256':hashlib.sha256(prefix.cpu().numpy().astype('<i8').tobytes()).hexdigest(),
                      'generated_ids':generated,'output_text':answer,'text_exact':answer==row['expected'],'trimmed_text_exact':answer.strip()==row['expected'],'whole_numeric_list_exact':bool(re.fullmatch(r'\s*\d+(?:\s+\d+)*\s*',answer)) and re.findall(r'\d+',answer)==row['references'],
                      'ended_eos':generated[-1] in eos,'continuation_seconds':time.monotonic()-tick,
                      'prefix_seconds':prefix_seconds,'selector_calls':oracle.calls,'forced_insertions':oracle.forced_insertions}
                    record['full_exact_and_eos']=record['text_exact'] and record['ended_eos'];record['trimmed_full_exact_and_eos']=record['trimmed_text_exact'] and record['ended_eos'];records.append(record)
                    stream.write(json.dumps(record)+'\n');stream.flush();print(json.dumps({k:record[k] for k in ('row_id','method','output_text','text_exact','ended_eos','continuation_seconds','whole_numeric_list_exact')}),flush=True)
    finally:oracle.close()
    meta.update(status='COMPLETE',seconds=time.monotonic()-start,outputs=len(records),peak_cuda_bytes=torch.cuda.max_memory_allocated())
    (a.output/'status.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
