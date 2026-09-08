"""A pre-identified natural error: same-budget source-block repair and matched wrong-block control.
F1 scores the entire completion; no first-line or substring extraction.
"""
import argparse,copy,hashlib,json,os,time,sys
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
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    tok=AutoTokenizer.from_pretrained(a.model);config=AutoConfig.from_pretrained(a.model)
    loader=AutoModelForImageTextToText if config.model_type=='qwen3_5' else AutoModelForCausalLM
    if config.model_type not in ('qwen2','qwen3_5'):raise ValueError('Unsupported model')
    model=loader.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    oracle=Oracle(model);eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    rows=[json.loads(x) for x in a.inputs.read_text().splitlines()]
    rows=[r for r in rows if r['row_id']=='hotpotqa_77']
    assert len(rows)==1
    text=tok.apply_chat_template([{'role':'user','content':rows[0]['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    enc=tok(text,return_offsets_mapping=True);off=enc['offset_mapping']
    title=text.index('\nCeltic nations\n');start=text.index('The six regions widely considered Celtic nations are',title);end=text.index('.',start)+1
    passage_end=text.index('\nPassage 6:',end)
    support=sorted({i//64 for i,(lo,hi) in enumerate(off) if hi>start and lo<end})
    excluded={i//64 for i,(lo,hi) in enumerate(off) if hi>title and lo<passage_end}
    wrong=[]
    for b in support:
        possible=[j for j in range(1,b+16) if j not in excluded and j not in wrong and j//16==b//16]
        if not possible:raise ValueError('No matched 1024-token distance-bin control')
        wrong.append(min(possible,key=lambda j:(abs(j-b),j)))
    annotation={'support_blocks':support,'wrong_blocks':sorted(wrong),'support_text':text[start:end],'locator':'Privileged source-sentence annotation from original document; only block indices intervene, no answer tokens inserted','control':'Equal block count, same 1024-token position bins, outside entire supporting passage'}
    (a.output/'evidence_annotation.json').write_text(json.dumps(annotation,indent=2))
    # All-keys gather qualification on real prefix tokens: exact Dense parity.
    with torch.inference_mode():
        text=tok.apply_chat_template([{'role':'user','content':rows[0]['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        qual_ids=tok(text,return_tensors='pt')['input_ids'].cuda()[:,:129]
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
    meta={'status':'RUNNING','pid':os.getpid(),'questions':len(rows),'methods':['Dense','NoPEMean','SupportRepair','WrongRepair'],
          'all_key_gather_bitwise_logit_parity':parity,'block':64,'local':2048,'remote_topk_per_query_head':32,'max_answer_tokens':384,'model':str(a.model),'model_type':config.model_type,'sparse_layer_indices':sorted(oracle.layers),'input_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),'code_sha256':frozen_code,
          'boundary':'All original full-attention layers sparse from first question token; native RoPE reader unchanged',
          'cost_boundary':'Oracles scan all keys; NOT a runtime/efficiency method; per-query-head diagnostic budget'}
    (a.output/'status.json').write_text(json.dumps(meta,indent=2));start=time.monotonic();current_group=None;records=[]
    try:
        with torch.inference_mode(),open(a.output/'outputs.jsonl','x') as stream:
            for row in rows:
                rendered=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                encoded=tok(rendered,return_offsets_mapping=True)
                ids=torch.tensor([encoded['input_ids']],device='cuda');cut_char=rendered.rfind('\n\nQuestion:')
                if cut_char<0:raise ValueError('Question boundary absent')
                cut=next(i for i,(_,end) in enumerate(encoded['offset_mapping']) if end>cut_char)
                prefix=ids[:,:cut];tail=ids[:,cut:];group=row['context_sha256']
                if current_group!=group:
                    oracle.mode='prefix';oracle.prefix_raw.clear();oracle.q.clear()
                    tick=time.monotonic();base=model(input_ids=prefix,use_cache=True,logits_to_keep=1).past_key_values
                    prefix_seconds=time.monotonic()-tick;base_ids=prefix.clone();current_group=group
                elif not torch.equal(prefix,base_ids):raise ValueError('Questions do not share identical token prefix')
                if tail.shape[1]+384>=1024:raise ValueError('Continuation buffer exceeds qualification')
                for method in meta['methods']:
                    oracle.reset('Dense' if method=='Dense' else 'NoPEMean',cut);oracle.forced_blocks=support if method=='SupportRepair' else wrong if method=='WrongRepair' else [];oracle.forced_insertions=0;cache=copy.deepcopy(base);tick=time.monotonic();last=None
                    def step(token,cache):
                        oracle.pos=cache.get_seq_length()
                        return model(input_ids=token,past_key_values=cache,position_ids=torch.tensor([[oracle.pos]],device='cuda'),
                          attention_mask=torch.ones((1,oracle.pos+1),device='cuda',dtype=torch.long),use_cache=True,logits_to_keep=1)
                    for j in range(tail.shape[1]):
                        last=step(tail[:,j:j+1],cache);cache=last.past_key_values
                    generated=[]
                    for j in range(384):
                        token=int(last.logits[0,-1].argmax());generated.append(token)
                        if token in eos:break
                        last=step(torch.tensor([[token]],device='cuda'),cache);cache=last.past_key_values
                    answer=tok.decode(generated,skip_special_tokens=True)
                    record={**row,'prompt':None,'method':method,'input_tokens':ids.shape[1],'prefix_tokens':cut,'question_ingest_tokens':tail.shape[1],
                      'prefix_sha256':hashlib.sha256(prefix.cpu().numpy().astype('<i8').tobytes()).hexdigest(),
                      'generated_ids':generated,'output_text':answer,'text_exact':any(answer.strip()==x for x in row['references']), 'normalized_exact':any(normalize_text(answer)==normalize_text(x) for x in row['references']), 'qa_f1':qa_f1_score(answer,row['references']),
                      'ended_eos':generated[-1] in eos,'continuation_seconds':time.monotonic()-tick,
                      'prefix_seconds':prefix_seconds,'selector_calls':oracle.calls,'forced_insertions':oracle.forced_insertions}
                    record['full_exact_and_eos']=record['text_exact'] and record['ended_eos'];records.append(record)
                    stream.write(json.dumps(record)+'\n');stream.flush();print(json.dumps({k:record[k] for k in ('row_id','method','output_text','text_exact','ended_eos','continuation_seconds','qa_f1')}),flush=True)
    finally:oracle.close()
    meta.update(status='COMPLETE',seconds=time.monotonic()-start,outputs=len(records),peak_cuda_bytes=torch.cuda.max_memory_allocated())
    (a.output/'status.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
