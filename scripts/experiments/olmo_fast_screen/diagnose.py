"""Source-subset, position and late-cache controls for the fixed 3B failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from scripts.experiments.scale_transport.position_visibility import check_decoder, _retain_dynamic_prompt_cache
from .prepare import sha_file, verify_weight_stats
from .run import atomic
from .runtime import install, verify
from .ruler_bench import score


def greedy(model, ids, positions, *, max_new_tokens, eos_token_id, cache=None):
    """Greedy full-prefill or cached-last-query decoding with explicit RoPE positions."""
    if not ids or len(ids)!=len(positions) or max_new_tokens<1:
        raise ValueError('invalid explicit-position generation')
    eos={eos_token_id} if isinstance(eos_token_id,int) else set(eos_token_id)
    if not eos:raise ValueError('missing EOS contract')
    device=model.device
    def forward(tokens,pos,past):
        length=0 if past is None else past.get_seq_length()
        return model(input_ids=torch.tensor([tokens],dtype=torch.long,device=device),
            position_ids=torch.tensor([pos],dtype=torch.long,device=device),
            attention_mask=torch.ones((1,length+len(tokens)),dtype=torch.long,device=device),
            past_key_values=past,use_cache=True,logits_to_keep=1)
    generated=[];trace=[]
    with torch.inference_mode():
        result=forward(ids,positions,cache)
        for step in range(max_new_tokens):
            logits=result.logits[0,-1].float()
            if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite logits')
            token=int(logits.argmax().item())
            values,indices=torch.topk(logits,min(5,logits.numel()))
            denominator=torch.logsumexp(logits,0)
            best_eos=max(float(logits[t]) for t in eos)
            non_eos=logits.clone();non_eos[list(eos)]=-torch.inf
            trace.append(dict(step=step,token_id=token,probability=float(torch.exp(logits[token]-denominator)),
                eos_margin=best_eos-float(non_eos.max()),top_ids=indices.tolist(),top_logits=values.tolist()))
            generated.append(token);cache=result.past_key_values
            del result
            if token in eos or step+1==max_new_tokens:break
            result=forward([token],[positions[-1]+step+1],cache)
    return dict(generated_ids=generated,token_trace=trace,ended_eos=generated[-1] in eos),cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();inputs=args.inputs.resolve();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    atomic(out/'status.json',dict(status='RUNNING',pid=os.getpid(),started_unix=time.time()))
    try:
        work(inputs,out)
    except BaseException as exc:
        atomic(out/'status.json',dict(status='FAILED',error=str(exc),traceback=traceback.format_exc()))
        raise


def work(inputs,out):
    started=time.monotonic()
    spec=json.loads((inputs/'manifest.json').read_text())
    prepared=Path(spec['prepared']);manifest=json.loads((prepared/'manifest.json').read_text())
    if sha_file(prepared/'manifest.json')!=spec['parent_manifest_sha256']:raise ValueError('parent changed')
    for name,h in spec['files'].items():
        if sha_file(inputs/name)!=h:raise ValueError('diagnostic layout changed')
    for name,h in manifest['prepared_files'].items():
        if sha_file(prepared/name)!=h:raise ValueError('frozen inputs changed')
    verify_weight_stats(manifest)
    for name,h in manifest['model_files_sha256'].items():
        if sha_file(Path(manifest['model_path'])/name)!=h:raise ValueError('model metadata changed')
    parameters=json.loads((prepared/'generation_config.json').read_text())
    # Transformers 5 serializes disabled legacy processors as None. Qwen is
    # decoder-only; these None values install no logits processor.
    checked=dict(parameters)
    for name,default in dict(encoder_no_repeat_ngram_size=0,encoder_repetition_penalty=1.,remove_invalid_values=False).items():
        if checked.get(name) is None:checked[name]=default
    check_decoder(checked)
    if parameters['repetition_penalty']!=1.:raise ValueError('this diagnostic requires unit repetition penalty')
    config=GenerationConfig.from_dict(parameters)
    rows={r['row_id']:r for r in map(json.loads,(prepared/'screen.jsonl').read_text().splitlines())}
    originals={r['row_id']:r for r in map(json.loads,(Path(spec['baseline_run'])/'MrProBM.jsonl').read_text().splitlines())}
    layouts=json.loads((inputs/'layouts.json').read_text());edits=json.loads((inputs/'edits.json').read_text())
    tables=json.loads((prepared/'tables.json').read_text())
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(x.strip().isdigit() for x in active.splitlines()):raise RuntimeError('another GPU process is active')
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    model=AutoModelForCausalLM.from_pretrained(manifest['model_path'],local_files_only=True,
        dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
    if model.config.model_type!='qwen2' or getattr(model.config,'use_sliding_window',False):
        raise ValueError('requires the qualified full-attention Qwen path')
    if sum(p.numel() for p in model.parameters())!=manifest['actual_parameters']:raise ValueError('model count mismatch')
    tokenizer=AutoTokenizer.from_pretrained(manifest['model_path'],local_files_only=True)
    root=Path(__file__).resolve().parents[3]
    deps=set(manifest['code_files'])|{'scripts/experiments/olmo_fast_screen/diagnose.py',
        'scripts/experiments/olmo_fast_screen/prepare_diagnosis.py','scripts/experiments/scale_transport/position_visibility.py'}
    atomic(out/'runtime.json',dict(parent_manifest_sha256=sha_file(prepared/'manifest.json'),
        diagnostic_manifest_sha256=sha_file(inputs/'manifest.json'),model=manifest['model_id'],
        code_files={name:sha_file(root/name) for name in deps},backend='Flash SDPA only',
        scope='Oracle diagnostic; not a task-score benchmark or deployment method'))
    results=[]
    def execute(name,arm,row,ids,positions,cache=None):
        if (out/'STOP').exists():raise RuntimeError('operator stop')
        atomic(out/'live.json',dict(stage=name,arm=arm,row_id=row['row_id'],completed=len(results),input_tokens=len(ids)))
        install(model,tables[arm]);verify(model,tables[arm])
        begin=time.monotonic()
        data,cache=greedy(model,ids,positions,max_new_tokens=row['max_new_tokens'],
                          eos_token_id=config.eos_token_id,cache=cache)
        new=data['generated_ids'];text=tokenizer.decode(new[:-1] if data['ended_eos'] else new,skip_special_tokens=False)
        data.update(mode=name,arm=arm,row_id=row['row_id'],references=row['references'],
            correct=score(row,text),output_text=text,input_tokens=len(ids),last_position=positions[-1],
            elapsed_seconds=time.monotonic()-begin)
        results.append(data)
        atomic(out/'results.json',results)
        print(json.dumps({k:data[k] for k in ('mode','arm','row_id','correct','elapsed_seconds')}),flush=True)
        return data,cache
    # One real, compact all-keep qualification, including removal of the prior generated cache.
    layout=layouts[0];row=rows[layout['row_id']];tiny=[row['prompt_ids'][j] for j in layout['keep_positions']]
    install(model,tables['MrProBM'])
    with torch.inference_mode():
        ids=torch.tensor([tiny],device='cuda')
        expected=model.generate(ids,attention_mask=torch.ones_like(ids),generation_config=config,max_new_tokens=12)[0,len(tiny):].tolist()
    trial,cache=greedy(model,tiny,list(range(len(tiny))),max_new_tokens=12,eos_token_id=config.eos_token_id)
    cache=_retain_dynamic_prompt_cache(cache,list(range(len(tiny)-1)))
    late,_=greedy(model,[tiny[-1]],[len(tiny)-1],max_new_tokens=12,eos_token_id=config.eos_token_id,cache=cache)
    if trial['generated_ids']!=expected or late['generated_ids']!=expected:
        raise ValueError('manual or all-keep late-cache decoding differs from HF generate')
    atomic(out/'qualification.json',dict(status='PASS',hf_ids=expected,manual_ids=trial['generated_ids'],late_ids=late['generated_ids']))
    del cache,ids
    # C and P differ only in positions; preserve the whole few-shot example for VT.
    for layout in layouts:
        row=rows[layout['row_id']];keep=layout['keep_positions']
        selected=[row['prompt_ids'][j] for j in keep]
        for arm in ('MrPro','MrProBM'):
            for mode,pos in [('C',list(range(len(keep)))),('P',keep)]:
                result,cache=execute(mode,arm,row,selected,pos)
                del cache
    # Only the four loss cases get dense replay and late filtering; the fifth is a gain contrast.
    for layout in layouts[:4]:
        row=rows[layout['row_id']];keep=layout['keep_positions']
        compact=next(r for r in results if r['row_id']==row['row_id'] and r['arm']=='MrProBM' and r['mode']=='C')
        if compact['correct']<1:
            print(json.dumps(dict(row_id=row['row_id'],status='COMPACT_NOT_FULLY_SOLVED_LATE_CONTROL_SKIPPED')),flush=True)
            continue
        original,cache=execute('O','MrProBM',row,row['prompt_ids'],list(range(len(row['prompt_ids']))))
        original['matches_saved_generated_ids']=original['generated_ids']==originals[row['row_id']]['generated_ids']
        # Autoregressive appends do not update past K/V. Discard last prompt query and all O-generated tokens.
        cache=_retain_dynamic_prompt_cache(cache,keep[:-1])
        late,cache=execute('L','MrProBM',row,[row['prompt_ids'][-1]],[len(row['prompt_ids'])-1],cache)
        late['original_replay_matches_saved']=original['matches_saved_generated_ids']
        del cache
    for edit in edits:
        row=rows[edit['row_id']]
        for arm in ('MrPro','MrProBM'):
            result,cache=execute(edit['variant'],arm,row,edit['prompt_ids'],list(range(len(edit['prompt_ids']))))
            del cache
    atomic(out/'results.json',results)
    atomic(out/'status.json',dict(status='COMPLETE',generations=len(results),elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':main()
