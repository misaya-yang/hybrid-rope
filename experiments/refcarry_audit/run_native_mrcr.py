"""Frozen native-model baseline on prepared MRCR counterfactual families.

Consumes exact prepared token IDs. Reuses question-blind history caches; full
generation and terminal EOS are saved. No adapter, training or GPU lifecycle call.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from .prepare_mrcr_pairs import digest
from .score_mrcr_pairs import score_text


def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit-families', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is not available; no automatic GPU start')
    args.output.mkdir(parents=True)
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines()]
    families = sorted({r['family_id'] for r in rows})
    if args.limit_families:
        allowed=set(families[:args.limit_families]);rows=[r for r in rows if r['family_id'] in allowed]
    rows.sort(key=lambda r:(r['variant']!='compact_control',r['family_id'],r['world'],r['requested_ordinal']))
    weights={p.name:file_hash(p) for p in args.model.glob('*.safetensors')}
    if not weights:raise ValueError('no local checkpoint weights found')
    meta=dict(status='LOADING',pid=os.getpid(),model_path=str(args.model),weight_sha256=weights,
        input_sha256=file_hash(args.inputs),questions=len(rows),method='Native',
        scope='development MRCR pairs; no method improvement or full benchmark claim',
        source_sha256={p.name:file_hash(p) for p in [Path(__file__),Path(__file__).with_name('score_mrcr_pairs.py'),Path(__file__).with_name('prepare_mrcr_pairs.py')]},
        torch=torch.__version__,device=torch.cuda.get_device_name())
    status=args.output/'status.json';status.write_text(json.dumps(meta,indent=2))
    code_dir=args.output/'code';code_dir.mkdir()
    for name in meta['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name),code_dir/name)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    config=AutoConfig.from_pretrained(args.model,local_files_only=True)
    loader=AutoModelForImageTextToText if config.model_type=='qwen3_5' else AutoModelForCausalLM
    model,loading=loader.from_pretrained(args.model,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys')):
        raise ValueError(f'checkpoint identity mismatch: {loading}')
    model=model.cuda().eval()
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    text_config=getattr(model.config,'text_config',model.config)
    meta.update(status='RUNNING',parameters=sum(p.numel() for p in model.parameters()),
        native_max_positions=text_config.max_position_embeddings,eos_token_ids=sorted(eos))
    status.write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
    start=time.monotonic();cached_group=None;base=None;base_ids=None;count=0
    try:
        with (args.output/'predictions.jsonl').open('x') as stream:
            for row in rows:
                ids=torch.tensor([row['input_ids']],device='cuda')
                if digest(row['input_ids'])!=row['input_ids_sha256']:raise ValueError('prepared token hash mismatch')
                rendered=tok.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
                if rendered!=row['input_ids']:raise ValueError('runtime tokenizer differs from prepared IDs')
                if ids.shape[1]+row['max_new_tokens']>text_config.max_position_embeddings:
                    raise ValueError('untruncated prompt and answer budget exceed native context')
                cut=row['prefix_tokens'];group=row['history_sha256'];tick=time.monotonic();prefill_seconds=0.
                if group!=cached_group:
                    base=None
                    torch.cuda.empty_cache()
                    began=time.monotonic()
                    out=model(ids[:,:cut],use_cache=True,logits_to_keep=1)
                    base=out.past_key_values;base_ids=ids[:,:cut].clone();cached_group=group
                    torch.cuda.synchronize();prefill_seconds=time.monotonic()-began
                    del out
                elif not torch.equal(base_ids,ids[:,:cut]):
                    raise ValueError('declared common histories have different tokens')
                cache=copy.deepcopy(base)
                # Cached multi-token suffixes require an explicit rectangular
                # mask in this Transformers version, unsupported by torch's
                # Flash SDPA. One-token question ingestion is fully causal and
                # also matches the prepared writer-to-reader intervention path.
                for index in range(cut, ids.shape[1]):
                    out=model(ids[:,index:index+1],past_key_values=cache,
                              use_cache=True,logits_to_keep=1)
                generated=[]
                for _ in range(row['max_new_tokens']):
                    token=int(out.logits[0,-1].argmax());generated.append(token)
                    if token in eos:break
                    out=model(torch.tensor([[token]],device='cuda'),past_key_values=cache,
                              use_cache=True,logits_to_keep=1)
                torch.cuda.synchronize();ended=generated[-1] in eos
                text=tok.decode(generated[:-1] if ended else generated,skip_special_tokens=False)
                scores=score_text(text,ended,row)
                record=dict(row_id=row['row_id'],family_id=row['family_id'],variant=row['variant'],
                    method='Native',input_ids_sha256=row['input_ids_sha256'],
                    generated_ids=generated,eos_token_ids=sorted(eos),output_text=text,
                    input_tokens=ids.shape[1],prefix_tokens=cut,max_new_tokens=row['max_new_tokens'],
                    prefill_seconds=prefill_seconds,seconds=time.monotonic()-tick,
                    **scores)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();count+=1
                print(json.dumps({k:record[k] for k in ('row_id','variant','input_tokens','seconds','official_sequence_ratio','full_exact_and_eos','ended_eos','wrong_occurrence_whole_string')}),flush=True)
                del out,cache
                meta.update(completed=count,elapsed_seconds=time.monotonic()-start)
                status.write_text(json.dumps(meta,indent=2))
        meta.update(status='COMPLETE',peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        meta.update(status='FAILED',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        status.write_text(json.dumps(meta,indent=2))


if __name__=='__main__':main()
