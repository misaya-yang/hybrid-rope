"""Independent-source B0 live-query moments for E04/E09/E10, no answer labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from .exact_probe import ExactBlockSelector
from .runtime import AttentionSettings, NosaReferenceForCausalLM, apply_rope
from .selector_controls import build_summary, summary_logmass
from .run import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=1024)
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    source = [json.loads(l) for l in Path(args.candidates).read_text().splitlines()]
    rows = []
    for row in source:
        before = 'Read and remember the complete context below.\n\nCONTEXT\n' + row['context'] + '\nEND CONTEXT'
        text = before + '\n\nQuestion: ' + row['input'] + '\nAnswer concisely using only the context. Output only the answer, then stop.'
        prompt = tokenizer.apply_chat_template([{'role':'user','content':text}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not 4096 < len(ids) <= 16128:
            continue
        rows.append({'row_id': 'cal_qasper_' + row['context_sha256'][:16], 'source_id': row.get('_id'),
                     'context_sha256':row['context_sha256'], 'prompt_ids':ids, 'prompt':prompt,
                     'calibration_role': 'validation' if len(rows) % 6 == 5 else 'fit',
                     'input_tokens':len(ids), 'answer_labels_used':False})
        if len(rows) == 24:break
    if not rows:raise ValueError('no independent intact calibration documents')
    data = root/'rows.jsonl'
    rendered = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if data.exists() and data.read_text()!=rendered:raise ValueError('frozen calibration changed')
    data.write_text(rendered)
    write_json(root/'manifest.json',{'documents':len(rows),'rows_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
                                   'source_candidates_sha256':hashlib.sha256(Path(args.candidates).read_bytes()).hexdigest(),
                                   'labels_used':False,'method':'B0 exact live prompt Q; no generated answer needed',
                                   's':32,'source':'unseen complete Qasper contexts; old DEV/TEST contexts excluded'})
    print(json.dumps({'prepared_documents':len(rows),'tokens':[r['input_tokens'] for r in rows]}),flush=True)
    if args.prepare_only:return
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.45)
    started=time.time()
    status={'status':'LOADING','pid':os.getpid(),'completed_documents':0,'total_documents':len(rows)}
    write_json(root/'status.json',status)
    sums, local_sums, counts, teacher = {}, {}, {}, []
    current_row = None
    class CaptureExact(ExactBlockSelector):
        def logmass(self, context):
            result = super().logmass(context)
            self.latest = (context, result)
            return result
    selector = CaptureExact("exact_mass")
    def observe(context, selected):
        h=context.k.shape[0];d=context.q.shape[-1]
        q=context.q.float().reshape(h,-1,d)
        moment=q.transpose(1,2)@q
        layer=context.layer_idx
        sums[layer]=sums.get(layer,torch.zeros_like(moment))+moment
        counts[layer]=counts.get(layer,0)+q.shape[1]
        # Uniform past-block origins, without key/query labels, for the local
        # shared frame. Repetition leaves the global second-moment basis intact.
        for fraction in (0., 1/3, 2/3, 1.):
            anchor = (context.query_positions.float() * fraction // 64) * 64
            rotated = apply_rope(context.q.float(), -anchor, context.rope_inv_freq, 1.).reshape(h,-1,d)
            local = rotated.transpose(1,2)@rotated
            local_sums[layer] = local_sums.get(layer,torch.zeros_like(local)) + local
        if int(context.query_positions[-1]) == current_row['input_tokens']-1:
            full = int(context.query_positions[-1])//64
            keys = context.k[:,:full*64].reshape(h,full,64,d)
            cis = context.cis[:,:full*64].reshape(h,full,64)
            summary = build_summary(keys,cis,'pc2')
            z = torch.cat((summary.mean,summary.extra['var_x'],summary.extra['cov_xy'],
                           summary.extra['var_y'],summary.log_weight[...,None]),-1)
            a = context.q[:,-1:].float().reshape(h,-1,1,d)/(d**.5)
            exact = selector.latest[1][:,:,-1].clone()
            pc2 = exact.clone()
            pc2[...,:full] = summary_logmass(a,summary,'pc2').squeeze(2)
            teacher.append({'row_id':current_row['row_id'],'role':current_row['calibration_role'],
                            'layer':layer,'q':a.squeeze(2).cpu(),'z':z.cpu(),
                            'exact':exact.cpu(),'pc2':pc2.cpu(),'full_blocks':full})
    model=NosaReferenceForCausalLM.from_pretrained(args.model,device='cuda',dtype=torch.bfloat16,
        selector=selector,settings=AttentionSettings(topk=64,select_blocks=16,
        attention_query_chunk_size=args.chunk_size),trace_callback=observe)
    write_json(root/'load_report.json',model.load_report)
    try:
        for row in rows:
            current_row = row
            status.update(status='RUNNING',row_id=row['row_id']);write_json(root/'status.json',status)
            before=time.perf_counter()
            output=model.prefill(torch.tensor([row['prompt_ids']],device='cuda'),chunk_size=args.chunk_size)
            if not torch.isfinite(output.logits).all():raise FloatingPointError('nonfinite calibration state')
            del output
            torch.cuda.synchronize()
            status['completed_documents']+=1
            print(json.dumps({'row_id':row['row_id'],'input_tokens':row['input_tokens'],'seconds':time.perf_counter()-before}),flush=True)
        bases={}; local_bases={}; eigenvalues={}; local_eigenvalues={}
        for layer,total in sums.items():
            values,vectors=torch.linalg.eigh(total.double()/counts[layer])
            bases[layer]=vectors[:,:,-32:].float().cpu()
            eigenvalues[layer]=values.cpu()
            lv,lu=torch.linalg.eigh(local_sums[layer].double()/(4*counts[layer]))
            local_bases[layer]=lu[:,:,-32:].float().cpu();local_eigenvalues[layer]=lv.cpu()
        torch.save({'bases':bases,'local_bases':local_bases,'local_eigenvalues':local_eigenvalues,'eigenvalues':eigenvalues,'counts':counts,
                    'source_rows_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
                    'method':'second moment of actual B0 post-RoPE Q; per-layer/KV shared basis'},root/'query_basis.pt')
        torch.save(teacher,root/'cutoff_teacher.pt')
        status.update(status='COMPLETE',elapsed_seconds=time.time()-started)
    except BaseException as e:
        status.update(status='FAILED',error_type=type(e).__name__,error=str(e));raise
    finally:write_json(root/'status.json',status)


if __name__=='__main__':main()
