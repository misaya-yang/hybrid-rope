"""One resident model, one cached MrPro screen, and a fixed reviewed candidate queue."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

from .bench import FAMILIES, digest, score, summarize, verdict
from .prepare import sha_file, verify_weight_stats


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def eligible_queue(queue):
    entries = queue['ordered_candidates']
    limit=queue.get('max_candidates',10)
    if not isinstance(limit,int) or not 1<=limit<=10:
        raise ValueError('invalid candidate limit')
    if len(entries) > limit or len({e['id'] for e in entries}) != len(entries):
        raise ValueError('candidate queue exceeds limit or has duplicate ids')
    selected = []
    for item in entries:
        if item['eligible']:
            if item['review_status'] != 'REVIEWED_FOR_GPU' or not all(item.get(k) for k in
                ('definition','hypothesis','failure_rule')):
                raise ValueError('candidate lacks a completed theoretical review')
            selected.append(item['id'])
    return selected


def worker(prepared, out, phase_deadline=None, baseline_only=False):
    prepared, out = Path(prepared), Path(out)
    manifest = json.loads((prepared/'manifest.json').read_text())
    for name, expected in manifest['prepared_files'].items():
        if sha_file(prepared/name) != expected:
            raise ValueError('prepared file drift: '+name)
    root = Path(__file__).resolve().parents[3]
    for name, expected in manifest['code_files'].items():
        if sha_file(root/name) != expected:
            raise ValueError('code drift: '+name)
    queue = eligible_queue(json.loads((prepared/'queue.json').read_text()))
    if not queue and not baseline_only:
        raise ValueError('no theoretically reviewed candidate; GPU load refused')
    if baseline_only:
        queue = []
    model_path = Path(manifest['model_path'])
    verify_weight_stats(manifest)
    for name, expected in manifest['model_files_sha256'].items():
        if sha_file(model_path/name) != expected:
            raise ValueError('model metadata/tokenizer drift: '+name)
    for name, expected in manifest['software'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('prepared software version changed: '+name)
    is_ruler = manifest.get('benchmark') == 'ruler_mixed_v1'
    if is_ruler:
        from .ruler_bench import score, summarize, verdict
    else:
        from .bench import score, summarize, verdict
    screen, qualification = rows(prepared/'screen.jsonl'), rows(prepared/'qualification.jsonl')
    tables = json.loads((prepared/'tables.json').read_text())
    reference = manifest.get('reference_arm', 'MrPro')
    if len({digest({k: tables[name].get(k) for k in ('tensor_sha256','gain','gain_by_slot')}) for name in [reference]+queue}) != len(queue)+1:
        raise ValueError('duplicate candidate/baseline tensors')
    out.mkdir(parents=True, exist_ok=True)
    for name in ['NativeQualification',reference]+queue:
        if (out/(name+'.jsonl')).exists() and not (out/(name+'.json')).exists():
            raise ValueError('partial previous arm requires explicit recovery, not automatic rerun: '+name)
    # Candidate additions do not invalidate an otherwise identical MrPro result.
    identity = digest({'model':[manifest['model_id'],manifest['revision'],manifest['weight_sha256']],
                       'screen':manifest['prepared_files']['screen.jsonl'],
                       'qualification':manifest['prepared_files']['qualification.jsonl'],
                       'decoding':manifest['prepared_files']['generation_config.json'],
                       'scorer':manifest['code_files']['scripts/experiments/olmo_fast_screen/'+('ruler_bench.py' if is_ruler else 'bench.py')],
                       'runner':manifest['code_files']['scripts/experiments/olmo_fast_screen/run.py'],
                       'runtime':manifest['code_files'].get('scripts/experiments/olmo_fast_screen/runtime.py'),
                       'software':manifest['software']})

    def progress(stage, expected_seconds):
        now = time.time()
        if (out/'STOP').exists() or (phase_deadline is not None and now >= phase_deadline):
            raise RuntimeError('stop requested or absolute phase deadline reached')
        atomic(out/'live.json', dict(stage=stage, start_unix=now,
               expected_seconds=expected_seconds, phase_deadline=phase_deadline,
               timing_policy='estimate only; never truncate an arm at the estimate'))

    progress('MODEL_LOAD', 180)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from scripts.experiments.cross_audit.tables import tensor_sha
    from .runtime import install, verify
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'],text=True)
    if any(line.strip().isdigit() and int(line.strip()) != os.getpid()
           for line in active.splitlines()):
        raise RuntimeError('GPU already has a compute process; no competing launch')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('requested CUDA/BF16 device is unavailable')
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, device_map={'':'cuda'}, attn_implementation='sdpa').eval()
    model.config.use_cache = True
    if sum(parameter.numel() for parameter in model.parameters()) != manifest['actual_parameters']:
        raise ValueError('loaded model parameter count differs from the prepared weight headers')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    decoding = GenerationConfig.from_dict(json.loads((prepared/'generation_config.json').read_text()))
    atomic(out/'runtime.json', dict(model_load_seconds=time.monotonic()-start,
        gpu=torch.cuda.get_device_name(0), gpu_memory=torch.cuda.get_device_properties(0).total_memory,
        backend='Flash SDPA only; no fallback enabled', software=manifest['software'],
        identity=identity, phase_deadline=phase_deadline))

    def evaluate(name, table_name, data):
        complete = out/(name+'.json')
        raw = out/(name+'.jsonl')
        if complete.exists():
            old = json.loads(complete.read_text())
            if old['identity'] != identity or old['raw_sha256'] != sha_file(raw):
                raise ValueError('cached result identity differs')
            if old['table'] != tables[table_name]:
                raise ValueError('cached deployment table/gain differs')
            if old['row_ids'] != [r['row_id'] for r in data]:
                raise ValueError('cached input rows differ')
            return rows(raw)
        values = np.asarray(tables[table_name]['values_float32'], dtype=np.float32)
        gain = tables[table_name]['gain']
        if tensor_sha(values) != tables[table_name]['tensor_sha256']:
            raise ValueError('table identity differs')
        install(model, tables[table_name])
        verify(model, tables[table_name])
        expected_seconds=60 if name=='NativeQualification' else 300
        progress(name, expected_seconds)
        before = time.monotonic()
        records = []
        with raw.open('x') as stream, torch.inference_mode():
            for row in data:
                if (out/'STOP').exists() or (phase_deadline is not None and time.time() >= phase_deadline):
                    raise RuntimeError('stop requested or phase deadline')
                if digest(row['prompt_ids']) != row['prompt_sha256']:
                    raise ValueError('prompt identity differs')
                ids = torch.tensor([row['prompt_ids']], device='cuda')
                started = time.monotonic()
                generated = model.generate(ids, attention_mask=torch.ones_like(ids),
                    generation_config=decoding, max_new_tokens=row['max_new_tokens'])
                new = generated[0,ids.shape[1]:].tolist()
                eos = decoding.eos_token_id
                ended = bool(new and new[-1] in (eos if isinstance(eos, list) else [eos]))
                text = tokenizer.decode(new[:-1] if ended else new, skip_special_tokens=False)
                record = {k:row[k] for k in ('row_id','group_id','family','length_cap','world','answer','prompt_sha256','input_tokens','task','references','max_new_tokens') if k in row}
                record.update(correct=score(row,text), output_text=text, generated_ids=new,
                              ended_eos=ended, elapsed_seconds=time.monotonic()-started)
                stream.write(json.dumps(record)+'\n'); stream.flush(); records.append(record)
                atomic(out/'live.json',dict(stage=name,start_unix=time.time()-(time.monotonic()-before),
                    completed_rows=len(records),total_rows=len(data),
                    elapsed_seconds=time.monotonic()-before,expected_seconds=expected_seconds,
                    estimate_exceeded=time.monotonic()-before>expected_seconds,
                    phase_deadline=phase_deadline,timing_policy='soft estimate; complete the fixed rows'))
                print(json.dumps({'arm':name,'completed':len(records),'total':len(data),
                                  'correct':record['correct']}), flush=True)
        elapsed = time.monotonic()-before
        verify(model, tables[table_name])
        atomic(complete, dict(status='COMPLETE',identity=identity,table=tables[table_name],
            raw_sha256=sha_file(raw),row_ids=[r['row_id'] for r in data],
            elapsed_seconds=elapsed,expected_seconds=expected_seconds,
            estimate_exceeded=elapsed>expected_seconds,summary=summarize(records)))
        return records

    if qualification:
        native = evaluate('NativeQualification','Native',qualification)
        compact = summarize(native)
        if compact['correct'] < 6 or any(compact['family_accuracy'].get(f,0) < .5 for f in FAMILIES):
            atomic(out/'decision.json', dict(status='ASSAY_UNQUALIFIED',native=compact,
                action='No candidate run. Inspect task/decoder/model ability before revising the benchmark.'))
            return
    baseline = evaluate(reference,reference,screen)
    for name in queue:
        candidate = evaluate(name,name,screen)
        result = verdict(candidate, baseline)
        atomic(out/(name+'_decision.json'),result)
        if result['status'] == 'DEVELOPMENT_WIN' and not manifest.get('complete_candidate_queue', False):
            atomic(out/'decision.json',dict(status='DEVELOPMENT_WIN',candidate=name,
                result=result,action='Stop screening and analyze the winner before defining deeper evaluation.'))
            return
    atomic(out/'decision.json', dict(status='BASELINE_ONLY_COMPLETE' if baseline_only else 'QUEUE_COMPLETE',
                                    reference=reference, candidates=queue))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--phase-deadline',type=float,help='Only an explicitly authorized total budget, never the five-minute estimate')
    p.add_argument('--baseline-only',action='store_true')
    args = p.parse_args()
    worker(args.prepared,args.out,args.phase_deadline,args.baseline_only)


if __name__ == '__main__':
    main()
