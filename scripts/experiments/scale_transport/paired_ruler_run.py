"""One frozen P2-middle versus MrPro comparison on existing RULER rows."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from scripts.experiments.cross_audit.runtime import cuda_runtime
from scripts.experiments.cross_audit.tables import tensor_sha
from scripts.experiments.scale_transport.run import install, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    start = time.monotonic()
    if time.time() >= plan['absolute_deadline_unix']:
        raise TimeoutError('phase deadline already elapsed')
    root, out = Path(plan['asset_root']), Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    raw = Path(plan['rows_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan['rows_sha256']:
        raise ValueError('frozen rows changed')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 20 or {r['task'] for r in rows} != {'niah_multikey_3', 'vt'}:
        raise ValueError('fixed two-task comparison requires exactly 20 source rows')
    arms = plan['arms']
    if [a['name'] for a in arms] != ['NativeShort', 'MrPro', 'P2Middle']:
        raise ValueError('one candidate and fixed references only')
    for arm in arms:
        if tensor_sha(arm['values_float32']) != arm['tensor_sha256']:
            raise ValueError('array hash mismatch')
    if arms[1]['gain'] != arms[2]['gain']:
        raise ValueError('frequency comparison must be gain matched')
    mr, candidate = [np.asarray(a['values_float32'], dtype=np.float32) for a in arms[1:]]
    if not np.array_equal(mr[:24], candidate[:24]) or not np.array_equal(mr[40:], candidate[40:]):
        raise ValueError('intervention changed outside slots 24--39')
    ready = json.loads((root/'model_ready.json').read_text())
    if ready['status'] != 'COMPLETE' or ready['revision'] != plan['model_revision']:
        raise ValueError('model revision/readiness mismatch')
    metric_path = Path(plan['metric_path'])
    if hashlib.sha256(metric_path.read_bytes()).hexdigest() != plan['metric_sha256']:
        raise ValueError('upstream scoring changed')
    spec = importlib.util.spec_from_file_location('ruler_metric', metric_path)
    metric = importlib.util.module_from_spec(spec); spec.loader.exec_module(metric)
    hardware = cuda_runtime()
    write(out/'progress.json', {'stage': 'loading', 'hardware': hardware})
    model = AutoModelForCausalLM.from_pretrained(root/'model', local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa').eval()
    tok = AutoTokenizer.from_pretrained(root/'model', local_files_only=True)
    eos = tok.eos_token_id
    generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=eos, pad_token_id=tok.pad_token_id or eos)
    write(out/'deployment.json', {'arms': arms, 'model_revision': ready['revision'],
        'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(), 'hardware': hardware})
    records = []
    with (out/'examples.jsonl').open('x') as output, torch.inference_mode():
        for row in rows:
            for arm in arms:
                if arm['name'] == 'NativeShort' and row['length_cap'] != 4096:
                    continue
                if time.time() >= plan['absolute_deadline_unix'] or time.monotonic()-start >= plan['run_budget_seconds']:
                    raise TimeoutError('same cumulative phase/run budget')
                # Same frozen prompt in both arms. The larger VT answer reserve
                # may extend total length up to 98 tokens beyond its old bucket.
                budget = 128
                if len(row['ids']) != row['input_tokens'] or not .95*row['length_cap'] <= len(row['ids']) < row['length_cap']:
                    raise ValueError('source prompt length drift')
                install(model, arm['values_float32'], arm['gain'])
                event = {'stage': 'generation', 'row': row['row_id'], 'arm': arm['name'],
                         'completed': len(records), 'elapsed_seconds': time.monotonic()-start}
                write(out/'progress.json', event); print(json.dumps(event), flush=True)
                before = time.monotonic(); torch.cuda.reset_peak_memory_stats()
                ids = torch.tensor(row['ids'], device='cuda')[None, :]
                result = model.generate(ids, generation_config=generation, max_new_tokens=budget, logits_to_keep=1)
                new = result[0, ids.shape[1]:].tolist()
                text = tok.decode(new, skip_special_tokens=True)
                prefix = tok.decode(new[:row['budget']], skip_special_tokens=True)
                if tensor_sha(model.model.rotary_emb.inv_freq.cpu().numpy()) != arm['tensor_sha256'] or model.model.rotary_emb.attention_scaling != arm['gain']:
                    raise RuntimeError('static deployment drift')
                record = {k: row[k] for k in ('row_id', 'task', 'length_cap', 'input_tokens', 'references', 'prompt_sha256')}
                record.update(arm=arm['name'], tensor_sha256=arm['tensor_sha256'], gain=arm['gain'],
                    source_budget=row['budget'], budget=budget, generated_ids=new, output_text=text,
                    actual_total_tokens=len(row['ids'])+len(new), eos=bool(new and new[-1] == eos),
                    official_score=metric.string_match_all([text], [row['references']]),
                    source_budget_score=metric.string_match_all([prefix], [row['references']]),
                    all_answers=all(ref.lower() in text.lower() for ref in row['references']),
                    seconds=time.monotonic()-before, peak_bytes=torch.cuda.max_memory_allocated())
                output.write(json.dumps(record)+'\n'); output.flush(); records.append(record)
    summary = []
    for arm in arms:
        for task in ('niah_multikey_3', 'vt'):
            for cap in (4096, 131072):
                subset = [r for r in records if (r['arm'],r['task'],r['length_cap']) == (arm['name'],task,cap)]
                if subset:
                    summary.append({'arm': arm['name'], 'task': task, 'length_cap': cap, 'rows': len(subset),
                        'official_score': metric.string_match_all([r['output_text'] for r in subset], [r['references'] for r in subset]),
                        'source_budget_score': sum(r['source_budget_score'] for r in subset)/len(subset),
                        'all_answers': sum(r['all_answers'] for r in subset), 'eos': sum(r['eos'] for r in subset)})
    write(out/'manifest.json', {'status': 'COMPLETE_DEVELOPMENT_COMPARISON', 'experiment_index': 1,
        'scope': 'One candidate; existing two-task development rows, not independent confirmation or SOTA.',
        'rows': len(records), 'summary': summary, 'elapsed_seconds': time.monotonic()-start,
        'examples_sha256': hashlib.sha256((out/'examples.jsonl').read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()
