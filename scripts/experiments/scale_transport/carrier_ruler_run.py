"""Single frozen signed-carrier table, qualified Flash SDPA, official RULER scoring.

No training, reference model reruns, parameter search, or automatic resume.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from scripts.experiments.cross_audit.runtime import cuda_runtime
from scripts.experiments.scale_transport.carrier import tensor_sha


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def official_postprocess(text):
    # Exact transformation in pinned upstream scripts/eval/evaluate.py.
    return re.sub(r'[\x00-\x1f]', '\n', text.strip()).strip()


def install_signed(model, values, gain):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (64,) or not np.isfinite(values).all() or not np.all(np.diff(values) < 0):
        raise ValueError('finite signed table in original decreasing slot order required')
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError('positive finite gain required')
    rotary = model.model.rotary_emb
    if rotary.rope_type != 'default':
        raise ValueError('static default RoPE source required')
    rotary.inv_freq = torch.tensor(values, device='cuda', dtype=torch.float32)
    rotary.original_inv_freq = rotary.inv_freq.clone()
    rotary.attention_scaling = float(gain)


def qualify_signed(model, values, gain):
    """Compare actual HF cos/sin against independent signed float32 phases."""
    rotary = model.model.rotary_emb
    positions = torch.tensor([[0, 1, 4095, 32767, 131071]], device='cuda')
    hidden = torch.zeros(1, 5, 2048, device='cuda', dtype=torch.float32)
    cos, sin = rotary(hidden, positions)
    phase = positions[0].float()[:, None]*torch.tensor(values, device='cuda')[None, :]
    expected_cos = torch.cat([phase.cos(), phase.cos()], dim=-1)*gain
    expected_sin = torch.cat([phase.sin(), phase.sin()], dim=-1)*gain
    error = max(float((cos[0]-expected_cos).abs().max()), float((sin[0]-expected_sin).abs().max()))
    if error > 2e-6:
        raise ValueError(f'actual signed HF rotary parity failed: {error}')
    with torch.inference_mode():
        logits = model(torch.tensor([[1, 2, 3, 4]], device='cuda'), use_cache=False, logits_to_keep=1).logits
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite signed model smoke logits')
    return {'signed_hf_cos_sin_max_abs': error, 'finite_short_forward': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    started = time.monotonic()
    if time.time() >= plan['absolute_deadline_unix']:
        raise TimeoutError('overnight deadline already elapsed')
    for path, expected in plan['input_files'].items():
        if digest(path) != expected:
            raise ValueError(f'frozen input drift: {path}')
    candidate = json.loads(Path(plan['candidate_path']).read_text())
    table = candidate['tables']['Carrier']
    values, gain = table['values_float32'], table['gain']
    if tensor_sha(values) != table['tensor_sha256']:
        raise ValueError('frozen signed table hash')
    rows = [json.loads(line) for line in Path(plan['rows_path']).read_text().splitlines()]
    if len(rows) != plan['expected_rows'] or len({r['row_id'] for r in rows}) != len(rows):
        raise ValueError('row count or duplicate identity')
    root, out = Path(plan['asset_root']), Path(plan['output'])
    ready = json.loads((root/'model_ready.json').read_text())
    if ready['status'] != 'COMPLETE' or ready['revision'] != candidate['model_revision']:
        raise ValueError('model revision/readiness drift')
    spec = importlib.util.spec_from_file_location('official_ruler_metric', plan['metric_path'])
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    out.mkdir(parents=True, exist_ok=False)
    hardware = cuda_runtime()
    write(out/'progress.json', {'stage': 'loading', 'hardware': hardware})
    model = AutoModelForCausalLM.from_pretrained(root/'model', local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa').eval()
    tok = AutoTokenizer.from_pretrained(root/'model', local_files_only=True)
    if (model.config.num_hidden_layers, model.config.num_attention_heads, model.config.num_key_value_heads) != (36, 16, 2):
        raise ValueError('qualified Qwen2.5-3B GQA architecture required')
    install_signed(model, values, gain)
    qualification = qualify_signed(model, values, gain)
    eos = tok.eos_token_id
    generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=eos, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos)
    write(out/'deployment.json', {'table': table, 'c': candidate['c'], 'model_revision': ready['revision'],
        'plan_sha256': digest(args.plan), 'hardware': hardware, 'qualification': qualification})
    records = []
    with (out/'examples.jsonl').open('x') as output, torch.inference_mode():
        for row in rows:
            if time.time() >= plan['absolute_deadline_unix'] or time.monotonic()-started >= plan['run_budget_seconds']:
                raise TimeoutError('same overnight/run deadline')
            if len(row['ids']) != row['input_tokens'] or not 0 < row['input_tokens'] < row['length_cap']:
                raise ValueError('frozen token count')
            if 'ids_sha256' in row and hashlib.sha256(np.asarray(row['ids'], dtype='<i4').tobytes()).hexdigest() != row['ids_sha256']:
                raise ValueError('token array drift')
            budget = plan.get('generation_budget_override') or row['budget']
            event = {'stage': 'generation', 'row': row['row_id'], 'completed': len(records),
                     'elapsed_seconds': time.monotonic()-started}
            write(out/'progress.json', event)
            print(json.dumps(event), flush=True)
            before = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            ids = torch.tensor(row['ids'], device='cuda')[None, :]
            result = model.generate(ids, generation_config=generation, max_new_tokens=budget, logits_to_keep=1)
            generated = result[0, ids.shape[1]:].tolist()
            text = tok.decode(generated, skip_special_tokens=True)
            prefix = tok.decode(generated[:row['budget']], skip_special_tokens=True)
            rotary = model.model.rotary_emb
            if tensor_sha(rotary.inv_freq.cpu().numpy()) != table['tensor_sha256'] or rotary.attention_scaling != gain:
                raise RuntimeError('static signed deployment drift')
            score = metric.string_match_part if row['task'].startswith('qa_') else metric.string_match_all
            record = {key: row[key] for key in ('row_id', 'task', 'length_cap', 'input_tokens', 'references', 'prompt_sha256')}
            record.update(arm='Carrier', tensor_sha256=table['tensor_sha256'], gain=gain,
                budget=budget, source_budget=row['budget'], generated_ids=generated, output_text=text,
                actual_total_tokens=len(row['ids'])+len(generated), eos=bool(generated and generated[-1] == eos),
                official_score=score([official_postprocess(text)], [row['references']]),
                source_budget_score=score([official_postprocess(prefix)], [row['references']]),
                all_answers=all(ref.lower() in text.lower() for ref in row['references']),
                seconds=time.monotonic()-before, peak_bytes=torch.cuda.max_memory_allocated())
            output.write(json.dumps(record)+'\n')
            output.flush()
            records.append(record)
            del result, ids
    summary = []
    for task, cap in sorted({(r['task'], r['length_cap']) for r in records}):
        selected = [r for r in records if (r['task'], r['length_cap']) == (task, cap)]
        score = metric.string_match_part if task.startswith('qa_') else metric.string_match_all
        summary.append({'task': task, 'length_cap': cap, 'rows': len(selected),
            'official_score': score([official_postprocess(r['output_text']) for r in selected], [r['references'] for r in selected]),
            'source_budget_score': sum(r['source_budget_score'] for r in selected)/len(selected),
            'eos': sum(r['eos'] for r in selected)})
    write(out/'manifest.json', {'status': 'COMPLETE', 'experiment_index': 2, 'scope': plan['scope'],
        'rows': len(records), 'summary': summary, 'elapsed_seconds': time.monotonic()-started,
        'examples_sha256': digest(out/'examples.jsonl'), 'plan_sha256': digest(args.plan)})


if __name__ == '__main__':
    main()
