"""Capture the two already-tested tables on the four existing short prompts.

Numerical readback of experiment 1, no generation, new candidate or tuning.
Stores unrotated QKV at all layers so direct table effects and state drift can
be compared on CPU without further model forwards.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.experiments.cross_audit.runtime import cuda_runtime
from scripts.experiments.cross_audit.tables import tensor_sha
from scripts.experiments.scale_transport.run import install, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=int, required=True)
    args = parser.parse_args()
    if not 0 < args.budget_seconds <= 180:
        raise ValueError('bounded short-input readback only')
    parent = json.loads(args.parent_plan.read_text())
    result_path = Path(parent['output'])/'manifest.json'
    completed = json.loads(result_path.read_text())
    if completed['status'] != 'COMPLETE_DEVELOPMENT_COMPARISON':
        raise ValueError('wait for the fixed comparison to complete')
    raw = Path(parent['rows_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != parent['rows_sha256']:
        raise ValueError('parent data drift')
    rows = [r for r in map(json.loads, raw.splitlines()) if r['length_cap'] == 4096]
    if len(rows) != 4:
        raise ValueError('only the four existing short controls')
    root = Path(parent['asset_root']); args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); hardware = cuda_runtime()
    model = AutoModelForCausalLM.from_pretrained(root/'model', local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa').eval()
    tok = AutoTokenizer.from_pretrained(root/'model', local_files_only=True)
    config = model.config
    if (config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads,
        config.hidden_size) != (36, 16, 2, 2048):
        raise ValueError('unexpected Qwen3B layout')
    active = {}; entries = []; handles = []
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for layer in range(36):
        module = model.model.layers[layer].self_attn
        projection = args.out/f'projection_{layer}.pt'
        torch.save(module.o_proj.weight.detach().cpu(), projection)
        entries.append({'file': projection.name, 'sha256': digest(projection)})
        def hook(module, values, kwargs, output, layer=layer):
            hidden = kwargs.get('hidden_states', values[0] if values else None)
            length = hidden.shape[1]
            pos = torch.arange(length-57, length, 8, device=hidden.device)
            q = module.q_proj(hidden[:, pos]).reshape(8, 16, 128).transpose(0, 1)
            k = module.k_proj(hidden).reshape(length, 2, 128).transpose(0, 1)
            v = module.v_proj(hidden).reshape(length, 2, 128).transpose(0, 1)
            cache = {'q': q.detach().cpu(), 'k': k.detach().cpu(), 'v': v.detach().cpu(),
                     'pos': pos.cpu(), 'y_actual': output[0][0, pos].detach().cpu()}
            path = args.out/f'{active["row_id"]}_{active["arm"]}_{layer}.pt'
            torch.save(cache, path)
            entries.append({'file': path.name, 'sha256': digest(path)})
        handles.append(module.register_forward_hook(hook, with_kwargs=True))
    events = []
    with torch.inference_mode():
        for row in rows:
            text = tok.decode(row['ids'], skip_special_tokens=False)
            encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            if encoded['input_ids'] != row['ids']:
                raise ValueError('cannot align source spans to original token IDs')
            input_path = args.out/f'{row["row_id"]}_input.json'
            write(input_path, {
                **{k:v for k,v in row.items() if k != 'ids'}, 'text': text,
                'ids': row['ids'], 'offsets': encoded['offset_mapping']})
            entries.append({'file': input_path.name, 'sha256': digest(input_path)})
            for arm in parent['arms'][1:]:
                if time.monotonic()-start >= args.budget_seconds or time.time() >= parent['absolute_deadline_unix']:
                    raise TimeoutError('readback budget')
                active.update(row_id=row['row_id'], arm=arm['name'])
                install(model, arm['values_float32'], arm['gain'])
                before = time.monotonic()
                result = model(torch.tensor(row['ids'], device='cuda')[None, :],
                               use_cache=False, logits_to_keep=1)
                if not torch.isfinite(result.logits).all():
                    raise ValueError('nonfinite logits')
                if tensor_sha(model.model.rotary_emb.inv_freq.cpu().numpy()) != arm['tensor_sha256']:
                    raise ValueError('table identity drift')
                scores, ids = result.logits[0, -1].float().topk(16)
                event = {**active, 'seconds': time.monotonic()-before,
                         'top_next_ids': ids.tolist(), 'top_next_logits': scores.tolist()}
                events.append(event); print(json.dumps(event), flush=True)
    for handle in handles:
        handle.remove()
    write(args.out/'manifest.json', {'status': 'COMPLETE_NUMERICAL_STATE_READBACK',
        'experiment_index': 1, 'new_candidates': 0, 'generated_answers': 0,
        'parent_plan_sha256': digest(args.parent_plan), 'parent_result_sha256': digest(result_path),
        'hardware': hardware, 'events': events, 'files': entries,
        'elapsed_seconds': time.monotonic()-start,
        'limits': 'Four previously exposed short prompts; numerical attribution inputs, not a new capability endpoint.'})


if __name__ == '__main__':
    main()
