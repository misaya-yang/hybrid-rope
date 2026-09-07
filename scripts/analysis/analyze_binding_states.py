"""Numerical 2x2 replay of already-captured Mr/P2 states and tables.

Measures attention to known source records and block-output changes. No model
forward, generation, optimizer, or new frequency candidate. CUDA accelerates
selected-query algebra (one query by <=4096 keys), never an LxL attention matrix.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch


def masks_for(row):
    text, offsets = row['text'], row['offsets']
    def mask(spans):
        return np.array([any(end > a and begin < b for a,b in spans)
                         for begin,end in offsets], dtype=bool)
    refs = row['references']
    if row['task'] == 'niah_multikey_3':
        uuid = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        records = list(re.finditer(r'One of the special magic uuids for ('+uuid+r') is: ('+uuid+r')\.', text))
        target = [m for m in records if m[2] in refs]
        other = [m for m in records if m[2] not in refs]
        if len(target) != 1 or not other:
            raise ValueError('UUID source-record alignment')
        spans = {'target_value': [m.span(2) for m in target],
                 'target_record': [m.span() for m in target],
                 'other_records': [m.span() for m in other]}
        # The concrete mistaken value from experiment 1; only a descriptive
        # source span, never a training/selection target.
        mistaken = '42161567-d5fb-42a2-a59d-6e8af6546470'
        wrong = [m for m in other if m[2] == mistaken]
        if wrong:
            spans['copied_wrong_value'] = [m.span(2) for m in wrong]
            spans['copied_wrong_record'] = [m.span() for m in wrong]
    else:
        assignments = list(re.finditer(r'VAR\s+([A-Z]+)\s*=\s*(?:VAR\s+[A-Z]+|\d+)', text))
        spans = {f'assignment_{ref}': [m.span() for m in assignments if m[1] == ref] for ref in refs}
        if any(len(s) != 1 for s in spans.values()):
            raise ValueError('VT source-assignment alignment')
        spans['target_record'] = [m.span() for m in assignments if m[1] in refs]
        spans['other_records'] = [m.span() for m in assignments if m[1] not in refs]
    masks = {name: mask(s) for name,s in spans.items()}
    if any(not m.any() for m in masks.values()):
        raise ValueError('empty aligned source span')
    return masks


def replay(cache, projection, frequency, gain, masks, device):
    q, k, v = [cache[n].to(device=device, dtype=torch.float32) for n in ('q','k','v')]
    pos = int(cache['pos'][-1]); q = q[:, -1]
    if k.shape[1] > 4096 or pos != k.shape[1]-1:
        raise ValueError('only last-query short-prefix readback is declared')
    heads, dim = q.shape; pairs = dim//2
    k = k.repeat_interleave(heads//k.shape[0], 0)
    v = v.repeat_interleave(heads//v.shape[0], 0)
    phase = (pos-torch.arange(pos+1, device=device))[:,None]*frequency
    c = q[:,None,:pairs]*k[:,:,:pairs] + q[:,None,pairs:]*k[:,:,pairs:]
    s = q[:,None,:pairs]*k[:,:,pairs:] - q[:,None,pairs:]*k[:,:,:pairs]
    per_slot = (c*phase.cos()+s*phase.sin())*(gain**2/math.sqrt(dim))
    logits = per_slot.sum(-1)
    probability = logits.softmax(-1)
    head_output = torch.bmm(probability[:,None,:], v).squeeze(1)
    output = head_output.reshape(-1) @ projection.T
    attention = {name: probability[:, torch.as_tensor(m, device=device)].sum(-1).cpu().tolist()
                 for name,m in masks.items()}
    target = torch.as_tensor(masks['target_record'], device=device)
    others = torch.as_tensor(masks['other_records'], device=device)
    contrast = per_slot[:,target].mean(1)-per_slot[:,others].mean(1)
    return output.double().cpu().numpy(), attention, contrast.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-plan', type=Path, required=True)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    # Keep numerical reductions independent of TF32's reduced mantissa.
    torch.backends.cuda.matmul.allow_tf32 = False
    parent = json.loads(args.parent_plan.read_text())
    frozen_raw = Path(parent['rows_path']).read_bytes()
    if hashlib.sha256(frozen_raw).hexdigest() != parent['rows_sha256']:
        raise ValueError('parent rows changed')
    frozen_rows = {r['row_id']:r for r in map(json.loads, frozen_raw.splitlines())}
    manifest = json.loads((args.states/'manifest.json').read_text())
    if manifest['status'] != 'COMPLETE_NUMERICAL_STATE_READBACK':
        raise ValueError('state collection incomplete')
    arms = {x['name']: x for x in parent['arms'][1:]}
    frequency = {n: torch.tensor(a['values_float32'], device=args.device) for n,a in arms.items()}
    hashes = {x['file']:x['sha256'] for x in manifest['files']}
    def read_tensor(name):
        path = args.states/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f'capture changed: {name}')
        return torch.load(path, map_location='cpu', weights_only=True)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); vector_rows = {}; records = []
    with (args.out/'rows.jsonl').open('x') as rows, torch.inference_mode():
        for source in sorted(args.states.glob('*_input.json')):
            row = json.loads(source.read_text())
            frozen = frozen_rows[row['row_id']]
            if any(row[k] != frozen[k] for k in ('ids', 'references', 'prompt_sha256', 'task')):
                raise ValueError('captured input differs from frozen parent')
            if source.name in hashes and hashlib.sha256(source.read_bytes()).hexdigest() != hashes[source.name]:
                raise ValueError('captured input metadata changed')
            masks = masks_for(row)
            for layer in range(36):
                projection = read_tensor(f'projection_{layer}.pt').to(args.device, dtype=torch.float32)
                outputs, attention, contrasts, parity = {}, {}, {}, {}
                for state in ('MrPro','P2Middle'):
                    cached = read_tensor(f'{row["row_id"]}_{state}_{layer}.pt')
                    for table in ('MrPro','P2Middle'):
                        key = f'{state}_state__{table}_table'
                        output, masses, contrast = replay(cached, projection, frequency[table], arms[table]['gain'], masks, args.device)
                        outputs[key] = output; attention[key] = masses; contrasts[key] = contrast
                        if state == table:
                            actual = cached['y_actual'][-1].float().numpy()
                            parity[state] = float(np.linalg.norm(output-actual)/max(np.linalg.norm(actual),1e-12))
                            if not math.isfinite(parity[state]) or parity[state] > .05:
                                raise ValueError(f'actual BF16 block parity {row["row_id"]}/{layer}: {parity[state]}')
                mm, mp, pm, pp = [outputs[k] for k in (
                    'MrPro_state__MrPro_table','MrPro_state__P2Middle_table',
                    'P2Middle_state__MrPro_table','P2Middle_state__P2Middle_table')]
                direct = .5*((mp-mm)+(pp-pm))
                state_change = .5*((pm-mm)+(pp-mp))
                total = pp-mm
                norm = float(total@total)
                if not np.allclose(direct+state_change,total,atol=1e-12,rtol=1e-12):
                    raise ValueError('factorial decomposition mismatch')
                rec = {'row_id': row['row_id'], 'layer': layer, 'actual_parity': parity,
                       'attention_mass_by_head': attention,
                       'mean_source_token_logit_contrast_by_slot': {k:v.tolist() for k,v in contrasts.items()},
                       'output_total_energy': norm, 'operator_component_energy': float(direct@direct),
                       'state_component_energy': float(state_change@state_change),
                       'component_cross_term': float(2*direct@state_change),
                       'operator_projection_on_total': float(direct@total/norm) if norm else None,
                       'state_projection_on_total': float(state_change@total/norm) if norm else None}
                rows.write(json.dumps(rec)+'\n'); rows.flush(); records.append(rec)
                for key,val in outputs.items(): vector_rows[f'{row["row_id"]}_{layer}_{key}'] = val
            print(json.dumps({'row_id':row['row_id'],'layers':36,'seconds':time.monotonic()-start}),flush=True)
    np.savez_compressed(args.out/'outputs.npz', **vector_rows)
    result = {'status':'COMPLETE_FIXED_CONDITION_NUMERICAL_REPLAY','experiment_index':1,
        'new_candidates':0,'new_model_forwards':0,'blocks':len(records),
        'seconds':time.monotonic()-start,'device':args.device,
        'max_actual_parity':max(v for r in records for v in r['actual_parity'].values()),
        'rows_sha256':hashlib.sha256((args.out/'rows.jsonl').read_bytes()).hexdigest(),
        'capture_manifest_sha256':hashlib.sha256((args.states/'manifest.json').read_bytes()).hexdigest(),
        'analysis_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limits':'Last prompt query on four exposed short rows. Symmetric block-level decomposition, not causal shares of final answer quality; components may oppose. No frequency optimizer.'}
    (args.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
