"""Compact real-prefix calibration: full-row denominators, selected raw K/V.

Candidate replay changes selected keys and holds the omitted contribution fixed.
It is explicitly a proposal filter, never a whole-model efficacy measurement.
"""
from __future__ import annotations

import hashlib
from bisect import bisect_right
import json
from pathlib import Path
import re
import time
import types

import numpy as np
import torch

from .worker import digest, save, sha


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


def reference_positions(tokenizer, ids, references):
    text = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != ids:
        raise ValueError('cannot map reference spans without exact token roundtrip')
    spans = []
    for ref in references:
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(ref), text, flags=re.IGNORECASE))
    merged=[]
    for start,end in sorted(spans):
        if merged and start<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(merged[-1][1],end))
        else:merged.append((start,end))
    ends=[end for start,end in merged]
    positions=[]
    for i,(left,right) in enumerate(encoded['offset_mapping']):
        if right<=left:continue
        candidate=bisect_right(ends,left)
        if candidate<len(merged) and merged[candidate][0]<right:positions.append(i)
    return positions


def capture_row(worker, folder, row_id, ids, query_positions, targets, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return json.loads((folder / 'complete.json').read_text())
    originals = []
    receipts = []
    target_cpu = torch.tensor(targets, dtype=torch.long)
    qpos = torch.tensor(query_positions, device='cuda')
    total = len(ids)

    def make_forward(original, layer_index):
        def forward(module, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            batch, length, _ = hidden_states.shape
            if batch != 1 or length != total or past_key_values is not None:
                raise ValueError('calibration expects one complete uncached prefix')
            q = module.q_proj(hidden_states[:, qpos]).view(1, len(query_positions), 16, 128).transpose(1, 2)
            k = module.k_proj(hidden_states).view(1, length, 2, 128).transpose(1, 2)
            v = module.v_proj(hidden_states).view(1, length, 2, 128).transpose(1, 2)
            cos, sin = position_embeddings
            qr = rotate(q, cos[:, qpos].unsqueeze(1), sin[:, qpos].unsqueeze(1))
            kr = rotate(k, cos.unsqueeze(1), sin.unsqueeze(1))
            scores = torch.einsum('ghqd,gkd->ghqk', qr[0].reshape(2, 8, -1, 128).float(), kr[0].float()) * module.scaling
            key_pos = torch.arange(length, device='cuda')
            valid = key_pos[None, :] <= qpos[:, None]
            scores.masked_fill_(~valid[None, None], -torch.inf)
            lse = scores.logsumexp(-1)
            probs = (scores - lse[..., None]).exp()
            output = torch.einsum('ghqk,gkd->ghqd', probs, v[0].float()).reshape(16, -1, 128)
            top = scores.topk(min(8, length), dim=-1).indices.reshape(-1)
            recent = torch.cat([torch.arange(max(0, p-32), p+1, device='cuda') for p in query_positions])
            # Include all reference positions up to 128, deterministically spaced
            # if FWE/QA contains many occurrences. Full target mass is still saved.
            target_positions = target_cpu.to('cuda')
            chosen_targets = target_positions
            if len(chosen_targets) > 128:
                chosen_targets = chosen_targets[torch.linspace(0, len(chosen_targets)-1, 128, device='cuda').long()]
            support = torch.linspace(0, length-1, 64, device='cuda').long()
            selected = torch.unique(torch.cat([top, recent, chosen_targets, support]), sorted=True)
            target_mask = torch.zeros(length, dtype=torch.bool, device='cuda')
            target_mask[target_positions] = True
            target_mass = probs[..., target_mask].sum(-1).reshape(16, -1)
            payload = dict(row_id=row_id, layer=layer_index, query_positions=qpos.cpu(), key_positions=selected.cpu(),
                q_raw=q[0].cpu(), k_raw=k[0, :, selected].cpu(), v_raw=v[0, :, selected].cpu(),
                selected_baseline_logits=scores[..., selected].reshape(16, len(query_positions), -1).cpu(),
                baseline_lse=lse.reshape(16, -1).cpu(), baseline_output=output.cpu(),
                full_target_mass=target_mass.cpu(), selected_target=target_mask[selected].cpu(),
                valid=valid[:, selected].cpu(), total_keys=length, metadata=metadata)
            path = folder / f'layer_{layer_index:02d}.pt'
            torch.save(payload, path)
            receipts.append(dict(layer=layer_index, selected_keys=len(selected), queries=len(qpos),
                full_target_tokens=len(target_positions), mean_target_mass=target_mass.mean().item()))
            del q, k, v, qr, kr, scores, probs, output, payload
            return original(hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values, **kwargs)
        return forward

    for i, layer in enumerate(worker.model.model.layers):
        attn = layer.self_attn
        originals.append((attn, attn.forward))
        attn.forward = types.MethodType(make_forward(attn.forward, i), attn)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            tokens = torch.tensor([ids], device='cuda')
            logits = worker.model(tokens, use_cache=False, logits_to_keep=1).logits
            last = logits[0, -1].float()
            top = last.topk(5)
        record = dict(status='COMPLETE', input_sha256=digest(ids), query_positions=query_positions,
            target_positions=targets, metadata=metadata, layer_receipts=receipts,
            last_top_ids=top.indices.tolist(), last_top_logits=top.values.tolist(),
            elapsed_seconds=time.monotonic()-started,
            scope='Real complete-prefix states; selected-key finite replay is conditional and approximate')
        save(folder / 'complete.json', record)
        return record
    finally:
        for attn, forward in originals:
            attn.forward = forward


def run(worker, job):
    root = worker.root / 'calibration'
    root.mkdir(exist_ok=True)
    rows = [r for r in worker.screen if r['row_id'].endswith(('_0', '_1'))]
    results = []
    for row in rows:
        prompt = row['prompt_ids']
        # Explicit correct-answer trajectory on calibration only. Four predictor
        # rows include the first answer token, not just a formatting-only query.
        answer = ', '.join(row['references'])
        ans_ids = worker.tokenizer.encode(' ' + answer, add_special_tokens=False)[:4]
        ids = prompt + ans_ids[:-1]
        qpositions = list(range(len(prompt)-1, len(ids)))
        targets = reference_positions(worker.tokenizer, prompt, row['references'])
        meta = dict(task=row['task'], length=row['length_cap'], split=int(row['row_id'].rsplit('_',1)[1]),
            references=row['references'], original_prompt_sha256=row['prompt_sha256'], answer_prefix_ids=ans_ids,
            reference_note='Literal reference occurrences in original prompt only; empty for absent QA aliases')
        result = capture_row(worker, root / row['row_id'], row['row_id'], ids, qpositions, targets, meta)
        results.append(dict(row=row['row_id'], elapsed_seconds=result['elapsed_seconds']))
        save(worker.root / 'live.json', dict(job=job['id'], phase='calibration', row=row['row_id'], completed=len(results), requested=len(rows)+4))
        print(json.dumps(dict(calibration=row['row_id'], seconds=result['elapsed_seconds'])), flush=True)
    for index, doc in enumerate(worker.nll_manifest['docs'][:4]):
        length = 32768
        ids = np.load(worker.nll_inputs / doc['file'])[:length].tolist()
        row_id = f'natural_{index:02d}_{length}'
        qpositions = [length-4, length-3, length-2, length-1]
        meta = dict(task='natural', length=length, split=index % 2, source=doc)
        result = capture_row(worker, root / row_id, row_id, ids, qpositions, [], meta)
        results.append(dict(row=row_id, elapsed_seconds=result['elapsed_seconds']))
        save(worker.root / 'live.json', dict(job=job['id'], phase='calibration', row=row_id, completed=len(results), requested=len(rows)+4))
    receipt = dict(status='COMPLETE', rows=results, table=worker.tables['MrPro'], source_sha256=sha(__file__),
        selection='RULER fixed IDs 0/1 for both lengths plus first four natural documents',
        usage='Proposal selection only; these historical rows are development, not independent validation')
    save(root / 'manifest.json', receipt)
    return receipt
