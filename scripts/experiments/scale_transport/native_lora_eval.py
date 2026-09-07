"""Generated Native retention before/after the one fixed long-context adapter.

The original model, frozen frequency parent and final adapted model see the
same validation rows. Full-string/EOS and text NLL are separate endpoints.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from scripts.experiments.cross_audit.contracts import score
from scripts.experiments.cross_audit.runtime import cuda_runtime
from scripts.experiments.cross_audit.training import causal_loss
from scripts.experiments.scale_transport.carrier import tensor_sha
from scripts.experiments.scale_transport.carrier_ruler_run import (
    digest, install_signed, load_frozen_adapter, write,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    started = time.monotonic()
    if time.time() >= plan['absolute_deadline_unix']:
        raise TimeoutError('same overnight deadline')
    for path, expected in plan['input_files'].items():
        if digest(path) != expected:
            raise ValueError(f'frozen input drift: {path}')
    if plan['arms'] != ['Native', 'Carrier', 'CarrierLoRA'] or not plan['adapter_path']:
        raise ValueError('one original baseline, frequency parent and fixed final adapter')
    rows = [json.loads(line) for line in Path(plan['rows_path']).read_text().splitlines()]
    if len(rows) != 128 or len({r['id'] for r in rows}) != 128 or any(r['split'] != 'validation' for r in rows):
        raise ValueError('exactly the frozen 128 Native validation rows required')
    if {g: sum(r['group'] == g for r in rows) for g in ('instruction', 'reasoning', 'position_format', 'text')} != {
            g: 32 for g in ('instruction', 'reasoning', 'position_format', 'text')}:
        raise ValueError('four equal Native strata required')
    root, out = Path(plan['asset_root']), Path(plan['output'])
    candidate = json.loads(Path(plan['candidate_path']).read_text())
    table = candidate['tables']['Carrier']
    if tensor_sha(table['values_float32']) != table['tensor_sha256']:
        raise ValueError('candidate frequency hash')
    ready = json.loads((root/'model_ready.json').read_text())
    if ready['status'] != 'COMPLETE' or ready['revision'] != candidate['model_revision']:
        raise ValueError('original model revision')
    out.mkdir(parents=True, exist_ok=False)
    hardware = cuda_runtime()
    model = AutoModelForCausalLM.from_pretrained(root/'model', local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa').eval()
    tokenizer = AutoTokenizer.from_pretrained(root/'model', local_files_only=True)
    native = model.model.rotary_emb.inv_freq.float().cpu().numpy().copy()
    if tensor_sha(native) != '138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e':
        raise ValueError('original Qwen3B Native clock')
    generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    write(out/'contract.json', dict(plan=plan, plan_sha256=digest(args.plan), hardware=hardware,
        model_revision=ready['revision'], table=table,
        endpoints='non-text full decoded string after removing only terminal EOS and outer whitespace, with all other special tokens retained; text all-token causal NLL separately',
        limits='Historical development validation pool. Source groups are uncertainty units; no composite score mixing NLL and generated accuracy.'))
    records, adapter = [], None
    with (out/'examples.jsonl').open('x') as output, torch.inference_mode():
        for arm in plan['arms']:
            values, gain = (native, 1.) if arm == 'Native' else (table['values_float32'], table['gain'])
            install_signed(model, values, gain)
            if arm == 'CarrierLoRA':
                model, adapter = load_frozen_adapter(model, plan['adapter_path'], table,
                    ready['revision'], plan['input_files'])
                write(out/'adapter_receipt.json', adapter)
            expected_sha = tensor_sha(values)
            for row in rows:
                if time.time() >= plan['absolute_deadline_unix'] or time.monotonic()-started >= plan['run_budget_seconds']:
                    raise TimeoutError('same frozen run/global deadline')
                before = time.monotonic()
                record = {k: row[k] for k in ('id', 'source_id', 'group', 'split')}
                record.update(arm=arm, table_sha256=expected_sha, gain=gain)
                if row['group'] == 'text':
                    if not 1 < len(row['input_ids']) <= 32768:
                        raise ValueError('physical Native text range')
                    ids = torch.tensor([row['input_ids']], device='cuda')
                    ce, count = causal_loss(model, ids, ids)
                    if not torch.isfinite(ce):
                        raise ValueError('nonfinite text NLL')
                    record.update(nll_sum=float(ce)*count, prediction_tokens=count)
                    del ce, ids
                else:
                    if not row['prompt_ids'] or len(row['prompt_ids'])+row['generation_budget'] > 32768:
                        raise ValueError('full Native generation reserve')
                    ids = torch.tensor([row['prompt_ids']], device='cuda')
                    result = model.generate(ids, generation_config=generation,
                        max_new_tokens=row['generation_budget'], logits_to_keep=1)
                    tokens = result[0, ids.shape[1]:].tolist()
                    ended_eos = bool(tokens and tokens[-1] == tokenizer.eos_token_id)
                    text = tokenizer.decode(tokens[:-1] if ended_eos else tokens, skip_special_tokens=False)
                    scored = score(dict(row, family='native_'+row['group']), text, ended_eos)
                    record.update(accepted_full_answers=row['accepted_full_answers'], generated_ids=tokens,
                        output_text=text, budget=row['generation_budget'],
                        full_answer_exact=scored['full_answer_exact'],
                        full_answer_exact_eos=scored['full_answer_exact_eos'], ended_with_eos=ended_eos)
                    del result, ids
                if (tensor_sha(model.model.rotary_emb.inv_freq.cpu().numpy()) != expected_sha
                        or model.model.rotary_emb.attention_scaling != gain):
                    raise ValueError('evaluation clock drift')
                record['seconds'] = time.monotonic()-before
                output.write(json.dumps(record)+'\n')
                output.flush()
                records.append(record)
                write(out/'progress.json', dict(arm=arm, row=row['id'], completed=len(records),
                    elapsed_seconds=time.monotonic()-started))
    summary = []
    for arm in plan['arms']:
        for group in ('instruction', 'reasoning', 'position_format', 'text'):
            selected = [r for r in records if r['arm'] == arm and r['group'] == group]
            item = dict(arm=arm, group=group, rows=len(selected),
                source_groups=len({r['source_id'] for r in selected}))
            if group == 'text':
                count = sum(r['prediction_tokens'] for r in selected)
                item.update(prediction_tokens=count, nll=sum(r['nll_sum'] for r in selected)/count)
            else:
                item.update(full_answer_exact=sum(r['full_answer_exact'] for r in selected),
                    full_answer_exact_eos=sum(r['full_answer_exact_eos'] for r in selected),
                    ended_with_eos=sum(r['ended_with_eos'] for r in selected))
            summary.append(item)
    write(out/'manifest.json', dict(status='COMPLETE', rows=len(records), summary=summary,
        examples_sha256=digest(out/'examples.jsonl'), plan_sha256=digest(args.plan),
        adapter=adapter, elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
