"""Prospective physical-64K LoRA with dense LM loss and original-Native replay.

One fixed adapter recipe, no frequency optimization or checkpoint selection.
Only a frozen plan may choose smoke or training; neither mode evaluates RULER.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from scripts.experiments.cross_audit.runtime import cuda_runtime
from scripts.experiments.cross_audit.training import causal_loss
from scripts.experiments.scale_transport.carrier import tensor_sha
from scripts.experiments.scale_transport.carrier_ruler_run import digest, install_signed, write
from scripts.lib.rope.generation_contract import stable_teacher_kl


MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
GROUPS = ('instruction', 'reasoning', 'position_format', 'text')


def learning_rate(step, horizon, peak):
    warmup = max(1, math.ceil(.05*horizon))
    if step <= warmup:
        return peak*step/warmup
    fraction = (step-warmup)/max(1, horizon-warmup)
    return peak*(.1+.9*.5*(1+math.cos(math.pi*fraction)))


def replay_order(rows, seed):
    groups = []
    for index, group in enumerate(GROUPS):
        chosen = sorted([r for r in rows if r['group'] == group], key=lambda r: r['id'])
        if len(chosen) != 32:
            raise ValueError('fixed 32 Native training rows in each stratum')
        random.Random(seed+index).shuffle(chosen)
        groups.append(chosen)
    return [groups[g][i] for i in range(32) for g in range(4)]


def original_teacher(model, wrapped, tokenizer, row, native, table, generation):
    """No live student graph may exist while the frequency buffers change."""
    model.eval()
    device = model.device
    try:
        install_signed(model, native, 1.)
        with wrapped.disable_adapter(), torch.no_grad():
            generated = []
            if row['group'] == 'text':
                source_ids = row['input_ids']
                positions = row['prediction_positions']
                prefix_scope = 'observed natural text prefixes'
            else:
                prompt = row['prompt_ids']
                input_tensor = torch.tensor([prompt], device=device)
                result = model.generate(input_tensor, generation_config=generation,
                    max_new_tokens=row['generation_budget'], logits_to_keep=1)
                generated = result[0, len(prompt):].tolist()
                if not generated:
                    raise ValueError('Native teacher returned no token')
                selected = np.unique(np.linspace(0, len(generated)-1, min(32, len(generated)), dtype=int))
                source_ids = prompt+generated[:-1]
                positions = (selected+len(prompt)-1).tolist()
                prefix_scope = 'actual original-Native greedy response prefixes, including terminal decision when emitted'
                del input_tensor, result
            if not 0 < len(source_ids) <= 32768:
                raise ValueError('Native teacher must remain in physical Native range')
            ids = torch.tensor([source_ids], device=device)
            pos = torch.tensor(positions, device=device)
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, pos]
            teacher_logits = F.linear(hidden, model.lm_head.weight).float().detach()
            if not torch.isfinite(teacher_logits).all():
                raise ValueError('nonfinite original teacher logits')
            trace = dict(id=row['id'], group=row['group'], source_id=row['source_id'],
                input_ids=source_ids, positions=positions, generated_ids=generated,
                prefix_scope=prefix_scope,
                eos=bool(generated and generated[-1] in (
                    [generation.eos_token_id] if isinstance(generation.eos_token_id, int)
                    else generation.eos_token_id)))
            return ids, pos, teacher_logits, trace
    finally:
        install_signed(model, table['values_float32'], table['gain'])
        model.train()


def native_teacher_kl(model, ids, positions, teacher_logits):
    if positions.numel() == 0 or positions.min() < 0 or positions.max() >= ids.shape[1]:
        raise ValueError('invalid Native prediction positions')
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, positions]
    student = F.linear(hidden, model.lm_head.weight).float()
    teacher = teacher_logits.to(device=student.device, dtype=torch.float32)
    if teacher.shape != student.shape or not torch.isfinite(teacher).all() or teacher.requires_grad:
        raise ValueError('fixed full-vocabulary original teacher required')
    return stable_teacher_kl(student, teacher), len(positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    started = time.monotonic()
    if time.time() >= plan['absolute_deadline_unix']-120:
        raise TimeoutError('phase deadline leaves no room for a complete update')
    if plan['mode'] not in ('smoke', 'train') or plan['schedule_steps'] != 128:
        raise ValueError('one declared 128-step recipe')
    steps = 2 if plan['mode'] == 'smoke' else 128
    if plan['optimizer_steps'] != steps or plan['rank'] != 16 or plan['alpha'] != 16:
        raise ValueError('no implicit rank or training-budget change')
    if plan['lr'] != 2e-5 or plan['native_kl_weight'] != 1.:
        raise ValueError('fixed prospective optimizer and loss weights')
    for path, expected in plan['input_files'].items():
        if digest(path) != expected:
            raise ValueError(f'frozen training input drift: {path}')
    root, out = Path(plan['asset_root']), Path(plan['output'])
    candidate = json.loads(Path(plan['candidate_path']).read_text())
    table = candidate['tables'][plan.get('table_key', 'Carrier')]
    if tensor_sha(table['values_float32']) != table['tensor_sha256']:
        raise ValueError('frequency identity')
    data = np.load(plan['cpt_path'], mmap_mode='r', allow_pickle=False)
    if data.shape != (128, 65537) or data.dtype != np.int32:
        raise ValueError('128 real 64K contexts plus observed next-token labels required')
    rows = [json.loads(s) for s in Path(plan['native_train_path']).read_text().splitlines()]
    if len(rows) != 128 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Native training split only')
    native_rows = replay_order(rows, plan['seed'])
    book_order = list(range(128))
    random.Random(plan['seed']).shuffle(book_order)
    out.mkdir(parents=True, exist_ok=False)
    (out/'teacher').mkdir()
    hardware = cuda_runtime()
    model = AutoModelForCausalLM.from_pretrained(root/'model', local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa')
    tokenizer = AutoTokenizer.from_pretrained(root/'model', local_files_only=True)
    native = model.model.rotary_emb.inv_freq.float().cpu().numpy().copy()
    if tensor_sha(native) != '138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e':
        raise ValueError('same original Native checkpoint clock required')
    ready = json.loads((root/'model_ready.json').read_text())
    if ready['status'] != 'COMPLETE' or ready['revision'] != candidate['model_revision']:
        raise ValueError('model revision/readiness drift')
    install_signed(model, table['values_float32'], table['gain'])
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(plan['seed'])
    wrapped = get_peft_model(model, LoraConfig(r=16, lora_alpha=16,
        target_modules=list(MODULES), lora_dropout=0., bias='none', task_type='CAUSAL_LM'))
    model = wrapped.get_base_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.train()
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named or any('lora_' not in name or not any(f'.{m}.' in name for m in MODULES) for name, _ in named):
        raise ValueError('trainable scope must remain seven linear LoRA modules')
    parameters = [p for _, p in named]
    optimizer = torch.optim.AdamW(parameters, lr=plan['lr'], betas=(.9, .95), weight_decay=0., fused=True)
    generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    write(out/'contract.json', dict(plan=plan, plan_sha256=digest(args.plan), hardware=hardware,
        model_revision=ready['revision'], table=table, native_tensor_sha256=tensor_sha(native),
        trainable_parameters=sum(p.numel() for p in parameters), trainable_names=[name for name, _ in named],
        book_order=book_order, native_order=[r['id'] for r in native_rows],
        teacher=f"original {ready['model_id']} weights with adapters disabled and Native clock/gain1; actual greedy prefixes for non-text rows",
        loss='all-64K-token causal CE mean + vocabulary-summed, position-mean Native forward KL, coefficient1',
        teacher_eos_token_ids=generation.eos_token_id,
        native_kl_gradient='existing stable_teacher_kl analytic first-order gradient; exactly zero at identical logits',
        checkpoint_policy='fixed final update only; smoke discards its adapter; no generation-based checkpoint selection'))
    records = []
    gradient_totals = {m: 0. for m in MODULES}
    with (out/'steps.jsonl').open('x') as log:
        for step in range(1, steps+1):
            if time.time() >= plan['absolute_deadline_unix']-120 or time.monotonic()-started >= plan['run_budget_seconds']-120:
                raise TimeoutError('remaining frozen run/global budget insufficient for another full update')
            before = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            teacher_start = time.monotonic()
            replay_ids, replay_pos, teacher_logits, trace = original_teacher(model, wrapped, tokenizer,
                native_rows[step-1], native, table, generation)
            teacher_path = out/'teacher'/f'{step:03d}.npz'
            np.savez(teacher_path, logits=teacher_logits.cpu().numpy(),
                input_ids=np.asarray(trace['input_ids'], dtype=np.int32), positions=np.asarray(trace['positions'], dtype=np.int32))
            trace.update(teacher_logits_file=teacher_path.name, teacher_logits_file_sha256=digest(teacher_path),
                probability_definition='FP32 softmax of saved full-vocabulary teacher logits')
            write(out/'teacher'/f'{step:03d}.json', trace)
            teacher_seconds = time.monotonic()-teacher_start
            ids = torch.tensor(np.asarray(data[book_order[step-1]], dtype=np.int64)[None, :], device='cuda')
            lr = learning_rate(step, 128, plan['lr'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ce, ce_count = causal_loss(model, ids, ids)
            if not torch.isfinite(ce):
                raise ValueError('nonfinite physical-64K language-model loss')
            ce.backward()
            ce_value = float(ce.detach())
            del ce, ids
            with torch.autocast('cuda', dtype=torch.bfloat16):
                kl, kl_count = native_teacher_kl(model, replay_ids, replay_pos, teacher_logits)
            if not torch.isfinite(kl):
                raise ValueError('nonfinite Native KL')
            kl.backward()
            kl_value = float(kl.detach())
            del kl, replay_ids, replay_pos, teacher_logits
            if plan['mode'] == 'smoke':
                for name, parameter in named:
                    if parameter.grad is not None:
                        scope = next(m for m in MODULES if f'.{m}.' in name)
                        gradient_totals[scope] += float(parameter.grad.float().square().sum())
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError('finite nonzero adapter gradient required')
            optimizer.step()
            rotary = model.model.rotary_emb
            if tensor_sha(rotary.inv_freq.float().cpu().numpy()) != table['tensor_sha256'] or rotary.attention_scaling != table['gain']:
                raise ValueError('student clock drift after Native teacher or update')
            torch.cuda.synchronize()
            record = dict(step=step, book_row=book_order[step-1], native_id=native_rows[step-1]['id'],
                native_group=native_rows[step-1]['group'], learning_rate=lr, cpt_ce=ce_value,
                native_kl=kl_value, cpt_prediction_tokens=ce_count, native_prediction_positions=kl_count,
                teacher_seconds=teacher_seconds, total_seconds=time.monotonic()-before,
                grad_norm=norm, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            log.write(json.dumps(record)+'\n')
            log.flush()
            records.append(record)
            print(json.dumps(record), flush=True)
    if plan['mode'] == 'smoke' and any(v <= 0 or not math.isfinite(v) for v in gradient_totals.values()):
        raise ValueError('attention or FFN LoRA scope has no finite gradient')
    adapter_files = {}
    if plan['mode'] == 'train':
        wrapped.save_pretrained(out/'adapter', safe_serialization=True)
        write(out/'adapter'/'deployment.json', dict(table=table, model_revision=ready['revision'],
            source_candidate_sha256=digest(plan['candidate_path']), optimizer_steps=steps))
        adapter_files = {p.name: digest(p) for p in (out/'adapter').iterdir() if p.is_file()}
    write(out/'manifest.json', dict(status='COMPLETE_64K_NUMERICAL_SMOKE' if plan['mode'] == 'smoke' else 'COMPLETE_FIXED_LORA_TRAINING_NOT_CAPABILITY',
        optimizer_steps=steps, physical_input_length=65536, cpt_prediction_tokens=sum(r['cpt_prediction_tokens'] for r in records),
        elapsed_seconds=time.monotonic()-started, gradient_scope_energy=gradient_totals if plan['mode'] == 'smoke' else None,
        steps_sha256=digest(out/'steps.jsonl'), adapter_files=adapter_files, table_sha256=table['tensor_sha256'],
        limits='Training/smoke execution only. Held-out generated long and Native endpoints are required; no RULER outcome is read here.'))


if __name__ == '__main__':
    main()
