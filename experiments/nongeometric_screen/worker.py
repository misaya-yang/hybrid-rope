"""One resident Qwen model; explicit jobs, paired historical scores, resumable rows."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def install_table(model, table):
    values = torch.tensor(table['values_float32'], dtype=torch.float32, device='cuda')
    if values.shape != (64,) or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError('expected 64 finite nonnegative frequencies in original slot order')
    gain = float(table['gain'])
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError('invalid gain')
    rotary = model.model.rotary_emb
    if rotary.rope_type != 'default':
        raise ValueError('cannot silently stack dynamic scaling')
    rotary.inv_freq = values
    rotary.original_inv_freq = values.clone()
    rotary.attention_scaling = gain


class Worker:
    def __init__(self, root, history):
        self.root, self.history = Path(root), Path(history)
        self.prepared = self.history / 'prepared_qwen3_01'
        self.manifest = json.loads((self.prepared / 'manifest.json').read_text())
        self.tables = json.loads((self.prepared / 'tables.json').read_text())
        self.screen = read_rows(self.prepared / 'screen.jsonl')
        self.baseline = {r['row_id']: r for r in read_rows(self.history / 'run_qwen3_01/MrPro.jsonl')}
        self.nll_inputs = self.history / 'prepared_nll_01'
        self.nll_manifest = json.loads((self.nll_inputs / 'manifest.json').read_text())
        self.nll_base = {(r['doc'], r['length']): r for r in read_rows(self.history / 'run_nll_01/MrPro.jsonl')}
        for name in ('screen.jsonl', 'generation_config.json', 'tables.json'):
            if sha(self.prepared / name) != self.manifest['prepared_files'][name]:
                raise ValueError('historical input/config drift: ' + name)
        if len(self.baseline) != 36 or len(self.nll_base) != 48:
            raise ValueError('historical baseline incomplete')
        for r in self.screen:
            if digest(r['prompt_ids']) != r['prompt_sha256'] or self.baseline[r['row_id']]['prompt_sha256'] != r['prompt_sha256']:
                raise ValueError('historical token input mismatch')
        for d in self.nll_manifest['docs']:
            if sha(self.nll_inputs / d['file']) != d['sha256']:
                raise ValueError('historical natural input mismatch')
        for name, expected in self.manifest['model_files_sha256'].items():
            if sha(Path(self.manifest['model_path']) / name) != expected:
                raise ValueError('model metadata drift: ' + name)
        for name, info in self.manifest['weight_stats'].items():
            p = Path(self.manifest['model_path']) / name
            stat = p.stat()
            if stat.st_size != info['size'] or stat.st_mtime_ns != info['mtime_ns']:
                if sha(p) != self.manifest['weight_files_sha256'][name]:
                    raise ValueError('model weights differ')
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.model = AutoModelForCausalLM.from_pretrained(self.manifest['model_path'], local_files_only=True,
            dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa').eval()
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.manifest['model_path'], local_files_only=True)
        self.decoding = GenerationConfig.from_dict(json.loads((self.prepared / 'generation_config.json').read_text()))
        if sum(p.numel() for p in self.model.parameters()) != self.manifest['actual_parameters']:
            raise ValueError('model parameter count mismatch')
        self.handles = []
        self.restore = None
        save(self.root / 'runtime.json', dict(model=self.manifest['model_id'], revision=self.manifest['revision'],
            torch=torch.__version__, transformers=importlib.import_module('transformers').__version__,
            gpu=torch.cuda.get_device_name(), prepared_sha256=sha(self.prepared / 'manifest.json'),
            nll_input_sha256=sha(self.nll_inputs / 'manifest.json'), baseline_ruler_sha256=sha(self.history / 'run_qwen3_01/MrPro.jsonl'),
            baseline_nll_sha256=sha(self.history / 'run_nll_01/MrPro.jsonl'), worker_sha256=sha(__file__),
            baseline_policy='Reuse existing paired rows; one path check only; no model weight updates'))

    def apply(self, spec):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.restore:
            self.restore()
            self.restore = None
        table = spec.get('table', self.tables['MrPro'])
        install_table(self.model, table)
        if spec.get('operator', 'static') != 'static':
            mod = importlib.import_module('experiments.nongeometric_screen.operators')
            self.restore = mod.install(self.model, spec, self.tables)

    def generate(self, row):
        from scripts.experiments.olmo_fast_screen.ruler_bench import score
        ids = torch.tensor([row['prompt_ids']], device='cuda')
        started = time.monotonic()
        with torch.inference_mode():
            generated = self.model.generate(ids, attention_mask=torch.ones_like(ids),
                generation_config=self.decoding, max_new_tokens=row['max_new_tokens'])
        new = generated[0, ids.shape[1]:].tolist()
        eos = self.decoding.eos_token_id
        ended = bool(new and new[-1] in (eos if isinstance(eos, list) else [eos]))
        text = self.tokenizer.decode(new[:-1] if ended else new, skip_special_tokens=False)
        record = {k: row[k] for k in ('row_id', 'family', 'length_cap', 'prompt_sha256', 'input_tokens', 'task', 'references', 'max_new_tokens')}
        record.update(correct=score(row, text), output_text=text, generated_ids=new, ended_eos=ended,
            elapsed_seconds=time.monotonic() - started)
        return record

    def qualification(self):
        path = self.root / 'qualification.json'
        if path.exists():
            return
        self.apply({'table': self.tables['MrPro']})
        row = self.screen[0]
        got = self.generate(row)
        old = self.baseline[row['row_id']]
        exact = got['generated_ids'] == old['generated_ids']
        save(path, dict(status='PASS' if exact else 'FAIL', row=got,
            original_generated_ids=old['generated_ids'], generated_ids_match=exact))
        if not exact:
            raise ValueError('new unmodified path failed historical greedy reproduction')

    def evaluate(self, job):
        from scripts.experiments.olmo_fast_screen.ruler_bench import summarize
        folder = self.root / 'results' / job['id']
        folder.mkdir(parents=True, exist_ok=True)
        spec = job['spec']
        contract = dict(spec=spec, decoding_sha256=sha(self.prepared / 'generation_config.json'))
        if (folder / 'contract.json').exists() and json.loads((folder / 'contract.json').read_text()) != contract:
            raise ValueError('cannot resume under different method/config')
        save(folder / 'contract.json', contract)
        self.apply(spec)
        for length in job.get('nll_lengths', [8192, 32768]):
            docs = self.nll_manifest['docs'][:job.get('nll_docs', 4)]
            existing = {(r['doc'], r['length']): r for r in read_rows(folder / 'nll.jsonl')}
            for doc in docs:
                if (doc['file'], length) in existing:
                    continue
                data = np.load(self.nll_inputs / doc['file'])
                ids = torch.tensor(data[:length].astype(np.int64), device='cuda')[None]
                target = torch.tensor(data[length-511:length+1].astype(np.int64), device='cuda')
                with torch.inference_mode():
                    logits = self.model(ids, use_cache=False, logits_to_keep=512).logits[0].float()
                    losses = torch.nn.functional.cross_entropy(logits, target, reduction='none')
                record = dict(doc=doc['file'], length=length, nll=losses.mean().item(),
                    token_nll=losses.tolist(), target_ids=target.tolist(), input_sha256=doc['sha256'],
                    baseline_nll=self.nll_base[(doc['file'], length)]['nll'])
                with (folder / 'nll.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                save(self.root / 'live.json', dict(job=job['id'], phase='nll', length=length, doc=doc['file']))
                del logits, losses, ids, target
        existing = {r['row_id']: r for r in read_rows(folder / 'ruler.jsonl')}
        data = [r for r in self.screen if job.get('panel', 'small') == 'full' or r['row_id'].endswith('_0')]
        if job.get('row_ids'):
            data = [r for r in self.screen if r['row_id'] in job['row_ids']]
        for row in data:
            if (self.root / 'STOP').exists():
                raise RuntimeError('user stop requested')
            if row['row_id'] in existing:
                continue
            record = self.generate(row)
            record['baseline_correct'] = self.baseline[row['row_id']]['correct']
            with (folder / 'ruler.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            existing[row['row_id']] = record
            save(self.root / 'live.json', dict(job=job['id'], phase='ruler', completed=len(existing),
                requested=len(data), last_row=row['row_id'], correct=record['correct'], baseline=record['baseline_correct']))
            print(json.dumps(dict(job=job['id'], row=row['row_id'], correct=record['correct'], baseline=record['baseline_correct'])), flush=True)
        records = list(existing.values())
        nll = read_rows(folder / 'nll.jsonl')
        summary = dict(status='COMPLETE', candidate=summarize(records), baseline=summarize([self.baseline[r['row_id']] for r in records]),
            nll_by_length={str(length): dict(n=sum(r['length'] == length for r in nll),
                delta=float(np.mean([r['nll'] - r['baseline_nll'] for r in nll if r['length'] == length]))) for length in sorted({r['length'] for r in nll})},
            wins=sum(r['correct'] > r['baseline_correct'] for r in records), losses=sum(r['correct'] < r['baseline_correct'] for r in records),
            scope='Historical development inputs; not independent confirmation', peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save(folder / 'summary.json', summary)
        return summary

    def run(self):
        self.qualification()
        while not (self.root / 'STOP').exists():
            pending = [p for p in sorted((self.root / 'queue').glob('*.json')) if not (self.root / 'done' / p.name).exists()]
            if not pending:
                save(self.root / 'live.json', dict(phase='READY_FOR_NEXT_EXPLICIT_JOB', pid=os.getpid()))
                time.sleep(3)
                continue
            path = pending[0]
            job = json.loads(path.read_text())
            started = time.monotonic()
            try:
                if job.get('action', 'evaluate') == 'evaluate':
                    result = self.evaluate(job)
                else:
                    self.apply({'table': self.tables['MrPro']})
                    result = importlib.import_module('experiments.nongeometric_screen.' + job['module']).run(self, job)
                save(self.root / 'done' / path.name, dict(status='COMPLETE', result=result, elapsed_seconds=time.monotonic()-started, job_sha256=sha(path)))
            except Exception as exc:
                import traceback
                save(self.root / 'failure.json', dict(job=job, error=repr(exc), traceback=traceback.format_exc()))
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--history', default='/root/autodl-tmp/bm_transfer_20260908')
    args = parser.parse_args()
    Worker(args.root, args.history).run()
