"""Read-only campaign audit; CPU checks, no model weights or GPU allocation.

Run via SSH stdin on the experiment host. Temporary fixtures are isolated.
"""
import contextlib
import importlib.util
import io
import json
import math
import hashlib
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path('/root/autodl-tmp/phase1_20260910')
sys.path.insert(0, str(ROOT / 'repoharness'))
from scripts.experiments.olmo_fast_screen.ruler_bench import score

def load(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
    assert len({r['row_id'] for r in rows}) == len(rows), path
    return {r['row_id']: r for r in rows}

panel = load('/root/autodl-tmp/olmo_fast_screen_20260908/prepared_holdout_union/screen.jsonl')
base = load(ROOT / 'holdout180/beta_b1p0.jsonl')
report = {'holdout': {}, 'score_mismatches': {}, 'record_fields': sorted(next(iter(base.values())))}
groups={}
for key,row in panel.items():
    h=hashlib.sha256(np.asarray(row['prompt_ids'],dtype='<i8').tobytes()).hexdigest()
    groups.setdefault(h,[]).append(key)
report['prompt_duplicates']=dict(rows=len(panel),unique_prompts=len(groups),repeated_groups=sum(len(v)>1 for v in groups.values()),examples=[v for v in groups.values() if len(v)>1][:4])
report['unique_holdout']={}
for name, rel in [('BM', 'holdout180/beta_b1p0.jsonl'),
                  ('b3', 'holdout180/beta_b3p0.jsonl'),
                  ('a1b64', 'holdout180/wide_b1p0.jsonl'),
                  ('b4wide', 'holdout180/wide_b4p0.jsonl'),
                  ('step42', 's42_out/pro_step42.jsonl')]:
    rows = load(ROOT / rel)
    assert rows.keys() == panel.keys() == base.keys()
    report['score_mismatches'][name] = sum(abs(score(panel[k], r['output_text']) - r['correct']) > 1e-12 for k, r in rows.items())
    report['holdout'][name] = {}
    report['unique_holdout'][name]={}
    for cap in (4096, 16384):
        keys = sorted(k for k in panel if panel[k]['length_cap'] == cap)
        a, b = (np.array([v[k]['correct'] for k in keys]) for v in (rows, base))
        d = a-b
        report['holdout'][name][cap] = dict(n=len(keys), mean=float(a.mean()),
            delta_pp=float(d.mean()*100), se_pp=float(d.std(ddof=1)/math.sqrt(len(d))*100),
            whole_delta_pp=float(((a>=.999).astype(float)-(b>=.999)).mean()*100),
            whole_repairs=int(((a>=.999)&(b<.999)).sum()),
            whole_breaks=int(((a<.999)&(b>=.999)).sum()))
        selected=[v for v in groups.values() if panel[v[0]]['length_cap']==cap]
        a,b=(np.array([np.mean([records[key]['correct'] for key in g]) for g in selected]) for records in (rows,base))
        d=a-b
        report['unique_holdout'][name][cap]=dict(n=len(selected),delta_pp=float(d.mean()*100),se_pp=float(d.std(ddof=1)/math.sqrt(len(d))*100),whole_delta_pp=float(((a>=.999).astype(float)-(b>=.999)).mean()*100),duplicate_score_disagreements=sum(len({rows[k]['correct'] for k in g})>1 for g in selected))
        tasks={}
        for task in sorted({panel[g[0]]['task'] for g in selected}):
            indices=[i for i,g in enumerate(selected) if panel[g[0]]['task']==task]
            td=d[indices]
            tasks[task]=dict(n=len(td),delta_pp=float(td.mean()*100),variance_of_mean=float(td.var(ddof=1)/len(td)))
        report['unique_holdout'][name][cap].update(task_cells=tasks,macro_delta_pp=float(np.mean([v['delta_pp'] for v in tasks.values()])),macro_se_pp=float(np.sqrt(sum(v['variance_of_mean'] for v in tasks.values()))/len(tasks)*100))

newpanel = load('/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl')
report['gain'] = {}
for p in sorted((ROOT/'olmo_gain').glob('*.jsonl')):
    r = load(p)
    report['gain'][p.stem] = dict(n=len(r), score_mismatches=sum(abs(score(newpanel[k], x['output_text'])-x['correct'])>1e-12 for k,x in r.items()), mean=float(np.mean([x['correct'] for x in r.values()])))
arch = load('/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01/MrProBM.jsonl')
rerun = load(ROOT/'olmo_gain/gain_bm_g1p138629436111989.jsonl')
assert arch.keys() == rerun.keys() == newpanel.keys()
report['bm_archive_vs_rerun'] = dict(n=len(arch), archive_score_mismatches=sum(abs(score(newpanel[k], r['output_text'])-r['correct'])>1e-12 for k,r in arch.items()), different_scores=sum(arch[k]['correct']!=rerun[k]['correct'] for k in arch), different_texts=sum(arch[k]['output_text']!=rerun[k]['output_text'] for k in arch), archive_mean=float(np.mean([r['correct'] for r in arch.values()])))
tables = json.loads(Path('/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/tables.json').read_text())
old = np.asarray(tables['MrProBM']['values_float32'],dtype=np.float32)
k = np.arange(1,19,dtype=np.float64)
w = k*(19-k)
m = np.zeros(64)
m[15:33] = np.cumsum(w/w.sum())
m[33:] = 1
new = (500000.**(-np.arange(64,dtype=np.float64)/64)*4.**(-m)).astype(np.float32)
rt = json.loads(Path('/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01/runtime.json').read_text())
report['bm_table_comparison'] = dict(archive_dictionary_sha=rt['table_identity']['MrProBM'],prepared_dictionary_sha=hashlib.sha256(json.dumps(tables['MrProBM'],sort_keys=True,separators=(',',':')).encode()).hexdigest(), prepared_tensor_sha=hashlib.sha256(old.tobytes()).hexdigest(), reconstructed_tensor_sha=hashlib.sha256(new.tobytes()).hexdigest(), differing_slots=int((old!=new).sum()), max_abs_difference=float(abs(old-new).max()), old_gain=tables['MrProBM']['gain'], new_gain=1.138629436111989)

spec = importlib.util.spec_from_file_location('reader', '/tmp/rope_pro_audit_20260911/qwen4x_power_read_before.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    reader.RUN = str(p)
    reader.MANIFEST = str(p/'books.json')
    # BM has strictly lower NLL in every synthetic sample.
    dd = np.array([-.020, -.019, -.021, -.020, -.018, -.022])
    (p/'rows.jsonl').write_text('\n'.join(json.dumps(dict(arm=a, per_doc=v)) for a,v in [('native',[0.]*6),('mrpro',[0.]*6),('beta_b1_BM',dd.tolist())]))
    (p/'books.json').write_text(json.dumps([dict(book=i) for i in range(6)]))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        reader.main()
    report['sign_fixture'] = dict(actual_better='BM', printed_verdict=out.getvalue().split('--- VERDICT')[-1])
groups = [0]*10 + [1] + [2]
d = np.array([0.]*10+[1.,2.])
mu = d.mean()
G = len(set(groups))
correct_se = math.sqrt(G/(G-1)*sum(sum(d[np.array(groups)==g]-mu)**2 for g in set(groups))/len(d)**2)
report['cluster_fixture'] = dict(piece_mean=float(mu), book_mean=float(np.mean([d[np.array(groups)==g].mean() for g in set(groups)])), reader_se=float(reader.clustered_se(d,groups)), piece_mean_cluster_se=correct_se)

books = json.loads(Path('/root/autodl-tmp/longtext/prepared_pg19_4x/rows.json').read_text())
report['qwen_books'] = dict(chunks=len(books), unique_books=len({r['book'] for r in books}), counts={str(b):sum(r['book']==b for r in books) for b in sorted({r['book'] for r in books})})

import torch
from transformers import Olmo2Config
from transformers.models.olmo2.modeling_olmo2 import Olmo2RotaryEmbedding
cfg = Olmo2Config(hidden_size=256, num_attention_heads=2, num_key_value_heads=2, num_hidden_layers=2, intermediate_size=512, vocab_size=256, eos_token_id=2, pad_token_id=0)
rot = Olmo2RotaryEmbedding(cfg)
rot.inv_freq = rot.inv_freq.clone().requires_grad_()
x = torch.zeros(1, 32, 256, dtype=torch.bfloat16)
pos = torch.arange(16000,16032)[None]
cos, sin = rot(x,pos)
report['stock_rotary'] = dict(cos_dtype=str(cos.dtype), cos_requires_grad=cos.requires_grad, inv_requires_grad=rot.inv_freq.requires_grad)
spec = importlib.util.spec_from_file_location('cm', '/tmp/rope_pro_audit_20260911/curvature_model.py')
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
f = cm.FrozenRoPE.__new__(cm.FrozenRoPE)
f.rotary = rot
f._grad_patched = False
f._patch_grad_rotary()
pc, ps = rot(x,pos)
report['patched_rotary'] = dict(cos_dtype=str(pc.dtype), cos_requires_grad=pc.requires_grad, max_cos_diff=float((pc.float()-cos).abs().max().detach()))
print(json.dumps(report, indent=2))
