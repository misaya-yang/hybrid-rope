#!/usr/bin/env python3
"""Exploratory CPU-only analysis for the round-10 question document (2026-09-05).

Purpose (planning aid, NOT an official verdict; touches frozen assets read-only):
 1. Reproduce the frozen paired_retention_intervals on the actual Z receipts and
    verify the CI matches out/Z/review.json exactly.
 2. Decompose which Native task stratum binds the task-CI lower bound.
 3. Nonparametric power simulation (numpy-vectorized, mirrors the frozen
    statistic exactly): if a FRESH independent Native confirmation pool of
    K source-groups/task is drawn from the same population as the observed
    groups, how often does the 95% paired-bootstrap lower bound reach >= 0.88?
 4. Pre-registered E3a decision thresholds: exact binomial critical values for
    Z_compact far-success counts against Z's observed far level.
 5. Breadth of Z's far gains across semantic groups (for mechanism section).

Outputs one JSON to stdout. Reads frozen receipts read-only; imports the frozen
engine function from code_release_008 (read-only import).
"""
import json, math, random, sys
from pathlib import Path
import numpy as np

B = Path('/root/autodl-tmp/ffn_review_execution_20260904')
OUT = Path('/root/autodl-tmp/claude_round10_20260905/out')
sys.path.insert(0, str(B / 'code_release_008'))
from scripts.lib.rope.generation_contract import paired_retention_intervals  # noqa: E402

import hashlib

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_receipt_rows(directory, name='native_evaluation.json'):
    """Mirror reviewer load_receipt: receipt + hash-verified examples.jsonl rows."""
    receipt = json.loads((directory / name).read_text())
    raw = directory / 'examples.jsonl'
    if sha(raw) != receipt['examples_sha256']:
        raise ValueError('raw validation receipt hash drift')
    return [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]

bn = load_receipt_rows(B / 'qwen_native_validation')
cn = load_receipt_rows(OUT / 'Z/native128')
review = json.loads((OUT / 'Z/review.json').read_text())

# --- 1. exact reproduction of the frozen interval -------------------------
ci = paired_retention_intervals(bn, cn, nll_task='text')
repro_ok = all(
    abs(ci['ci95'][k][i] - review['paired_retention_uncertainty']['ci95'][k][i]) < 1e-12
    for k in ('ppl', 'task', 'task_eos') for i in (0, 1))

# --- 2. per-task decomposition --------------------------------------------
def group_stats(rows, field):
    grouped = {}
    for r in rows:
        if r['task'] == 'text':
            continue
        grouped.setdefault(r['task'], {}).setdefault(r['group'], []).append(r[field])
    return {t: {g: sum(v) for g, v in gs.items()} for t, gs in grouped.items()}

def group_counts(rows):
    grouped = {}
    for r in rows:
        if r['task'] == 'text':
            continue
        grouped.setdefault(r['task'], {}).setdefault(r['group'], []).append(1)
    return {t: {g: len(v) for g, v in gs.items()} for t, gs in grouped.items()}

L = group_stats(bn, 'score_eos'); R = group_stats(cn, 'score_eos')
N = group_counts(bn)
per_task = {}
for t in L:
    la = sum(L[t].values()); lb = sum(R[t].values())
    per_task[t] = {'groups': len(L[t]), 'keys': sum(N[t].values()),
                   'left_correct': la, 'right_correct': lb,
                   'retention': round(lb / la, 4) if la else None}

# --- 3. power simulation for a fresh confirmation pool --------------------
# Empirical distribution of source groups per task: (n_keys, L_g, R_g).
# Frozen statistic: per resample, for each task draw K groups with replacement
# (K = number of groups in pool), each draw contributes ALL its keys;
# task_a/task_b = per-task mean left/right score over drawn keys;
# sample value = sum_t task_b / sum_t task_a.
emp = {t: np.array([[N[t][g], L[t][g], R[t][g]] for g in sorted(L[t])], dtype=float)
       for t in L}

def simulate(K, trials=2000, attenuate=1.0, seed=905):
    rng = np.random.default_rng(seed)
    lowers = np.empty(trials)
    for i in range(trials):
        pools = {t: emp[t][rng.integers(0, len(emp[t]), K)] for t in emp}
        ta = np.zeros(1000); tb = np.zeros(1000)
        for t, arr in pools.items():
            idx = rng.integers(0, K, (1000, K))
            sel = arr[idx]                                   # (1000, K, 3)
            na = sel[..., 0].sum(1); la = sel[..., 1].sum(1)
            ra = sel[..., 2].sum(1) * attenuate
            ta += la / na; tb += ra / na
        lowers[i] = np.sort(tb / ta)[int(0.025 * 999)]
    return {'K_groups_per_task': K, 'trials': trials, 'attenuate': attenuate,
            'prob_task_lower_ge_088': round(float((lowers >= 0.88).mean()), 4),
            'median_lower': round(float(np.median(lowers)), 4),
            'p05_lower': round(float(np.percentile(lowers, 5)), 4),
            'p95_lower': round(float(np.percentile(lowers, 95)), 4)}

power = [simulate(K, attenuate=a) for K in (16, 32, 64, 128) for a in (1.0, 0.95)]

# --- 4. E3a pre-registered thresholds --------------------------------------
def binom_cdf(n, p, k):
    from math import comb
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))

def binom_critical(n, p0, alpha=0.05):
    c = -1
    for k in range(n + 1):
        if binom_cdf(n, p0, k) <= alpha:
            c = k
        else:
            break
    return c

e3a = {}
for name, (x, n) in (('all_groups_far', (23, 32)),
                     ('qualified_cohort_far', (17, 26))):
    p0 = x / n
    c = binom_critical(n, p0)
    e3a[name] = {
        'Z_successes': x, 'n': n, 'p0_retention_null': round(p0, 4),
        'critical_value_alpha05_one_sided': c,
        'rule': (f'Z_compact far <= {c}/{n} rejects same-level retention '
                 f'(one-sided exact binomial, alpha<=0.05)'),
        'size_at_null': round(binom_cdf(n, p0, c), 5),
        'power_if_true_p_halved': round(1 - binom_cdf(n, p0 / 2, c - 1), 4),
        'prob_le_c_if_p05': round(binom_cdf(n, 0.5, c), 5),
        'prob_le_c_if_p025': round(binom_cdf(n, 0.25, c), 5),
    }

# --- 5. breadth of far gains across semantic groups ------------------------
fam = review['family_comparisons']['single_evidence']['all_groups']
breadth = {
    'far_gained_groups_of_total': f"{fam['far']['gained']}/32",
    'far_lost': fam['far']['lost'],
    'near_gained/lost': f"{fam['near']['gained']}/{fam['near']['lost']}",
    'compact_lost': fam['compact']['lost'],
    'interpretation': 'broad gain, zero loss: not a few-item artifact'}

print(json.dumps({'reproduction_matches_review': repro_ok,
                  'reproduced_ci95': ci['ci95'],
                  'groups_per_task': ci['groups_per_task'],
                  'per_task_retention': per_task,
                  'fresh_pool_power': power,
                  'e3a_thresholds': e3a,
                  'far_gain_breadth': breadth,
                  'caveats': [
                      'exchangeability assumption for fresh-pool simulation: new content is assumed drawn from the same effect distribution as observed groups',
                      'planning aid only; official verdict remains the frozen reviewer output']},
                 indent=2))
