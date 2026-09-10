"""Read-only, rerunnable summaries of NOSA/PC2 and PM-Keep generation results.

Usage: python -m experiments.position_overnight.report --runs RUN [RUN ...] --output REPORT_DIR
Only observed paired rows enter differences. Independent material/document clusters
are resampled together; metric families and runs are never pooled into a macro score.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
from statistics import mean

METRICS = ('official_recall', 'exact_plus_eos', 'qa_f1')
PC2_CANDIDATES = ('pc2', 'pc2_unweighted', 'pc2_rank1')
PC2_REFERENCES = ('native', 'cobs_rank2', 'split2')
PM_REFERENCES = ('E', 'F', 'K', 'E_author_policy')


def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def read_rows(path, warnings):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_bytes().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('record is not an object')
            rows.append(row)
        except (ValueError, UnicodeDecodeError) as error:
            warnings.append(f'{path}:{number}: incomplete/invalid JSON row ignored, source unchanged: {error}')
    return rows


def numeric(row, metric):
    value = row.get(metric, row.get('metrics', {}).get(metric))
    if isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1:
        return float(value)
    return None


def budget_label(row, kind, contract):
    if kind == 'PC2':
        return 'topk=' + str(row.get('topk', contract.get('topk', 'unknown')))
    # F is intentionally reusable across allocations: its cached config may be
    # from an earlier run. Pair it under this run's immutable allocation contract.
    config = contract.get('config') or row.get('config') or {}
    return 'keep_fraction={};sink={};recent={}'.format(
        config.get('keep_fraction', 'unknown'), config.get('sink_tokens', 4), config.get('recent_tokens', 256))


def cluster_id(row, kind):
    if row.get('material_cluster_id'):
        return 'material:' + str(row['material_cluster_id'])
    if kind == 'PM' and row.get('doc_id'):
        return 'document:' + str(row['doc_id'])
    if row.get('context_sha256'):
        return 'context:' + str(row['context_sha256'])
    return 'row:' + str(row['row_id'])


def quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def paired_bootstrap(pairs, replicates, seed):
    """Resample clusters, preserving their rows; report the paired row-mean estimand."""
    clusters = defaultdict(list)
    for _, unit, candidate, reference in pairs:
        clusters[unit].append(candidate - reference)
    if len(clusters) < 2:
        return None, None, len(clusters), 'insufficient independent units for a bootstrap interval'
    totals = [(sum(values), len(values)) for values in clusters.values()]
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled = [totals[rng.randrange(len(totals))] for _ in totals]
        estimates.append(sum(x[0] for x in sampled) / sum(x[1] for x in sampled) * 100)
    note = 'descriptive paired cluster bootstrap; conditional on completed shared rows'
    if len(totals) < 10:
        note += '; few independent units, interval may be unstable'
    return quantile(estimates, .025), quantile(estimates, .975), len(totals), note


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def summarize(runs, output, replicates=1000, seed=20260909):
    warnings, summaries, differences, completion, strongest = [], [], [], [], []
    baseline_ids = set()
    unique_runs = list(dict.fromkeys(Path(r).resolve() for r in runs))
    output = Path(output).resolve()
    if output in unique_runs:
        raise ValueError('Use a separate output directory; the source run directory is read-only')
    for run in unique_runs:
        for filename, kind, arm_field, contract_file in (
            ('generations.jsonl', 'PC2', 'selector', 'contract.json'),
            ('per_example.jsonl', 'PM', 'arm', 'manifest.json'),
        ):
            source = run / filename
            if not source.exists():
                continue
            contract = read_json(run / contract_file)
            observations, conflicting, duplicate_count = {}, set(), 0
            for row in read_rows(source, warnings):
                if not row.get('row_id') or not row.get(arm_field):
                    warnings.append(f'{source}: record lacks row_id or {arm_field}; excluded')
                    continue
                row = dict(row)
                row['_budget'] = budget_label(row, kind, contract)
                row['_arm'] = str(row[arm_field])
                key = (str(row['row_id']), row['_arm'], row['_budget'])
                if key in observations:
                    duplicate_count += 1
                    prior = observations[key]
                    if any(numeric(row, m) != numeric(prior, m) for m in METRICS):
                        conflicting.add(key)
                else:
                    observations[key] = row
                if row.get('baseline_cache_key'):
                    baseline_ids.add(str(row['baseline_cache_key']))
            for key in conflicting:
                observations.pop(key, None)
                warnings.append(f'{source}: conflicting duplicate {key!r} excluded from all statistics')
            rows = list(observations.values())
            planned_arms = contract.get('selectors' if kind == 'PC2' else 'arms', [])
            planned_rows = contract.get('row_ids' if kind == 'PC2' else 'rows', [])
            completed_by_arm = {arm: sorted({str(r['row_id']) for r in rows if r['_arm'] == arm})
                                for arm in sorted({r['_arm'] for r in rows} | set(planned_arms))}
            shared = set(planned_rows or {str(r['row_id']) for r in rows})
            for arm in planned_arms:
                shared.intersection_update(completed_by_arm.get(arm, []))
            completion.append({
                'run': str(run), 'kind': kind, 'source': str(source),
                'status': read_json(run / 'status.json').get('status', 'unknown'),
                'planned_rows': len(planned_rows) if planned_rows else None,
                'planned_arms': planned_arms, 'unique_observed_rows': len({str(r['row_id']) for r in rows}),
                'observed_row_arm_pairs': len(rows), 'duplicates_ignored': duplicate_count,
                'conflicts_excluded': len(conflicting), 'completed_row_ids_by_arm': completed_by_arm,
                'all_planned_arms_completed_row_ids': sorted(shared) if planned_arms else None,
                'reused_baseline_observations': sum(bool(r.get('reused_baseline')) for r in rows),
            })
            cells = defaultdict(dict)
            for row in rows:
                for metric in METRICS:
                    value = numeric(row, metric)
                    if value is None:
                        continue
                    cell = (str(row.get('task', 'unknown')), str(row.get('split', 'unknown')),
                            str(row.get('length_cap', 'not_recorded')), row['_budget'], metric)
                    cells[cell].setdefault(row['_arm'], {})[str(row['row_id'])] = (value, cluster_id(row, kind), row)
            for cell, arms in sorted(cells.items()):
                task, split, length_group, budget, metric = cell
                metadata = dict(run=str(run), kind=kind, task=task, split=split,
                                length_group=length_group, budget=budget, metric=metric)
                for arm, values in sorted(arms.items()):
                    summaries.append(dict(metadata, arm=arm, n_rows=len(values),
                                          n_units=len({v[1] for v in values.values()}),
                                          mean_percent=mean(v[0] for v in values.values()) * 100,
                                          reused_rows=sum(bool(v[2].get('reused_baseline')) for v in values.values()),
                                          completed_row_ids=sorted(values)))
                candidates = PC2_CANDIDATES if kind == 'PC2' else ('P',)
                references = PC2_REFERENCES if kind == 'PC2' else PM_REFERENCES + ('C', 'U')
                for candidate in candidates:
                    if candidate not in arms:
                        continue
                    for reference in references:
                        if reference not in arms:
                            continue
                        ids = sorted(arms[candidate].keys() & arms[reference].keys())
                        pairs = []
                        for row_id in ids:
                            a, b = arms[candidate][row_id], arms[reference][row_id]
                            if a[1] != b[1]:
                                warnings.append(f'{run}: cluster mismatch for {row_id} {candidate}/{reference}; pair excluded')
                                continue
                            pairs.append((row_id, a[1], a[0], b[0]))
                        salt = int(hashlib.sha256(repr((str(run), cell, candidate, reference)).encode()).hexdigest()[:16], 16)
                        low, high, n_units, note = paired_bootstrap(pairs, replicates, seed + salt)
                        differences.append(dict(metadata, candidate=candidate, reference=reference,
                            comparison_role='mechanism_control' if reference in ('C', 'U') else 'quality_reference',
                            n_candidate_observed=len(arms[candidate]), n_reference_observed=len(arms[reference]),
                            n_pairs=len(pairs), n_units=n_units,
                            candidate_mean_percent=mean(p[2] for p in pairs) * 100 if pairs else None,
                            reference_mean_percent=mean(p[3] for p in pairs) * 100 if pairs else None,
                            delta_percentage_points=mean(p[2] - p[3] for p in pairs) * 100 if pairs else None,
                            ci95_low_pp=low, ci95_high_pp=high, uncertainty_note=note,
                            paired_row_ids=[p[0] for p in pairs]))
                    refs = [r for r in (PC2_REFERENCES if kind == 'PC2' else PM_REFERENCES) if r in arms]
                    if refs:
                        common = set(arms[candidate])
                        for ref in refs:
                            common.intersection_update(arms[ref])
                        if common:
                            ref_means = {ref: mean(arms[ref][i][0] for i in common) * 100 for ref in refs}
                            best = max(ref_means, key=ref_means.get)
                            strongest.append(dict(metadata, candidate=candidate, reference=best, n_common_rows=len(common),
                                reference_means_percent=ref_means,
                                scope='highest observed reference on identical rows shared by all available references; not population superiority'))
    if not completion:
        warnings.append('No generations.jsonl or per_example.jsonl found in the supplied run directories')
    result = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'runs': completion,
        'summary': summaries, 'paired_differences': differences, 'strongest_observed_references': strongest,
        'warnings': warnings, 'bootstrap_replicates': replicates, 'seed': seed,
        'unique_baseline_cache_keys_across_runs': len(baseline_ids),
        'policy': 'Percent = metric fraction × 100. Metrics/runs are separate. Missing observations are never zero. '
                  'Duplicate row/arm/budget records are counted once. Reused controls are references within each run, '
                  'never pooled across runs as independent replications. Intervals are descriptive, not acceptance or stopping decisions.',
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / 'summary.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for filename, entries in (('summary.csv', summaries), ('paired_differences.csv', differences)):
        filtered = [{k: v for k, v in entry.items() if k not in ('completed_row_ids', 'paired_row_ids')} for entry in entries]
        stream = io.StringIO()
        if filtered:
            writer = csv.DictWriter(stream, fieldnames=list(filtered[0]))
            writer.writeheader()
            writer.writerows(filtered)
        atomic_text(output / filename, stream.getvalue())
    lines = ['# 当前观测与待决问题', '', result['policy'], '',
             '本报告只读取已保存输出；不执行实验、不触发停止或关机，不凭小样本推断 accept。', '']
    for entry in completion:
        counts = ', '.join(f'{arm}={len(ids)}' for arm, ids in entry['completed_row_ids_by_arm'].items())
        lines.append(f"- `{entry['run']}` ({entry['kind']}, {entry['status']}): {counts}; "
                     f"观察到 {entry['unique_observed_rows']} 个独特 row；完整 row IDs 在 summary.json。")
    lines += ['', '| 任务/长度/预算 | 比较 | 指标 | 共同 rows / 独立 units | 差值 pp | 配对 95% 区间 pp |',
              '|---|---|---|---:|---:|---|']
    for d in differences:
        delta = '未决' if d['delta_percentage_points'] is None else f"{d['delta_percentage_points']:+.2f}"
        interval = '独立样本不足' if d['ci95_low_pp'] is None else f"[{d['ci95_low_pp']:+.2f}, {d['ci95_high_pp']:+.2f}]"
        label = '机制对照' if d['comparison_role'] == 'mechanism_control' else '质量参考'
        lines.append(f"| {Path(d['run']).name}/{d['task']}/{d['split']}/{d['length_group']}/{d['budget']} | "
                     f"{d['candidate']}−{d['reference']} ({label}) | {d['metric']} | {d['n_pairs']} / {d['n_units']} | {delta} | {interval} |")
    lines += ['', '正差值仅表示当前共同样本上候选较好；区间跨零、独立材料少或比较尚未完成时，收益仍未决。'
              '即使区间不跨零，也未校正开发集选择和多重比较，不能据此宣布稳定收益。', '']
    for entry in strongest:
        lines.append(f"- {Path(entry['run']).name}/{entry['task']}/{entry['metric']}/{entry['budget']}: "
                     f"在所有已出现参考臂共同的 {entry['n_common_rows']} rows 上，最高观察参考为 `{entry['reference']}`；"
                     '不把不同缺失集合下的均值直接排名。')
    if warnings:
        lines += ['', '读取提示：'] + ['- ' + w for w in warnings]
    atomic_text(output / 'DECISION.md', '\n'.join(lines) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260909)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error('--bootstrap must be >= 100; default is 1000')
    result = summarize(args.runs, args.output, args.bootstrap, args.seed)
    print(json.dumps({'output': str(args.output.resolve()), 'runs': len(result['runs']),
                      'summary_cells': len(result['summary']), 'paired_comparisons': len(result['paired_differences']),
                      'warnings': len(result['warnings'])}))


if __name__ == '__main__':
    main()
