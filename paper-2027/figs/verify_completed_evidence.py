"""Reproduce recorded score means and the public-parameter NCP table on CPU."""
from pathlib import Path
import collections
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'runtime' if (ROOT / 'runtime/experiments').is_dir() else ROOT.parent))
from experiments.native_contrastive_proximal_20260915.tables import build_ncp_arrays


def main():
    data = json.loads((HERE / 'completed_evidence_inputs.json').read_text())
    reports = data['reports']
    mapping = {'olmo_clean': 'olmo_clean16k_ruler200', 'x4': 'llama_clean_matched_dose_c',
               'x5': 'llama_clean_native8k'}
    means = {}
    row_count = 0
    for case, arms in data['score_rows'].items():
        means[case] = {}
        keys = None
        for arm, rows in arms.items():
            identities = {(r[0], r[2]) for r in rows}
            assert len(identities) == len(rows)
            if keys is not None:
                assert keys == identities
            keys = identities
            row_count += len(rows)
            cells = collections.defaultdict(list)
            for _, task, length, score in rows:
                cells[(str(length), task)].append(score)
            by_length = {length: float(np.mean([np.mean(v) for (l, _), v in cells.items() if l == length]))
                         for length in sorted({l for l, _ in cells})}
            means[case][arm] = by_length
            if case in mapping:
                for length, value in by_length.items():
                    expected = reports[mapping[case]]['summaries'][arm]['by_length'][length]['task_macro_official']
                    assert abs(value - expected) < 1e-12
            elif case == 'olmo_qa':
                assert abs(by_length['16384'] - reports['olmo_naturalqa631']['arms'][arm]['macro_f1']) < 1e-12
    for arm, case in [('native', 'native'), ('ncp', 'ncp')]:
        assert abs(means[case][arm]['4096'] - reports['olmo_native_ncp']['arm_macro_scores'][arm]) < 1e-12
    native = np.array(data['ncp_native']['values_float32'], dtype=np.float32)
    expected = np.array(data['ncp_receipt']['values_float32'], dtype=np.float32)
    built = build_ncp_arrays(native, native_length=4096)
    assert np.array_equal(built['candidate'], expected)
    digest = hashlib.sha256(expected.astype('<f4').tobytes()).hexdigest()
    assert digest == data['ncp_receipt']['table_sha256_float32']
    assert all(built['checks'].values())
    # Independent integration of the Fourier risk for representative spans.
    nodes, weights = np.polynomial.legendre.leggauss(160)
    t = (nodes + 1) / 2
    j = np.arange(1, len(built['fourier_coefficients']) + 1)
    max_error = 0.0
    for phi in [0, .1, 1, np.pi, 4.7, 10]:
        values = built['a0'] + np.cos(phi * t[:, None] * j) @ built['fourier_coefficients']
        integral = np.dot(weights * (1-t), values)
        closed = built['a0'] + np.dot(built['fourier_coefficients'], np.sinc(j*phi/(2*np.pi))**2)
        max_error = max(max_error, abs(integral-closed))
    assert max_error < 1e-10
    lb = reports['llama_longbench_v2_8k32k']
    for arm in ['tailspline', 'mrpro']:
        assert abs(np.mean([r[arm+'_score'] for r in lb['paired_rows']]) - lb['overall'][arm]) < 1e-12
    result = {'status': 'PASS', 'model_execution': False, 'stored_score_rows_checked': row_count,
              'ncp_exact_fp32_sha256': digest, 'ncp_checks': built['checks'],
              'ncp_changed_pairs': len(built['changed_indices']),
              'ncp_stationarity_residual': built['maximum_stationarity_residual'],
              'fourier_risk_integration_max_error': max_error, 'means': means,
              'intervals': 'Retained from paired reports; not re-estimated by this check.'}
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
