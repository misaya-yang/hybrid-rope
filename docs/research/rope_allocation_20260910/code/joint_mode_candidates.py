#!/usr/bin/env python3
"""CPU-only exact rank-one retiming of lowest-order transition relations.

No model loading, role inference, optimization, gradient selection or GPU use.
Each candidate starts from the actual MrPro FP32 tensor, changes only the
specified adjacent pair/triple, and targets that native relation divided by 4.
FP64 projection identities and deployed FP32 rounding errors are separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NATIVE_SHA = '138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e'
MR_SHA = '33cbe3a40994ac2a79126a14ce30282867bd6d49b74c1ada4d4cce8a7e76016f'
SCALE = 4.0
FIRST, LAST = 24, 39


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f4').tobytes()).hexdigest()


def read_table(path, expected_sha):
    entry = json.loads(Path(path).read_text())['table']
    raw = np.asarray(entry['values_float32'], dtype=np.float64)
    values = raw.astype('<f4')
    if raw.shape != (64,) or not np.array_equal(raw, values.astype(np.float64)):
        raise ValueError('source must be exactly represented FP32 values of shape64')
    if tensor_sha(values) != expected_sha or entry['tensor_sha256'] != expected_sha:
        raise ValueError('source tensor hash mismatch')
    if not np.isfinite(values).all() or not np.all(values > 0) or not np.all(np.diff(values) < 0):
        raise ValueError('source must be positive and strictly decreasing')
    if not math.isfinite(entry['gain']) or entry['gain'] <= 0:
        raise ValueError('invalid gain')
    return values.astype(np.float64), entry


def project(reference, native, n):
    target = float(n @ native) / SCALE
    delta_clock = target - float(n @ reference)
    candidate = reference + n * delta_clock / float(n @ n)
    return candidate, target, delta_clock


def build(native_path, mr_path):
    native, ne = read_table(native_path, NATIVE_SHA)
    mr, me = read_table(mr_path, MR_SHA)
    gain = float(me['gain'])
    if gain != 1 + .1 * math.log(SCALE) or ne['gain'] != 1:
        raise ValueError('gain differs from frozen Native/MrPro contracts')
    source_m = -np.log(mr/native)/math.log(SCALE)
    rows = []
    for order, pattern in ((1, [1., -1.]), (2, [1., -2., 1.])):
        for start in range(FIRST, LAST - order + 1):
            slots = list(range(start, start + order + 1))
            n = np.zeros(64); n[slots] = pattern
            ideal, target, delta_clock = project(mr, native, n)
            deployed = ideal.astype('<f4').astype(np.float64)
            dn = ideal-mr
            dn32 = deployed-mr
            norm_sq = float(n@n)
            orth = dn - n * float(n@dn)/norm_sq
            orth32 = dn32 - n * float(n@dn32)/norm_sq
            error = float(n@ideal)-target
            error32 = float(n@deployed)-target
            rounding_bound = float(np.abs(n)@np.abs(deployed-ideal))
            positive = bool(np.isfinite(deployed).all() and np.all(deployed > 0))
            crossings = np.flatnonzero(np.diff(deployed) >= 0).tolist()
            outside = np.ones(64,dtype=bool); outside[slots] = False
            exact_tol = 32*np.finfo(float).eps*max(float(np.max(np.abs(mr[slots]))), 1e-300)*norm_sq
            checks = dict(finite_positive=positive, strictly_decreasing=not crossings,
                crossing_left_indices=crossings, unchanged_outside_relation=bool(np.array_equal(deployed[outside],mr[outside])),
                endpoints_bitwise_equal=bool(np.array_equal(deployed[[0,-1]],mr[[0,-1]])),
                gain_exactly_preserved=True,
                exact_projection_identity_pass=bool(abs(error)<=exact_tol),
                exact_orthogonal_complement_pass=bool(float(np.max(np.abs(orth)))<=exact_tol),
                float32_relation_within_rounding_bound=bool(abs(error32)<=rounding_bound+exact_tol))
            if not all(checks[k] for k in ('unchanged_outside_relation','endpoints_bitwise_equal',
                 'exact_projection_identity_pass','exact_orthogonal_complement_pass','float32_relation_within_rounding_bound')):
                raise AssertionError(checks)
            m = -np.log(deployed/native)/math.log(SCALE) if positive else np.full(64,np.nan)
            native_clock = float(n@native); mr_clock = float(n@mr)
            relation = dict(order=order,slots_zero_based=slots,coefficients=pattern,
                native_clock=native_clock, mr_clock=mr_clock, target_clock=target,
                ideal_candidate_clock=float(n@ideal), deployed_candidate_clock=float(n@deployed),
                mr_clock_over_native=mr_clock/native_clock if native_clock else None,
                mr_effective_period_extension=native_clock/mr_clock if mr_clock else None,
                desired_period_extension=SCALE,
                mr_clock_relative_error_to_target=(mr_clock-target)/target if target else None,
                mr_already_retimes_native_by4=math.isclose(mr_clock,target,rel_tol=1e-6,abs_tol=0),
                source_period_tokens=2*math.pi/abs(native_clock) if native_clock else None,
                mr_period_tokens=2*math.pi/abs(mr_clock) if mr_clock else None,
                candidate_period_tokens=2*math.pi/abs(float(n@deployed)) if n@deployed else None,
                target_minus_mr_clock=delta_clock,
                exact_relation_error=error,deployed_relation_error=error32,
                deployed_relation_relative_error=error32/target if target else None,
                float32_relation_rounding_bound=rounding_bound,
                exact_orthogonal_residual_max_abs=float(np.max(np.abs(orth))),
                deployed_orthogonal_residual_max_abs=float(np.max(np.abs(orth32))),
                raw_frequency_sum_delta=float(dn.sum()),raw_frequency_sum_delta_float32=float(dn32.sum()))
            name=f'JointMode_d{order}_s'+'_'.join(map(str,slots))
            rows.append(dict(name=name,status='CPU_VALID_CANDIDATE' if positive and not crossings else 'CPU_INVALID_NOT_FOR_MODEL',
                table=dict(values_float32=deployed.tolist(),tensor_sha256=tensor_sha(deployed),gain=gain),
                relation=relation,checks=checks,
                delta_frequency_float64=dn.tolist(),delta_frequency_float32=dn32.tolist(),
                delta_log_period=(-np.log(deployed/mr)).tolist() if positive else None,
                delta_log_frequency=np.log(deployed/mr).tolist() if positive else None,
                delta_m=(m-source_m).tolist() if positive else None,
                m_float64=m.tolist() if positive else None,
                sum_m=float(m.sum()) if positive else None,
                delta_sum_m=float((m-source_m).sum()) if positive else None,
                compression_box_0_to1=bool(positive and np.all(m>=-1e-7) and np.all(m<=1+1e-7)),
                slots_faster_than_native=np.flatnonzero(deployed>native).tolist(),
                slots_faster_than_mr=np.flatnonzero(deployed>mr).tolist(),
                max_abs_delta_phase_at_128k=float(np.max(np.abs(dn32))*131072),
                max_abs_delta_log_period=float(np.max(np.abs(np.log(deployed/mr)))) if positive else None))
    if len(rows)!=29 or len({r['table']['tensor_sha256'] for r in rows})!=29:
        raise AssertionError('expected 15 pair and14 triple distinct candidates')
    return dict(status='CPU_DERIVED_FAMILY_NO_ROLE_OR_CAPABILITY_QUALIFICATION',
        formula='nu_c = nu_M + n * (n^T omega_native / 4 - n^T nu_M) / (n^T n)',
        coordinate='actual runtime inverse frequency; global64 slots; no sorting or clamping',
        scale=SCALE,transition_slots_zero_based=[FIRST,LAST],orders=[1,2],candidate_count=len(rows),
        source=dict(native_contract=str(native_path),native_contract_sha256=digest(native_path),native_tensor_sha256=NATIVE_SHA,
            mr_contract=str(mr_path),mr_contract_sha256=digest(mr_path),mr_tensor_sha256=MR_SHA,
            mr_gain=gain,mr_sum_m=float(source_m.sum()),mr_m_float64=source_m.tolist()),
        generator_sha256=digest(__file__),numpy_version=np.__version__,candidates=rows,
        valid_candidates=sum(r['status']=='CPU_VALID_CANDIDATE' for r in rows),
        qualification='Not selected by clocks, amplitudes, geometry or native gradients alone. Use actual whole-model long/native gradient sign only as a direction filter, then exact finite full-model losses and generated task endpoints. Candidate changes can have many radians of finite phase; no linear extrapolation guarantee.',
        limitations=['A correct source relation clock does not establish that the mode carries useful computation.',
            'Relation phase coefficients and previous-layer hidden states can change in the full model.',
            'Common raw-frequency sums are preserved by zero-sum n; sum of log-compression exponents is not fixed.',
            'FP32 deployment perturbs exact projection identities by reported rounding errors.',
            'Candidates deliberately permit acceleration relative to Mr and report any violation of the native compression box.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-contract',type=Path,default=ROOT/'results/nongeometric_screen_20260909/native_reference/contract.json')
    p.add_argument('--mr-contract',type=Path,default=ROOT/'results/nongeometric_screen_20260909/long_nll/MrPro_contract.json')
    p.add_argument('--out',type=Path,default=ROOT/'.agents/rope_unification_20260910/joint_mode_candidates.json')
    a=p.parse_args();result=build(a.native_contract,a.mr_contract)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.out),candidate_count=result['candidate_count'],valid_candidates=result['valid_candidates'],
        mr_sum_m=result['source']['mr_sum_m'],delta_sum_m_range=[min(x['delta_sum_m'] for x in result['candidates']),max(x['delta_sum_m'] for x in result['candidates'])]),indent=2))
    for x in result['candidates']:
        r=x['relation'];print(x['name'],x['status'],'Mr_extension',round(r['mr_effective_period_extension'],6),
            'sum_m_delta',round(x['delta_sum_m'],9),'phase128k',round(x['max_abs_delta_phase_at_128k'],6),
            'box',x['compression_box_0_to1'])

if __name__=='__main__':main()
