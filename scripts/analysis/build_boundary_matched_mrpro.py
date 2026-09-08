"""The single author-selected discrete minimum-gap-roughness MrPro replacement."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from .project_mrpro_transition import exponent, fp32, tensor_sha


def increments(n):
    if not isinstance(n,int) or n < 1:
        raise ValueError('positive integer transition width required')
    return [Fraction(6*i*(n+1-i), n*(n+1)*(n+2)) for i in range(1,n+1)]


def cumulative(n,q):
    if not 0 <= q <= n:
        raise ValueError('transition coordinate outside [0,N]')
    return Fraction(q*(q+1)*(3*n+2-2*q), n*(n+1)*(n+2))


def roughness(values):
    extended = [Fraction(),*values,Fraction()]
    return sum((b-a)**2 for a,b in zip(extended,extended[1:]))


def independent_minimum(n):
    # Gaussian elimination on L z=1, then normalize to sum(z)=1.
    # L is the Dirichlet tridiagonal Laplacian of the stated objective.
    matrix = [[Fraction(2 if i==j else -1 if abs(i-j)==1 else 0)
               for j in range(n)]+[Fraction(1)] for i in range(n)]
    for column in range(n):
        pivot = matrix[column][column]
        matrix[column] = [value/pivot for value in matrix[column]]
        for row in range(n):
            if row == column:continue
            coefficient = matrix[row][column]
            matrix[row] = [a-coefficient*b for a,b in zip(matrix[row],matrix[column])]
    solution = [row[-1] for row in matrix]
    total = sum(solution)
    return [value/total for value in solution]


def build(native,mr,gain,scale,low,high,p2=None):
    n = high-low
    steps = increments(n)
    if steps != independent_minimum(n):
        raise AssertionError('closed form disagrees with independent constrained solution')
    intended = [float(cumulative(n,max(0,min(n,j-low)))) for j in range(len(native))]
    values = [fp32(w*scale**(-m)) if low<j<high else mr[j]
              for j,(w,m) in enumerate(zip(native,intended))]
    realized, reference = exponent(native,values,scale), exponent(native,mr,scale)
    p2_m = exponent(native,p2,scale) if p2 is not None else None
    source_steps = [Fraction(2*i,n*(n+1)) for i in range(1,n+1)]
    r_bm,r_mr = roughness(steps),roughness(source_steps)
    if n>1 and not r_bm<r_mr:
        raise AssertionError('nontrivial case did not improve the specified objective')
    rows = []
    for j in range(len(native)):
        row = dict(slot_zero_based=j,slot_one_based=j+1,
            mr_exponent=reference[j],bm_exponent_intended=intended[j],bm_exponent_realized=realized[j],
            exponent_delta_to_mr=intended[j]-reference[j],native_frequency=native[j],
            mr_frequency=mr[j],bm_frequency=values[j],frequency_ratio_to_mr=values[j]/mr[j])
        if j+1<len(native):
            row.update(native_right_log_gap=math.log(native[j]/native[j+1]),
                       mr_right_log_gap=math.log(mr[j]/mr[j+1]),
                       bm_right_log_gap=math.log(values[j]/values[j+1]))
        else:row.update(native_right_log_gap=None,mr_right_log_gap=None,bm_right_log_gap=None)
        if p2 is not None:
            row.update(p2_exponent=p2_m[j],p2_frequency=p2[j],
                       bm_exponent_delta_to_p2=intended[j]-p2_m[j])
        rows.append(row)
    checks = dict(
        finite_positive=all(math.isfinite(w) and w>0 for w in values),
        strictly_decreasing=all(a>b for a,b in zip(values,values[1:])),
        fast_band_bitwise_equal=values[:low+1]==mr[:low+1],
        slow_band_bitwise_equal=values[high:]==mr[high:],
        support_endpoints_bitwise_equal=(values[0],values[-1])==(mr[0],mr[-1]),
        intended_exponent_range=[min(intended),max(intended)],
        realized_exponent_range=[min(realized),max(realized)],
        strictly_increasing_middle_exponents=all(a<b for a,b in zip(intended[low:high],intended[low+1:high+1])),
        sum_radix_increments=float(sum(steps)),
        exponent_sum_increase_vs_mr=math.fsum(a-b for a,b in zip(intended,reference)),
        changed_slots=[j for j,(a,b) in enumerate(zip(values,mr)) if a!=b],
        amplitude_preserved=gain,
        max_exponent_rounding_error=max(abs(a-b) for a,b in zip(intended,realized)),
    )
    if not all(checks[k] for k in ('finite_positive','strictly_decreasing','fast_band_bitwise_equal',
        'slow_band_bitwise_equal','support_endpoints_bitwise_equal','strictly_increasing_middle_exponents')):
        raise AssertionError(checks)
    return dict(low=low,high=high,N=n,scale=scale,gain=gain,values_float32=values,
        tensor_sha256=tensor_sha(values),exponents_intended=intended,exponents_realized=realized,
        radix_increments=[float(v) for v in steps],checks=checks,rows=rows,
        theory=dict(objective='sum_{i=0}^N (epsilon_{i+1}-epsilon_i)^2; endpoint epsilon=0, nonnegative, sum=1',
            independent_exact_laplacian_solution=True,
            mr_roughness=float(r_mr),bm_roughness=float(r_bm),ratio=float(r_bm/r_mr),
            fraction_reduction=float(1-r_bm/r_mr),
            terminal_increment_mr=float(source_steps[-1]),terminal_increment_bm=float(steps[-1]),
            finite_terminal_jump_not_zero=True,
            theorem_scope='Unique minimizer of this discrete spectral objective. No theorem of task-score improvement.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args();owner=args.root/'docs/research'
    op=owner/'ROPE_OLMO_MRPRO_SOURCE_20260908.json'
    qp=owner/'ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json'
    np=owner/'ROPE_RECOVERED_QWEN_P2_20260907.json'
    o,q,h=[json.loads(p.read_text()) for p in (op,qp,np)]
    oe=o['tables']['MrPro'];on=o['tables']['Native']['values_float32']
    olmo=build(on,oe['values_float32'],oe['gain'],o['scale'],
               oe['construction']['low'],oe['construction']['high'])
    qn=h['native_float32'];qe=q['tables']['MrPro']
    turns=[w*q['native_length']/(2*math.pi) for w in qn]
    low=max(j for j,v in enumerate(turns) if v>32);high=min(j for j,v in enumerate(turns) if v<1)
    qwen=build(qn,qe['values_float32'],qe['gain'],q['scale'],low,high,
              p2=q['tables']['FullLagP2']['values_float32'])
    report=dict(status='FROZEN_SINGLE_MRPRO_BM_REVIEWED_FOR_ONE_BOUNDED_SCREEN',
        candidate_id='MrProBM',target_model=o['model_id'],target_revision=o['revision'],
        target_actual_parameters=o['actual_parameters'],target_native_length=o['native_length'],target_base=o['base'],
        definition='epsilon_i=6*i*(N+1-i)/(N*(N+1)*(N+2)); m_q=q*(q+1)*(3*N+2-2*q)/(N*(N+1)*(N+2)); frequency=Native*s^(-m_q)',
        source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (op,qp,np)},
        target_olmo=olmo,source_qwen_comparison=qwen,
        review=dict(decision='Proceed with the author-requested single short development comparison; no second candidate.',
            predicted_geometric_effect='Positive compression throughout the middle, less concentration of log-gap expansion at the slow boundary, same cumulative scale and unchanged outer bands.',
            task_hypothesis='The redistribution may improve combined retrieval, binding and update accuracy without a 4K regression; this is the bounded empirical discriminator, not an established consequence of smoothness.',
            failure_modes=['More compression at every middle slot can harm native/local processing.',
                           'Smoothness across rotary slots need not match learned Q/K sensitivity.',
                           'Only part of the old P2 movement agrees; earlier P2 middle slots move in the opposite direction.',
                           'A weak baseline or compact qualification failure makes this benchmark unsuitable for ranking.'],
            prediction_falsifier='No positive total score gain or a 4K correct-count decrease does not promote this candidate. Do not tune the formula or add smoothing variants on these outcomes.',
            novelty='Not adjudicated; a supplied proposal and a verified discrete optimizer, not a novelty claim.'),
        execution='Array arithmetic only; no GPU or model inference.')
    path=owner/'ROPE_MRPRO_BM_CANDIDATE_20260908.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for name,data in [('OLMO',olmo),('QWEN',qwen)]:
        with (owner/f'ROPE_MRPRO_BM_{name}_SLOTS_20260908.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data['rows'][0]));writer.writeheader();writer.writerows(data['rows'])
    print(json.dumps({'status':report['status'],'olmo_theory':olmo['theory'],'olmo_checks':olmo['checks'],
                      'olmo_tensor_sha256':olmo['tensor_sha256'],'qwen_tensor_sha256':qwen['tensor_sha256']},indent=2))


if __name__=='__main__':main()
