from fractions import Fraction

import torch
from scipy.optimize import linprog

from scripts.lib.rope.gap_capped import cumulative_minimum, gap_capped_inv_freq
from scripts.experiments.cross_audit.tables import native_table, transform


def test_closed_form_is_simultaneous_linear_program_optimum():
    for n in (2, 3, 7, 17, 18):
        m = cumulative_minimum(n)
        cap = Fraction(2, n+1)
        increments = [b-a for a, b in zip(m, m[1:])]
        assert sum(increments) == 1 and all(0 <= x <= cap for x in increments)
        for q in range(1, n):
            solved = linprog([1.]*q+[0.]*(n-q), A_eq=[[1.]*n], b_eq=[1.],
                bounds=[(0, float(cap))]*n, method='highs')
            assert solved.success and abs(solved.fun-float(m[q])) < 1e-10
            assert m[q] <= Fraction(q*(q+1), n*(n+1))


def test_endpoint_gain_and_maximum_gap_match_original_mrpro_limits():
    for base, length in ((1e6, 32768), (500000., 4096)):
        native = native_table(128, base)
        mr, gain, _ = transform(native, dim=128, base=base, reference_length=length, scale=4., method='mrpro')
        candidate, actual_gain, meta = gap_capped_inv_freq(torch.from_numpy(native),
            base=base, reference_length=length, scale=4.)
        mr = torch.from_numpy(mr)
        assert actual_gain == gain
        assert torch.equal(candidate[:meta['low']+1], mr[:meta['low']+1])
        assert torch.equal(candidate[meta['high']:], mr[meta['high']:])
        assert torch.all(candidate >= mr) and torch.all(candidate <= torch.from_numpy(native))
        assert torch.log(candidate[:-1]/candidate[1:]).max() <= torch.log(mr[:-1]/mr[1:]).max()+1e-6
