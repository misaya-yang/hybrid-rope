"""Least cumulative compression under MrPro's original maximum radix increment.

This is a conditional geometric optimum, not a task-performance theorem.
"""
import math
from fractions import Fraction

import torch


def cumulative_minimum(width):
    if type(width) is not int or width < 1:
        raise ValueError('positive integer width required')
    cap = Fraction(2, width+1)
    return [max(Fraction(0), 1-(width-q)*cap) for q in range(width+1)]


def gap_capped_inv_freq(native_inv_freq, *, base, reference_length, scale):
    if (native_inv_freq.ndim != 1 or native_inv_freq.dtype != torch.float32
            or native_inv_freq.numel() < 2):
        raise ValueError('native FP32 vector required')
    if not math.isfinite(base) or base <= 1 or reference_length <= 0 or not math.isfinite(scale) or scale < 1:
        raise ValueError('invalid native geometry or scale')
    native = native_inv_freq.detach().cpu()
    dim = 2*native.numel()
    expected = 1/(base**(torch.arange(0, dim, 2, dtype=torch.float32)/dim))
    if not torch.allclose(native, expected, rtol=3e-7, atol=0):
        raise ValueError('input is not the declared native grid')
    if scale == 1:
        return native_inv_freq.clone(), 1., dict(identity=True, method='gap_capped')
    turns = [float(w)*reference_length/(2*math.pi) for w in native]
    fast = [j for j, t in enumerate(turns) if t > 32]
    slow = [j for j, t in enumerate(turns) if t < 1]
    if not fast or not slow or slow[0] <= fast[-1]:
        raise ValueError('no valid 32/1-turn transition')
    low, high = fast[-1], slow[0]
    width = high-low
    middle = cumulative_minimum(width)
    exponents = [float(middle[min(width, max(0, j-low))]) for j in range(len(native))]
    values = [w if j <= low else w/scale if j >= high else w*scale**(-exponents[j])
              for j, w in enumerate(native.tolist())]
    result = torch.tensor(values, dtype=torch.float32, device=native_inv_freq.device)
    if not torch.isfinite(result).all() or not torch.all(result > 0) or not torch.all(result[:-1] > result[1:]):
        raise ValueError('invalid table')
    return result, 1+.1*math.log(scale), dict(method='gap_capped', low=low, high=high,
        N=width, cap=float(Fraction(2, width+1)), scale=scale, base=base,
        reference_length=reference_length, exponents=exponents,
        formula='m_q=max(0,1-2*(N-q)/(N+1))',
        scope='Componentwise least cumulative compression with nonnegative increments bounded by the original MrPro maximum; no task-gain guarantee.')
