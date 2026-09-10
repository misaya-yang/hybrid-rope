"""CPU arithmetic checks for conditional softmax mixed-frequency identities.

The artificial coefficients below validate mathematics, not Qwen activations.
No model loading, parameter fitting, or candidate-table generation is performed.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np


def bessel_i(n, amplitude):
    n = abs(int(n))
    term = (amplitude/2)**n/math.factorial(n)
    value = term
    for k in range(1,200):
        term *= (amplitude/2)**2/(k*(k+n))
        value += term
        if abs(term) <= 1e-17*max(abs(value),1e-300):
            return value
    raise ArithmeticError('Bessel series failed to converge')


def main():
    root = Path(__file__).resolve().parents[2]
    out = root/'results/nongeometric_screen_20260909/planned_controls/softmax_harmonic_audit.json'
    nmax, length = 9, 8192
    ds = np.arange(length)
    amplitudes = np.array([1.,1.])
    phase = np.array([.17,-.31])
    reports = {}
    for name,frequencies in [('slow_difference',[.51,.5103]),('rapid_difference',[.51,.53])]:
        f = np.array(frequencies)
        score = (amplitudes*np.cos(ds[:,None]*f-phase)).sum(1)
        direct = np.exp(score)
        reconstructed = np.zeros(length,dtype=np.complex128)
        for n in itertools.product(range(-nmax,nmax+1),repeat=2):
            coefficient = math.prod(bessel_i(n[j],amplitudes[j]) for j in range(2))
            coefficient *= np.exp(-1j*np.dot(n,phase))
            reconstructed += coefficient*np.exp(1j*ds*np.dot(n,f))
        kept_mass = math.prod(sum(bessel_i(n,a) for n in range(-nmax,nmax+1)) for a in amplitudes)
        tail_bound = math.exp(amplitudes.sum())-kept_mass
        error = float(np.max(np.abs(direct-reconstructed)))
        if error > tail_bound+1e-12:
            raise AssertionError('Fourier reconstruction exceeds absolute tail bound')
        probability = direct/direct.sum()
        reports[name] = {'frequencies':frequencies,
            'individual_cycles':[float(x*length/(2*math.pi)) for x in f],
            'difference_phase_over_window':float((f[1]-f[0])*length),
            'first_half_attention_mass':float(probability[:length//2].sum()),
            'quarter_attention_masses':[float(v.sum()) for v in np.array_split(probability,4)],
            'max_fourier_reconstruction_error':error,
            'absolute_truncation_bound':tail_bound}
    native = np.array(json.loads((root/'results/nongeometric_screen_20260909/reference_tables.json').read_text())['Native']['values_float32'])
    mixed = float(native[28]-2*native[29]+native[30])
    mixed_amplitudes = []
    for amplitude in [.3,1.,2.]:
        relative = 2*bessel_i(1,amplitude)**2*bessel_i(2,amplitude)/bessel_i(0,amplitude)**3
        mixed_amplitudes.append({'assumed_equal_logit_amplitude':amplitude,
            'mixed_cosine_amplitude_relative_to_torus_constant':relative,
            'scope':'Bessel coefficient ratio in a 3-frequency conditional row, not an observed model amplitude'})
    report = {'scope':'CPU identity and conditional-row examples only; no model evidence',
        'content_phases':phase.tolist(),'two_frequency_checks':reports,
        'native_qwen_28_minus_2x29_plus30':{'angular_frequency':mixed,'period_tokens':2*math.pi/mixed,'source_phase_at_32768':mixed*32768},
        'three_frequency_amplitude_checks':mixed_amplitudes}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
