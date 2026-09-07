"""Author-selected low-frequency carrier removal: one closed-form scalar.

NumPy-only construction and independent algebra checks. No model optimization.
Complex coordinates are q_first + i*q_second in the split-half RoPE layout.
"""
from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def tensor_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<f4').tobytes()).hexdigest()


def moments(coefficients, omega, length, nodes=64):
    x, w = np.polynomial.legendre.leggauss(nodes)
    u, w = (x+1)*length/2, w/2  # normalized integral; common L cancels in c0
    phase = np.exp(1j*np.asarray(omega)[:,None]*u)
    f = np.einsum('hj,jn->hn', coefficients, phase, optimize=False)
    g = np.einsum('hj,jn->hn', coefficients*omega, phase, optimize=False)
    denominator = float(np.sum((f.real**2+f.imag**2)*w))
    numerator = float(np.sum((g.real*f.real+g.imag*f.imag)*w))
    second = float(np.sum((g.real**2+g.imag**2)*w))
    if not all(math.isfinite(x) for x in (denominator,numerator,second)) or denominator < 0:
        raise ValueError('invalid carrier integral')
    return dict(numerator=numerator, denominator=denominator, second_moment=second,
                c0=numerator/denominator if denominator else 0.0)


def analytic_moments(coefficients, omega, length):
    delta = np.asarray(omega)[:,None]-np.asarray(omega)[None,:]
    half_phase = delta*length/2
    kernel = np.exp(1j*half_phase)*np.sinc(half_phase/np.pi)
    def product(a, b):
        return float(np.einsum('hj,jk,hk->', a, kernel, b.conj(), optimize=False).real)
    return dict(numerator=product(coefficients*omega, coefficients),
                denominator=product(coefficients, coefficients),
                second_moment=product(coefficients*omega, coefficients*omega))


def build(native, native_length, scale, c0):
    native = np.asarray(native, dtype=np.float64)
    if native.ndim != 1 or not np.isfinite(native).all() or not np.all(native > 0) or not np.all(np.diff(native)<0):
        raise ValueError('positive ordered Native table required')
    if not native_length > 0 or not scale > 1 or not math.isfinite(c0):
        raise ValueError('finite scale>1 and carrier required')
    fast = np.flatnonzero(native*native_length > 32*2*np.pi)
    slow = np.flatnonzero(native*native_length < 2*np.pi)
    band = np.flatnonzero(native*native_length <= np.pi/2)
    if not len(fast) or not len(slow) or not len(band):
        raise ValueError('Native table lacks the declared bands')
    low, high, begin = int(fast[-1]), int(slow[0]), int(band[0])
    if high <= low or begin < high:
        raise ValueError('carrier band must be inside the PI tail')
    gamma = np.clip((np.arange(len(native))-low)/(high-low), 0, 1)
    yarn = (native*((1-gamma)+gamma/scale)).astype(np.float32)
    t = np.clip(np.arange(len(native))-low, 0, high-low)
    mr = (native/scale**(t*(t+1)/((high-low)*(high-low+1)))).astype(np.float32)
    cap = (1-1/scale)*native[begin]
    carrier = float(np.clip(c0, 0, cap))
    candidate = yarn.copy()
    candidate[band] = ((native[band]-carrier)/scale).astype(np.float32)
    if not np.isfinite(candidate).all() or not np.all(np.diff(candidate)<0):
        raise ValueError('signed slot ordering invalid')
    return dict(native=native.astype(np.float32), yarn=yarn, mr=mr, candidate=candidate,
                c0=float(c0), c=carrier, cap=float(cap), band_start=begin,
                low=low, high=high, scale=float(scale), native_length=int(native_length),
                gain=1+.1*math.log(scale))


def self_check():
    rng = np.random.default_rng(20260907)
    native = (1e6**(-np.arange(64)/64)).astype(np.float32)
    band = np.flatnonzero(native.astype(float)*32768 <= np.pi/2)
    coefficients = rng.normal(size=(7,len(band)))+1j*rng.normal(size=(7,len(band)))
    omega = native[band].astype(float)
    numeric = moments(coefficients, omega, 32768)
    analytic = analytic_moments(coefficients, omega, 32768)
    for key in analytic:
        if not math.isclose(numeric[key], analytic[key], rel_tol=1e-11, abs_tol=1e-20):
            raise AssertionError(('analytic quadrature parity',key))
    coarse = moments(coefficients, omega, 32768, nodes=32)
    if not math.isclose(coarse['c0'], numeric['c0'], rel_tol=1e-11, abs_tol=1e-18):
        raise AssertionError('quadrature convergence')
    c0, d, n, a = [numeric[k] for k in ('c0','denominator','numerator','second_moment')]
    objective = lambda c: a-2*c*n+c*c*d
    epsilon = float(omega.max())*.1
    for sign in (-1,1):
        if not math.isclose(objective(c0+sign*epsilon)-objective(c0),d*epsilon**2,rel_tol=1e-10,abs_tol=1e-20):
            raise AssertionError('closed-form minimizer')
    plan = build(native,32768,4,1.0)  # deliberately exercise the upper clip and negative frequencies
    assert plan['band_start']==47 and plan['low']==23 and plan['high']==40
    assert np.array_equal(plan['candidate'][:47],plan['yarn'][:47])
    assert np.any(plan['candidate'][47:]<0)
    assert max(abs(plan['candidate'][47:].astype(float)))*131072 <= np.pi/2+1e-7
    assert plan['c']*32768/4 <= 3*np.pi/32+1e-12
    real_values = (omega-plan['c'])/4
    assert np.allclose(np.diff(real_values),np.diff(omega)/4,rtol=1e-12,atol=1e-20)
    # Standard-library complex arithmetic avoids a shared array-kernel oracle.
    coefficient = coefficients[0]
    envelope_errors=[]
    for distance in (0,4096,65536,131072):
        before=sum(z*cmath.exp(1j*w*distance/4) for z,w in zip(coefficient,omega))
        after=sum(z*cmath.exp(1j*(w-plan['c'])*distance/4) for z,w in zip(coefficient,omega))
        envelope_errors.append(abs(after-cmath.exp(-1j*plan['c']*distance/4)*before))
    assert max(envelope_errors)<1e-12
    # Independent real absolute rotations verify the signed-frequency convention.
    def rotate(q, phase):
        return np.array([q[0]*math.cos(phase)-q[1]*math.sin(phase),
                         q[0]*math.sin(phase)+q[1]*math.cos(phase)])
    q,k=np.array([.2,-.7]),np.array([1.1,.3]); frequency=-.17; p,r=9,2
    direct=float(rotate(q,p*frequency)@rotate(k,r*frequency))
    complex_value=((q[0]+1j*q[1])*(k[0]-1j*k[1])*cmath.exp(1j*frequency*(p-r))).real
    reflected=float(rotate(q*np.array([1,-1]),p*(-frequency))@rotate(k*np.array([1,-1]),r*(-frequency)))
    assert abs(direct-complex_value)<1e-12 and abs(direct-reflected)<1e-12
    zero=moments(np.zeros_like(coefficients),omega,32768)
    zero_plan=build(native,32768,4,zero['c0'])
    assert zero['denominator']==0 and np.array_equal(zero_plan['candidate'],zero_plan['yarn'])
    return dict(status='INDEPENDENT_NUMPY_ALGEBRA_CHECKED_NO_CAPABILITY_CLAIM',
                quadrature_c0_difference=abs(coarse['c0']-numeric['c0']),
                carrier_identity_max_abs=max(envelope_errors),signed_rotation_error=abs(direct-complex_value),
                checked=['analytic_integral','quadrature_convergence','quadratic_minimizer',
                         'band_spacing_and_phase_bounds','complex_carrier_identity',
                         'signed_rotation_and_reflection','zero_background_reference_fallback'])


def construct(means_dir, out):
    manifest = json.loads((means_dir/'manifest.json').read_text())
    raw = (means_dir/'means.npz').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['means_npz_sha256']:
        raise ValueError('Native mean artifact drift')
    data = np.load(means_dir/'means.npz',allow_pickle=False)
    native = data['native'].astype(np.float64)
    if tensor_sha(native)!=manifest['native_tensor_sha256']:
        raise ValueError('Native frequency identity')
    if (manifest['query_heads'],manifest['kv_heads'],manifest['head_dim']) != (16,2,128):
        raise ValueError('declared Qwen3B GQA layout')
    band=np.flatnonzero(native*32768<=np.pi/2); docs=[d['doc'] for d in manifest['docs']]
    coefficients=[]
    for layer in manifest['layers']:
        for qdoc in docs:
            q=data[f'{qdoc}_{layer}_q'].astype(float);qc=q[:,:64]+1j*q[:,64:]
            for kdoc in docs:
                if qdoc==kdoc:continue
                k=data[f'{kdoc}_{layer}_k'].astype(float);kc=k[:,:64]+1j*k[:,64:]
                coefficients.append((qc*kc[np.arange(16)//8].conj())[:,band])
    coefficients=np.concatenate(coefficients,axis=0)
    integral=moments(coefficients,native[band],32768,nodes=64)
    exact=analytic_moments(coefficients,native[band],32768)
    for key in exact:
        if not math.isclose(integral[key],exact[key],rel_tol=1e-10,abs_tol=1e-20):
            raise ValueError('real-data integral verification')
    plan=build(native,32768,4,integral['c0'])
    result={k:v for k,v in plan.items() if not isinstance(v,np.ndarray)}
    result.update(status='FROZEN_UNIQUE_CANDIDATE' if plan['c'] else 'REFERENCE_ONLY',
        source_means_sha256=manifest['means_npz_sha256'],source_manifest_sha256=hashlib.sha256((means_dir/'manifest.json').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_revision=manifest['model_revision'],coefficient_rows=len(coefficients),
        pairing='all 56 ordered distinct C-document pairs, six layers, all 16 query heads; exact GQA mapping',
        integral=integral,analytic_integral=exact,self_check=self_check(),
        negative_slots=np.flatnonzero(plan['candidate']<0).tolist(),
        actual_native_added_phase=plan['c']*32768/4,
        max_target_absolute_phase=float(max(abs(plan['candidate'][47:].astype(float)))*131072),
        tables={name:{'values_float32':plan[key].tolist(),'tensor_sha256':tensor_sha(plan[key]),'gain':plan['gain']}
                for name,key in [('YaRN','yarn'),('MrPro','mr'),('Carrier','candidate')]},
        limits='One Native-background scalar; no validation answers, optimization sweep, weight update or capability proof.')
    with out.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    return {k:result[k] for k in ('status','c0','c','cap','band_start','negative_slots','actual_native_added_phase','max_target_absolute_phase')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--means-dir',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check(),indent=2))
    else:
        if args.means_dir is None or args.out is None:parser.error('--means-dir and --out required')
        print(json.dumps(construct(args.means_dir,args.out),indent=2))
