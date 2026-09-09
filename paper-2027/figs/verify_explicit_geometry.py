"""Evaluate fully specified basis counterexamples; no model execution."""
from pathlib import Path
import json
import numpy as np


def cross_gram(x, y, length):
    d, s = (x-y)*length, (x+y)*length
    a = lambda z: np.sinc(z/np.pi)
    b = lambda z: 0.0 if z == 0 else 2*np.sin(z/2)**2/z
    return .5*np.array([[a(d)+a(s), b(s)-b(d)],
                       [b(s)+b(d), a(d)-a(s)]])


def metrics(frequencies, length, quadrature=False):
    if quadrature:
        t, weight = np.polynomial.legendre.leggauss(128)
        t, weight = (t+1)*length/2, weight/2
        basis = [np.column_stack([np.cos(t*x), np.sin(t*x)])
                 for x in frequencies]
        gram = lambda i,j: basis[i].T @ (weight[:,None]*basis[j])
    else:
        gram = lambda i,j: cross_gram(frequencies[i], frequencies[j], length)
    self_gram = [gram(i,i) for i in range(len(frequencies))]
    cosine, full = [], []
    for i in range(len(frequencies)):
        for j in range(i):
            h = gram(i,j)
            full.append(.5*np.trace(np.linalg.solve(self_gram[i],h)
                                   @ np.linalg.solve(self_gram[j],h.T)))
            cosine.append(h[0,0]**2/(self_gram[i][0,0]*self_gram[j][0,0]))
    collision = float(np.mean(full))
    return np.array([np.mean(cosine), collision,
                     2*len(frequencies)/(1+(len(frequencies)-1)*collision)])


def main():
    cases = [
        ("phase_A", [1,5/8,3/8,1/4], 8*np.pi),
        ("phase_B", [1,6.01/8,4.01/8,1/4], 8*np.pi),
    ]
    for factor in [1,2,4]:
        for name, values in [("A",[1,.99,.02,.01]), ("B",[1,2/3,1/3,.01])]:
            cases.append((f"length_{factor}_{name}", values, factor*2*np.pi))
    result = {}
    for name, values, length in cases:
        analytic, numerical = metrics(values,length), metrics(values,length,True)
        error = float(np.max(np.abs(analytic-numerical)))
        assert error < 1e-11
        result[name] = {"frequencies":values,"uniform_interval_length":length,
                        "cosine_collision":float(analytic[0]),
                        "full_collision":float(analytic[1]),
                        "renyi2_rank":float(analytic[2]),
                        "max_quadrature_discrepancy":error}
    assert result["phase_A"]["cosine_collision"] < 1e-25
    assert result["phase_B"]["cosine_collision"] > 1e-5
    assert result["phase_A"]["renyi2_rank"] < result["phase_B"]["renyi2_rank"]
    assert result["length_1_A"]["full_collision"] < result["length_1_B"]["full_collision"]
    for factor in [2,4]:
        assert result[f"length_{factor}_A"]["full_collision"] > result[f"length_{factor}_B"]["full_collision"]
    path = Path(__file__).with_name("explicit_geometry_examples.json")
    path.write_text(json.dumps({"computation":"Closed-form basis geometry, checked by independent quadrature.",
                               "cases":result},indent=2)+"\n")
    print(f"Verified {len(cases)} basis calculations; maximum discrepancy "
          f"{max(x['max_quadrature_discrepancy'] for x in result.values()):.3g}.")


if __name__ == "__main__":
    main()
