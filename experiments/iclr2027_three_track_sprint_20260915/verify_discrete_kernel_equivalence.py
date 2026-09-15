#!/usr/bin/env python3
"""CPU examples for the no-alias integer-position kernel corollary; no model runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def rotation(frequencies, distance):
    matrix = np.zeros((2 * len(frequencies), 2 * len(frequencies)))
    for k, omega in enumerate(frequencies):
        c, s = np.cos(omega * distance), np.sin(omega * distance)
        matrix[2*k:2*k+2, 2*k:2*k+2] = [[c, -s], [s, c]]
    return matrix


def verify():
    omega = np.array([.13, .7, .7, 1.9])
    perm = np.array([3, 1, 0, 2])
    coordinates = np.array([[2*k, 2*k+1] for k in perm]).ravel()
    p = np.eye(8)[coordinates]
    residual = max(float(np.max(np.abs(p @ rotation(omega, d) @ p.T
                                      - rotation(omega[perm], d))))
                   for d in [-31, -1, 0, 1, 13, 1024])
    assert residual < 1e-12
    # Equal trace does not imply equal full rotation spectrum.
    left, right = np.arccos([.9, .1]), np.arccos([.7, .3])
    trace1 = abs(np.trace(rotation(left, 1)) - np.trace(rotation(right, 1)))
    trace2 = abs(np.trace(rotation(left, 2)) - np.trace(rotation(right, 2)))
    assert trace1 < 1e-12 and trace2 > .1
    alias = omega + 2*np.pi
    alias_error = max(float(np.max(np.abs(rotation(omega, d)-rotation(alias, d))))
                      for d in [-17, 0, 1, 8])
    fractional_error = float(np.max(np.abs(rotation(omega, .5)-rotation(alias, .5))))
    assert alias_error < 1e-12 and fractional_error > 1
    def softmax(x):
        e = np.exp(x - np.max(x)); return e/e.sum()
    logits = np.array([-.8, .2, 1.3])
    target = 1.7*logits + 4
    assert np.allclose(softmax(target), softmax(1.7*logits))
    nonaffine = np.array([-.5, .4, .8])
    fitted = np.linalg.lstsq(np.column_stack([logits, np.ones(3)]), nonaffine, rcond=None)[0]
    nonaffine_error = float(np.linalg.norm(nonaffine-(fitted[0]*logits+fitted[1])))
    assert nonaffine_error > .1
    # Necessary assumption: d=1 alone permits arbitrary two-sided maps.
    a, b = rotation(left, 1), rotation(right, 1)
    fixed_b = a.T @ b
    assert np.allclose(a @ fixed_b, b) and not np.allclose(fixed_b, np.eye(4))
    return {"status": "PASS", "evidence_role": "CPU mathematical examples, not model performance",
            "checks": {"permutation_with_repeated_frequencies_max_error": residual,
                       "same_trace_one_step_error": float(trace1), "different_trace_two_step_gap": float(trace2),
                       "integer_alias_error": alias_error, "fractional_alias_failure": fractional_error,
                       "nonaffine_gain_residual": nonaffine_error, "zero_distance_assumption_necessary": True},
            "assumptions": ["frequencies in (0, pi)", "full bilinear kernel for all content vectors",
                            "invertible position-independent linear Q/K maps", "distances include 0 and 1"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
