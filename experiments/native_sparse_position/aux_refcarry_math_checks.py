"""CPU-only mathematical checks for the user-supplied RefCarry proposal.

These are finite algebraic witnesses, not pretrained-model capability evidence.
Run: python3 experiments/native_sparse_position/aux_refcarry_math_checks.py
"""

import cmath
import json
import math
import random


def softmax(xs):
    values = [math.exp(x - max(xs)) for x in xs]
    return [x / sum(values) for x in values]


def inner(xs, ys):
    return sum((x.conjugate() * y).real for x, y in zip(xs, ys))


def main():
    rng = random.Random(20260909)
    maximum_factorization_error = 0.0
    for _ in range(100):
        frequencies = [1.0, 0.1, 0.01]
        positions = [0, 3, 17, 41]
        weights = [rng.random() for _ in positions]
        weights = [w / sum(weights) for w in weights]
        query = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in frequencies]
        key = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in frequencies]
        target = 43
        expected_logit = sum(
            w * inner(query, [cmath.exp(1j * f * (target - a)) * k
                              for f, k in zip(frequencies, key)])
            for a, w in zip(positions, weights)
        )
        moments = [sum(w * cmath.exp(1j * f * a) for a, w in zip(positions, weights))
                   for f in frequencies]
        factored_logit = inner(
            [m * q for m, q in zip(moments, query)],
            [cmath.exp(1j * f * target) * k for f, k in zip(frequencies, key)],
        )
        maximum_factorization_error = max(maximum_factorization_error,
                                          abs(expected_logit - factored_logit))
    assert maximum_factorization_error < 1e-12

    # One RoPE pair, frequency pi/2, anchor positions 0 and 1, q=1.
    # These are post-RoPE keys. Raw keys exist by applying the inverse rotation.
    post_rotary_keys = [complex(10, -10), complex(-10, 10), complex(1, 1)]
    anchor_queries = [1 + 0j, 1j]
    logits = [[inner([q], [k]) for k in post_rotary_keys] for q in anchor_queries]
    individual_reads = [softmax(row) for row in logits]
    average_logits = [sum(row[j] for row in logits) / 2 for j in range(3)]
    moment_read = softmax(average_logits)
    mixture_read = [sum(row[j] for row in individual_reads) / 2 for j in range(3)]
    geometric_read = [math.sqrt(individual_reads[0][j] * individual_reads[1][j])
                      for j in range(3)]
    geometric_read = [p / sum(geometric_read) for p in geometric_read]
    assert [max(range(3), key=row.__getitem__) for row in individual_reads] == [0, 1]
    assert max(range(3), key=moment_read.__getitem__) == 2
    assert max(abs(x - y) for x, y in zip(moment_read, geometric_read)) < 1e-12

    # Identical complete first-harmonic moments, different marginalized reads.
    # Positions 0/2 produce query directions +/-x; 1/3 produce +/-y.
    collision_keys = [4 + 0j, -4 + 0j, 0j]
    collision_moments = []
    collision_reads = []
    for directions in ([1 + 0j, -1 + 0j], [1j, -1j]):
        collision_moments.append(sum(directions) / 2)
        reads = [softmax([inner([q], [k]) for k in collision_keys])
                 for q in directions]
        collision_reads.append([sum(row[j] for row in reads) / 2 for j in range(3)])
    assert collision_moments == [0j, 0j]
    collision_tv = sum(abs(x - y) for x, y in zip(*collision_reads)) / 2
    assert collision_tv > 0.3

    # Probability normalization is known: one real number mu[1] recovers both
    # Fourier coordinates although the uncentered feature matrix has rank two.
    phi0 = complex(1, 0)
    phi1 = cmath.exp(1j)
    simplex_reconstruction_error = 0.0
    for n in range(101):
        mu1 = n / 100
        original = (1 - mu1) * phi0 + mu1 * phi1
        recovered = phi0 + mu1 * (phi1 - phi0)
        simplex_reconstruction_error = max(simplex_reconstruction_error,
                                            abs(original - recovered))
    assert abs(math.sin(1)) > 0.8  # determinant, hence uncentered rank = 2
    assert simplex_reconstruction_error < 1e-12

    # Equal normalization and a pure phase change do not entail movement of
    # the selected physical location when content keys differ across locations.
    fixed_keys = [0j, 0j, 100 + 0j, 1j, 0j, 0j]
    selected_targets = []
    for anchor in [0, 1]:
        q = cmath.exp(0.7j * anchor)
        scores = [inner([q], [k]) for k in fixed_keys]
        selected_targets.append(max(range(len(scores)), key=scores.__getitem__))
    assert selected_targets == [2, 2]

    print(json.dumps({
        "scope": "CPU algebra only; no language-model or generation claim",
        "factorization_trials": 100,
        "maximum_factorization_error": maximum_factorization_error,
        "two_anchor_logits": logits,
        "moment_softmax": moment_read,
        "mixture_of_anchor_softmaxes": mixture_read,
        "third_target_probability_moment": moment_read[2],
        "third_target_probability_mixture": mixture_read[2],
        "same_moment_different_marginal_reads": collision_reads,
        "same_moment_marginal_read_total_variation": collision_tv,
        "uncentered_phi_rank": 2,
        "sufficient_simplex_sketch_dimension": 1,
        "simplex_reconstruction_error": simplex_reconstruction_error,
        "selected_targets_after_unit_anchor_shift": selected_targets,
    }, indent=2))


if __name__ == "__main__":
    main()
