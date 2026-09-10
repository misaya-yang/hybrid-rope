"""CPU algebra check: phase-consistent reads need not give schedule-consistent states.

This is an untrained, small causal attention stack, not a Jet-Long model run or
an accuracy benchmark. All cached keys are stored BEFORE rotation, so the old
mixed-rotation-cache issue is absent by construction. No model downloads.
"""
import json
import math
import random


def mv(w, x):
    return [sum(a * b for a, b in zip(row, x)) for row in w]


def normalize(x):
    s = math.sqrt(sum(a * a for a in x) / len(x) + 1e-6)
    return [a / s for a in x]


def rotate(x, p):
    out = []
    for j, omega in enumerate((1.0, 0.13)):
        a, b = x[2 * j:2 * j + 2]
        c, s = math.cos(omega * p), math.sin(omega * p)
        out.extend((a * c - b * s, a * s + b * c))
    return out


def softmax(xs):
    ys = [math.exp(x - max(xs)) for x in xs]
    return [y / sum(ys) for y in ys]


def run(tokens, weights, chunk_ends, mode, window=4, local=1):
    assert chunk_ends[-1] == len(tokens)
    assert all(a < b for a, b in zip([0] + chunk_ends, chunk_ends))
    caches = [[] for _ in weights]
    states = [[] for _ in weights]
    for t, token in enumerate(tokens):
        end = next(e for e in chunk_ends if e > t)
        horizon = {"call": end, "row": t + 1, "fixed": len(tokens)}[mode]
        group = max(1, math.ceil(horizon / window))
        x = token[:]
        for layer, (wq, wk, wv, wo) in enumerate(weights):
            z = normalize(x)
            q, k, v = mv(wq, z), mv(wk, z), mv(wv, z)
            caches[layer].append((k, v))
            logits = []
            for s, (key, _) in enumerate(caches[layer]):
                qp, kp = (t, s) if t - s <= local else (t // group, s // group)
                qr, kr = rotate(q, qp), rotate(key, kp)
                logits.append(sum(a * b for a, b in zip(qr, kr)) / 2)
            probs = softmax(logits)
            out = [sum(p * entry[1][d] for p, entry in zip(probs, caches[layer]))
                   for d in range(4)]
            update = mv(wo, out)
            x = [a + 0.6 * b for a, b in zip(x, update)]
            states[layer].append(x[:])
    return states, caches


def distance(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def main():
    rng = random.Random(19)
    tokens = [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(12)]
    weights = [[[[rng.uniform(-0.5, 0.5) for _ in range(4)]
                 for _ in range(4)] for _ in range(4)] for _ in range(2)]
    partitions = [[12], [4, 8, 12], list(range(1, 13)), [3, 7, 9, 12]]
    report = {"kind": "untrained_cpu_counterexample", "seed": 19,
              "native_window": 4, "length": 12, "local_window": 1,
              "partitions": partitions, "modes": {}}
    for mode in ("call", "row", "fixed"):
        runs = [run(tokens, weights, p, mode) for p in partitions]
        reference = runs[0][0]
        last_errors = [[distance(reference[layer][-1], states[layer][-1])
                        for layer in range(2)] for states, _ in runs]
        all_errors = [max(distance(a, b) for la, lb in zip(reference, states)
                          for a, b in zip(la, lb)) for states, _ in runs]
        report["modes"][mode] = {"last_hidden_errors_by_layer": last_errors,
                                  "max_all_hidden_errors": all_errors}
        if mode in ("row", "fixed"):
            assert max(all_errors) < 1e-12
        else:
            assert max(e[0] for e in last_errors) < 1e-12
            assert max(e[1] for e in last_errors) > 1e-5
            # First-layer raw K/V are identical; second-layer V already differs.
            for layer in range(2):
                err = max(distance(a[1], b[1])
                          for a, b in zip(runs[0][1][layer], runs[1][1][layer]))
                report[f"call_layer_{layer + 1}_value_cache_error"] = err
    # Appending tokens changes old states only for the call-wide horizon policy.
    for mode in ("call", "row"):
        short = run(tokens[:4], weights, [4], mode)[0]
        long = run(tokens, weights, [12], mode)[0]
        err = max(distance(a, b) for la, lb in zip(short, long)
                  for a, b in zip(la, lb[:4]))
        report[f"{mode}_prefix_extension_error"] = err
        assert (err < 1e-12) if mode == "row" else (err > 1e-5)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
