"""Dependency-light contracts for single-table generation experiments.

These metrics do not replace official benchmark scores. Exact strings are
scored only when a task declares a complete answer, not a bag of references.
"""
from __future__ import annotations

import math
import random


def stable_teacher_kl(student_logits,teacher_logits):
    """KL value with its analytic first-order gradient, exactly zero at identity.

    Autodiff through -sum(p*log_softmax(q)) can retain a normalization-roundoff
    gradient when p=q. Adam's small epsilon amplifies it. The exact gradient
    with respect to student logits is softmax(student)-softmax(teacher).
    This first-order implementation is for Adam/SGD, not Hessian computation.
    """
    logp=teacher_logits.log_softmax(-1);logq=student_logits.log_softmax(-1)
    value=(logp.exp()*(logp-logq)).sum(-1).mean()
    delta=(student_logits.softmax(-1)-teacher_logits.softmax(-1)).detach()
    surrogate=(delta*student_logits).sum(-1).mean()
    return value.detach()+(surrogate-surrogate.detach())


def token_exact_eos(generated, answer_tokens, eos_token_id):
    """No prefix, substring, normalization, or first-number credit."""
    if not answer_tokens or eos_token_id in answer_tokens:
        raise ValueError("answer must be nonempty and exclude EOS")
    return list(generated) == [*answer_tokens, eos_token_id]


def retention_verdict(native_nll, candidate_nll, native_task, candidate_task):
    values = (native_nll, candidate_nll, native_task, candidate_task)
    if not all(math.isfinite(v) for v in values) or native_task <= 0:
        raise ValueError("finite metrics and resolving Native task control required")
    if not 0 <= candidate_task <= 1 or not 0 < native_task <= 1:
        raise ValueError("task scores must lie in [0,1]")
    delta = candidate_nll - native_nll
    ppl = math.exp(min(700.0, -delta))
    task = candidate_task / native_task
    return {"ppl_retention": ppl, "task_retention": task,
            "strict_088_pass": ppl >= .88 and task >= .88,
            "historical_0875_pass": ppl >= .875 and task >= .875,
            "decision": "PASS" if min(ppl, task) >= .88 else
                        "MARGINAL" if min(ppl, task) >= .875 else "STOP"}


def projection_parameter_count(config, modules, rank):
    """Standard bias-free Q/K/V/O PEFT count, including GQA dimensions."""
    h = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv = int(config.get("num_key_value_heads", heads))
    dim = int(config.get("head_dim", h // heads))
    shapes = {"q_proj": (h, heads * dim), "k_proj": (h, kv * dim),
              "v_proj": (h, kv * dim), "o_proj": (heads * dim, h)}
    if "intermediate_size" in config:
        intermediate = int(config["intermediate_size"])
        shapes.update(gate_proj=(h, intermediate), up_proj=(h, intermediate), down_proj=(intermediate, h))
    if rank <= 0 or not modules or len(set(modules)) != len(modules):
        raise ValueError("invalid rank or modules")
    return int(config["num_hidden_layers"]) * rank * sum(sum(shapes[m]) for m in modules)


def matched_projection_rank(config, modules, qk_rank=64):
    target = projection_parameter_count(config, ("q_proj", "k_proj"), qk_rank)
    per_rank = projection_parameter_count(config, modules, 1)
    rank = max(1, round(target / per_rank))
    actual = per_rank * rank
    if abs(actual / target - 1) > .01:
        raise ValueError("integer uniform rank cannot match QK within 1%; declare a new budget")
    return rank, actual


def paired_retention_intervals(native, candidate, *, resamples=1000, seed=20260904, nll_task='pg19'):
    """Within-task SOURCE-GROUP bootstrap, paired arms, fixed checkpoint/tasks."""
    def index(rows):
        result = {}
        for row in rows:
            key = (row["task"], row["asset_sha256"])
            if key in result:
                raise ValueError("duplicate paired retention row")
            result[key] = row
        return result
    left, right = index(native), index(candidate)
    if left.keys() != right.keys() or not left:
        raise ValueError("unpaired retention bootstrap")
    grouped = {}
    for key, row in left.items():
        if row["group"] != right[key]["group"]:
            raise ValueError("source-group drift")
        grouped.setdefault(row["task"], {}).setdefault(row["group"], []).append(key)
    if nll_task not in grouped or len(grouped) < 2 or resamples < 2:
        raise ValueError("NLL and task groups required")
    rng = random.Random(seed)
    samples = {"ppl": [], "task": [], "task_eos": []}
    for _ in range(resamples):
        task_a, task_b, eos_a, eos_b = [], [], [], []
        for task, groups in sorted(grouped.items()):
            group_names = sorted(groups)
            keys = [key for _ in group_names for key in groups[rng.choice(group_names)]]
            if task == nll_task:
                delta = sum(right[k]["nll"] - left[k]["nll"] for k in keys)/len(keys)
                samples["ppl"].append(math.exp(min(700., -delta)))
            else:
                task_a.append(sum(left[k]["score"] for k in keys)/len(keys))
                task_b.append(sum(right[k]["score"] for k in keys)/len(keys))
                eos_a.append(sum(left[k]["score_eos"] for k in keys)/len(keys))
                eos_b.append(sum(right[k]["score_eos"] for k in keys)/len(keys))
        if sum(task_a) <= 0 or sum(eos_a) <= 0:
            return {"status": "UNRESOLVED_ZERO_BOOTSTRAP_DENOMINATOR", "resamples": resamples}
        samples["task"].append(sum(task_b)/sum(task_a))
        samples["task_eos"].append(sum(eos_b)/sum(eos_a))
    intervals = {}
    for name, values in samples.items():
        ordered = sorted(values)
        intervals[name] = [ordered[int(.025*(resamples-1))], ordered[int(.975*(resamples-1))]]
    return {"status": "PAIRED_SOURCE_GROUP_BOOTSTRAP", "ci95": intervals,
            "all_lower_bounds_ge_088": all(v[0] >= .88 for v in intervals.values()),
            "groups_per_task": {t: len(g) for t,g in grouped.items()},
            "resamples": resamples, "seed": seed,
            "scope": "conditional on this checkpoint and fixed tasks; not training-seed uncertainty"}


def kl_argmax_radius(probabilities):
    """Minimum forward KL to change a unique teacher top-1 (ties: radius 0).

    A pointwise preservation certificate, not a correctness or mean-KL guarantee.
    """
    p = [float(value) for value in probabilities]
    if len(p) < 2 or any(not math.isfinite(v) or v < 0 for v in p) or abs(sum(p)-1.) > 1e-9:
        raise ValueError("a normalized finite probability distribution is required")
    a, b = sorted(p, reverse=True)[:2]
    if a == b:
        return 0.
    return a * math.log(2*a/(a+b)) + (b * math.log(2*b/(a+b)) if b else 0.)
