"""Retained P2 result audit and WITHDRAWN historical operator reference.

No model, new frequency table, training, or capability experiment is executed.
Random vectors below check rotary/softmax algebra only; they are not task data.
The near/far proposal was withdrawn by the author on 2026-09-08. Its historical
functions remain for auditing the claim, not as an active research candidate.
"""
from __future__ import annotations

import argparse
import cmath
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "docs/research"
WINDOW = 512
REFERENCE_CAP = 65536


def mean_fraction(values):
    return sum(map(Fraction, values), Fraction()) / len(values)


def rotate_split(values, phases):
    """Independent real 2x2 rotations in Qwen's fixed (j, j+64) slots."""
    half = len(phases)
    result = [0.0] * len(values)
    for j, phase in enumerate(phases):
        c, s = math.cos(phase), math.sin(phase)
        x, y = values[j], values[j + half]
        result[j], result[j + half] = c * x - s * y, s * x + c * y
    return result


def absolute_score(q, k, qpos, kpos, frequencies, gain):
    qr = rotate_split(q, [qpos * w for w in frequencies])
    kr = rotate_split(k, [kpos * w for w in frequencies])
    return gain * gain * sum(x * y for x, y in zip(qr, kr)) / math.sqrt(len(q))


def relative_score(q, k, distance, frequencies, gain):
    half = len(frequencies)
    value = sum(
        (complex(q[j], q[j + half]).conjugate()
         * complex(k[j], k[j + half])
         * cmath.exp(-1j * distance * w)).real
        for j, w in enumerate(frequencies)
    )
    return gain * gain * value / math.sqrt(len(q))


def proposal_score(q, k, qpos, kpos, native, p2, gain, cap, *, cap_remote=True):
    distance = qpos - kpos
    if distance < 0:
        raise ValueError("causal pairs only")
    if distance <= WINDOW:
        return absolute_score(q, k, qpos, kpos, native, gain)
    factor = max(1.0, cap / REFERENCE_CAP) if cap_remote else 1.0
    return absolute_score(q, k, qpos / factor, kpos / factor, p2, gain)


def softmax_read(scores, values):
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    partition = sum(weights)
    output = [sum(w * v[d] for w, v in zip(weights, values)) / partition
              for d in range(len(values[0]))]
    return output, peak + math.log(partition)


def algebra_checks(native, p2, gain):
    rng = random.Random(20260908)
    dim = 2 * len(native)
    q = [rng.gauss(0, 1) for _ in range(dim)]
    k = [rng.gauss(0, 1) for _ in range(dim)]
    errors = {name: 0.0 for name in (
        "complex_vs_absolute", "local_native_at_same_gain", "remote64_original_p2",
        "remote128_doubled_distance_to64", "joint_translation", "shared_softmax_merge")}
    old_local_change = 0.0
    for distance in (1, 31, 127, 511, 512, 513, 1024, 8192, 32768, 65535):
        qpos, kpos = 65535, 65535 - distance
        old = relative_score(q, k, distance, p2, gain)
        direct = absolute_score(q, k, qpos, kpos, p2, gain)
        errors["complex_vs_absolute"] = max(errors["complex_vs_absolute"], abs(old - direct))
        proposed = proposal_score(q, k, qpos, kpos, native, p2, gain, 65536)
        if distance <= WINDOW:
            target = relative_score(q, k, distance, native, gain)
            errors["local_native_at_same_gain"] = max(errors["local_native_at_same_gain"], abs(proposed - target))
            old_local_change = max(old_local_change, abs(old - target))
        else:
            errors["remote64_original_p2"] = max(errors["remote64_original_p2"], abs(proposed - old))
            longer = proposal_score(q, k, 2 * qpos, 2 * kpos, native, p2, gain, 131072)
            errors["remote128_doubled_distance_to64"] = max(errors["remote128_doubled_distance_to64"], abs(longer - old))
        shifted = proposal_score(q, k, qpos + 1000000, kpos + 1000000, native, p2, gain, 65536)
        errors["joint_translation"] = max(errors["joint_translation"], abs(shifted - proposed))

    distances = [1, 20, 100, 512, 513, 2048, 10000, 65536, 131000]
    keys = [[rng.gauss(0, 1) for _ in range(dim)] for _ in distances]
    values = [[rng.gauss(0, 1) for _ in range(7)] for _ in distances]
    scores = [proposal_score(q, key, 131071, 131071 - delta, native, p2, gain, 131072)
              for key, delta in zip(keys, distances)]
    direct_output, _ = softmax_read(scores, values)
    groups = [[i for i, delta in enumerate(distances) if (delta <= WINDOW) == local]
              for local in (True, False)]
    branch_outputs, lses = [], []
    for indices in groups:
        output, lse = softmax_read([scores[i] for i in indices], [values[i] for i in indices])
        branch_outputs.append(output)
        lses.append(lse)
    merged, _ = softmax_read(lses, branch_outputs)
    errors["shared_softmax_merge"] = max(abs(x - y) for x, y in zip(merged, direct_output))
    wrong_merge = [(x + y) / 2 for x, y in zip(*branch_outputs)]
    wrong_merge_error = max(abs(x - y) for x, y in zip(wrong_merge, direct_output))
    if old_local_change <= 1e-4 or wrong_merge_error <= 1e-4:
        raise AssertionError("nontrivial rotary and branch-weight examples required")
    if max(errors.values()) > 1e-8:
        raise AssertionError(errors)
    return dict(scope="Float64 fixed-vector identities only; no model-quality evidence",
                seed=20260908, max_abs_errors=errors,
                nonzero_old_p2_vs_native_local_score_difference=old_local_change,
                incorrect_equal_branch_average_max_abs_error=wrong_merge_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: OWNER / filename for name, filename in (
        ("result", "ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json"),
        ("candidate", "ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json"),
        ("historical", "ROPE_RECOVERED_QWEN_P2_20260907.json"))}
    loaded = {name: json.loads(path.read_text()) for name, path in paths.items()}
    data, candidate, historical = (loaded[name] for name in ("result", "candidate", "historical"))
    native = historical["native_float32"]
    p2, mr = (candidate["tables"][name] for name in ("FullLagP2", "MrPro"))
    if not len(native) == len(p2["values_float32"]) == len(mr["values_float32"]) == 64:
        raise ValueError("actual Qwen 64-slot arrays required")
    first = {cell["task"]: cell for cell in data["summary64"]}
    aligned = []
    for cell in data["checkpoint_transfer_3b64"]["summary"]:
        old = first[cell["task"]]
        differences = dict(zip(old["row_ids"], old["paired"]["mrpro"]["differences_fraction"]))
        old_values = [differences[row_id] for row_id in cell["row_ids"]]
        new_values = cell["paired"]["differences_fraction"]
        aligned.append(dict(task=cell["task"], row_ids=cell["row_ids"],
                            p2_minus_mr_1p5b_pp=float(100 * mean_fraction(old_values)),
                            p2_minus_mr_3b_pp=float(100 * mean_fraction(new_values)),
                            differences_1p5b=old_values, differences_3b=new_values))
    frequencies = p2["values_float32"]
    geometry = []
    for j in range(23, 41):
        wm, wp, wn = mr["values_float32"][j], frequencies[j], native[j]
        geometry.append(dict(slot=j, native_turns_at32768=wn * 32768 / (2 * math.pi),
                             p2_exponent=math.log(wn / wp, 4), mr_exponent=math.log(wn / wm, 4),
                             p2_over_mr_frequency=wp / wm,
                             signed_p2_minus_mr_phase_at65536=(wp - wm) * 65536))
    output = dict(
        status="RETAINED_RESULT_ANALYSIS_WITH_WITHDRAWN_OPERATOR_REFERENCE",
        inputs={name: {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in paths.items()},
        model=data["model"], model_revision=data["model_revision"],
        alignment_scope="Same row IDs in the retained paired owner; remote prompt tensors not reread in this CPU analysis",
        aligned_64k=aligned, geometry=geometry,
        gains=dict(p2=p2["gain"], mr=mr["gain"],
                   p2_over_mr_logit_multiplier=(p2["gain"] / mr["gain"]) ** 2),
        tail40plus_arrays_equal=frequencies[40:] == mr["values_float32"][40:],
        proposal=dict(status="WITHDRAWN_BY_AUTHOR_NOT_FOR_IMPLEMENTATION_OR_EXPERIMENT",
                      local_window=WINDOW, reference_request_cap=REFERENCE_CAP,
                      local="Native frequencies at original relative distance",
                      remote="Frozen P2 frequencies at distance / max(1, frozen_request_cap / 65536)",
                      gain=p2["gain"], one_common_softmax=True,
                      phase_cap_is_development_informed=True,
                      model_execution="NOT_IMPLEMENTED_OR_RUN", weights="FROZEN"),
        algebra=algebra_checks(native, frequencies, p2["gain"]),
        limits=["No uniquely identified cause of the model scores",
                "Operator equality conditions on fixed Q/K; longer inputs change states and key count",
                "Local Native angles do not imply whole-model Native behavior or Native gain",
                "This does not estimate a probability of success or establish new prior art"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "aligned_64k": aligned,
                      "gains": output["gains"], "tail40plus_arrays_equal": output["tail40plus_arrays_equal"],
                      "algebra": output["algebra"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
