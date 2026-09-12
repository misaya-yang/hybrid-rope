"""The H-stage verdict: did the locked candidate upgrade over MR?

Plan sections 0.2, 6.4, 6.5, 7 and 9.2.

Three contrasts, all on the same frozen H rows:

    G_new      = A(M) - A(MR)                       (does it beat the incumbent)
    G_old      = A(MR) - A(YARN)                    (what the last upgrade was worth)
    C_upgrade  = A(M) - 2 A(MR) + A(YARN) = G_new - G_old

The plan requires G_new > 0 **and** C_upgrade >= 0, so that a case where YaRN
happens to beat MR cannot be parleyed into a low bar.  Both get a joint 95%
lower bound from the same source-clustered paired bootstrap.

This tool does not score anything and does not pick a winner.  It assumes the
JSONL it is handed was already scored by the frozen RULER scorer, and it refuses
to run at all unless every arm covers exactly the frozen expected-data index.

Usage (plan section 9.2):

    python upgrade_report.py --mr results/H/MR.jsonl \
        --candidate results/H/LOCKED.jsonl --yarn results/H/YARN_INDEX.jsonl \
        --expected-data data/H.jsonl --out upgrade_H.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import paired_report as P

# plan section 0.2
PRACTICAL_GAIN = 0.03            # >= 3pp macro gain over MR: worth RECORDING
HERO_GAIN = 0.05                 # >= 5pp over the BEST baseline: worth locking H
RETENTION_TARGET = 0.88          # 8K retention
STRICT_TOLERANCE = -0.02         # strict whole-question, vs MR
QA_TOLERANCE = -0.03             # natural QA
LENGTH_TOLERANCE = -0.05         # any long length vs MR
DISASTER_RETENTION = 0.65        # plan section 6.3
DISASTER_FORMAT_FAIL = 0.50


def _load(path):
    return P.load_jsonl(path) if path else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mr", required=True, help="H rows for MrRoPE-Pro")
    ap.add_argument("--candidate", required=True, help="H rows for the single locked candidate")
    ap.add_argument("--yarn", required=True, help="H rows for official YaRN (index ramp)")
    ap.add_argument("--bm", default=None, help="optional H rows for BM")
    ap.add_argument("--native", default=None,
                    help="optional native rows; required for the 8K retention gate")
    ap.add_argument("--strict", default=None,
                    help="optional whole-question strict rows for the candidate")
    ap.add_argument("--strict-mr", default=None, help="strict rows for MR")
    ap.add_argument("--qa", default=None, help="optional natural-QA rows for the candidate")
    ap.add_argument("--qa-mr", default=None, help="natural-QA rows for MR")
    ap.add_argument("--expected-data", required=True,
                    help="the frozen H row index; arms must match it exactly")
    ap.add_argument("--lengths", default="16384,32768",
                    help="the two long lengths the main statistic averages")
    ap.add_argument("--guard-length", type=int, default=8192)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    long_lengths = [int(x) for x in a.lengths.split(",") if x.strip()]

    expected = _load(a.expected_data)
    arms = {"MR": P.index_arm(_load(a.mr), "MR"),
            "M": P.index_arm(_load(a.candidate), "M"),
            "YARN": P.index_arm(_load(a.yarn), "YARN")}
    if a.bm:
        arms["BM"] = P.index_arm(_load(a.bm), "BM")

    try:
        P.require_aligned(arms, expected)
    except ValueError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2

    # -- accuracy table ----------------------------------------------------
    acc = {k: P.macro_accuracy(list(v.values())) for k, v in arms.items()}

    # Native is a separate 8K reference, not a member of the long H panel.
    # Requiring it to contain the H rows made the intended short-only run
    # impossible and also polluted the bootstrap identity set.
    native = None
    if a.native:
        native_all = P.index_arm(_load(a.native), "native")
        guard_expected = [r for r in expected if P._length(r) == a.guard_length]
        if not guard_expected:
            print("REFUSING: frozen expected-data has no guard-length rows for native", file=sys.stderr)
            return 2
        try:
            # Some legacy runners emitted the whole H index for native.  Keep
            # only the guard rows after validating that the complete guard is
            # present; long native rows are irrelevant to retention and must
            # never enter the long statistic.
            native = {rid: r for rid, r in native_all.items()
                      if P._length(r) == a.guard_length}
            P.require_aligned({"native": native}, guard_expected)
        except ValueError as exc:
            print(f"REFUSING: native guard is not aligned to frozen guard index: {exc}",
                  file=sys.stderr)
            return 2

    # -- contrasts ---------------------------------------------------------
    #
    # G_new > 0 and C_upgrade >= 0 together do NOT imply A(M) > A(YARN).  The
    # counterexample is exact arithmetic: A(YARN)=10, A(MR)=8, A(M)=9 gives
    # G_new = +1 > 0 and C_upgrade = 9-16+10 = +3 >= 0, yet the candidate loses
    # to YaRN.  With BM in the H table the same hole is wider.  So the baseline
    # that a candidate must clear is the MAXIMUM over the comparators, not MR.
    comparators = [n for n in ("MR", "YARN", "BM") if n in arms]
    contrasts = [
        {"name": "G_new", "terms": [("M", +1.0), ("MR", -1.0)], "lengths": long_lengths},
        {"name": "G_old", "terms": [("MR", +1.0), ("YARN", -1.0)], "lengths": long_lengths},
        {"name": "C_upgrade", "terms": [("M", +1.0), ("MR", -2.0), ("YARN", +1.0)],
         "lengths": long_lengths},
        {"name": "G_vs_yarn", "terms": [("M", +1.0), ("YARN", -1.0)], "lengths": long_lengths},
    ]
    if "BM" in arms:
        contrasts.append({"name": "G_vs_bm", "terms": [("M", +1.0), ("BM", -1.0)],
                          "lengths": long_lengths})
    contrasts.append({"name": "G_vs_best", "kind": "max_of", "subject": "M",
                      "comparators": comparators, "lengths": long_lengths})
    try:
        boot = P.cluster_bootstrap(
            arms, contrasts, n_boot=a.n_boot, seed=a.seed,
            expected_rows=expected, required_tasks=sorted({P._task(r) for r in expected
                                                            if P._length(r) in long_lengths}),
            required_lengths=long_lengths)
    except ValueError as exc:
        print(f"REFUSING: bootstrap panel is invalid: {exc}", file=sys.stderr)
        return 2

    # -- the main statistic: equal-weight macro over the two long lengths ---
    required_tasks = sorted({P._task(r) for r in expected
                             if P._length(r) in long_lengths})
    try:
        d_long = P.paired_stats(
            list(arms["MR"].values()), list(arms["M"].values()),
            use_lengths=set(long_lengths), expected_rows=expected,
            required_tasks=required_tasks)
        d_mr_yarn = P.paired_stats(
            list(arms["YARN"].values()), list(arms["MR"].values()),
            use_lengths=set(long_lengths), expected_rows=expected,
            required_tasks=required_tasks)
    except ValueError as exc:
        print(f"REFUSING: long-panel statistic is invalid: {exc}", file=sys.stderr)
        return 2

    # -- guard: 8K retention ----------------------------------------------
    retention = None
    if native is not None:
        nat = P.macro_accuracy(list(native.values()), required_tasks=required_tasks,
                               required_lengths=[a.guard_length])
        cand = P.macro_accuracy(list(arms["M"].values()))
        n8, c8 = nat["by_length"].get(a.guard_length), cand["by_length"].get(a.guard_length)
        if n8 is not None and c8 is not None:
            retention = {"native": n8, "candidate": c8,
                         "ratio": (c8 / n8) if n8 > 0 else float("nan"),
                         "target": RETENTION_TARGET}

    # -- guards: strict and QA --------------------------------------------
    def _guard(cand_path, mr_path, tol, key, score_keys):
        if not cand_path:
            return None
        c = P.index_arm(_load(cand_path), "cand")
        rows_c = list(c.values())
        if mr_path:
            m = P.index_arm(_load(mr_path), "mr")
            try:
                P.require_aligned({"cand": c, "mr": m})
            except ValueError as exc:
                raise ValueError(f"{key} guard arms are not aligned: {exc}") from exc
        else:
            m = None
        metrics = {}
        for score_key in score_keys:
            acc_c = P.macro_accuracy(rows_c, score_key=score_key)["macro"]
            item = {"candidate_macro": acc_c, "tolerance": tol, "passed": None}
            if m is not None:
                item["mr_macro"] = P.macro_accuracy(
                    list(m.values()), score_key=score_key)["macro"]
                item["delta"] = acc_c - item["mr_macro"]
                item["passed"] = bool(item["delta"] >= tol)
            metrics[score_key] = item
        # Keep the historical top-level fields for consumers, while exposing
        # both dedicated QA metrics and requiring all selected metrics to pass.
        first = metrics[score_keys[0]]
        out = {**first, "metrics": metrics,
               "score_keys": list(score_keys),
               "passed": (None if m is None else all(x["passed"] for x in metrics.values()))}
        return out

    strict = _guard(a.strict, a.strict_mr, STRICT_TOLERANCE, "strict",
                    ["strict_score"])
    qa = _guard(a.qa, a.qa_mr, QA_TOLERANCE, "qa", ["qa_em", "qa_f1"])

    # -- verdict (plan section 6.5, with the section 0.2 baseline fixed) ---
    g_new = boot["contrasts"]["G_new"]
    c_up = boot["contrasts"]["C_upgrade"]
    g_best = boot["contrasts"]["G_vs_best"]
    g_yarn = boot["contrasts"]["G_vs_yarn"]
    g_bm = boot["contrasts"].get("G_vs_bm")
    both_lengths_nonneg = all(v >= 0 for v in d_long["by_length"].values())

    gates = {
        "gain_at_least_3pp_vs_MR": bool(d_long["delta_macro"] >= PRACTICAL_GAIN),
        "both_lengths_non_negative": bool(both_lengths_nonneg),
        "G_new_lower_bound_positive": bool(g_new["lower_simultaneous_95"] > 0),
        "C_upgrade_lower_bound_non_negative": bool(c_up["lower_simultaneous_95"] >= 0),
        "beats_yarn_point": bool(g_yarn["point"] > 0),
        "beats_bm_point": None if g_bm is None else bool(g_bm["point"] > 0),
        "beats_best_baseline_lower_bound_positive": bool(g_best["lower_simultaneous_95"] > 0),
        "hero_gain_at_least_5pp_over_best": bool(g_best["point"] >= HERO_GAIN),
        "gain_at_least_5pp_at_32K": None,
        "retention_8K": None,
        "strict": None if strict is None else strict["passed"],
        "qa": None if qa is None else qa["passed"],
    }
    if 32768 in d_long["by_length"]:
        gates["gain_at_least_5pp_at_32K"] = bool(d_long["by_length"][32768] >= HERO_GAIN)
    if retention is not None:
        gates["retention_8K"] = bool(retention["ratio"] >= RETENTION_TARGET)

    # an interval that still admits both "useful gain" and "no gain"
    underpowered = bool(
        d_long["delta_macro"] - P.Z_ALPHA_2 * d_long["se_paired"] < PRACTICAL_GAIN
        and d_long["delta_macro"] + P.Z_ALPHA_2 * d_long["se_paired"] > 0.0
    )
    # The core is now "beats every baseline", not "beats MR by the historical
    # increment".  A candidate that loses to YaRN or BM is not an upgrade no
    # matter how it does against MR.
    core = [gates["beats_best_baseline_lower_bound_positive"]]
    failed_hard = (g_best["point"] + P.Z_ALPHA_2 * g_best["se"] < -PRACTICAL_GAIN
                   if np.isfinite(g_best["se"]) and g_best["se"] > 0 else False)

    guards_unevaluated = [k for k in ("retention_8K", "strict", "qa") if gates.get(k) is None]
    if "BM" not in arms:
        guards_unevaluated.append("beats_BM (no --bm rows supplied)")

    if failed_hard or gates["strict"] is False or gates["qa"] is False:
        verdict = "FAIL"
    elif all(core) and not guards_unevaluated:
        verdict = "PASS"
    elif all(core) and guards_unevaluated:
        verdict = "INCOMPLETE_GUARDS"
    elif underpowered and not any(core):
        verdict = "UNDERPOWERED"
    else:
        verdict = "FAIL"

    hero_result = bool(gates["beats_best_baseline_lower_bound_positive"]
                       and gates["hero_gain_at_least_5pp_over_best"])

    report = {
        "verdict": verdict,
        "hero_result": hero_result,
        "baseline_that_must_be_beaten": comparators,
        "why_not_MR_only": (
            "G_new > 0 and C_upgrade >= 0 do not imply A(M) > A(YARN). With "
            "A(YARN)=10, A(MR)=8, A(M)=9 both hold (G_new=+1, C_upgrade=+3) while the "
            "candidate loses to YaRN. The binding comparison is therefore against "
            "max(MR, YARN, BM), whose lower bound must be strictly positive."),
        "thresholds": {
            "record": PRACTICAL_GAIN,
            "hero": HERO_GAIN,
            "note": ("3pp is worth recording but is close to the measurement floor for "
                     "this family; only >= 5pp over the best baseline justifies spending "
                     "V/H budget, per the 2026-09-11 correction"),
        },
        "verdict_definitions": {
            "PASS": "the section 0.2 main gain and every gate requirement are supported by evidence",
            "FAIL": "the interval excludes >= 3pp practical gain, or a key capability is damaged",
            "UNDERPOWERED": "the interval still admits both a useful gain and no gain",
            "INCOMPLETE_GUARDS": ("the main gain is supported but at least one section 0.2 gate was "
                                  "not measured. Plan section 6.5 does not define this case; the "
                                  "label is this tool's, not the plan's, and must not be reported "
                                  "as PASS"),
        },
        "guards_unevaluated": guards_unevaluated,
        "plan_gap_note": (
            "plan section 6.5 lists three outcomes; 'a gate was never measured' is not among "
            "them. Reporting such a run as PASS would claim evidence that does not exist, and "
            "reporting it as FAIL would claim a violation that was not observed."),

        "main_statistic": {
            "definition": "half of [A_16(M)-A_16(MR) + A_32(M)-A_32(MR)], equal-task macro",
            "lengths": long_lengths,
            **d_long,
        },
        "yarn_vs_mr": d_mr_yarn,
        "contrasts": boot,
        "accuracies": acc,
        "retention_8K": retention,
        "strict_guard": strict,
        "qa_guard": qa,
        "gates": gates,
        "constants": {
            "record_gain_vs_MR": PRACTICAL_GAIN, "hero_gain_over_best": HERO_GAIN,
            "retention_target": RETENTION_TARGET, "strict_tolerance": STRICT_TOLERANCE,
            "qa_tolerance": QA_TOLERANCE,
        },
        "power_note": ("S is a screen, not a result (plan section 6.3): at the S sample "
                       "sizes a 3pp ranking is very unreliable.  This report is for H."),
        "caveats": [],
    }
    if retention is None:
        report["caveats"].append(
            "no --native rows supplied, so the 8K retention gate was not evaluated; "
            "the plan does not let it default to pass")
    if strict is None:
        report["caveats"].append("no --strict rows supplied; strict guard not evaluated")
    if qa is None:
        report["caveats"].append("no --qa rows supplied; natural-QA guard not evaluated")
    if boot["rejection_fraction"] > 0.05 if np.isfinite(boot["rejection_fraction"]) else False:
        report["caveats"].append(
            f"{boot['rejection_fraction']:.1%} of bootstrap resamples left an empty cell; "
            "the interval approximation needs review (plan section 9.2)")

    Path(a.out).write_text(json.dumps(report, indent=2, default=float))
    print(f"verdict: {verdict}   hero_result: {hero_result}")
    print(f"  must beat: max({', '.join(comparators)})")
    print(f"  delta vs MR over {long_lengths}: {d_long['delta_macro']:+.4f} "
          f"(se {d_long['se_paired']:.4f}, n={d_long['n_pairs']})")
    print(f"  by length: " + ", ".join(f"{k}: {v:+.4f}" for k, v in d_long["by_length"].items()))
    print(f"  G_vs_best  = {g_best['point']:+.4f}  lower95 {g_best['lower_simultaneous_95']:+.4f}  <-- binding")
    print(f"  G_new      = {g_new['point']:+.4f}  lower95 {g_new['lower_simultaneous_95']:+.4f}")
    print(f"  C_upgrade  = {c_up['point']:+.4f}  lower95 {c_up['lower_simultaneous_95']:+.4f}")
    for k, v in gates.items():
        print(f"  gate {k:44s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
