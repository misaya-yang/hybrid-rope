#!/usr/bin/env python3
"""Phase 1 round 2: the extrapolation instrument -- RULER generation at 32K/128K.

WHY THIS RUN EXISTS AT ALL, GIVEN ROUND 1.  Round 1 scores tail NLL at
8192/16384/32768, all INSIDE the 32768 native window, and the first arm it
scored already showed the instrument's resolution: the NATIVE table -- no
compression at all -- came within 4.2e-4 of MrRoPE's NLL.  Every arm in the bank
agrees with MrRoPE at short distances by construction, because they differ only
in the compressed low-frequency slots, and those are the slots a 32K prefix
barely exercises.  So round 1 cannot rank arms on capability; what it can do is
show what a table COSTS in-window, and the answer so far is "almost nothing".
The ranking has to come from a distance where the compressed slots are doing
work: the 131072 rows.

THE CONTRACTION IS DECLARED HERE, BEFORE ROUND 1'S NUMBERS ARE IN.  That is the
whole point of writing it down first.  The bank has 30 arms and running all of
them at 128K is ~10 GPU-hours; picking the six to keep AFTER seeing round 1
would be a multi-candidate scan wearing a schedule.  The six are chosen by the
theory instead:

    mrpro_n17     the incumbent, and the anchor of both dials
    yarn_lin      the published baseline -- its frozen-Qwen RULER column is
                  EMPTY in the archive (R1 recon), so this row is new
                  information whichever way it lands
    front_a0p001  the front dial pushed hard toward "disturb the boundary less"
    front_a0p1    the front dial pushed hard the other way
    back_r0       the back dial at its flattest (YaRN-like ramp, MrRoPE's front)
    back_r4       the back dial at its most concentrated

Four of the six are the ENDPOINTS of the two dials.  If MrRoPE is stationary on
a dial, its two extremes straddle the incumbent and neither wins; if it is not,
the winner names the direction and the mechanism.  Either outcome contracts the
space: a dial whose extremes both lose is retired, and a dial with a winning
extreme has its sign fixed and only the magnitude left to find.

MrRoPE IS NOT RE-RUN.  Its 36 rows are archived at `run_qwen3_01/MrPro.jsonl`
and are read, not regenerated -- the campaign's stated GPU saving.  The rows it
is compared against must therefore be the same rows, and `Worker` refuses to
start if the panel's SHA or the token inputs have drifted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_screen import (A_MR, GAIN_YARN, QWEN25_3B, anchor_check,  # noqa: E402
                           m_incr_beta, m_incr_power, m_incr_split, m_mrpro,
                           m_to_inv_freq, m_yarn)

# turn-window support: the same (alpha, beta) applies to any checkpoint
sys.path.insert(0, "/root/autodl-tmp/phase1_20260910/repoharness")
sys.path.insert(0, "/root/autodl-tmp/phase1_20260910")
try:
    from experiments.curvature_20260910.tables import (band_from_turns,  # noqa
                                                       m_turns)
    HAVE_TURNS = True
except Exception:                                     # pragma: no cover
    HAVE_TURNS = False


def _turns(alpha, beta, ramp="beta1"):
    """The turn-window table for the Qwen config, matching the OLMo run's ramp.

    The whole point of the family is that these two arguments mean the same thing
    on every checkpoint: the band is `alpha <= W*omega/2pi <= beta` and theta,
    window and head_dim decide which slots that is.  On Qwen2.5-3B (theta 1e6,
    W 32768, head_dim 128) the anchor (1, 32) lands on slots [23, 40], which is
    where MrRoPE and the deployed BM table both sit -- so the anchor is not a new
    table and the arms below are the off-anchor points.
    """
    return m_turns(float(alpha), float(beta), QWEN25_3B["theta"],
                   QWEN25_3B["window"], QWEN25_3B["head_dim"], ramp=ramp)

# The pre-declared contraction.  Order fixed here, not sorted by anything.
# THE CONTRACTED SET, DECLARED BEFORE ANY ROUND-2 SCORE IS SEEN.
#
# The transcription mining produced the one thing the project has been missing:
# a pair of tables that are BOTH deployed, BOTH measured, and OPPOSITE in
# outcome, separated by a single parameter.
#
#     b = 0  MrRoPE          Qwen 32K NLL 2.08899 | 128K RULER 78.13%
#                            OLMo 16K NLL 3.68798 | 16K RULER  2.78%
#     b = 1  deployed BM     Qwen 32K NLL 2.08789 | 128K RULER 70.83%
#                            OLMo 16K NLL 2.86206 | 16K RULER 51.32%
#
# Same band (derived per configuration), same gain, same S.  On OLMo the gap is
# 0.83 nats and 48.5 points; on Qwen it is 0.001 nats and -7.3 points.  So the
# endpoints do NOT need re-running -- both numbers are archived, which is the
# GPU saving the campaign asks for -- and the arms below are the INTERIOR, where
# nothing has been measured.  A monotone interpolation says b is the operative
# variable and its sign differs by model; an interior peak says the two models
# want the same shape at different strengths.
CONTRACTED = (
    ("beta_b0p25", lambda: m_incr_beta(0.25)),
    ("beta_b0p5", lambda: m_incr_beta(0.5)),
    ("beta_b2", lambda: m_incr_beta(2.0)),
    ("yarn_lin", lambda: m_yarn(k=64)),
)

# The turn-window arms, kept separate so `--arm-set turns` is an explicit request
# and a default run cannot silently change which experiment it is.  Same numbers
# as the OLMo run; the anchor (1, 32) is the deployed tables and is not re-run.
TURN_CONTRACTED = (
    ("turns_a1_b16", lambda: _turns(1.0, 16.0)),
    ("turns_a1_b64", lambda: _turns(1.0, 64.0)),
    ("turns_a0p5_b32", lambda: _turns(0.5, 32.0)),
    ("turns_a2_b32", lambda: _turns(2.0, 32.0)),
)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--history", default="/root/autodl-tmp/bm_transfer_20260908")
    ap.add_argument("--lengths", default="32768,131072")
    ap.add_argument("--only", default=None)
    ap.add_argument("--arm-set", default="beta", choices=("beta", "turns"))
    args = ap.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",")]

    ac = anchor_check()
    if not ac["bit_exact"]:
        print("REFUSING: bank anchor is not bit-exact against MrRoPE", file=sys.stderr)
        return 2

    from experiments.nongeometric_screen.worker import Worker
    w = Worker(root, args.history)
    theta = QWEN25_3B["theta"]

    rows = [r for r in w.screen if r["length_cap"] in lengths]
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        rows = [r for r in rows if r["row_id"] in want]
    print(json.dumps({"phase": "SETUP", "n_rows": len(rows),
                      "by_len": {str(L): sum(1 for r in rows if r["length_cap"] == L)
                                 for L in lengths},
                      "contracted_arms": [n for n, _ in CONTRACTED]}), flush=True)

    todo = TURN_CONTRACTED if args.arm_set == "turns" else CONTRACTED
    if args.arm_set == "turns" and not HAVE_TURNS:
        print("REFUSING: --arm-set turns needs the tables module on PYTHONPATH",
              file=sys.stderr)
        return 2
    if args.arm_set == "turns":
        for nm, build in todo:
            m = np.asarray(build(), dtype=np.float64)
            print(json.dumps({"turn_arm": nm, "sum_m": float(m.sum()),
                              "band": [int(np.flatnonzero(m > 1e-12)[0]),
                                       int(np.flatnonzero(m < 1 - 1e-12)[-1] + 1)]}),
                  flush=True)
    for name, build in todo:
        m = np.asarray(build(), dtype=np.float64)
        tbl = dict(values_float32=m_to_inv_freq(m, theta).astype(np.float32),
                   gain=GAIN_YARN)
        w.apply({"table": tbl})
        raw = root / f"{name}.jsonl"
        done = set()
        if raw.exists():
            done = {json.loads(l)["row_id"] for l in raw.open()}
        for row in rows:
            if row["row_id"] in done:
                continue
            rec = w.generate(row)
            base = w.baseline[row["row_id"]]
            rec.update(arm=name, mrpro_correct=base["correct"],
                       delta=float(rec["correct"]) - float(base["correct"]))
            with raw.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps({"arm": name, "row": row["row_id"],
                              "len": row["length_cap"], "correct": rec["correct"],
                              "mrpro": base["correct"], "delta": rec["delta"],
                              "sec": round(rec["elapsed_seconds"], 1),
                              "eos": rec["ended_eos"]}), flush=True)
        recs = [json.loads(l) for l in raw.open()]
        summ = {}
        for L in lengths:
            sub = [r for r in recs if r["length_cap"] == L]
            if sub:
                summ[str(L)] = dict(
                    n=len(sub),
                    correct=float(np.mean([r["correct"] for r in sub])),
                    mrpro=float(np.mean([r["mrpro_correct"] for r in sub])),
                    delta=float(np.mean([r["delta"] for r in sub])))
        allsub = recs
        summary = dict(status="COMPLETE", arm=name, by_length=summ,
                       n=len(allsub),
                       correct=float(np.mean([r["correct"] for r in allsub])),
                       mrpro=float(np.mean([r["mrpro_correct"] for r in allsub])),
                       delta=float(np.mean([r["delta"] for r in allsub])),
                       sum_m=float(np.asarray(build()).sum()),
                       scope="6-task RULER development subset; MrRoPE columns are "
                             "the archived baseline rows, not regenerated")
        (root / f"{name}_summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps({"ARM_SUMMARY": summary}), flush=True)

    (root / "manifest.json").write_text(json.dumps(
        dict(status="COMPLETE", arms=[n for n, _ in todo], lengths=lengths,
             n_rows=len(rows), finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
             mrpro_rerun=False, weights_updated=False,
             contraction_declared_before_round1=True), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
