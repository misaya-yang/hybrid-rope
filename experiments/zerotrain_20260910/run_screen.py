"""Driver: score the declared bank against one frozen checkpoint.

    python -m experiments.zerotrain_20260910.run_screen \
        --model /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
        --corpus /root/autodl-tmp/bm_transfer_20260908/prepared_nll_01/rows.jsonl \
        --keep 512 --out runs/screen_01

Order of operations, and why each step is where it is:

  1. THE ALGEBRA GATE, first and free.  `selftest.py` runs before the model is
     loaded, because a bank whose r=1 is not bit-identical to MrRoPE produces a
     full leaderboard of confident nonsense and the mistake is invisible in the
     numbers.
  2. THE CORPUS CHECK, before the model too.  A corpus whose documents are not
     longer than window + keep gives an in-window metric, and the flag that says
     so is printed next to the result rather than inferred.
  3. ONE ARM, THEN A DECISION.  The timing figures this project has are
     projections from archived rows, not measurements of this package, so the
     driver runs the FIRST arm, prints its per-document cost, and only then
     commits to the rest.  The budget for the whole screen is printed before the
     second arm -- not after.
  4. THE REST, in the bank's declared order.  There is no early stopping on
     score: dropping a losing arm is the multi-candidate scanning the project
     forbids.  It stops on a time budget or a stop file, and it records what it
     did not reach as `not_run` rather than returning a shorter table.

Nothing is trained, no checkpoint is written, and the model's weights are read
only.  The only writes are the receipt and the log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description="zero-training table screen")
    ap.add_argument("--model", required=True, help="frozen checkpoint path")
    ap.add_argument("--corpus", required=True, help="JSONL with `ids` or `text`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=512,
                    help="scored trailing positions; base log-probs are "
                         "keep*vocab*4 bytes per document")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--limit", type=int, default=8, help="documents to use")
    ap.add_argument("--budget-seconds", type=float, default=3600.0)
    ap.add_argument("--skip-gate", action="store_true",
                    help="only for an already-gated tree; recorded in the receipt")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"), args=vars(args),
                   trained_anything=False, wrote_checkpoint=False)

    # ---------------------------------------------------------------- 1. gate
    if not args.skip_gate:
        from . import selftest as ST
        gate = ST.run(verbose=False)
        receipt["algebra_gate"] = dict(ok=gate["ok"], n_ok=gate["n_ok"],
                                       n_total=gate["n_total"],
                                       failures=[r["name"] for r in gate["results"]
                                                 if not r["ok"]])
        print(f"algebra gate: {gate['n_ok']}/{gate['n_total']} "
              f"{'PASS' if gate['ok'] else 'FAIL'}")
        if not gate["ok"]:
            (out / "receipt.json").write_text(json.dumps(receipt, indent=1))
            print("REFUSING to run the screen on an ungated bank; every number "
                  "below would be confident nonsense")
            return 2
    else:
        receipt["algebra_gate"] = dict(ok=None, skipped=True)

    from ..curvature_20260910.model import FrozenRoPE
    from ..curvature_20260910 import tables as T
    from . import bank as B
    from .corpus import load_docs, describe
    from .score import Scorer

    frozen = FrozenRoPE(args.model, dtype=args.dtype)
    window = int(getattr(frozen.model.config, "max_position_embeddings", 32768))
    corp = load_docs(args.corpus, tokenizer=frozen.tokenizer, limit=args.limit)
    cinfo = describe(corp, args.keep, window)
    receipt["corpus"] = {k: v for k, v in cinfo.items()}
    print(f"corpus: {cinfo['n_docs']} docs, lengths "
          f"{cinfo['min_length']}..{cinfo['max_length']}, window {window}, "
          f"keep {args.keep}")
    if not cinfo["scored_positions_all_beyond_window"]:
        print("REFUSING: some documents are not longer than window + keep, so the "
              "long_nll metric would be an in-window number")
        (out / "receipt.json").write_text(json.dumps(receipt, indent=1))
        return 2

    scorer = Scorer(frozen, corp["docs"], keep=args.keep, label=str(out))
    warm = scorer.warmup()
    receipt["warmup"] = warm
    print(f"warmup: {warm['warmup_seconds']:.1f}s for {warm['n_docs']} docs, "
          f"native long NLL {warm['native_nll']:.6f}")

    tables = [B.build(a) for a in B.arms(gain=args.gain)]

    # ------------------------------------------------- 3. one arm, then decide
    first = scorer.score(tables[0])
    per_doc = first["seconds_per_doc"]
    remaining = len(tables) - 1
    est = remaining * per_doc * len(corp["docs"])
    print(f"first arm ({first['name']}): {first['seconds']:.1f}s "
          f"({per_doc:.2f}s/doc), peak "
          f"{('%.2f GiB' % (first['peak_bytes'] / 2**30)) if first['peak_bytes'] else 'n/a'}")
    print(f"remaining {remaining} arms -> about {est / 60:.1f} min; "
          f"budget {args.budget_seconds / 60:.1f} min")
    receipt["first_arm"] = first
    receipt["estimate_remaining_seconds"] = est

    rows = [dict(first, index=0, status="ok", elapsed_cumulative=first["seconds"])]
    t0 = time.time()

    def progress(r):
        print(f"  [{r['index']:2d}/{len(tables)}] {r['name']:20s} "
              f"kl={r['native_kl']:.6f}  long_nll={r['long_nll']:.6f}  "
              f"gain={r['long_nll_gain']:+.6f}  {r['seconds']:.1f}s")

    rest = scorer.screen(tables[1:], progress=progress,
                         stop_file=str(out / "STOP"),
                         budget_seconds=max(0.0, args.budget_seconds - (time.time() - t0)))
    for r in rest["rows"]:
        r["index"] = r.get("index", 0) + 1
    rows.extend(rest["rows"])

    # ------------------------------------------------------------- leaderboard
    ok = [r for r in rows if r.get("status") == "ok"]
    ref = next((r for r in ok if r["name"] == "mrpro_n17"), None)
    anchor = next((r for r in ok if r["name"] == "incr_r1"), None)
    for r in ok:
        if ref:
            r["long_nll_vs_mrpro"] = r["long_nll"] - ref["long_nll"]
            r["native_kl_vs_mrpro"] = r["native_kl"] - ref["native_kl"]
    ok_sorted = sorted(ok, key=lambda r: r["long_nll"])

    receipt["rows"] = rows
    receipt["leaderboard"] = [{k: r.get(k) for k in
                               ("name", "native_kl", "long_nll",
                                "long_nll_vs_mrpro", "native_kl_vs_mrpro",
                                "seconds")} for r in ok_sorted]
    receipt["anchor_check"] = dict(
        mrpro_scored=ref is not None, anchor_scored=anchor is not None,
        identical_tables=None if not (ref and anchor) else
        bool(abs(ref["long_nll"] - anchor["long_nll"]) < 1e-12
             and abs(ref["native_kl"] - anchor["native_kl"]) < 1e-12),
        note="incr_r1 and mrpro_n17 are the same table bit-for-bit, so they MUST "
             "score identically; a difference is a harness bug and the whole "
             "leaderboard is void")
    # --------------------------------------------------- the theory reading
    # The leaderboard ranks arms; this is the part the screen exists for.  The
    # threshold below is expressed as a FRACTION of the dial's own slope scale
    # rather than an absolute number, because "close to zero" has no absolute
    # meaning when the two derivatives are measured in arbitrary units.
    from .kkt_readout import readout
    rd = readout(rows)
    for lab, d in (rd.get("dials") or {}).items():
        if d.get("residual_over_scale") is not None:
            d["kkt_consistent"] = bool(d["residual_over_scale"] <= 0.25)
    rd["threshold_used"] = 0.25
    rd["threshold_basis"] = ("|residual| / max(|dL/d(dial)|, |price*dD/d(dial)|) "
                             "<= 0.25: the Lagrangian changes by less than a "
                             "quarter of the larger term it is made of, so the "
                             "dial cannot be said to be above or below its price")
    receipt["kkt_readout"] = rd

    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    receipt["n_ran"] = len(ok)
    receipt["n_declared"] = len(tables)
    receipt["note"] = ("long_nll is language modelling on held-out documents, "
                       "not a task score; the screen ranks arms and does not "
                       "establish that the winner is better at anything")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str))

    print("\n" + "=" * 78)
    print(f"{'arm':22s} {'native_kl':>11s} {'long_nll':>11s} "
          f"{'vs MrPro':>10s} {'d_kl':>10s}")
    for r in ok_sorted:
        print(f"{r['name']:22s} {r['native_kl']:11.6f} {r['long_nll']:11.6f} "
              f"{r.get('long_nll_vs_mrpro', float('nan')):+10.6f} "
              f"{r.get('native_kl_vs_mrpro', float('nan')):+10.6f}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['name']:22s} NOT RUN -- {r.get('reason')}")
    print(f"\nanchor check (incr_r1 == mrpro_n17): {receipt['anchor_check']['identical_tables']}")
    pr = rd.get("pareto") or {}
    if pr.get("price") is not None:
        print(f"\nPareto price at {rd['anchor']}: d(long_nll)/d(native_kl) = "
              f"{pr['price']:+.4f}  (over {pr['n']} neighbours)")
    for lab, d in (rd.get("dials") or {}).items():
        if d.get("residual") is None:
            print(f"  {lab}: not readable ({d.get('reason')})")
            continue
        print(f"  {lab}: n={d['n']}  dD/d(dial)={d['d_native_kl_d_dial']:+.4e}  "
              f"dL/d(dial)={d['d_long_nll_d_dial']:+.4e}  "
              f"residual={d['residual']:+.4e}  "
              f"consistent={d.get('kkt_consistent')}")
        print(f"      -> {d['reading']}")
    print(f"receipt -> {out / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
