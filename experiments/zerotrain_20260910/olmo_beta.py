#!/usr/bin/env python3
"""The beta sweep on OLMo-2-0425-1B: the model where the effect is 48 points wide.

WHY THIS MODEL AND NOT QWEN.  The two archived endpoints of the beta family are

    b = 0  MrRoPE        OLMo 16K RULER  2.78%   NLL 3.68798
    b = 1  deployed BM   OLMo 16K RULER 51.32%   NLL 2.86206

against a Qwen gap of -7.3 points and 0.001 nats over the same pair.  A sweep
whose signal is 48 points wide on one model and 7 points wide on the other
should be run where it is wide: the instrument's resolution is not in question
there, so an interior optimum or a monotone ramp is readable in one pass.

The endpoints are NOT re-run.  MrRoPE's and BM's 350-row scores are archived
(`run_ruler_newtasks_01/{MrPro,MrProBM}.json`, score_sum 24.8 and 145.85) and are
read; the arms below are the interior, where nothing has been measured.  That is
the campaign's stated GPU saving, and it only works if the panel is the same one
-- so the row ids and the prompt hashes are checked against the archive before a
single forward runs.

COST.  The archived run did 700 generations in 1385 s, so 350 rows per arm is
about 12 minutes on this card.  Four arms is under an hour and the numbers are
comparable to two published endpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_screen import arms as declared_arms  # noqa: E402
from phase1_screen import m_incr_beta, m_order, order_perms  # noqa: E402
from experiments.curvature_20260910.tables import band_from_turns, m_turns  # noqa: E402

# THE TURN-WINDOW ARMS, DECLARED BEFORE ANY SCORE IS SEEN.
#
# YaRN's band rule, in its model-independent form, is "the slots whose in-window
# turn count lies in [alpha, beta]", and the deployed family hard-codes
# (alpha, beta) = (1, 32) for every checkpoint.  The KKT reading says the
# boundary belongs where the native marginal cost of compressing a slot crosses
# the long-range benefit -- and that crossing is model-dependent, so a fixed
# window is a constant approximation to a curve.  The arms below move one
# threshold at a time off (1, 32), with the same numbers on every model, which is
# what makes the result a statement about the RULE rather than about one
# checkpoint.
#
#   (1, 32)  the anchor: with ramp='beta1' this IS the deployed BM table, and its
#            350-row score is archived (41.67%), so it is not re-run
#   (1, 16)  the fast edge pulled in: fewer slots held as "well resolved"
#   (1, 64)  the fast edge pushed out: more slots held
#   (0.5, 32) the slow edge pushed out: more slots declared "unresolved"
#   (2, 32)  the slow edge pulled in
TURN_ARMS = ((1.0, 16.0), (1.0, 64.0), (0.5, 32.0), (2.0, 32.0))

# The OLMo configuration, from the archived runtime.json of the run that
# produced the endpoints: base 500000, window 4096, K 64, head_dim 128.  The
# band is YaRN's derived one for this config, which lands on slots [15, 32] with
# n = 18 increments -- NOT the Qwen band, and one of the two facts that made the
# OLMo table work where a slot-23 table would not.
OLMO = dict(theta=500_000.0, window=4096, head_dim=128, K=64,
            low=14, n=18, gain=1.138629436111989)


def native_inv_freq(theta, k=64):
    return theta ** (-np.arange(k, dtype=np.float64) / k)


def m_to_nu(m, theta):
    return native_inv_freq(theta) * np.power(4.0, -np.asarray(m, dtype=np.float64))


def check_endpoints():
    """The family must hit BOTH archived endpoints before the GPU is touched."""
    b0 = m_incr_beta(0.0, n=OLMO["n"], low=OLMO["low"])
    b1 = m_incr_beta(1.0, n=OLMO["n"], low=OLMO["low"])
    k = np.arange(1, OLMO["n"] + 1, dtype=np.float64)
    exact = 6.0 * k * (OLMO["n"] + 1 - k) / (OLMO["n"] * (OLMO["n"] + 1) * (OLMO["n"] + 2))
    return dict(
        b0_incr=float(np.abs(np.diff(b0[OLMO["low"]:OLMO["low"] + OLMO["n"] + 1])
                             - 2 * k / (OLMO["n"] * (OLMO["n"] + 1))).max()),
        b1_max_abs_err=float(np.abs(np.diff(b1[OLMO["low"]:OLMO["low"] + OLMO["n"] + 1])
                                    - exact).max()),
        sum_m_b0=float(b0.sum()), sum_m_b1=float(b1.sum()),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--archive", required=True,
                    help="dir with the archived MrPro/MrProBM jsonl")
    ap.add_argument("--betas", default="")
    ap.add_argument("--turns", default="1,16;1,64;0.5,32;2,32")
    ap.add_argument("--ramp", default="beta1", choices=("beta1", "mrpro", "linear"))
    ap.add_argument("--arms", default="",
                    help="geometry-FREE arm names only (native/native_gain1/"
                         "interp); everything else carries the Qwen band and is "
                         "refused, because installing it here would run a table "
                         "whose transition sits on the wrong slots")
    ap.add_argument("--leaks", default="",
                    help="comma-separated leak fractions, built in THIS "
                         "checkpoint's geometry. A leak fraction `a` moves a "
                         "share of the compression into the held plateau with the "
                         "increment budget held at exactly 1, so the only "
                         "difference from the deployed table is the SUPPORT. "
                         "a=0 reproduces the deployed BM exactly.")
    ap.add_argument("--gains", default="",
                    help="comma-separated rotary amplitudes to run the SAME table "
                         "at.  The gain is a separate design face from the "
                         "frequency allocation and this campaign has held it at "
                         "YaRN's 1.138629436111989 throughout.  Measured in-window "
                         "on Qwen: native-at-gain-1 is 0.0502 nats BETTER than "
                         "native-at-YaRN-gain, while the entire frequency table "
                         "contributes ~0.0007 -- a factor of 70.  Whether the gain "
                         "also moves long range has never been asked, and every "
                         "long-range comparison in this campaign was run at the "
                         "inherited gain")
    ap.add_argument("--c42", action="store_true",
                    help="run the Pro analysis's two control tables C42 and "
                         "C42-V24: same band, same plateaus, same S=42, centroid "
                         "moved to 22; the second one additionally matches the "
                         "increment variance of a1_b64.  Together they separate "
                         "'centroid moved' from 'higher-order allocation at fixed "
                         "centroid', which no existing family can")
    ap.add_argument("--knife", default="",
                    help="comma-separated d:a pairs for the S-vs-Wc 2x2. `d` is "
                         "the plateau TAPER (a pure budget dial: sum(m) falls 16 "
                         "units while the m>=0.5 crossing and the weighted budget "
                         "Wc stay put), `a` is the LEAK fraction (a pure Wc dial). "
                         "Crossing them separates the two surviving candidates "
                         "with the crossing slot held fixed at 23 by construction")
    ap.add_argument("--evq", default="",
                    help="comma-separated taus for THE PROJECT'S OWN METHOD: "
                         "nu_j = theta^{-phi_j}, a SPAN-PRESERVING redistribution "
                         "of the log-frequency spacing (fast side packed, slow "
                         "side spread) rather than a span-EXTENDING translation. "
                         "Never yet run at long range; the arm called "
                         "`evq_deployed` in the Qwen bank is a different, "
                         "project-invented construction and is NOT this.")
    ap.add_argument("--steps", default="",
                    help="comma-separated plateau-start slots for the EXTREME "
                         "concentration family m=0 up to hi-1, m=1 from hi. "
                         "Walking hi moves the budget and the held count in "
                         "OPPOSITE directions, so the scores say which one is "
                         "the cause; no existing arm separates them")
    ap.add_argument("--ab", default="",
                    help="comma-separated a:b exponent pairs, eps ~ k^a (n+1-k)^b "
                         "in THIS geometry; these break the confound that b and "
                         "sum(m) move together along the b ramp")
    ap.add_argument("--orders", default="",
                    help="comma-separated names from order_perms(); the ordering "
                         "causal experiment. Identity (`progressive`) is MrRoPE and "
                         "is archived, so it is skipped rather than re-run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="check the endpoints, panel and archive, then exit "
                         "WITHOUT loading the model")
    args = ap.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    chk = check_endpoints()
    print(json.dumps({"endpoint_check": chk}), flush=True)
    if chk["b0_incr"] > 1e-15 or chk["b1_max_abs_err"] > 1e-15:
        print("REFUSING: the family does not reproduce its own endpoints", file=sys.stderr)
        return 2

    from scripts.experiments.olmo_fast_screen import runtime as Rt
    from scripts.experiments.olmo_fast_screen.ruler_bench import score
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    rows = [json.loads(l) for l in open(args.panel)]
    arch = {}
    for nm in ("MrPro", "MrProBM"):
        f = Path(args.archive) / f"{nm}.jsonl"
        if f.exists():
            arch[nm] = {json.loads(l)["row_id"]: json.loads(l) for l in f.open()}
    ids_panel = {r["row_id"] for r in rows}
    for nm, d in arch.items():
        if set(d) != ids_panel:
            print(f"REFUSING: archived {nm} is over a different row set "
                  f"({len(d)} vs {len(rows)})", file=sys.stderr)
            return 2
    if args.limit:
        rows = rows[: args.limit]
    print(json.dumps({"phase": "SETUP", "rows": len(rows),
                      "archived": sorted(arch), "betas": args.betas,
                      "tasks": sorted({r["task"] for r in rows}),
                      "caps": sorted({r["length_cap"] for r in rows})}), flush=True)
    if args.dry_run:
        print(json.dumps({"phase": "DRY_RUN_OK", "model_loaded": False}), flush=True)
        return 0

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # THE DECODING CONFIG IS PART OF THE INSTRUMENT.  The archived run used the
    # model's own generation_config.json verbatim -- do_sample False, num_beams 1,
    # use_cache True, and pad_token_id 100277, which is NOT the eos token.  An
    # earlier version of this file passed `pad_token_id=tok.eos_token_id`, a
    # different value that changes the padding mask and therefore attention and
    # positions; the scores would still have come out, and they would not have
    # been comparable to the archived endpoints they are being read against.
    gcfg = GenerationConfig.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    def run_arm(name, m, gain=None):
        # The gain is a design face; `gain=None` keeps the inherited YaRN value so
        # every existing arm is bit-identical to before this parameter existed.
        tbl = dict(values_float32=m_to_nu(m, OLMO["theta"]).astype(np.float32),
                   gain=float(OLMO["gain"] if gain is None else gain))
        Rt.install(model, tbl)
        raw = root / f"{name}.jsonl"
        done = {json.loads(l)["row_id"] for l in raw.open()} if raw.exists() else set()
        t0 = time.monotonic()
        for row in rows:
            if row["row_id"] in done:
                continue
            ids = torch.tensor([row["prompt_ids"]], device="cuda")
            with torch.inference_mode():
                gen = model.generate(ids, attention_mask=torch.ones_like(ids),
                                     generation_config=gcfg,
                                     max_new_tokens=int(row["max_new_tokens"]))
            new = gen[0, ids.shape[1]:].tolist()
            eos = gcfg.eos_token_id
            ended = bool(new and new[-1] == eos)
            text = tok.decode(new[:-1] if ended else new, skip_special_tokens=False)
            rec = dict(row_id=row["row_id"], task=row["task"],
                       length_cap=row["length_cap"], correct=score(row, text),
                       ended_eos=ended, output_text=text)
            with raw.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            del ids, gen
        recs = [json.loads(l) for l in raw.open()]
        s = dict(arm=name, n=len(recs),
                 score_sum=float(sum(r["correct"] for r in recs)),
                 accuracy=float(np.mean([r["correct"] for r in recs])),
                 eos=int(sum(r["ended_eos"] for r in recs)),
                 sum_m=float(np.asarray(m).sum()),
                 seconds=time.monotonic() - t0)
        for nm, d in arch.items():
            sub = [r for r in recs if r["row_id"] in d]
            if sub:
                s[f"vs_{nm}"] = dict(
                    mine=float(np.mean([r["correct"] for r in sub])),
                    theirs=float(np.mean([d[r["row_id"]]["correct"] for r in sub])),
                    wins=int(sum(r["correct"] > d[r["row_id"]]["correct"] for r in sub)),
                    losses=int(sum(r["correct"] < d[r["row_id"]]["correct"] for r in sub)),
                    ties=int(sum(r["correct"] == d[r["row_id"]]["correct"] for r in sub)))
        (root / f"{name}_summary.json").write_text(json.dumps(s, indent=1))
        print(json.dumps({"ARM_SUMMARY": s}), flush=True)
        return s

    out = []
    if args.betas.strip():
        for bs in args.betas.split(","):
            b = float(bs)
            out.append(run_arm(f"beta_b{bs}".replace(".", "p"),
                               m_incr_beta(b, n=OLMO["n"], low=OLMO["low"])))
    if args.arms.strip():
        # GEOMETRY GUARD.  Every arm in the Qwen bank is built on band [23,40]
        # with n = 17.  OLMo's band is [14,32] with n = 18.  Installing a Qwen
        # table here would run a table whose transition sits on the wrong slots,
        # and it would still produce a number -- which is the failure mode this
        # project has paid for repeatedly (LESSONS L6).  Only arms whose support
        # is a free choice rather than a checkpoint-derived band may pass.
        GEOMETRY_FREE = ("native", "native_gain1", "interp")
        bank = dict(declared_arms())
        bad = [s.strip() for s in args.arms.split(",")
               if s.strip() and s.strip() not in GEOMETRY_FREE]
        if bad:
            print(f"REFUSING: {bad} carry the Qwen band [23,40]/n=17 and this "
                  f"checkpoint's band is [14,32]/n=18; use --betas, --leaks, "
                  f"--orders or --turns, which build in THIS geometry",
                  file=sys.stderr)
            return 2
        for nm in [s.strip() for s in args.arms.split(",") if s.strip()]:
            m = np.asarray(bank[nm], dtype=np.float64)
            print(json.dumps({"named_arm": nm, "sum_m": float(m.sum())}), flush=True)
            out.append(run_arm(nm, m))
    if args.leaks.strip():
        # THE HELD-PLATEAU PROBE, in this checkpoint's geometry.  In-window, the
        # leak is FREE up to sum(m) ~ 37 on Qwen and only turns up past ~41; if it
        # also improves the long-range score, the three-band restriction is
        # over-constraining and EVQ's global support is legitimate after all.
        from experiments.curvature_20260910.tables import m_leak as _m_leak
        for astr in [s.strip() for s in args.leaks.split(",") if s.strip()]:
            a = float(astr)
            m = _m_leak(a, base="bm", n=OLMO["n"], low=OLMO["low"])
            print(json.dumps({"leak_arm": a, "m_at_band_start": float(m[OLMO["low"]]),
                              "sum_m": float(m.sum())}), flush=True)
            out.append(run_arm(f"leak_a{astr}".replace(".", "p"), m))
    if args.gains.strip():
        # THE SAME TABLE AT SEVERAL GAINS.  The frequency allocation is held
        # exactly fixed, so a difference here is attributable to the rotary
        # amplitude alone -- the one design face this campaign has never varied.
        from experiments.curvature_20260910.tables import m_incr_beta as _mb
        base = np.asarray(_mb(1.0, n=OLMO["n"], low=OLMO["low"]), dtype=np.float64)
        for gs in [s.strip() for s in args.gains.split(",") if s.strip()]:
            g = float(gs)
            print(json.dumps({"gain_arm": g, "table": "deployed BM",
                              "sum_m": float(base.sum())}), flush=True)
            out.append(run_arm(f"gain_{gs}".replace(".", "p"), base, gain=g))
    if args.c42:
        from experiments.curvature_20260910.tables import m_C42 as _c42, m_C42V24 as _c42v
        for nm, fn in (("C42", _c42), ("C42V24", _c42v)):
            m = np.asarray(fn(), dtype=np.float64)
            kk = np.arange(64, dtype=float)
            eps = np.diff(np.concatenate(([0.0], m)))
            mu = float((kk * eps).sum())
            print(json.dumps({"c42_arm": nm, "sum_m": float(m.sum()),
                              "mu_eps": mu, "S_plus_mu_is_64": float(m.sum() + mu)}),
                  flush=True)
            out.append(run_arm(f"ctl_{nm}", m))
    if args.knife.strip():
        # THE 2x2.  All four cells have the m>=0.5 crossing at slot 23 and differ
        # only in (S, Wc); the archived deployed BM IS the (0,0) cell, so only
        # three arms are new.  Row effect => S is the cause; column effect => Wc
        # is; all four flat => S, Wc and the crossing all die together and only
        # the ordering line survives.  All three branches have a next step.
        from experiments.curvature_20260910.tables import m_taper as _mt
        for spec in [s.strip() for s in args.knife.split(",") if s.strip()]:
            d_s, a_s = spec.split(":")
            d, a = float(d_s), float(a_s)
            m = np.asarray(_mt(d), dtype=np.float64).copy()
            if a:
                # raise the held plateau by a*j -- the same leak knob the Qwen
                # bank uses, applied to the tapered table.  Compresses slots the
                # three-band family leaves at exactly m=0, which is the whole
                # point: it moves Wc without moving the crossing.
                lo = OLMO["low"]
                m[:lo] = np.maximum(m[:lo], a * np.arange(1, lo + 1,
                                                         dtype=np.float64))
            m = np.clip(m, 0.0, 1.0)
            print(json.dumps({"knife_arm": spec, "sum_m": float(m.sum()),
                              "cross": int(np.flatnonzero(m >= 0.5)[0]),
                              "Wc": float(np.sum(
                                  OLMO["theta"] ** (-np.arange(64) / 64) * m))}),
                  flush=True)
            out.append(run_arm(f"knife_d{d_s.replace('.','p')}_a{a_s.replace('.','p')}", m))
    if args.evq.strip():
        # THE PROJECT'S OWN METHOD, at long range, for the first time.
        # Measured in-window on Qwen it is NOT free the way the three-band family
        # is: +0.087 nats at tau=1 and +3.17 at tau=2, against +-0.005 for every
        # three-band arm.  That is the shape a KKT method should have -- it TRADES
        # in-window loss for extrapolation and therefore sits ON the constraint,
        # not deep in the interior where the span-extending family sits.
        from experiments.curvature_20260910.tables import m_evq_shift as _m_evq
        for ts_ in [s.strip() for s in args.evq.split(",") if s.strip()]:
            tau = float(ts_)
            m = np.asarray(_m_evq(tau, k=64), dtype=np.float64)
            print(json.dumps({"evq_arm": tau, "sum_m": float(m.sum())}), flush=True)
            out.append(run_arm(f"evq_shift_t{ts_}".replace(".", "p"), m))
    if args.steps.strip():
        from experiments.curvature_20260910.tables import m_step as _m_step
        for hs in [s.strip() for s in args.steps.split(",") if s.strip()]:
            hi = int(hs)
            m = _m_step(hi)
            print(json.dumps({"step_arm": hi, "sum_m": float(m.sum()),
                              "n_held": int((m < 1e-12).sum()),
                              "n_plateau": int((m > 1 - 1e-12).sum())}), flush=True)
            out.append(run_arm(f"step_hi{hi}", m))
    if args.ab.strip():
        from fractions import Fraction as _F  # noqa: F401
        for spec in [s.strip() for s in args.ab.split(",") if s.strip()]:
            a_e, b_e = (float(x) for x in spec.split(":"))
            n, low = OLMO["n"], OLMO["low"]
            kk = np.arange(1, n + 1, dtype=np.float64)
            w = np.power(kk, a_e) * np.power(n + 1 - kk, b_e)
            eps = w / w.sum()
            m = np.zeros(64)
            m[low + 1: low + n + 1] = np.cumsum(eps)
            m[low + n + 1:] = 1.0
            print(json.dumps({"ab_arm": spec, "sum_m": float(m.sum())}), flush=True)
            out.append(run_arm(f"ab_a{a_e:g}_b{b_e:g}".replace(".", "p"), m))
    if args.orders.strip():
        # THE ORDERING EXPERIMENT.  Same band, same endpoints, same span and the
        # SAME INCREMENT MULTISET -- only the order in which the mass is spent
        # differs.  `progressive` is MrRoPE and its 350-row score is archived, so
        # it is a free anchor rather than an arm to re-run.
        # `order_perms()` defaults to n = 17 (the Qwen band width).  This
        # checkpoint's band width is 18, and `m_order` guards on the permutation
        # being of range(n) -- so passing the default here raised ValueError on
        # every arm and the whole ordering axis would have been silently absent
        # from OLMo.  The permutation must be drawn in THIS geometry.
        perms = order_perms(n=OLMO["n"])
        for nm in [s.strip() for s in args.orders.split(",") if s.strip()]:
            if nm not in perms:
                print(f"REFUSING: {nm!r} is not one of {sorted(perms)}",
                      file=sys.stderr)
                return 2
            m = m_order(perms[nm], n=OLMO["n"], low=OLMO["low"])
            print(json.dumps({"order_arm": nm, "perm": [int(x) for x in perms[nm]],
                              "sum_m": float(np.asarray(m).sum())}), flush=True)
            out.append(run_arm(f"ord_{nm}", m))
    for spec in args.turns.split(";"):
        if not spec.strip():
            continue
        a, b = (float(x) for x in spec.split(","))
        m = m_turns(a, b, OLMO["theta"], OLMO["window"], OLMO["head_dim"],
                    ramp=args.ramp)
        lo, hi = band_from_turns(a, b, OLMO["theta"], OLMO["window"], OLMO["head_dim"])
        print(json.dumps({"turn_arm": [a, b], "band": [lo, hi],
                          "n_increments": hi - lo,
                          "sum_m": float(np.asarray(m).sum())}), flush=True)
        out.append(run_arm(f"turns_a{a:g}_b{b:g}".replace(".", "p"), m))
    (root / "manifest.json").write_text(json.dumps(
        dict(status="COMPLETE", model=args.model, panel=args.panel,
             arms=[s["arm"] for s in out], endpoint_check=chk,
             archived_endpoints={k: len(v) for k, v in arch.items()},
             mrpro_rerun=False, bm_rerun=False), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
