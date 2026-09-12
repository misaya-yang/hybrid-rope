"""Rejection tests for the budget and paired-statistics tools.

Plan section 9.1: "budget and paired-statistics rejection tests on artificial
fixtures".  Every fixture here is synthetic.  None of these numbers is a model
result and none may be quoted as one.

The point is the negative direction: a tool that reports a number on a partial
or misaligned panel is worse than a tool that crashes, because the number looks
like a result.  So the assertions below are mostly "this input must be refused".

Run:  python test_tools.py            (exit 0 = all pass)
      python test_tools.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import paired_report as P
import budget as B
import coverage_report as C
import llama_runner as L

HERE = Path(__file__).resolve().parent
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append({"test": name, "pass": bool(cond), "detail": str(detail)})
    return bool(cond)


def expect_raises(name, fn, exc=Exception):
    try:
        fn()
    except exc as e:
        return check(name, True, f"refused: {type(e).__name__}")
    except Exception as e:
        return check(name, False, f"wrong exception {type(e).__name__}: {e}")
    return check(name, False, "did not refuse")


# --------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------


def fake_rows(n_tasks=2, lengths=(16384, 32768), per_cell=4, p=0.5, seed=0, prefix="r"):
    rng = __import__("numpy").random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        for ln in lengths:
            for i in range(per_cell):
                rows.append({
                    "row_id": f"{prefix}_{t}_{ln}_{i}",
                    "task": f"task{t}",
                    "length_cap": ln,
                    "source_id": f"doc_{t}_{i}",
                    "correct": int(rng.random() < p),
                })
    return rows


# --------------------------------------------------------------------------
# 1. power arithmetic must reproduce plan section 6.6
# --------------------------------------------------------------------------


def test_power():
    # Plan section 6.6's four quoted numbers are STANDARD ERRORS (sqrt(0.2/96)
    # = 4.56pp).  Asserting them against a 95% MDE would be wrong by 1.96x.
    m = P.mde(q=0.2, n=96, per_length_n=48)
    check("SE at S combined N=96 is 4.56pp (plan section 6.6)",
          abs(m["combined"]["se"] - 0.0456) < 5e-4, f"{m['combined']['se']:.4f}")
    check("SE at S per-length N=48 is 6.45pp", abs(m["per_length"]["se"] - 0.0645) < 5e-4,
          f"{m['per_length']['se']:.4f}")
    mh = P.mde(q=0.2, n=256, per_length_n=128)
    check("SE at H combined N=256 is 2.80pp", abs(mh["combined"]["se"] - 0.0280) < 5e-4,
          f"{mh['combined']['se']:.4f}")
    check("SE at H per-length N=128 is 3.95pp", abs(mh["per_length"]["se"] - 0.0395) < 5e-4,
          f"{mh['per_length']['se']:.4f}")
    check("the 95% MDE is 1.96x the SE, kept as a separate field",
          abs(m["combined"]["mde_95"] / m["combined"]["se"] - P.Z_ALPHA_2) < 1e-9, "")
    n = P.n_for_power(delta=0.05, q=0.2, alpha=0.05, power=0.80)
    check("~627 independent pairs for 5pp at 80% power", abs(n - 627) <= 2, f"{n}")


# --------------------------------------------------------------------------
# 2. budget must reproduce the plan's own worked table
# --------------------------------------------------------------------------


def test_budget():
    c = B.Counts()
    res = B.evaluate(60, c, 1.2, 2.5, 4.7)
    check("per-candidate S = 6.992 min", abs(res["minutes_per_s_candidate"] - 6.992) < 1e-3,
          f"{res['minutes_per_s_candidate']:.4f}")
    check("60-candidate total = 599.880", abs(res["total_minutes"] - 599.880) < 1e-3,
          f"{res['total_minutes']:.4f}")
    want = {"S candidates": 419.520, "S controls": 27.968, "V tables": 38.272,
            "H tables": 76.544, "native short reference": 2.576,
            "gain probes (new low-g cells)": 7.360, "operator parity / overhead reserve": 7.640,
            "model preflight + pilot": 12.000, "exception reserve": 8.000}
    for it in res["items"]:
        if it["item"] in want:
            check(f"line item {it['item']} = {want[it['item']]}",
                  abs(it["minutes"] - want[it["item"]]) < 1e-3, f"{it['minutes']:.4f}")
    for c_min, expect in ((6.992, 60), (8.0, 49), (9.0, 41), (10.0, 35)):
        got = B.fit_candidates(c, 1.2, 2.5, 4.7, s_minutes=c_min)["max_candidates"]
        check(f"fit at {c_min} min/candidate gives {expect} candidates", got == expect, f"got {got}")
    low = B.fit_candidates(c, 1.2, 2.5, 4.7, s_minutes=100.0)
    check("an infeasible budget reports fewer than the 40 floor", not low["meets_minimum_40"],
          f"max={low['max_candidates']}")


# --------------------------------------------------------------------------
# 3. paired statistics must refuse partial and misaligned panels
# --------------------------------------------------------------------------


def test_paired_rejections():
    a = fake_rows(seed=1, prefix="a")
    b = fake_rows(seed=2, prefix="b")
    expect_raises("misaligned row ids are refused",
                  lambda: P.require_aligned({"A": P.index_arm(a), "B": P.index_arm(b)}),
                  ValueError)

    b2 = [dict(r) for r in b]
    b2[0]["row_id"] = a[0]["row_id"]
    for i, r in enumerate(b2):
        r["row_id"] = a[i]["row_id"]
    b2[0]["row_id"] = "extra_row"
    expect_raises("an extra row is refused",
                  lambda: P.require_aligned({"A": P.index_arm(a), "B": P.index_arm(b2)}),
                  ValueError)

    partial = a[: len(a) // 2]
    expect_raises("a partial panel is refused against the frozen index",
                  lambda: P.require_aligned({"A": P.index_arm(partial)}, expected_rows=a),
                  ValueError)

    dup = a + [dict(a[0])]
    expect_raises("a duplicated row is refused", lambda: P.index_arm(dup), ValueError)

    # and the positive control: a matched pair must go through
    ok = P.require_aligned({"A": P.index_arm(a), "B": P.index_arm(a)})
    check("an exactly aligned pair is accepted", len(ok) == len(a), f"n={len(ok)}")


def test_paired_arithmetic():
    base = fake_rows(seed=3, p=0.2, prefix="a")
    cand = [dict(r) for r in base]
    for r in cand:
        r["correct"] = 1 if r["source_id"].endswith("0") else r["correct"]
    st = P.paired_stats(base, cand)
    check("identical arms give delta 0",
          abs(P.paired_stats(base, base)["delta_macro"]) < 1e-12, "")
    check("a strictly better arm gives delta > 0", st["delta_macro"] > 0, f"{st['delta_macro']:.4f}")
    check("paired SE is finite and positive", st["se_paired"] > 0, f"{st['se_paired']:.4f}")
    check("n_pairs counts every matched row", st["n_pairs"] == len(base), f"{st['n_pairs']}")


# --------------------------------------------------------------------------
# 4. the bootstrap must reject empty cells and say how often
# --------------------------------------------------------------------------


def test_baseline_hole():
    """The exact counterexample: beating MR while losing to YaRN must not pass.

    A(YARN)=0.10, A(MR)=0.08, A(M)=0.09  =>  G_new = +1pp > 0 and
    C_upgrade = 0.09 - 0.16 + 0.10 = +3pp >= 0, yet A(M) < A(YARN).  Under the
    plan's original two-condition test this would have been reported as an
    upgrade.  It must now be FAIL.
    """
    tmp = Path(tempfile.mkdtemp())
    N = 100
    mr = graded_rows(0.08, per_cell=N, prefix="h")
    yarn = graded_rows(0.10, per_cell=N, prefix="h")
    cand = graded_rows(0.09, per_cell=N, prefix="h")
    bm = graded_rows(0.05, per_cell=N, prefix="h")
    ep = tmp / "exp.jsonl"
    ep.write_text("\n".join(json.dumps(r) for r in mr), encoding="utf-8")

    def dump(rows, name):
        p = tmp / name
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return str(p)

    out = tmp / "hole.json"
    rc = subprocess.run([sys.executable, str(HERE / "upgrade_report.py"),
                         "--mr", dump(mr, "mr.jsonl"), "--candidate", dump(cand, "c.jsonl"),
                         "--yarn", dump(yarn, "y.jsonl"), "--bm", dump(bm, "b.jsonl"),
                         "--expected-data", str(ep), "--n-boot", "200", "--out", str(out)],
                        capture_output=True, text=True)
    check("the counterexample run completes", rc.returncode == 0, rc.stderr[-200:])
    if rc.returncode != 0:
        return
    rep = json.loads(out.read_text())
    check("beats-MR and C_upgrade>0 does NOT yield PASS",
          rep["verdict"] != "PASS", rep["verdict"])
    check("the recorded reason is that it loses to YaRN",
          rep["gates"]["beats_yarn_point"] is False, "")
    check("G_new is positive (the old test would have been fooled)",
          rep["contrasts"]["contrasts"]["G_new"]["point"] > 0,
          f"{rep['contrasts']['contrasts']['G_new']['point']:.4f}")
    check("C_upgrade is positive (the old test would have been fooled)",
          rep["contrasts"]["contrasts"]["C_upgrade"]["point"] > 0,
          f"{rep['contrasts']['contrasts']['C_upgrade']['point']:.4f}")
    check("G_vs_best turns negative, which is the binding gate",
          rep["contrasts"]["contrasts"]["G_vs_best"]["point"] < 0,
          f"{rep['contrasts']['contrasts']['G_vs_best']['point']:.4f}")
    check("hero_result is False", rep["hero_result"] is False, "")


def test_bootstrap():
    rows = fake_rows(n_tasks=2, per_cell=3, seed=5)
    arms = {"MR": P.index_arm(rows), "M": P.index_arm(rows), "YARN": P.index_arm(rows)}
    contrasts = [
        {"name": "G_new", "terms": [("M", 1.0), ("MR", -1.0)]},
        {"name": "G_old", "terms": [("MR", 1.0), ("YARN", -1.0)]},
        {"name": "C_upgrade", "terms": [("M", 1.0), ("MR", -2.0), ("YARN", 1.0)]},
    ]
    r = P.cluster_bootstrap(arms, contrasts, n_boot=300, seed=1)
    check("identical arms give a zero point estimate for every contrast",
          all(abs(v["point"]) < 1e-12 for v in r["contrasts"].values()), "")
    check("bootstrap reports a rejection fraction", math.isfinite(r["rejection_fraction"]),
          f"{r['rejection_fraction']:.3f}")
    check("identical arms give a pointwise-zero interval",
          all(abs(v["upper_simultaneous_95"]) < 1e-9 for v in r["contrasts"].values()), "")

    rng = __import__("numpy").random.default_rng(7)
    noisy = [dict(x) for x in rows]
    for x in noisy:
        x["correct"] = int(rng.random() < 0.5)
    arms2 = {"MR": P.index_arm(rows), "M": P.index_arm(noisy), "YARN": P.index_arm(rows)}
    r2 = P.cluster_bootstrap(arms2, contrasts, n_boot=300, seed=1)
    check("a noisy arm gets a non-degenerate interval",
          r2["contrasts"]["G_new"]["se"] > 0, f"{r2['contrasts']['G_new']['se']:.4f}")


# --------------------------------------------------------------------------
# 5. upgrade_report must refuse a partial H panel end to end
# --------------------------------------------------------------------------


def graded_rows(acc, per_cell=10, seed=11, prefix="h"):
    """Deterministic rows with an EXACT accuracy, so the verdict is reproducible.

    Sampling a Bernoulli would make the test flaky near a gate boundary; here the
    first `round(acc*per_cell)` rows of every cell are correct.
    """
    rows = fake_rows(n_tasks=2, lengths=(16384, 32768), per_cell=per_cell,
                     seed=seed, prefix=prefix)
    k = int(round(acc * per_cell))
    counters = {}
    for r in rows:
        key = (r["task"], r["length_cap"])
        i = counters.get(key, 0)
        counters[key] = i + 1
        r["correct"] = int(i < k)
    return rows


def test_upgrade_report_cli():
    tmp = Path(tempfile.mkdtemp())
    # 2 tasks x 2 lengths x 40 = 160 rows over 80 source clusters.  The size
    # matters: C_upgrade has three terms, so its joint lower bound is materially
    # harder to clear than G_new's -- at 40 rows it lands just below zero and the
    # verdict is (correctly) FAIL.  That asymmetry is the point of section 0.2.
    N = 40
    exp = graded_rows(0.5, per_cell=N, prefix="h")
    mr = graded_rows(0.5, per_cell=N, prefix="h")
    # cand 0.9 > mr 0.5 > yarn 0.4  =>  G_new = +0.4, G_old = +0.1, C_upgrade = +0.3
    cand = graded_rows(0.9, per_cell=N, prefix="h")
    yarn = graded_rows(0.4, per_cell=N, prefix="h")
    bm = graded_rows(0.45, per_cell=N, prefix="h")

    def dump(rows, name):
        p = tmp / name
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return str(p)

    ep, mp, cp, yp = (dump(exp, "exp.jsonl"), dump(mr, "mr.jsonl"),
                      dump(cand, "cand.jsonl"), dump(yarn, "yarn.jsonl"))
    bp = dump(bm, "bm.jsonl")

    out = tmp / "up.json"
    r = subprocess.run([sys.executable, str(HERE / "upgrade_report.py"),
                        "--mr", mp, "--candidate", cp, "--yarn", yp, "--bm", bp,
                        "--expected-data", ep, "--n-boot", "200", "--out", str(out)],
                       capture_output=True, text=True)
    check("upgrade_report runs on an aligned panel", r.returncode == 0, r.stderr[-200:])
    if r.returncode == 0:
        rep = json.loads(out.read_text())
        check("a dominant candidate without measured guards is NOT reported as PASS",
              rep["verdict"] != "PASS", rep["verdict"])
        check("it is labelled INCOMPLETE_GUARDS and names what is missing",
              rep["verdict"] == "INCOMPLETE_GUARDS" and "retention_8K" in rep["guards_unevaluated"],
              f"{rep['verdict']} {rep['guards_unevaluated']}")
        check("the plan gap is stated rather than hidden", "plan_gap_note" in rep, "")
        check("missing guards are recorded as caveats",
              any("retention" in c for c in rep["caveats"]), str(rep["caveats"])[:120])
        check("C_upgrade is around +0.30 for the graded fixture",
              abs(rep["contrasts"]["contrasts"]["C_upgrade"]["point"] - 0.30) < 1e-9,
              f"{rep['contrasts']['contrasts']['C_upgrade']['point']:.4f}")

        # same panel, now WITH the guards measured -> PASS becomes available
        nat = dump(graded_rows(0.9, per_cell=N, prefix="h"), "native.jsonl")
        out2 = tmp / "up_pass.json"
        r2p = subprocess.run([sys.executable, str(HERE / "upgrade_report.py"),
                              "--mr", mp, "--candidate", cp, "--yarn", yp, "--bm", bp,
                              "--expected-data", ep,
                              "--native", nat, "--strict", cp, "--strict-mr", mp,
                              "--qa", cp, "--qa-mr", mp, "--guard-length", "16384",
                              "--n-boot", "200", "--out", str(out2)],
                             capture_output=True, text=True)
        check("the guarded run completes", r2p.returncode == 0, r2p.stderr[-200:])
        if r2p.returncode == 0:
            rep2 = json.loads(out2.read_text())
            check("with every guard measured and passing the verdict is PASS",
                  rep2["verdict"] == "PASS", f"{rep2['verdict']} {rep2.get('guards_unevaluated')}")

    part = dump(exp[: len(exp) // 2], "partial.jsonl")
    r2 = subprocess.run([sys.executable, str(HERE / "upgrade_report.py"),
                         "--mr", mp, "--candidate", part, "--yarn", yp,
                         "--expected-data", ep, "--n-boot", "100", "--out", str(tmp / "up2.json")],
                        capture_output=True, text=True)
    check("a partial candidate panel is refused (exit 2)", r2.returncode == 2,
          f"rc={r2.returncode}")
    check("the refusal says REFUSING", "REFUSING" in r2.stderr, r2.stderr[-160:])

    # unequal-length arms are refused
    short = [r for r in yarn if r["length_cap"] == 16384]
    r3 = subprocess.run([sys.executable, str(HERE / "upgrade_report.py"),
                         "--mr", mp, "--candidate", cp, "--yarn", dump(short, "yshort.jsonl"),
                         "--expected-data", ep, "--n-boot", "100", "--out", str(tmp / "up3.json")],
                        capture_output=True, text=True)
    check("an arm missing a whole length is refused", r3.returncode == 2, f"rc={r3.returncode}")


# --------------------------------------------------------------------------
# 6. operators: identities and pseudo-method detection
# --------------------------------------------------------------------------


def test_operators():
    import operators as O
    rep = O.selftest(None)
    check("all algebraic identity checks pass", rep["all_pass"],
          f"{rep['n_pass']}/{rep['n_checks']}")
    check("four section 1.1 pseudo-methods are identified",
          rep["n_pseudo_identified"] == 4, f"{rep['n_pseudo_identified']}")
    check("60 configurations compile", len(O.config_ids()) == 60, f"{len(O.config_ids())}")
    g = O.Geometry.from_native(None)
    check("band is derived as (18, 35)", (g.low, g.high) == (18, 35), f"{(g.low, g.high)}")

    # a mutated stock array must be refused rather than silently compiled against
    bad = g.omega.copy()
    bad[0] *= 1.01
    expect_raises("a mismatched stock inv_freq is refused",
                  lambda: O.Geometry.from_native(bad, window=8192, theta=500000.0), ValueError)


def test_rotation():
    """The rotation application, and the pair-layout convention it depends on."""
    import operators as O
    g = O.Geometry.from_native(None)
    rng = np.random.default_rng(4)
    P, K = 16, g.K

    for layout in ("half", "adjacent"):
        x = rng.normal(size=(2, P, 2 * K))
        ph = rng.normal(size=(P, K)) * 3.0
        got = O.apply_rotation(x, ph, pair_layout=layout)
        want = O.reference_rotation(x, ph, pair_layout=layout)
        check(f"rotation ({layout}) matches the complex reference",
              np.max(np.abs(got - want)) < 1e-10, f"{np.max(np.abs(got - want)):.2e}")

    # the zero operator is the identity
    x = rng.normal(size=(P, 2 * K))
    got = O.apply_rotation(x, np.zeros((P, K)))
    check("zero phase is the identity", np.allclose(got, x, atol=1e-12), "")

    # a full turn is the identity
    got = O.apply_rotation(x, np.full((P, K), 2 * np.pi))
    check("a whole 2*pi turn is the identity", np.allclose(got, x, atol=1e-10),
          f"{np.max(np.abs(got - x)):.2e}")

    # the two layouts are genuinely different operators on the same frequencies
    ph = g.nu_mrpro[None, :] * np.arange(P)[:, None]
    a = O.apply_rotation(x, ph, pair_layout="half")
    b = O.apply_rotation(x, ph, pair_layout="adjacent")
    check("pair layout changes the operator (not cosmetic)",
          not np.allclose(a, b, atol=1e-6), f"{np.max(np.abs(a - b)):.4f}")

    # amp and diag are applied, and in the order the plan writes
    amp = 1.0 + rng.random((P, K))
    got = O.apply_rotation(x, ph, amp=amp)
    check("amplitude scales the rotated pairs", np.allclose(got, a * np.tile(amp, 2), atol=1e-10), "")
    dg = rng.normal(size=(K, 2)) + 2.0
    got = O.apply_rotation(np.ones((1, 2 * K)), np.zeros((1, K)), diag=dg)
    want = np.concatenate([dg[:, 0], dg[:, 1]])[None, :]
    check("pair metric is applied before the rotation", np.allclose(got, want, atol=1e-12), "")

    # ---- the RoPE invariant: a pure-frequency operator depends on t-p only ----
    # q content and k content are fixed; only the positions move.  For a
    # frequency-only operator the score <q(p), k(t)> must depend on t-p alone.
    for cid in ("D01a", "D03a", "D06a"):
        op = O.build(cid, g)
        q0 = rng.normal(size=2 * K)[None, :]
        k0 = rng.normal(size=2 * K)[None, :]
        base = None
        agreed = True
        worst = 0.0
        for p0 in (0, 37, 1000, 20000):
            for d in (0, 1, 17, 4096):
                pp = np.array([[float(p0)]])
                tt = np.array([[float(p0 + d)]])
                qr = O.apply_rotation(q0, op.q_phase(pp[0]))
                kr = O.apply_rotation(k0, op.k_phase(tt[0]))
                s = float(np.sum(qr * kr))
                if base is None:
                    base = {}
                key = d
                if key not in base:
                    base[key] = s
                else:
                    worst = max(worst, abs(s - base[key]))
                    if abs(s - base[key]) > 1e-9:
                        agreed = False
        check(f"{cid}: pure-frequency operator is relative-position only", agreed,
              f"max drift {worst:.2e} over p0 and t-p")


def _panel_fixture(tmp, mutate=None):
    """8 tasks x {4096,16384,32768} sources, 2 rows per source."""
    import random
    rng = random.Random(7)
    rows = []
    for t in ("niah_single_2", "niah_multikey_2", "niah_multivalue", "niah_multiquery",
              "vt", "fwe", "squad", "hotpotqa"):
        for ln in (4096, 16384, 32768):
            n = 40 if ln == 4096 else 30
            for i in range(n):
                for j in range(2):
                    rows.append({"row_id": f"{t}_{ln}_{i}_{j}", "task": t, "length_cap": ln,
                                 "source_id": f"doc_{t}_{ln}_{i}", "correct": 0})
    if mutate:
        mutate(rows)
    p = tmp / "panel.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_split_panel():
    """The P/S/V/H splitter must refuse every form of source reuse."""
    tmp = Path(tempfile.mkdtemp())
    py = [sys.executable, str(HERE / "split_panel.py")]

    good = _panel_fixture(tmp)
    r = subprocess.run(py + ["--rows", str(good), "--out", str(tmp / "ok")],
                       capture_output=True, text=True)
    check("a clean panel splits", r.returncode == 0, r.stderr[-160:])
    if r.returncode == 0:
        man = json.loads((tmp / "ok" / "split_manifest.json").read_text())
        # no source may appear in two stages
        allsrc = []
        for stage in ("P", "S", "V", "H"):
            rs = [json.loads(l) for l in (tmp / "ok" / f"{stage}.jsonl").read_text().splitlines()]
            allsrc.append({x["source_id"] for x in rs})
        union = set().union(*allsrc)
        check("stages are disjoint by source", sum(len(x) for x in allsrc) == len(union),
              f"{sum(len(x) for x in allsrc)} placements, {len(union)} sources")
        check("the manifest carries a sha per stage",
              all("sha256" in v for v in man["stages"].values()), "")

    def steal_from_4096(rows):
        for x in rows:
            if x["task"] == "squad" and x["length_cap"] == 16384 and int(x["source_id"].split("_")[-1]) >= 6:
                x["source_id"] = x["source_id"].replace("16384", "4096")
    r = subprocess.run(py + ["--rows", str(_panel_fixture(tmp, steal_from_4096)),
                             "--out", str(tmp / "xlen")], capture_output=True, text=True)
    check("a source at two lengths is refused", r.returncode == 2, f"rc={r.returncode}")

    def steal_from_other_task(rows):
        for x in rows:
            if x["task"] == "vt" and x["length_cap"] == 16384 and int(x["source_id"].split("_")[-1]) >= 6:
                x["source_id"] = x["source_id"].replace("vt", "fwe")
    r = subprocess.run(py + ["--rows", str(_panel_fixture(tmp, steal_from_other_task)),
                             "--out", str(tmp / "xtask")], capture_output=True, text=True)
    check("a source used by two tasks is refused", r.returncode == 2, f"rc={r.returncode}")
    check("the refusal names the source and both stages",
          "again" in r.stderr or "used more than once" in r.stderr, r.stderr[-160:])

    def thin(rows):
        rows[:] = [x for x in rows
                   if not (x["length_cap"] == 16384 and int(x["source_id"].split("_")[-1]) >= 6)]
    r = subprocess.run(py + ["--rows", str(_panel_fixture(tmp, thin)),
                             "--out", str(tmp / "thin")], capture_output=True, text=True)
    check("a cell too thin to fill S is refused", r.returncode == 2, f"rc={r.returncode}")


def test_scoring_contract():
    multi = {"task": "niah_multivalue", "references": ["17", "23"]}
    check("strict multi-answer requires every gold",
          L.strict_task_score(multi, "17") == 0.0, "")
    check("strict multi-answer accepts complete recall",
          L.strict_task_score(multi, "17, 23") == 1.0, "")
    check("strict multi-answer explicitly allows extra text",
          L.strict_task_score(multi, "answer: 17, 23") == 1.0, "")
    check("full-string metric rejects the same extra text",
          L.full_string_exact(multi, "answer: 17, 23") == 0.0, "")
    check("full-string metric preserves gold order",
          L.full_string_exact(multi, "23, 17") == 0.0, "")
    check("full-string metric accepts punctuation-only separators",
          L.full_string_exact(multi, "17, 23") == 1.0, "")
    qa = {"task": "qa_1", "references": ["The Eiffel Tower"]}
    em, f1 = L.qa_scores(qa, "eiffel tower")
    check("QA keeps standard article-insensitive EM", em == 1.0, em)
    check("QA keeps standard token F1", f1 == 1.0, f1)
    class Scorer:
        source_sha256 = "scorer"
        scoring_contract_revision = "v1"
        scoring_contract_sha256 = "contract"
        @staticmethod
        def score(row, text):
            return 1.0
    identity_row = {
        "row_id": "r", "task": "niah_single_2", "length_cap": 8192,
        "source_document_id": "s", "semantic_group_id": "g", "split": "P",
        "prompt_ids": [1, 2], "prompt_sha256": L.ids_digest([1, 2]),
        "references": ["17"], "max_new_tokens": 2,
        "scorer_revision": "upstream-v1", "evidence_positions": [1],
        "distractor_positions": [],
    }
    rec = L.row_template(identity_row, "MR", "17", [17], "MR", "op",
                         scorer=Scorer(), eos=False, cap_hit=False)
    check("unterminated rows remain scored but are explicitly labeled",
          rec["validity_status"] == "VALID_SCORED_NO_EOS", rec["validity_status"])
    check("result rows preserve the complete resume identity",
          L.row_identity(rec) == L.row_identity(identity_row), "")


def test_controlled_modules():
    g = __import__("operators").Geometry.from_native()
    mr = L.build_control("MR", g).nu()
    bm = L.build_control("BM", g).nu()
    check("L1 MR a=1 reuses the L0 frequency table",
          np.array_equal(L.build_arm("L1_MR_A100", g).nu(), mr), "")
    check("L1 BM a=1 reuses the L0 frequency table",
          np.array_equal(L.build_arm("L1_BM_A100", g).nu(), bm), "")
    check("L2 base-band MR reuses L0",
          np.array_equal(L.build_arm("L2_MR_BF32_BS1", g).nu(), mr), "")
    check("L2 base-band BM reuses L0",
          np.array_equal(L.build_arm("L2_BM_BF32_BS1", g).nu(), bm), "")
    smooth = L.build_arm("MR_AREA_SMOOTH", g)
    log_scale = math.log(g.scale)
    smooth_m = -np.log(smooth.nu() / g.omega) / log_scale
    mr_m = -np.log(mr / g.omega) / log_scale
    check("MR-area smooth candidate exactly matches MR exponent area",
          math.isclose(float(smooth_m.sum()), float(mr_m.sum()),
                       rel_tol=0.0, abs_tol=1e-12),
          f"smooth={smooth_m.sum()} mr={mr_m.sum()}")
    check("MR-area smooth candidate preserves both plateau endpoints",
          smooth_m[g.low] == 0.0 and math.isclose(smooth_m[g.high], 1.0)
          and np.allclose(smooth_m[g.high + 1:], 1.0), smooth_m)
    smooth_increments = np.diff(smooth_m[g.low:g.high + 1])
    check("MR-area smooth candidate tapers both boundary increments",
          smooth_increments[0] < smooth_increments[1]
          and smooth_increments[-1] < smooth_increments[-2], smooth_increments)
    bridge = L.build_arm("RIBB_A2_B1P5", g)
    bridge_m = -np.log(bridge.nu() / g.omega) / log_scale
    bm_m = -np.log(bm / g.omega) / log_scale
    interior = slice(g.low + 1, g.high)
    check("RIBB alpha2 beta1.5 is pointwise between MR and BM",
          np.all(bridge_m[interior] > mr_m[interior])
          and np.all(bridge_m[interior] < bm_m[interior]), bridge_m)
    check("RIBB alpha2 beta1.5 preserves exact plateau endpoints",
          bridge_m[g.low] == 0.0 and math.isclose(bridge_m[g.high], 1.0)
          and np.allclose(bridge_m[g.high + 1:], 1.0), bridge_m)
    bridge_g1 = L.build_arm("RIBB_A2_B1P5_G1", g)
    check("RIBB gain-one diagnostic changes only Q/K gain",
          np.array_equal(bridge_g1.nu(), bridge.nu())
          and np.all(bridge_g1.q_amp(np.array([0.0])) == 1.0)
          and np.all(bridge.q_amp(np.array([0.0])) == g.gain), "")
    for base_name in ("MR", "BM"):
        base = L.build_arm(base_name, g)
        gain_one = L.build_arm(f"{base_name}_G1", g)
        check(f"{base_name} gain-one control changes only Q/K gain",
              np.array_equal(gain_one.nu(), base.nu())
              and np.all(gain_one.q_amp(np.array([0.0])) == 1.0)
              and np.all(base.q_amp(np.array([0.0])) == g.gain), "")
    g16 = __import__("operators").Geometry.from_native(
        g.native_inv_freq, window=g.window, theta=g.theta, scale=16.0,
        K=g.K, head_dim=g.head_dim, low=g.low, high=g.high, n=g.n)
    mr16 = L.build_control("MR", g16).nu()
    f16g1 = L.build_arm("MR_FS16_GS1", g16)
    f1g16 = L.build_arm("MR_FS1_GS16", g16)
    f16g4 = L.build_arm("MR_FS16_GS4", g16)
    f4g16 = L.build_arm("MR_FS4_GS16", g16)
    check("scale/gain control keeps the s16 table while removing gain",
          np.array_equal(f16g1.nu(), mr16)
          and np.all(f16g1.q_amp(np.array([0.0])) == 1.0), "")
    check("scale/gain control isolates s16 gain on native frequencies",
          np.array_equal(f1g16.nu(), g16.omega)
          and np.all(f1g16.q_amp(np.array([0.0])) == g16.gain), "")
    check("scale/gain control pairs s16 frequencies with s4 gain",
          np.array_equal(f16g4.nu(), mr16)
          and np.allclose(f16g4.q_amp(np.array([0.0])), 1.0 + 0.1 * math.log(4)), "")
    check("scale/gain control pairs s4 frequencies with s16 gain",
          np.array_equal(f4g16.nu(), mr)
          and np.all(f4g16.q_amp(np.array([0.0])) == g16.gain), "")
    names = (list(L.CONTROLLED_MODULES) + list(L.MATCHED_CONTROLS)
             + list(L.SCALE_GAIN_CONTROLS) + list(L.TABLE_GAIN_CONTROLS))
    built = [L.build_arm(name, g) for name in names]
    check("every controlled-module/matched-control table is finite",
          all(np.all(np.isfinite(op.q_phase(np.array([0., 8192., 32767.]))))
              for op in built), f"n={len(built)}")


def test_coverage_auc():
    caps = (8192, 16384, 32768)
    constant = {str(cap): 0.42 for cap in caps}
    check("log-length AUC preserves a constant curve",
          math.isclose(C.log_length_auc(constant, caps), 0.42,
                       rel_tol=0.0, abs_tol=1e-15), constant)
    two_point = {"8192": 0.2, "32768": 0.8}
    check("two-point normalized log-length AUC is the endpoint mean",
          math.isclose(C.log_length_auc(two_point, (8192, 32768)), 0.5,
                       rel_tol=0.0, abs_tol=1e-15), two_point)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    test_power()
    test_budget()
    test_paired_rejections()
    test_paired_arithmetic()
    test_baseline_hole()
    test_bootstrap()
    test_upgrade_report_cli()
    test_operators()
    test_rotation()
    test_split_panel()
    test_scoring_contract()
    test_controlled_modules()
    test_coverage_auc()

    n_pass = sum(1 for r in RESULTS if r["pass"])
    for r in RESULTS:
        if not r["pass"]:
            print(f"  FAIL {r['test']}  {r['detail']}")
    print(f"{n_pass}/{len(RESULTS)} tool tests pass")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"n_pass": n_pass, "n_tests": len(RESULTS), "tests": RESULTS,
             "nature": "artificial fixtures; none of these numbers is a model result"},
            indent=2))
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
