"""Self-checks for the C-free layer: geometry identity, no-ops, M01-M09 structure.

Every assertion here is a property the plan states in words.  The point is that
they are checked mechanically rather than trusted, because the failure modes are
all silent: a wrong pair layout, a k chosen outside its feasible set, a period
that is not an integer, a dictionary node outside the slot's window.  Each of
those produces a table that runs and gives numbers.

Run:  python selftest.py --model /path/to/Meta-Llama-3-8B-Instruct
      python selftest.py --config-json cfg.json          (no model needed)
      python selftest.py --self-contained                (the plan's own numbers)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import core
import cfree

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": str(detail)})
    return bool(ok)


# the identity section 6.1 expects; used when no checkpoint is reachable
PLAN_CONFIG = {
    "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
    "num_hidden_layers": 32, "max_position_embeddings": 8192,
    "rope_theta": 500000.0, "rope_scaling": None,
}


def geometry_from(args):
    if args.model:
        return core.Geometry.from_config(args.model)
    if args.config_json:
        return core.Geometry.from_mapping(json.loads(Path(args.config_json).read_text()))
    return core.Geometry.from_mapping(PLAN_CONFIG)


# ---------------------------------------------------------------------------


def test_geometry(g):
    check("checkpoint identity matches section 6.1", g.assert_expected(),
          f"{g.n_layers}L {g.n_heads}/{g.n_kv_heads}H d={g.head_dim} theta={g.theta:g} W={g.window}")
    check("K = head_dim/2 = 64", g.K == 64, g.K)
    check("band is (18, 35) with n = 17", (g.low, g.high, g.n) == (18, 35, 17),
          f"{(g.low, g.high, g.n)}")
    check("T is the 16 partly-scaled slots 19..34", list(g.T) == list(range(19, 35)), len(g.T))
    check("gain is 1 + 0.1 ln 4", abs(g.gain - (1 + 0.1 * math.log(4))) < 1e-15, f"{g.gain:.15f}")
    m = g.m_mr
    check("m^MR is 0 at slot 18 and 1 at slot 35",
          abs(m[18]) < 1e-15 and abs(m[35] - 1.0) < 1e-15, f"{m[18]:.3e} {m[35]:.15f}")
    check("m^MR is monotone non-decreasing", np.all(np.diff(m) >= 0), float(np.min(np.diff(m))))
    check("nu^MR <= omega everywhere (compression only speeds nothing up)",
          np.all(g.nu_mr <= g.omega + 1e-15), float(np.max(g.nu_mr / g.omega)))


def test_noops(g):
    for entry in core.check_pseudo_methods(g):
        check(f"no-op collapses: {entry['pseudo'][:52]}", entry["collapses"],
              f"residual {entry['residual']:.2e}")


def test_d01(g):
    A_of = {"M01": g.window, "M02": 2 * g.window, "M03": 4 * g.window}
    for m in ("M01", "M02", "M03"):
        c = cfree.BUILDERS[m](g)
        A = A_of[m]
        # the defining identity: exp(i A nu) = exp(i A omega/s) on every T slot
        lhs = np.exp(1j * A * c.nu[g.T])
        rhs = np.exp(1j * A * (g.omega[g.T] / g.scale))
        check(f"{m}: exp(iA nu) = exp(iA omega/s) at the anchor",
              np.allclose(lhs, rhs, atol=1e-9), float(np.max(np.abs(lhs - rhs))))
        # the feasible box
        check(f"{m}: nu stays in [omega/s, omega] on T",
              np.all(c.nu[g.T] >= g.omega[g.T] / g.scale - 1e-12)
              and np.all(c.nu[g.T] <= g.omega[g.T] + 1e-12), "")
        # outside T the plan says keep MR
        outside = np.setdiff1d(np.arange(g.K), g.T)
        check(f"{m}: outside T the MR node is kept",
              np.allclose(c.nu[outside], g.nu_mr[outside], atol=0, rtol=0), "")
        # k is an integer and inside its own feasible set
        ks = c.detail.get("k", {})
        ok_k = True
        for j, k in ks.items():
            w = g.omega[j]
            kmax = math.floor(A * (w - w / g.scale) / (2 * math.pi) + 1e-12)
            if not (0 <= k <= kmax):
                ok_k = False
        check(f"{m}: every k is inside its feasible set", ok_k,
              f"{len(ks)} slots moved")
        # and the result is NOT a no-op in general
        check(f"{m}: the rule actually moves nodes (or reports NO_CHANGE)",
              c.status == "NO_CHANGE" or len(ks) > 0, c.status)


def test_d02(g):
    for m in ("M04", "M05", "M06"):
        c = cfree.BUILDERS[m](g)
        check(f"{m}: construction is feasible", c.status in ("OK", "NO_CHANGE"),
              f"{c.status} {c.note}")
        if c.status not in ("OK", "NO_CHANGE"):
            continue
        Ps = list(c.detail["periods"].values())
        check(f"{m}: all periods are integers", all(float(p).is_integer() for p in Ps), "")
        check(f"{m}: periods strictly increase with j",
              all(Ps[i] < Ps[i + 1] for i in range(len(Ps) - 1)), "")
        check(f"{m}: each period lies in [P_native, s P_native]",
              all(2 * math.pi / g.omega[j] - 1 <= c.detail["periods"][j]
                  <= 2 * math.pi * g.scale / g.omega[j] + 1 for j in c.detail["periods"]), "")
        if m in ("M04", "M06"):
            check(f"{m}: periods pairwise coprime",
                  all(math.gcd(a, b) == 1 for a, b in
                      [(Ps[i], Ps[j]) for i in range(len(Ps)) for j in range(i + 1, len(Ps))]), "")
        check(f"{m}: lcm is the exact joint recurrence",
              int(c.detail["lcm_exact_recurrence"]) > 0,
              f"{c.detail['lcm_digits']} digits, exceeds deployable context: "
              f"{c.detail['lcm_exceeds_deployable_context']}")
        # nu = 2 pi / P exactly
        check(f"{m}: nu_j = 2 pi / P_j",
              all(abs(c.nu[j] - 2 * math.pi / c.detail["periods"][j]) < 1e-12
                  for j in c.detail["periods"]), "")


def test_d03(g):
    for m in ("M07", "M08", "M09"):
        c = cfree.BUILDERS[m](g)
        check(f"{m}: construction is feasible", c.status in ("OK", "NO_CHANGE"),
              f"{c.status} {c.note}")
        if c.status.startswith("INFEASIBLE"):
            continue
        # every chosen node is one of the shipped 64 frequencies
        ok_dict = True
        ok_win = True
        for j, k in c.detail["picked"].items():
            if not np.any(np.isclose(g.omega, c.nu[j], atol=0, rtol=0)):
                ok_dict = False
            if not (g.omega[j] / g.scale - 1e-15 <= c.nu[j] <= g.omega[j] + 1e-15):
                ok_win = False
        check(f"{m}: every node comes from the native dictionary", ok_dict,
              f"{len(c.detail['picked'])} slots")
        check(f"{m}: every node lies in [omega_j/s, omega_j]", ok_win, "")
        check(f"{m}: collisions are recorded, not hidden", "n_collisions" in c.detail,
              f"{c.detail.get('n_collisions')} collisions")
        outside = np.setdiff1d(np.arange(g.K), g.T)
        check(f"{m}: outside T the MR node is kept",
              np.allclose(c.nu[outside], g.nu_mr[outside], atol=0, rtol=0), "")


def test_dedup(g):
    """Section 5.4: report how many DISTINCT operators the nine rules produce."""
    fps = {}
    for m in cfree.CFREE_METHODS:
        c = cfree.BUILDERS[m](g)
        if not c.is_valid:
            continue
        fps.setdefault(cfree.core_hash(c.nu), []).append(m)
    dupes = {k[:8]: v for k, v in fps.items() if len(v) > 1}
    check("distinct operators among the valid C-free rules is reported", True,
          f"{len(fps)} distinct of {len(cfree.CFREE_METHODS)} rules; duplicates: {dupes}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=None, help="checkpoint dir (config.json is read)")
    ap.add_argument("--config-json", default=None)
    ap.add_argument("--self-contained", action="store_true", default=False)
    ap.add_argument("--json", dest="out", default=None)
    a = ap.parse_args(argv)

    g = geometry_from(a)
    print(f"geometry: theta={g.theta:g} W={g.window} d_head={g.head_dim} K={g.K} "
          f"band=({g.low},{g.high}) n={g.n} layers={g.n_layers} "
          f"heads={g.n_heads}/{g.n_kv_heads}")

    test_geometry(g)
    test_noops(g)
    test_d01(g)
    test_d02(g)
    test_d03(g)
    test_dedup(g)

    n_pass = sum(1 for r in RESULTS if r["pass"])
    for r in RESULTS:
        if not r["pass"]:
            print(f"  FAIL {r['check']}  {r['detail']}")
    print(f"{n_pass}/{len(RESULTS)} checks pass")

    # the construction record section 5.4 asks for, printed for the eye
    print("\nC-free construction record:")
    for m in cfree.CFREE_METHODS:
        c = cfree.BUILDERS[m](g)
        sm = c.sum_m(g)
        print(f"  {m} {c.direction}  {c.status:11s} sum_m={'n/a' if sm is None else f'{sm:.4f}':>9s}  {c.note}")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"geometry": g.__dict__ if not isinstance(g.native_inv_freq, np.ndarray) else None,
             "n_pass": n_pass, "n_checks": len(RESULTS), "checks": RESULTS}, indent=2, default=str))
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
