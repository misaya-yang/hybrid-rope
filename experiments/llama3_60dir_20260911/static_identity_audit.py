"""static_identity_audit.json -- are the 18 static tables new, or re-runs?

Plan section 9.1 lists `static_identity_audit.json` as "bit-level duplicate check
of the new 18 static tables against the archived arrays", and section 6.1 makes
history dedup a precondition for spending GPU time:

    "history dedup by: intervention object + region + reference frequency +
     actual parameters + gain + scope.  A different array only rules out a
     bit-level duplicate; it does not mean a new mechanism.  If an isomorphic
     configuration has already been run, reuse the original material rather than
     adding a new name to fill the count."

Two things the plan leaves implicit, resolved here and stated in the output:

1. **Which 18.**  The plan never enumerates them.  They are exactly the
   configurations whose scope leaves the operator a static frequency table:
   `frequency` (D01, D02), `frequency_assignment` (D03, D06),
   `signed_frequency` (D04), `dc_frequency` (D05) -- 6 directions x 3 = 18.
   Everything from D07 on changes amplitude, phase intercept, pair metric or the
   position map, and is not a "static table" at all.

2. **Which space to compare in.**  The archived tables live in the Qwen
   geometry (theta 1e6, W 32768, band [23,40]); the review's tables are defined
   for Llama-3-8B (theta 5e5, W 8192, band [18,35]).  Comparing `nu_j` across
   those two is meaningless.  So each config is rebuilt **in the archive's own
   geometry** and the comparison is made on the dimensionless m-profile
   `m_j = log_s(omega_j / nu_j)`, which is the geometry-independent description
   of a policy.  A match there means "we have run this policy before", which is
   the question section 6.1 actually asks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import operators as O

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ARCHIVE = REPO / "analysis/unify_20260910/tables/ground_truth_tables.json"

STATIC_SCOPES = ("frequency", "frequency_assignment", "signed_frequency", "dc_frequency")

# the archive's geometry, read from its own meta.constants
QWEN = dict(theta=1_000_000.0, window=32768, scale=4.0)


def static_configs():
    return [c for c in O.config_ids() if O.scope_of(c) in STATIC_SCOPES]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--out", type=Path, default=HERE / "static_identity_audit.json")
    ap.add_argument("--close-tol", type=float, default=1e-6,
                    help="max |dm| counted as the same policy")
    ap.add_argument("--spectrum-tol", type=float, default=1e-6,
                    help="sorted-profile tolerance; the archive's m is fp32-derived, so "
                         "bit-exact comparison is unattainable and machine epsilon is wrong")
    a = ap.parse_args(argv)

    if not a.archive.exists():
        raise SystemExit(f"archive not found: {a.archive}")
    arch = json.loads(a.archive.read_text(encoding="utf-8"))
    methods = arch["methods"]
    const = arch["meta"]["constants"]
    geom_q = O.Geometry.from_native(None, window=int(const["W"]), theta=float(const["base"]),
                                    scale=float(const["S"]))

    g_llama = O.Geometry.from_native(None, window=8192, theta=500000.0, scale=4.0)

    cfgs = static_configs()
    results = []
    for cid in cfgs:
        op_q = O.build(cid, geom_q)          # rebuilt in the ARCHIVE's geometry
        op_l = O.build(cid, g_llama)         # as it will actually run
        # m exists only where nu > 0.  For D04 (negated frequency) and D05
        # (identity slot) the m-profile comparison is UNDEFINED, not "different"
        # -- reporting a bare nan/inf in a column of numbers is how an undefined
        # quantity gets read as a measurement downstream.
        m_ok = bool(np.all(op_q.nu() > 0) and np.all(np.isfinite(op_q.nu())))
        if m_ok:
            m_q = np.log(geom_q.omega / op_q.nu()) / np.log(geom_q.scale)
        else:
            m_q = None

        if m_q is not None:
            # some archive entries carry a scalar or absent m_j; comparing against
            # those is not a comparison, so they are excluded and counted.
            usable = {n: np.asarray(m.get("m_j"), dtype=np.float64)
                      for n, m in methods.items()}
            usable = {n: v for n, v in usable.items()
                      if v.ndim == 1 and v.shape[0] == geom_q.K
                      and np.all(np.isfinite(v))}
            skipped = sorted(set(methods) - set(usable))
            best, best_d = None, None
            for name, ma in usable.items():
                d = float(np.max(np.abs(m_q - ma)))
                if best_d is None or d < best_d:
                    best, best_d = name, d
            exact = [name for name, ma in usable.items()
                     if np.array_equal(m_q.astype(np.float32), ma.astype(np.float32))]
            close = [name for name, ma in usable.items()
                     if float(np.max(np.abs(m_q - ma))) < a.close_tol]
            # D03 is DEFINED by keeping the spectrum and changing the assignment,
            # so an elementwise difference is expected and a sorted-match is by
            # design, not a duplicate finding.  Both are recorded separately so
            # the one cannot be mistaken for the other.
            # The archive's m_j is INVERTED FROM FP32 DEPLOYED VALUES, so it
            # carries ~1e-7 of re-derivation noise and a bit-exact spectrum
            # comparison can never succeed against it.  atol is set to that
            # noise floor, not to machine epsilon.
            # SPECTRUM = the multiset of FREQUENCIES, i.e. of log(nu), NOT of m.
            # m_j is defined relative to slot j's own native frequency, so moving
            # a frequency from slot 19 to slot 20 makes m negative at slot 20 even
            # though the spectrum is untouched.  Comparing sorted(m) would call a
            # pure permutation a spectral change, which is exactly backwards for
            # D03 -- whose whole point is that the spectrum is preserved.
            lognu_new = np.log(op_q.nu())
            spectrum = []
            for name, ma in usable.items():
                lognu = np.log(geom_q.omega) - ma * np.log(geom_q.scale)
                if np.allclose(np.sort(lognu_new), np.sort(lognu), atol=a.spectrum_tol):
                    spectrum.append(name)
        else:
            usable, skipped = {}, sorted(methods)
            best, best_d, exact, close, spectrum = None, None, [], [], []

        results.append({
            "config": cid,
            "scope": O.scope_of(cid),
            "policy": op_q.policy,
            "sum_m_llama": float(np.sum(np.log(g_llama.omega / op_l.nu()) / np.log(g_llama.scale)))
            if np.all(op_l.nu() > 0) else None,
            "sum_m_qwen": float(np.sum(m_q)) if m_q is not None else None,
            "m_coordinate_defined": m_ok,
            "m_undefined_reason": None if m_ok else
            ("zero frequency: the slot is an identity rotation and has no m" if np.any(op_q.nu() == 0)
             else "negative frequency: m = log(omega/nu)/log s is undefined"),
            "bit_identical_to": exact,
            "same_policy_as": close,
            "same_spectrum_as": spectrum,
            "archived_methods_compared": len(usable),
            "archived_methods_skipped": skipped,
            "closest_archived": best,
            "closest_max_abs_dm": best_d,
            "is_new": (not close) if m_ok else None,
        })

    n_new = sum(1 for r in results if r["is_new"] is True)
    n_undef = sum(1 for r in results if r["is_new"] is None)
    out = {
        "question": "have any of the 18 static tables already been run?",
        "n_static_configs": len(results),
        "n_new": n_new,
        "n_already_run": len(results) - n_new - n_undef,
        "n_m_coordinate_undefined": n_undef,
        "n_spectrum_preserving": sum(1 for r in results if r["same_spectrum_as"]),
        "spectrum_definition": ("the multiset of log(nu), i.e. of frequencies -- NOT of m: "
                                "m is slot-relative, so a permutation makes m negative at "
                                "the receiving slot while leaving the spectrum intact"),
        "archive": str(a.archive.relative_to(REPO)),
        "archive_geometry": {"theta": geom_q.theta, "window": geom_q.window,
                             "scale": geom_q.scale, "low": geom_q.low,
                             "high": geom_q.high, "n": geom_q.n},
        "review_geometry": {"theta": g_llama.theta, "window": g_llama.window,
                            "scale": g_llama.scale, "low": g_llama.low,
                            "high": g_llama.high, "n": g_llama.n},
        "method": ("each config is rebuilt in the ARCHIVE's geometry and compared on the "
                   "dimensionless m-profile m_j = log_s(omega_j/nu_j); nu_j is not comparable "
                   "across the two geometries"),
        "which_18": ("the configurations whose scope leaves the operator a static frequency "
                     "table: " + ", ".join(STATIC_SCOPES)),
        "close_tolerance": a.close_tol,
        "spectrum_tolerance": a.spectrum_tol,
        "fp32_note": ("the archive's m_j is inverted from fp32 deployed values and carries "
                      "~1e-7 of re-derivation noise, so bit-identity against it is not "
                      "attainable; tolerances are set to that floor"),
        "results": results,
        "limitation_d03": (
            "D03's slot ranges are HARD-CODED ([19,34] for the adjacent/cyclic policies, "
            "[35,40] for the block swap) because the plan names them literally. Those are "
            "Llama-band slots; the archive's band is [23,40], so rebuilding D03 in the "
            "archive's geometry does not exercise the same region. The three spectrum "
            "matches above are therefore weaker evidence than they look -- D03c in "
            "particular rolls over slots that are mostly below the Qwen band, where the "
            "table is all zeros and the roll is nearly trivial. A fair D03 dedup needs the "
            "plan's ranges re-expressed band-relative, which the plan does not do."),
        "caveat": ("a difference here rules out only a bit-level duplicate.  It does NOT "
                   "establish a new mechanism -- section 6.1: 数组不同只排除逐位重复，"
                   "不代表新机制.  The archive also holds 38 methods whose own band/low "
                   "differ, so a small max|dm| against a differently-banded method is not "
                   "by itself evidence of a duplicate; read the `closest_archived` column "
                   "together with that method's band."),
    }
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(results)} static configs: {n_new} new, "
          f"{len(results) - n_new - n_undef} matching an archived policy, "
          f"{n_undef} with an undefined m-coordinate")
    print(f"  spectrum-preserving (by design, NOT a duplicate): "
          f"{sum(1 for r in results if r['same_spectrum_as'])}")
    print(f"  {'config':8s} {'scope':22s} {'closest archived':26s} max|dm|")
    for r in results:
        d = "   n/a (m undefined)" if r["closest_max_abs_dm"] is None else f"{r['closest_max_abs_dm']:.3e}"
        if r["is_new"] is None:
            tag = "   (m undefined: no comparison possible)"
        elif r["same_policy_as"]:
            tag = "   <-- SAME POLICY"
        elif r["same_spectrum_as"]:
            tag = "   <-- SPECTRUM match (by design for D03, not a duplicate)"
        else:
            tag = ""
        print(f"  {r['config']:8s} {r['scope']:22s} {str(r['closest_archived']):26s} {d}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
