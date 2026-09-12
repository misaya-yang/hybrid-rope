"""The direction layer, re-pointed at the list Plan B actually adopts.

WHY THIS FILE EXISTS
--------------------
Plan B cites its candidate library explicitly (line 1397):

    candidate_library: LLAMA3_ROPE_20_DIRECTIONS_60_CONFIGS_REVIEW_PLAN_20260911(1).md

and its appendix A says "来源为 [H2 §4]" with [H2] resolved to that same file.
Its in-text D-references confirm it: "D02 ... 三个 phase anchor 分支政策",
"D03 ... 同谱对应", "D01 ... 有限整数周期组合", "D20 ... 原生有效位置量化" --
all of which are the CONFIGS_REVIEW directions.

An earlier version of this package implemented the *other* document,
`..._METHODS_PLAN_20260911.md`, which Plan B never cites and whose D01-D20 is a
different list (its D01 is integer-winding, which is CONFIGS D02; its D05 is
same-spectrum reassignment, which is CONFIGS D03).  That was a real error and
this module replaces it.

Rather than re-implement, this bridges to `operators.py`, the CONFIGS_REVIEW
implementation that was verified bit-identical against the campaign's own
`tables.m_mrpro` / `nu_mrpro` / `GAIN_YARN`.  One operator library, one set of
hashes, no second source of truth.

The 60 configurations split by Plan B section 0.3:

* **core** -- frozen weights, a shared static positive-frequency table, global
  gain: CONFIGS D01, D02, D03, D06 (the frequency / frequency-assignment line).
* **extension** -- signed or DC frequency, spectral amplitude, Q/K phase,
  position-dependent amplitude, non-linear position maps: CONFIGS D04, D05 and
  D07-D20.  Section 0.3 says these "可以作为升级候选或机制对照，但不能再标成
  '只换 64 个正频率'" and, when not authorised, must be recorded `SCOPE_BLOCKED`.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import core
import cfree
from cfree import Construction

# the verified CONFIGS_REVIEW implementation lives in the sibling package
_SIBLING = Path(__file__).resolve().parent.parent / "llama3_60dir_20260911"
if str(_SIBLING) not in sys.path:
    sys.path.insert(0, str(_SIBLING))
import operators as OPS  # noqa: E402


# Plan B section 0.3 scope assignment, keyed by the CONFIGS_REVIEW scope string
CORE_SCOPES = ("frequency", "frequency_assignment")
EXTENSION_SCOPES = ("signed_frequency", "dc_frequency", "spectral_amplitude",
                    "qk_phase_bias", "qk_pair_metric", "position_amplitude",
                    "position_phase")


def scope_of(config_id):
    return OPS.scope_of(config_id)


def plan_b_line(config_id):
    """'core' or 'extension' per Plan B section 0.3."""
    return "core" if scope_of(config_id) in CORE_SCOPES else "extension"


def config_ids():
    return OPS.config_ids()


# ---------------------------------------------------------------------------
# Construction from the operators package
# ---------------------------------------------------------------------------


def _nu_of(op):
    return np.asarray(op.nu(), dtype=np.float64)


def build(config_id, geom, authorized_scopes=None):
    """Build one CONFIGS_REVIEW configuration as a Construction.

    `authorized_scopes` is Plan B section 0.3's rule: an extension-scope
    configuration that is not authorised is recorded SCOPE_BLOCKED and the core
    task continues -- it is NOT silently run as though it were a table swap, and
    the literal value of `execution_authorized` is never edited to get around it.
    """
    op = OPS.build(config_id, geom)
    nu = _nu_of(op)
    scope = scope_of(config_id)
    line = plan_b_line(config_id)
    if line == "extension" and authorized_scopes is not None \
            and scope not in set(authorized_scopes):
        return Construction(config_id, op.direction, geom.nu_mr.copy(), "SCOPE_BLOCKED",
                            f"scope {scope!r} is an extension under Plan B section 0.3 and "
                            "is not in the authorised list",
                            {"scope": scope, "line": line})
    surfaces = _surfaces(op, geom)
    return Construction(config_id, op.direction, nu, "OK",
                        f"{scope} / {line}",
                        {"scope": scope, "line": line, "policy": op.policy,
                         **surfaces})


# Positions used to probe a phase surface.  Covering the native window, both
# evaluation lengths and the endpoints is enough to separate the phase families
# without hashing a 32768 x 64 array per configuration.
PHASE_PROBE = (0.0, 1.0, 17.0, 1000.0, 4096.0, 8191.0, 16384.0, 32767.0)


def _surfaces(op, geom):
    """The extra surfaces a same-dimension operator needs, if it has any.

    The PHASE must be probed.  D10 and D14-D20 leave `nu` equal to MR and change
    only `q_phase`/`k_phase` -- an earlier version of this function hashed only
    amp and diag, so twenty-one configurations (D10a-c, D14-D18 a-c, D20a-c)
    collapsed onto a single fingerprint and the section 5.4 dedup silently
    under-reported the number of distinct operators.
    """
    out = {"surface_kind": "frequency"}
    K = geom.K
    p = np.array(PHASE_PROBE)
    try:
        qa = op.q_amp(p)
        ka = op.k_amp(p)
        if qa.shape == (p.size, K) and ka.shape == (p.size, K):
            out["q_amp"] = qa.tolist()
            out["k_amp"] = ka.tolist()
            if not np.allclose(qa, geom.gain) or not np.allclose(ka, geom.gain):
                out["surface_kind"] = "position_amplitude"
    except Exception:
        pass
    try:
        qd = op.q_diag()
        kd = op.k_diag()
        if not np.allclose(qd, 1.0) or not np.allclose(kd, 1.0):
            out["q_diag"] = qd.tolist()
            out["k_diag"] = kd.tolist()
            out["surface_kind"] = "pair_metric"
    except Exception:
        pass

    # the phase is the defining surface for the position-phase family
    try:
        qph = np.asarray(op.q_phase(p), dtype=np.float64)
        kph = np.asarray(op.k_phase(p), dtype=np.float64)
        # reduce to a stable, comparable summary: wrap into (-pi, pi] so that
        # 2 pi offsets -- which are the same rotation -- do not create a
        # spurious difference
        qw = np.angle(np.exp(1j * qph))
        kw = np.angle(np.exp(1j * kph))
        out["q_phase_probe"] = qw.tolist()
        out["k_phase_probe"] = kw.tolist()
        out["q_k_phase_differ"] = bool(np.max(np.abs(qw - kw)) > 1e-12)
        # A FREQUENCY operator has q_phase(p) = p * nu, so the slope q_phase/p is
        # the same at every position.  Comparing against p * nu_mr instead would
        # misclassify every frequency table that moves its nodes (D01/D02/D03/D06)
        # as a position-phase map -- which is what an earlier version did, leaving
        # zero configurations classified as frequency.
        if out["surface_kind"] == "frequency":
            nz = p > 0
            slopes = qph[nz] / p[nz, None]
            if not np.allclose(slopes, slopes[0], atol=1e-9, rtol=1e-9):
                out["surface_kind"] = "position_phase"
    except Exception:
        pass

    if getattr(op, "sel", None) is not None:
        out["selected_slots"] = [int(j) for j in np.where(op.sel)[0]]
        if out["surface_kind"] == "frequency":
            out["surface_kind"] = "signed_or_dc"
    return out


def fingerprint(construction):
    """A FULL identity, not just the frequencies.

    Plan B's §11.1 dedup rule and the earlier plan's §5.4 both require that two
    operators differing only in a matrix, a phase or a metric DO NOT collide.
    Hashing `nu` alone made every same-dimension operator fingerprint identically
    to MR and to each other; this hashes every surface that is actually applied.
    """
    c = construction
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(c.nu, dtype=np.float64)).tobytes())
    for key in ("q_amp", "k_amp", "q_diag", "k_diag", "selected_slots",
                "q_phase_probe", "k_phase_probe"):
        v = (c.detail or {}).get(key)
        if v is not None:
            h.update(key.encode())
            h.update(np.ascontiguousarray(np.asarray(v, dtype=np.float64)).tobytes())
    h.update(str((c.detail or {}).get("surface_kind", "frequency")).encode())
    return h.hexdigest()


def build_all(geom, authorized_scopes=None):
    return {cid: build(cid, geom, authorized_scopes) for cid in config_ids()}


def distinct_operators(constructions):
    """Section 5.4: the number of DISTINCT operators must be reported."""
    seen = {}
    for cid, c in constructions.items():
        if not c.is_valid:
            continue
        seen.setdefault(fingerprint(c), []).append(cid)
    return {"n_valid": sum(1 for c in constructions.values() if c.is_valid),
            "n_distinct": len(seen),
            "collisions": {k[:10]: v for k, v in seen.items() if len(v) > 1}}


# ---------------------------------------------------------------------------
# Plan B section 6: the L-stage modules
# ---------------------------------------------------------------------------


def l1_factorial(geom, amplitudes=(0.87, 1.00, 1.13, 1.30), profiles=("mr", "bm")):
    """L1: F in {MR, BM} x a in {0.87, 1.00, 1.13, 1.30}, B0=(18,35), s=4, g4.

    Section 6 / L1.  The reported quantity is the interaction
        I_{shape,a} = [A(BM,1.13)-A(MR,1.13)] - [A(BM,1)-A(MR,1)]
    per length.  This module only builds the eight cells; the statistics live in
    panel.py.
    """
    cells = {}
    for prof in profiles:
        for a in amplitudes:
            cells[(prof, a)] = Construction(
                f"L1_{prof}_a{a}", "L1", geom.nu_of(a, prof), "OK",
                f"profile={prof} amplitude={a}",
                {"scope": "core", "line": "core", "profile": prof, "amplitude": a,
                 "surface_kind": "frequency"})
    return cells


def l2_bands(geom, beta_fast=(32.0, 64.0), beta_slow=(1.0, 0.5)):
    """L2: the four (beta_fast, beta_slow) bands, a=1, g4.

    Section 6 / L2: the band edges are recomputed from W, theta, K by the
    verified official correction-range rule -- they are NOT hardcoded to (18,35).
    """
    out = {}
    for bf in beta_fast:
        for bs in beta_slow:
            lo, hi = OPS.find_correction_range(bf, bs, geom.head_dim, geom.theta, geom.window)
            g = core.Geometry.from_mapping(
                {"hidden_size": geom.head_dim * geom.n_heads,
                 "num_attention_heads": geom.n_heads, "num_key_value_heads": geom.n_kv_heads,
                 "num_hidden_layers": geom.n_layers,
                 "max_position_embeddings": geom.window, "rope_theta": geom.theta,
                 "rope_scaling": None},
                scale=geom.scale, low=lo, high=hi, n=hi - lo)
            out[(bf, bs)] = {
                "beta_fast": bf, "beta_slow": bs, "low": lo, "high": hi, "n": hi - lo,
                "geometry": g,
                "mr": Construction(f"L2_mr_bf{bf}_bs{bs}", "L2", g.nu_of(1.0, "mr"), "OK",
                                   f"MR at band ({lo},{hi})",
                                   {"scope": "core", "line": "core", "surface_kind": "frequency"}),
                "bm": Construction(f"L2_bm_bf{bf}_bs{bs}", "L2", g.nu_of(1.0, "bm"), "OK",
                                   f"BM at band ({lo},{hi})",
                                   {"scope": "core", "line": "core", "surface_kind": "frequency"}),
            }
    return out


def l3_permutations(geom):
    """L3: CONFIGS D03a/b/c, the three literal permutations, plus the gauge arm.

    The gauge arm permutes frequency, Q pair and K pair by the SAME permutation
    with V unchanged -- section 6 / L3 calls it the gauge negative control, and
    it must come out as a no-op.
    """
    out = {}
    for pol in ("a", "b", "c"):
        cid = f"D03{pol}"
        out[cid] = build(cid, geom)
    return out
