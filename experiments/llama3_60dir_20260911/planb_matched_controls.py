"""Plan B matched controls and a machine-readable parent registry.

This module is deliberately independent of the experimental runner.  It only
constructs controls whose algebra is fixed by Plan B and the existing
``operators.Operator`` API.  Controls needing model-derived M-dev activations
are represented as explicit ``BLOCKED`` records until those activations are
provided; no fallback statistic or guessed calibration is used.

The registry is for audit/queue construction.  It does not authorize scope,
select candidates, or produce a task result.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import operators as O


TOL = 1e-10


def _T(g):
    """The partly compressed slots ell+1 .. h-1."""
    return np.arange(g.low + 1, g.high, dtype=np.int64)


class _FrequencyControl(O.Operator):
    """An explicit static-frequency control with the normal global gain."""

    def __init__(self, geom, nu, name, policy, parent=None):
        super().__init__(geom)
        self._nu = np.asarray(nu, dtype=np.float64)
        if self._nu.shape != (geom.K,) or not np.all(np.isfinite(self._nu)):
            raise ValueError(f"{name}: frequency table must be finite shape ({geom.K},)")
        self.config = name
        self.direction = name
        self.policy = policy
        self.scope = "control"
        self.parent = parent


class _StaticGainControl(O.Operator):
    """MR frequencies with a fixed Q/K gain pair totaling ``score_gain``."""

    def __init__(self, geom, score_gain, name, parent="MR", q_gain=None, k_gain=None):
        super().__init__(geom)
        if not np.isfinite(score_gain) or score_gain <= 0:
            raise ValueError("static endpoint gain must be finite and positive")
        self._nu = geom.nu_mrpro.copy()
        self.config = name
        self.direction = name
        self.policy = f"MR frequency; static score multiplier={score_gain:.17g}"
        self.scope = "control"
        self.parent = parent
        self.score_gain = float(score_gain)
        self.q_gain = float(score_gain / geom.gain if q_gain is None else q_gain)
        self.k_gain = float(geom.gain if k_gain is None else k_gain)
        if not np.isclose(self.q_gain * self.k_gain, self.score_gain,
                          atol=1e-12, rtol=1e-12):
            raise ValueError("Q/K gain factors do not equal the declared score multiplier")

    def q_amp(self, p):
        return np.full((len(p), self.geom.K), self.q_gain, dtype=np.float64)

    def k_amp(self, t):
        return np.full((len(t), self.geom.K), self.k_gain, dtype=np.float64)


class _DropControl(O.Operator):
    """D05 matched DROP: zero Q/K pairs on a D05 mask, preserve content path."""

    def __init__(self, geom, mask_name="D05c"):
        super().__init__(geom)
        d05 = O.build(mask_name, geom)
        self._nu = geom.nu_mrpro.copy()
        self.mask = np.asarray(d05.sel, dtype=bool).copy()
        self.config = f"DROP_{mask_name}"
        self.direction = "DROP"
        self.policy = f"zero Q/K pairs on {mask_name} mask; MR phase parent"
        self.scope = "mechanism_control"
        self.parent = "MR"

    def q_amp(self, p):
        a = np.full((len(p), self.geom.K), self.geom.gain, dtype=np.float64)
        a[:, self.mask] = 0.0
        return a

    def k_amp(self, t):
        a = np.full((len(t), self.geom.K), self.geom.gain, dtype=np.float64)
        a[:, self.mask] = 0.0
        return a


class _AreaControl(_FrequencyControl):
    """D06 AREA: same outside bands, with T-area matched to one D06 arm."""

    def __init__(self, geom, d06_id):
        if d06_id not in {"D06a", "D06b", "D06c"}:
            raise ValueError("AREA requires one of D06a/D06b/D06c")
        d06 = O.build(d06_id, geom)
        m_d06 = np.log(geom.omega / d06.nu()) / math.log(geom.scale)
        if not np.all(np.isfinite(m_d06)):
            raise ValueError("AREA requires a positive D06 frequency table")
        t = _T(geom)
        m_mr = geom.m_mrpro
        outside = np.ones(geom.K, dtype=bool)
        outside[t] = False
        if not np.allclose(m_d06[outside], m_mr[outside], atol=TOL, rtol=0):
            raise ValueError(
                f"BLOCKED: {d06_id} changes slots outside T; fix the upstream "
                "D06 operator before constructing AREA")
        denom = float(np.sum(m_mr[t]))
        if denom <= 0:
            raise ValueError("AREA has zero MR middle-band area")
        factor = (float(np.sum(m_d06)) - float(np.sum(m_mr[outside]))) / denom
        if not np.isfinite(factor) or factor < 0:
            raise ValueError(f"AREA factor invalid: {factor!r}")
        m = m_mr.copy()
        m[t] *= factor
        nu = geom.omega * geom.scale ** (-m)
        super().__init__(geom, nu, f"AREA_{d06_id}",
                         f"MR outside T; T m-area matched to {d06_id}; factor={factor:.17g}",
                         parent="MR")
        self.reference_d06 = d06_id
        self.m_area = m
        self.factor = float(factor)


class _GaugeControl:
    """D03 gauge negative control, with Q/K pair and frequency permutation.

    The existing Operator API has no V tensor, so this object exposes the CPU
    score-level operation.  It permutes Q and K pairs identically and leaves V
    untouched at the caller boundary.  This is enough to test the required
    score invariance without pretending it is a complete HF hook.
    """

    def __init__(self, geom, d03_id="D03a"):
        if d03_id not in {"D03a", "D03b", "D03c"}:
            raise ValueError("gauge requires D03a/D03b/D03c")
        self.parent = _FrequencyControl(geom, geom.nu_mrpro, "MR", "MrRoPE-Pro")
        self.candidate = O.build(d03_id, geom)
        self.perm = np.asarray(self.candidate.perm, dtype=np.int64)
        self.config = f"GAUGE_{d03_id}"
        self.direction = "D03_GAUGE"
        self.policy = "same permutation on frequency, Q pairs and K pairs; V unchanged"
        self.scope = "mechanism_control"
        self.parent_id = "MR"
        self.reference_d03 = d03_id

    @staticmethod
    def _permute_pairs(x, perm):
        x = np.asarray(x, dtype=np.float64)
        k = x.shape[-1] // 2
        if x.shape[-1] != 2 * k:
            raise ValueError("Q/K last dimension must be even")
        return np.concatenate([x[..., :k][..., perm], x[..., k:][..., perm]], axis=-1)

    def score(self, q, k, positions_q, positions_k):
        """Return parent score and gauge score for matched Q/K rows."""
        q = np.asarray(q, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        if q.shape != k.shape or q.ndim != 2:
            raise ValueError("score expects matched q/k arrays of shape (N, head_dim)")
        pq = np.asarray(positions_q, dtype=np.float64)
        pk = np.asarray(positions_k, dtype=np.float64)
        if pq.shape != (q.shape[0],) or pk.shape != (k.shape[0],):
            raise ValueError("positions must have shape (N,)")
        # ``apply_rotation`` treats its penultimate axis as position.  Put the
        # matched rows there (1, N, D), rather than (N, 1, D), which would
        # broadcast every row against every phase and silently select row 0.
        parent_q = O.apply_rotation(q[None, :, :], self.parent.q_phase(pq),
                                    amp=self.parent.q_amp(pq), pair_layout="half")[0]
        parent_k = O.apply_rotation(k[None, :, :], self.parent.k_phase(pk),
                                    amp=self.parent.k_amp(pk), pair_layout="half")[0]
        qg = self._permute_pairs(q, self.perm)
        kg = self._permute_pairs(k, self.perm)
        gauge_q = O.apply_rotation(qg[None, :, :], self.candidate.q_phase(pq),
                                   amp=self.candidate.q_amp(pq), pair_layout="half")[0]
        gauge_k = O.apply_rotation(kg[None, :, :], self.candidate.k_phase(pk),
                                   amp=self.candidate.k_amp(pk), pair_layout="half")[0]
        return ((parent_q * parent_k).sum(axis=-1) / math.sqrt(self.parent.geom.head_dim),
                (gauge_q * gauge_k).sum(axis=-1) / math.sqrt(self.parent.geom.head_dim))


def build_mr(geom):
    return _FrequencyControl(geom, geom.nu_mrpro, "MR", "MrRoPE-Pro Eq.14")


def build_resonance_yarn(geom):
    """Complete Resonance-YaRN parent: official index YaRN then nearest periods."""
    q = geom.q
    u = q / float(geom.n)
    nu = geom.omega * ((1.0 - u) + u / geom.scale)
    sel = np.arange(geom.K) > geom.low
    periods = 2.0 * math.pi / nu[sel]
    rounded = np.maximum(np.floor(periods + 0.5), 1.0)
    nu[sel] = 2.0 * math.pi / rounded
    return _FrequencyControl(geom, nu, "ResonanceYaRN",
                             "official YaRN index ramp + nearest integer period",
                             parent="OfficialYaRN")


def build_sign(geom, d04_id):
    if d04_id not in {"D04a", "D04b", "D04c"}:
        raise ValueError("SIGN requires D04a/D04b/D04c")
    op = O.build(d04_id, geom)
    op.config = f"SIGN_{d04_id}"
    op.direction = "SIGN"
    op.scope = "mechanism_control"
    op.parent = "MR"
    return op


def build_dc(geom, d05_id):
    if d05_id not in {"D05a", "D05b", "D05c"}:
        raise ValueError("DC requires D05a/D05b/D05c")
    op = O.build(d05_id, geom)
    op.config = f"DC_{d05_id}"
    op.direction = "DC"
    op.scope = "mechanism_control"
    op.parent = "MR"
    return op


def _score_rows(op, q, k, positions_q, positions_k, gain_override=None):
    """Return exact CPU score rows used by the D07 variance calibration."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    pq = np.asarray(positions_q, dtype=np.float64)
    pk = np.asarray(positions_k, dtype=np.float64)
    if q.shape != k.shape or q.ndim != 2 or pq.shape != (len(q),) or pk.shape != (len(k),):
        raise ValueError("q/k must be matched (N,head_dim) rows with (N,) positions")
    qr = O.apply_rotation(q[None, :, :], op.q_phase(pq), op.q_amp(pq), pair_layout="half")[0]
    kr = O.apply_rotation(k[None, :, :], op.k_phase(pk), op.k_amp(pk), pair_layout="half")[0]
    if gain_override is not None:
        qr = qr * (gain_override / op.geom.gain)
        kr = kr * (gain_override / op.geom.gain)
    return (qr * kr).sum(axis=-1) / math.sqrt(op.geom.head_dim)


def build_matched_global(geom, q, k, positions_q, positions_k, target_id="D07a"):
    """Build D07a's unlabeled global-variance-matched control.

    The activation pool is an explicit input and is not inspected for labels.
    ``v0≈0`` is a hard BLOCKED condition rather than a fallback gain.
    """
    if target_id != "D07a":
        raise ValueError("Plan B L6 specifies D07a for the matched-global control")
    mr = build_mr(geom)
    target = O.build(target_id, geom)
    s0 = _score_rows(mr, q, k, positions_q, positions_k)
    s1 = _score_rows(target, q, k, positions_q, positions_k)
    v0 = float(np.var(s0))
    v1 = float(np.var(s1))
    if not np.isfinite(v0) or not np.isfinite(v1) or v0 <= 1e-20:
        raise ValueError("BLOCKED: parent logit variance is zero or non-finite")
    g_match = geom.gain * (v1 / v0) ** 0.25
    if not np.isfinite(g_match) or g_match <= 0:
        raise ValueError("BLOCKED: matched global gain is non-finite")
    return _StaticGainControl(geom, g_match ** 2, "D07a_MATCHED_GLOBAL", parent="MR",
                              q_gain=g_match, k_gain=g_match), {
        "target": target_id, "v_parent": v0, "v_target": v1,
        "gain_qk": float(g_match), "score_multiplier": float(g_match ** 2),
        "calibration": "unlabeled fixed M-dev activations; g4*(v1/v0)**0.25",
    }


def build_static_endpoint(geom, d13_id):
    if d13_id not in {"D13a", "D13b", "D13c"}:
        raise ValueError("static endpoint requires D13a/D13b/D13c")
    op = O.build(d13_id, geom)
    p = np.array([float(geom.target - 1)], dtype=np.float64)
    # D13 uses n=p+1 and u=max(n,W); therefore p=D-1 is exactly t(D).
    t_const = float(op.q_amp(p)[0, 0] * op.k_amp(p)[0, 0])
    out = _StaticGainControl(geom, t_const, f"{d13_id}_STATIC_ENDPOINT", parent="MR")
    out.reference_d13 = d13_id
    out.endpoint_position = int(geom.target - 1)
    out.endpoint_score_multiplier = t_const
    return out


def _spec(direction, parents, controls, status="READY_CPU"):
    return {"direction": direction, "parent_ids": list(parents),
            "required_controls": list(controls), "construction_status": status}


REGISTRY = {}
for _d in range(1, 21):
    _did = f"D{_d:02d}"
    for _p in "abc":
        REGISTRY[f"{_did}{_p}"] = _spec(_did, ["MR"], [], "READY_CPU")

for _cid in ("D01a", "D01b", "D01c"):
    REGISTRY[_cid] = _spec("D01", ["MR", "ResonanceYaRN"], ["ResonanceYaRN"], "READY_CPU")
for _cid in ("D03a", "D03b", "D03c"):
    REGISTRY[_cid] = _spec("D03", ["MR"], ["D03_GAUGE"], "READY_CPU_SCORE_ONLY")
for _cid in ("D04a", "D04b", "D04c"):
    REGISTRY[_cid] = _spec("D04", ["MR"], ["SIGN"], "READY_CPU")
for _cid in ("D05a", "D05b", "D05c"):
    REGISTRY[_cid] = _spec("D05", ["MR"], ["DC", "DROP", "SIGN"], "READY_CPU")
for _cid in ("D06a", "D06b", "D06c"):
    REGISTRY[_cid] = _spec("D06", ["MR"], ["AREA"], "READY_CPU")
for _cid in ("D07a", "D07b", "D07c"):
    REGISTRY[_cid] = _spec("D07", ["MR"], ["D07_MATCHED_GLOBAL"], "BLOCKED_UNTIL_MDEV_ACTIVATIONS")
for _cid in ("D08a", "D08b", "D08c", "D09a", "D09b", "D09c"):
    REGISTRY[_cid]["required_controls"] = ["actual_variance_or_channel_profile"]
    REGISTRY[_cid]["construction_status"] = "BLOCKED_UNTIL_SPECIFIC_PARENT_LOGS"
for _cid in ("D10a", "D10b", "D10c"):
    REGISTRY[_cid]["required_controls"] = ["QK_PHASE_PARITY"]
for _cid in ("D11a", "D11b", "D11c"):
    REGISTRY[_cid]["required_controls"] = ["PAIR_METRIC_PARITY"]
for _cid in ("D12a", "D12b", "D12c"):
    REGISTRY[_cid]["required_controls"] = ["XPOS_DISTINCTION"]
for _cid in ("D13a", "D13b", "D13c"):
    REGISTRY[_cid]["required_controls"] = ["D13_STATIC_ENDPOINT"]
for _cid in ("D17a", "D17b", "D17c", "D18a", "D18b", "D18c"):
    REGISTRY[_cid]["parent_ids"] = ["D16b"]
for _cid in ("D19a", "D19b", "D19c"):
    REGISTRY[_cid]["parent_ids"] = ["MR", "D05c"]
for _cid in ("D20a", "D20b", "D20c"):
    REGISTRY[_cid]["parent_ids"] = ["MR", "D01a"]


def registry_document():
    """Return a JSON-safe registry including deterministic module provenance."""
    path = Path(__file__)
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return {
        "protocol": "RoPE_Integrated_Experiment_Guide_Codex_20260911 Plan B",
        "module_sha256": digest,
        "n_candidate_configs": len(REGISTRY),
        "candidate_configs": REGISTRY,
        "controls": {
            "MR": {"builder": "build_mr", "status": "READY_CPU"},
            "ResonanceYaRN": {"builder": "build_resonance_yarn", "status": "READY_CPU"},
            "D03_GAUGE": {"builder": "_GaugeControl", "status": "READY_CPU_SCORE_ONLY",
                           "limitation": "existing API has no V hook; score parity only"},
            "DC": {"builder": "build_dc", "status": "READY_CPU"},
            "DROP": {"builder": "_DropControl", "status": "READY_CPU"},
            "SIGN": {"builder": "build_sign", "status": "READY_CPU"},
            "AREA": {"builder": "_AreaControl", "status": "READY_CPU"},
            "D07_MATCHED_GLOBAL": {"builder": "build_matched_global",
                                    "status": "BLOCKED_UNTIL_MDEV_ACTIVATIONS"},
            "D13_STATIC_ENDPOINT": {"builder": "build_static_endpoint", "status": "READY_CPU"},
        },
        "scope_note": "This registry does not grant execution authority.",
    }


def selftest(geom=None):
    """Small deterministic algebra checks; never touches a model or GPU."""
    g = geom or O.Geometry.from_native()
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})

    mr = build_mr(g)
    ry = build_resonance_yarn(g)
    add("registry has exactly 60 candidate configs", len(REGISTRY) == 60)
    add("ResonanceYaRN has an OfficialYaRN parent", ry.parent == "OfficialYaRN")
    add("D01 resonance periods are integral", np.allclose(
        2 * math.pi / ry.nu()[np.arange(g.K) > g.low],
        np.round(2 * math.pi / ry.nu()[np.arange(g.K) > g.low]), atol=1e-9))

    for cid in ("D04a", "D04b", "D04c"):
        sign = build_sign(g, cid)
        add(f"{cid} SIGN leaves absolute frequency unchanged",
            np.allclose(np.abs(sign.nu()), mr.nu(), atol=0, rtol=0))
    for cid in ("D05a", "D05b", "D05c"):
        dc = build_dc(g, cid)
        add(f"{cid} DC retains all Q/K slots", np.all(dc.q_amp(np.array([0.])) > 0))
    drop = _DropControl(g, "D05c")
    add("DROP mask is nonempty", bool(np.any(drop.mask)))
    add("DROP zeros exactly its mask", np.all(drop.q_amp(np.array([0.]))[0, drop.mask] == 0)
        and np.all(drop.q_amp(np.array([0.]))[0, ~drop.mask] == g.gain))

    for cid in ("D06a", "D06b", "D06c"):
        area = _AreaControl(g, cid)
        t = _T(g)
        m_d = np.log(g.omega / O.build(cid, g).nu()) / math.log(g.scale)
        m_a = np.log(g.omega / area.nu()) / math.log(g.scale)
        outside = np.setdiff1d(np.arange(g.K), t)
        add(f"{cid} AREA preserves outside-T m", np.allclose(m_a[outside],
                                                               g.m_mrpro[outside], atol=1e-10))
        add(f"{cid} AREA matches total m", abs(float(m_a.sum() - m_d.sum())) < 1e-10)

    for cid in ("D13a", "D13b", "D13c"):
        ep = build_static_endpoint(g, cid)
        d13 = O.build(cid, g)
        p = np.array([float(g.target - 1)])
        want = float(d13.q_amp(p)[0, 0] * d13.k_amp(p)[0, 0])
        got = float(ep.q_amp(p)[0, 0] * ep.k_amp(p)[0, 0])
        add(f"{cid} static endpoint matches t(D)", abs(want - got) < 1e-12)

    gauge = _GaugeControl(g, "D03a")
    rng = np.random.default_rng(20260911)
    q = rng.normal(size=(16, g.head_dim)); k = rng.normal(size=(16, g.head_dim))
    p = np.arange(16, dtype=np.float64); s0, sg = gauge.score(q, k, p, p)
    add("D03 gauge score is invariant", np.allclose(s0, sg, atol=1e-10, rtol=1e-10),
        np.max(np.abs(s0 - sg)))
    add("D03 gauge leaves V outside this control", True, "caller-owned V is untouched")

    return {"checks": checks, "n_checks": len(checks),
            "n_pass": sum(int(x["pass"]) for x in checks),
            "all_pass": all(x["pass"] for x in checks),
            "registry": registry_document()}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, ensure_ascii=False))
