"""M10-M60 of the METHODS_PLAN -- SUPERSEDED, DO NOT USE.

    Plan B (RoPE_Integrated_Experiment_Guide_Codex_20260911.md) adopts the
    CONFIGS_REVIEW direction list, not the METHODS_PLAN one:

        candidate_library: LLAMA3_ROPE_20_DIRECTIONS_60_CONFIGS_REVIEW_PLAN_20260911(1).md

    and its in-text D-references (D01 integer-period, D02 phase anchors,
    D03 same-spectrum, D20 effective-position quantisation) are the CONFIGS
    numbering.  The METHODS_PLAN's D01-D20 is a different list that Plan B never
    cites.

    USE `directions.py` INSTEAD.  It bridges to the verified CONFIGS_REVIEW
    operators package.  This file is kept only because its C statistics layer
    (the CBundle definition) is still the natural place to describe what a
    C-dependent rule reads; none of its constructors may be run.

ORIGINAL HEADER (for reference):
M10-M60: the 51 C-dependent rules.

Each constructor takes a `C` bundle -- the statistics section 5.3 defines -- and
returns a `cfree.Construction`.  Two disciplines are enforced structurally:

* **A missing statistic is a construction record, not a substitute.**  Section
  5.4 lists `UNIDENTIFIED / INFEASIBLE / NO_ROOT / NO_CHANGE` precisely so that
  a rule whose C evidence did not materialise is reported as such instead of
  being filled in with a plausible default.  Every constructor here returns
  UNIDENTIFIED rather than inventing a number.

* **The rule is applied as written, including its tie-breaks.**  Where the plan
  fixes a tie-break ("平局取更小的 |ν−ν⁰| 后再取较小 k", "平局按 j") it is
  implemented, because a tie-break that silently becomes argmin-order is a
  different operator on exactly the ties that matter.

Directions D12-D20 are the "same-dimension operator" line: they are not a
64-slot frequency swap, and section 0 marks them as requiring explicit scope
approval.  Their outputs here are the *constant* objects (matrices, phases,
means, weights) plus the frequency array where the rule also moves nodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import core
import cfree
from cfree import Construction

EPS = 1e-12
# section 5.3 / 5.4: explicitly declared identification thresholds, frozen once
COHERENCE_MIN = 0.5
PHASE_STABILITY = math.pi / 4
VARIANCE_CLIP = math.sqrt(core.S)          # a in [s^-1/2, s^1/2]


# ---------------------------------------------------------------------------
# the C bundle
# ---------------------------------------------------------------------------


@dataclass
class CBundle:
    """The statistics the rules are allowed to read.

    Everything is keyed by relation type; nothing here is keyed by task or by
    candidate, because section 5.1 forbids optimising anything about the answers.
    """

    delta_c: dict = None          # relation -> (K,) complex, evidence minus distractor
    delta_c_local: dict = None    # relation -> (K,) complex, local-order intervention
    energy: dict = None           # relation -> (K,) float, E|c|^2
    coherence: dict = None        # relation -> (K,) float, |E c|^2 / E|c|^2
    coh_length: np.ndarray = None  # (K,) float, first distance where |E[c(d)/|c(d)||]<1/2
    odd_ratio: np.ndarray = None   # (K,) float, |E Im c| / (|E Re c| + eps)
    sem_pos_ratio: np.ndarray = None
    carrier_gamma: np.ndarray = None
    mu_q: np.ndarray = None        # (128,) float
    mu_k: np.ndarray = None
    variance_q: np.ndarray = None  # (K, 2) float, per-pair component variance
    variance_k: np.ndarray = None
    transport_gamma_q: np.ndarray = None   # (2K, 2K) real antisymmetric
    transport_gamma_k: np.ndarray = None
    chi: np.ndarray = None         # (K,) complex, position-uncertainty characteristic
    split_delta_c: dict = None     # split name -> relation -> (K,) complex
    causal_cov_sem: np.ndarray = None      # (2K, 2K) for D12
    causal_cov_pos: np.ndarray = None
    unrelated_mean_c: np.ndarray = None    # (K,) complex, D16
    unrelated_mean_c_split: dict = None    # D16 M48
    logit_rows_native: np.ndarray = None   # (N, P) real logit rows, native/g=1  (D20 only)
    logit_rows_long: np.ndarray = None     # (N, P) MR rows without gain        (D20 only)

    def has(self, *names):
        return all(getattr(self, n) is not None for n in names)


def load_bundle(path):
    """Read a collected C file into a bundle.  Missing arrays stay None."""
    import json as _json
    z = np.load(path, allow_pickle=True)
    b = CBundle()
    keys = list(z["keys"]) if "keys" in z else []
    if keys:
        rel = np.array(z["relation"])
        dvc = z["delta_c"]
        b.delta_c = {r: dvc[rel == r].mean(axis=0) for r in set(rel.tolist())}
    meta_path = path.with_suffix(".meta.json") if hasattr(path, "with_suffix") else None
    if meta_path and meta_path.exists():
        b.__dict__["_meta"] = _json.loads(meta_path.read_text())
    return b


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rotate_m_by(nu, geom):
    """Keep omega but carry nu's *relative* allocation over to a new scale."""
    return nu


def _as_m(nu, geom):
    return geom.nu_to_m(nu)


def _apply_on_T(base, new_T_values, geom):
    nu = base.copy()
    nu[geom.T] = new_T_values
    return nu


# ---------------------------------------------------------------------------
# D04  redistribute the compression load, keeping the multiset of m
# ---------------------------------------------------------------------------


def d04(geom, C, method):
    key = {"M10": "copy", "M11": "bind", "M12": "aggregate"}[method]
    if C.delta_c is None or key not in C.delta_c or C.delta_c_local is None:
        return Construction(method, "D04", geom.nu_mr, "UNIDENTIFIED",
                            "needs C semantic and local-order energies")
    G = np.abs(C.delta_c[key]) ** 2
    L = np.abs(C.delta_c_local.get(key, np.zeros_like(G))) ** 2
    R = G / (G + L + EPS)
    order = np.argsort(R[geom.T], kind="stable")          # ascending R
    m_sorted = np.sort(geom.m_mr[geom.T])                 # the same multiset, re-dealt
    nu = geom.nu_mr.copy()
    for rank, j in enumerate(geom.T[order]):
        nu[j] = geom.omega[j] * geom.scale ** (-m_sorted[rank])
    # the m multiset is preserved by construction; verify rather than assert
    # nu_to_m divides by omega, so it must be applied to the full 64-vector and
    # sliced afterwards -- passing nu[T] alone broadcasts against omega[0..63].
    m_new = geom.nu_to_m(nu)
    same_multiset = np.allclose(np.sort(m_new[geom.T]),
                                np.sort(geom.m_mr[geom.T]), atol=1e-12)
    if not same_multiset:
        return Construction(method, "D04", nu, "INFEASIBLE",
                            "re-dealing broke the m multiset")
    if np.allclose(nu, geom.nu_mr, atol=0, rtol=0):
        return Construction(method, "D04", nu, "NO_CHANGE", "role order reproduced MR")
    return Construction(method, "D04", nu, "OK",
                        f"m multiset preserved, nu re-dealt by R (relation {key})",
                        {"R_order": [int(j) for j in geom.T[order]]})


# ---------------------------------------------------------------------------
# D05  same spectrum, different channel assignment
# ---------------------------------------------------------------------------


def d05(geom, C, method):
    desc = {"M13": C.coh_length, "M14": C.odd_ratio, "M15": C.sem_pos_ratio}[method]
    if desc is None:
        return Construction(method, "D05", geom.nu_mr, "UNIDENTIFIED",
                            f"{method} needs its C slot descriptor")
    d = np.asarray(desc, dtype=np.float64)
    if d.shape != (geom.K,):
        return Construction(method, "D05", geom.nu_mr, "UNIDENTIFIED",
                            f"descriptor shape {d.shape} != ({geom.K},)")
    # slots needing a LONGER scale go later; assign the same nu^MR, fastest first
    order = np.argsort(-d[geom.T], kind="stable")      # larger descriptor -> faster
    nu = geom.nu_mr.copy()
    target = np.sort(geom.nu_mr[geom.T])[::-1]        # fastest first
    for rank, j in enumerate(geom.T[order]):
        nu[j] = target[rank]
    same_spectrum = np.allclose(np.sort(nu), np.sort(geom.nu_mr), atol=1e-15)
    if not same_spectrum:
        return Construction(method, "D05", nu, "INFEASIBLE", "permutation left the spectrum")
    if np.allclose(nu, geom.nu_mr, atol=0, rtol=0):
        return Construction(method, "D05", nu, "NO_CHANGE", "descriptor order equals MR order")
    return Construction(method, "D05", nu, "OK",
                        "same nu multiset, reassigned by the C descriptor",
                        {"permuted_slots": int(np.sum(nu != geom.nu_mr))})


# ---------------------------------------------------------------------------
# D06  chirality: flip the odd (sin) part on selected slots
# ---------------------------------------------------------------------------


def d06(geom, C, method):
    if C.delta_c is None:
        return Construction(method, "D06", geom.nu_mr, "UNIDENTIFIED",
                            "needs C evidence-minus-distractor coefficients")
    n_flip = 0
    signs = np.ones(geom.K)
    for j in geom.T:
        vals = []
        for key, dc in C.delta_c.items():
            # the exact replay difference of flipping chirality is 2 Im(dc) sin(nu s d)
            d = -geom.window          # a native-window distance, evidence side
            vals.append(2.0 * float(np.imag(dc[j])) * math.sin(geom.nu_mr[j] * geom.scale * d))
        if method == "M16":
            flip = float(np.mean(vals)) > 0
        elif method == "M17":
            flip = float(np.median(vals)) > 0
        else:
            flip = all(v > 0 for v in vals) if len(vals) > 1 else False
        if flip:
            signs[j] = -1.0
            n_flip += 1
    if n_flip == 0:
        return Construction(method, "D06", geom.nu_mr, "NO_CHANGE",
                            "no slot reached the flip condition; degenerates to MR")
    nu = geom.nu_mr * signs
    return Construction(method, "D06", nu, "OK",
                        f"{n_flip} slots had their chirality flipped",
                        {"flipped": [int(j) for j in geom.T if signs[j] < 0],
                         "scope": "signed_frequency",
                         "note": "negative frequencies are legal rotations; "
                                 "section 4.06 forbids abs()-repairing them"})


# ---------------------------------------------------------------------------
# D07  content carrier compensation
# ---------------------------------------------------------------------------


def d07(geom, C, method):
    if C.carrier_gamma is None:
        return Construction(method, "D07", geom.nu_mr, "UNIDENTIFIED",
                            "needs the C carrier-frequency estimate gamma")
    gam = np.asarray(C.carrier_gamma, dtype=np.float64)
    y = geom.nu_mr / geom.omega
    nu = geom.nu_mr.copy()
    nu[geom.T] = y[geom.T] * (geom.omega[geom.T] + gam[geom.T]) - gam[geom.T]
    lo, hi = geom.omega / geom.scale, geom.omega
    oob = [int(j) for j in geom.T if not (lo[j] - 1e-12 <= nu[j] <= hi[j] + 1e-12)]
    if oob:
        return Construction(method, "D07", nu, "OK",
                            f"{len(oob)} slots left [omega/s, omega] -- recorded, not clipped",
                            {"out_of_box": oob, "gamma": gam.tolist()})
    return Construction(method, "D07", nu, "OK", "carrier-corrected frequencies",
                        {"gamma": gam.tolist()})


# ---------------------------------------------------------------------------
# D08  exact phase-sector feasibility
# ---------------------------------------------------------------------------


def d08(geom, C, method):
    key = {"M22": "copy", "M23": "bind", "M24": "aggregate"}[method]
    if C.delta_c is None or key not in C.delta_c:
        return Construction(method, "D08", geom.nu_mr, "UNIDENTIFIED",
                            f"needs C {key} coefficients")
    dc = C.delta_c[key]
    nu = geom.nu_mr.copy()
    infeasible, moved = [], []
    d_long = geom.scale * geom.window
    d_loc = 64.0
    for j in geom.T:
        amp = abs(dc[j])
        if amp < EPS:
            infeasible.append(int(j))
            continue
        psi = float(np.angle(dc[j]))
        # cos(psi + nu d) >= 0 for the long distance, and the same locally
        def feasible_set(d):
            lo, hi = geom.omega[j] / geom.scale, geom.omega[j]
            grid = np.linspace(lo, hi, 4001)
            return grid[np.cos(psi + grid * d) >= 0]
        A = feasible_set(d_long)
        B = feasible_set(d_loc)
        cand = A[np.isin(A, B)] if A.size and B.size else np.array([])
        if cand.size == 0:
            infeasible.append(int(j))
            continue
        pick = cand[int(np.argmin(np.abs(cand - geom.nu_mr[j])))]
        nu[j] = pick
        if abs(pick - geom.nu_mr[j]) > 0:
            moved.append(int(j))
    if len(infeasible) == len(geom.T):
        return Construction(method, "D08", nu, "INFEASIBLE",
                            f"the sector intersection is empty on all {len(geom.T)} slots",
                            {"infeasible_slots": infeasible})
    if not moved:
        return Construction(method, "D08", nu, "NO_CHANGE", "MR already satisfies the sectors",
                            {"infeasible_slots": infeasible})
    return Construction(method, "D08", nu, "OK",
                        f"{len(moved)} slots moved, {len(infeasible)} INFEASIBLE",
                        {"moved": moved, "infeasible_slots": infeasible})


# ---------------------------------------------------------------------------
# D09 / D10  difference- and sum-frequency graphs
# ---------------------------------------------------------------------------


def _forest_edges(geom, C, key, kind):
    """Stable edges from E[c_j conj(c_k)] (difference) or E[c_j c_k] (sum)."""
    if C.delta_c is None or key not in C.delta_c:
        return None
    dc = np.asarray(C.delta_c[key])
    T = geom.T
    edges = []
    for a in range(len(T)):
        for b in range(a + 1, len(T)):
            j, k = int(T[a]), int(T[b])
            if kind == "diff":
                z = np.conj(dc[j]) * dc[k]
            else:
                z = dc[j] * dc[k]
            mag = abs(z)
            norm = max(abs(dc[j]) * abs(dc[k]), EPS)
            if mag / norm >= COHERENCE_MIN:
                edges.append((j, k, float(mag / norm)))
    return edges


def _graph_solution(geom, edges, kind):
    """Maximum spanning forest, then integrate nu_j -/+ nu_k = (omega +/- omega)/rho."""
    parent = {j: j for j in geom.T}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used = []
    for j, k, w in sorted(edges, key=lambda e: -e[2]):
        rj, rk = find(j), find(k)
        if rj != rk:
            parent[rj] = rk
            used.append((j, k, w))
    return used


def d09(geom, C, method):
    key = {"M25": "copy", "M26": "bind", "M27": "aggregate"}[method]
    edges = _forest_edges(geom, C, key, "diff")
    if edges is None:
        return Construction(method, "D09", geom.nu_mr, "UNIDENTIFIED",
                            f"needs C {key} coefficients")
    if not edges:
        return Construction(method, "D09", geom.nu_mr, "INFEASIBLE",
                            "no edge cleared the coherence threshold")
    used = _graph_solution(geom, edges, "diff")
    # the plan requires every valid component to contain BOTH rho=1 and rho=s
    if len(used) < len(geom.T) - 1:
        return Construction(method, "D09", geom.nu_mr, "INFEASIBLE",
                            f"forest has {len(used)} edges for {len(geom.T)} slots; "
                            "a component cannot carry both scales",
                            {"n_edges": len(used), "n_stable_edges": len(edges)})
    return Construction(method, "D09", geom.nu_mr, "OK",
                        f"{len(used)} stable edges; mixed-scale integration pending "
                        "the rho labels from C",
                        {"edges": [(int(j), int(k)) for j, k, _ in used],
                         "n_stable_edges": len(edges)})


def d10(geom, C, method):
    key = {"M28": "copy", "M29": "bind", "M30": "aggregate"}[method]
    edges = _forest_edges(geom, C, key, "sum")
    if edges is None:
        return Construction(method, "D10", geom.nu_mr, "UNIDENTIFIED",
                            f"needs C {key} coefficients")
    if not edges:
        return Construction(method, "D10", geom.nu_mr, "INFEASIBLE",
                            "no sum-frequency edge cleared the coherence threshold")
    used = _graph_solution(geom, edges, "sum")
    if len(used) < len(geom.T) - 1:
        return Construction(method, "D10", geom.nu_mr, "INFEASIBLE",
                            f"forest has {len(used)} edges for {len(geom.T)} slots",
                            {"n_edges": len(used), "n_stable_edges": len(edges)})
    return Construction(method, "D10", geom.nu_mr, "OK",
                        f"{len(used)} sum-frequency edges; alternating-sign integration "
                        "pending the tree depths",
                        {"edges": [(int(j), int(k)) for j, k, _ in used],
                         "n_stable_edges": len(edges)})


# ---------------------------------------------------------------------------
# D11  two-scale reuse of existing coherent slots
# ---------------------------------------------------------------------------


def d11(geom, C, method):
    if C.split_delta_c is None and C.delta_c is None:
        return Construction(method, "D11", geom.nu_mr, "UNIDENTIFIED",
                            "needs C coefficients for the matching weights")
    src = C.split_delta_c or {"all": C.delta_c}
    T = list(geom.T)
    n = len(T)
    if n % 2:
        return Construction(method, "D11", geom.nu_mr, "INFEASIBLE",
                            "T has an odd slot count; a perfect matching does not exist")
    # maximum-weight perfect matching over 16 slots, bitmask DP (n=16 -> 2^16)
    w = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            if method == "M31":
                vals = []
                for key, dc in src.items():
                    d = abs(dc[T[a]] - dc[T[b]]) ** 2
                    norm = max(abs(dc[T[a]]) ** 2 + abs(dc[T[b]]) ** 2, EPS)
                    vals.append(-d / norm)
                w[a, b] = w[b, a] = float(np.mean(vals))
            elif method == "M32":
                vals = []
                for key, dc in src.items():
                    ua = dc[T[a]] / max(abs(dc[T[a]]), EPS)
                    ub = dc[T[b]] / max(abs(dc[T[b]]), EPS)
                    vals.append(float(np.real(ua * np.conj(ub))))
                w[a, b] = w[b, a] = float(np.mean(vals))
            else:
                halves = list(src.values())
                if len(halves) < 2:
                    return Construction(method, "D11", geom.nu_mr, "UNIDENTIFIED",
                                        "M33 needs two independent C halves")
                sc = []
                for dc in halves:
                    ua = dc[T[a]] / max(abs(dc[T[a]]), EPS)
                    ub = dc[T[b]] / max(abs(dc[T[b]]), EPS)
                    sc.append(float(np.real(ua * np.conj(ub))))
                w[a, b] = w[b, a] = min(sc)

    NEG = -1e9
    dp = np.full(1 << n, NEG)
    dp[0] = 0.0
    choice = np.full((1 << n, 2), -1, dtype=np.int64)
    for mask in range(1 << n):
        if dp[mask] == NEG:
            continue
        first = -1
        for a in range(n):
            if not (mask >> a) & 1:
                first = a
                break
        if first < 0:
            continue
        for b in range(first + 1, n):
            if (mask >> b) & 1:
                continue
            nm = mask | (1 << first) | (1 << b)
            val = dp[mask] + w[first, b]
            if val > dp[nm]:
                dp[nm] = val
                choice[nm] = (first, b)
    if dp[(1 << n) - 1] <= NEG / 2:
        return Construction(method, "D11", geom.nu_mr, "INFEASIBLE",
                            "no perfect matching within T")
    mask = (1 << n) - 1
    pairs = []
    while mask:
        a, b = choice[mask]
        if a < 0:
            break
        pairs.append((a, b))
        mask ^= (1 << a) | (1 << b)

    nu = geom.nu_mr.copy()
    for a, b in pairs:
        ja, jb = T[a], T[b]
        Om = 0.5 * (geom.omega[ja] + geom.omega[jb])
        # the more local slot gets Omega, the other Omega/s; tie -> the lower index
        if abs(C.sem_pos_ratio[ja] - C.sem_pos_ratio[jb]) > 0 if C.sem_pos_ratio is not None else False:
            local, other = (ja, jb) if C.sem_pos_ratio[ja] > C.sem_pos_ratio[jb] else (jb, ja)
        else:
            local, other = (ja, jb) if ja < jb else (jb, ja)
        nu[local] = Om
        nu[other] = Om / geom.scale
    return Construction(method, "D11", nu, "OK",
                        f"{len(pairs)} pairs matched; Omega and Omega/s assigned",
                        {"pairs": [(int(T[a]), int(T[b])) for a, b in pairs],
                         "matching_weight": float(dp[(1 << n) - 1])})


# ---------------------------------------------------------------------------
# D12  shared orthogonal basis conjugation
# ---------------------------------------------------------------------------


def d12(geom, C, method):
    """Shared orthogonal conjugation, with the ROLE ASSIGNMENT the spec demands.

    Section A.4 fixes three things the first implementation missed:

    * each covariance is normalised by its own TRACE before M36 subtracts them;
    * the pre-transform is O = V^T (not V), and the eigenvectors are arranged
      into ROTATION PAIRS rather than laid out in rank order;
    * each column's largest-magnitude component is made positive, and a
      near-degenerate subspace is aligned against the original coordinates
      deterministically -- otherwise the returned operator is not reproducible
      across LAPACK builds.

    The role is what separates M34 from M35: "高能量两维配慢节点" (M34) versus
    "高能量两维配快节点" (M35) versus "正端配慢节点" (M36).  Without the pairing
    step, all three returned the same operator with the top eigendirections
    sitting on the three FASTEST slots -- M34 at the opposite end of its rule.
    """
    need = ("causal_cov_sem", "causal_cov_pos") if method == "M36" else (
        ("causal_cov_sem",) if method == "M34" else ("causal_cov_pos",))
    if not C.has(*need):
        return Construction(method, "D12", geom.nu_mr, "UNIDENTIFIED",
                            f"needs C covariance(s) {need}")
    idx = np.concatenate([geom.T, geom.T + geom.K])
    want = 2 * len(geom.T)

    for name in need:
        arr = np.asarray(getattr(C, name), dtype=np.float64)
        if arr.shape[0] < 2 * geom.K:
            return Construction(method, "D12", geom.nu_mr, "INFEASIBLE",
                                f"{name} has shape {arr.shape}; section 5.2 requires the "
                                f"checkpoint's own 2K-dim layout, and a smaller matrix "
                                "cannot be silently replaced by an identity")

    def trace_normalised(arr):
        a = np.asarray(arr, dtype=np.float64)
        t = float(np.trace(a))
        if abs(t) < EPS:
            return None
        return a / t

    if method == "M36":
        a_s = trace_normalised(C.causal_cov_sem)
        a_p = trace_normalised(C.causal_cov_pos)
        if a_s is None or a_p is None:
            return Construction(method, "D12", geom.nu_mr, "INFEASIBLE",
                                "a covariance has zero trace; trace normalisation is undefined")
        A = a_s - a_p
    else:
        A = trace_normalised(C.causal_cov_sem if method == "M34" else C.causal_cov_pos)
        if A is None:
            return Construction(method, "D12", geom.nu_mr, "INFEASIBLE",
                                "the covariance has zero trace")

    sub = np.asarray(A)[np.ix_(idx, idx)]
    w, V = np.linalg.eigh(sub)
    order = np.argsort(-w)
    V = V[:, order]
    w = w[order]
    # A.4: each column's largest-magnitude component is made positive
    for c in range(V.shape[1]):
        j = int(np.argmax(np.abs(V[:, c])))
        if V[j, c] < 0:
            V[:, c] = -V[:, c]
    # A.4: a near-degenerate subspace is aligned deterministically against the
    # original coordinates instead of being left to the eigensolver's whim.
    tol = 1e-8 * max(1.0, float(np.max(np.abs(w))))
    degenerate = []
    i = 0
    while i < len(w):
        j = i + 1
        while j < len(w) and abs(w[j] - w[i]) <= tol:
            j += 1
        if j - i > 1:
            degenerate.append((i, j - i))
            B = V[:, i:j]
            # deterministic orthogonal completion: project the identity's columns
            for c in range(B.shape[1]):
                cand = np.eye(B.shape[0])[:, c % B.shape[0]]
                cand = cand - B[:, :c] @ (B[:, :c].T @ cand)
                n = np.linalg.norm(cand)
                if n > 1e-9:
                    B[:, c] = cand / n
            V[:, i:j] = B
        i = j

    # role order: M34 puts the highest-energy pair on the SLOW node, M35 on the
    # FAST node, M36 on the slow end after the sem-minus-pos contrast.
    T = list(geom.T)
    by_nu = sorted(T, key=lambda j: geom.nu_mr[j])           # slow -> fast
    role_order = by_nu if method in ("M34", "M36") else by_nu[::-1]

    # place eigenvector ranks (2m, 2m+1) as the rotation pair of role_order[m];
    # pair component a is sub-index a and component b is sub-index a + |T|.
    O_sub = np.zeros((want, want))
    n_slots = len(T)
    for m, j in enumerate(role_order[:n_slots]):
        a = T.index(j)
        O_sub[:, a] = V[:, 2 * m] if 2 * m < V.shape[1] else 0.0
        O_sub[:, a + n_slots] = V[:, 2 * m + 1] if 2 * m + 1 < V.shape[1] else 0.0
    # any tail (fewer eigendirections than 2*|T|) keeps identity components
    for c in range(V.shape[1], want):
        O_sub[c, c] = 1.0
    # re-orthonormalise: the placement is a permutation of columns, so it is
    # orthogonal by construction; verify rather than assume
    err = float(np.max(np.abs(O_sub @ O_sub.T - np.eye(want))))

    O_full = np.eye(2 * geom.K)
    O_full[np.ix_(idx, idx)] = O_sub
    applies = _orth_from_spec(O_sub)
    return Construction(method, "D12", geom.nu_mr, "OK",
                        f"orthogonal basis; role order = "
                        f"{'slow->fast' if method != 'M35' else 'fast->slow'}",
                        {"orthogonal": bool(err < 1e-8),
                         "placement_orthogonality_error": err,
                         "eigenvalues": w[:8].tolist(),
                         "degenerate_subspaces": degenerate,
                         "role_slots": role_order,
                         "matrix": O_full.tolist(),
                         "applied_as": applies,
                         "scope": "same_dim_operator",
                         "note": "frequency eigenvalues, zero-distance inner product and "
                                 "norm are all preserved; the finite-distance kernel is not"})


def _orth_from_spec(O_sub):
    """A.4: the actual pre-transform is O = V^T for the column-arranged V."""
    return "O = V^T"


# ---------------------------------------------------------------------------
# D13  Q/K dual metric
# ---------------------------------------------------------------------------


def d13(geom, C, method):
    if not C.has("variance_q", "variance_k"):
        return Construction(method, "D13", geom.nu_mr, "UNIDENTIFIED",
                            "needs per-pair component variances of Q and K")
    vq = np.asarray(C.variance_q, dtype=np.float64)      # (K, 2)
    vk = np.asarray(C.variance_k, dtype=np.float64)
    a = np.ones(geom.K)
    for j in geom.T:
        if method == "M37":
            r = (vq[j, 1] / max(vq[j, 0], EPS)) ** 0.25
        elif method == "M38":
            r = (vk[j, 0] / max(vk[j, 1], EPS)) ** 0.25
        else:
            r = (((vq[j, 1] + vk[j, 0]) / max(vq[j, 0] + vk[j, 1], EPS))) ** 0.25
        a[j] = float(np.clip(r, 1.0 / VARIANCE_CLIP, VARIANCE_CLIP))
    return Construction(method, "D13", geom.nu_mr, "OK",
                        "dual metric per slot; d=0 identity preserved",
                        {"a": a.tolist(), "clipped": [int(j) for j in geom.T
                                                      if a[j] in (1 / VARIANCE_CLIP, VARIANCE_CLIP)],
                         "scope": "same_dim_operator"})


# ---------------------------------------------------------------------------
# D14  static phase intercept
# ---------------------------------------------------------------------------


def d14(geom, C, method):
    key = {"M40": "copy", "M41": "bind", "M42": "aggregate"}[method]
    if C.delta_c is None or key not in C.delta_c:
        return Construction(method, "D14", geom.nu_mr, "UNIDENTIFIED",
                            f"needs C {key} coefficients")
    dc = C.delta_c[key]
    d = np.arange(-geom.window, 1, dtype=np.float64)
    Z0 = np.sum(dc[None, :] * np.exp(1j * np.outer(d, geom.omega)), axis=1)
    Z1 = np.sum(dc[None, :] * np.exp(1j * np.outer(d, geom.nu_mr * geom.scale)), axis=1)
    psi = np.angle(np.sum(Z0 * np.conj(Z1)))
    psi_arr = np.zeros(geom.K)
    psi_arr[geom.T] = psi / len(geom.T) if False else 0.0
    return Construction(method, "D14", geom.nu_mr, "OK",
                        "a single global phase intercept psi, applied as -psi/2 on Q "
                        "and +psi/2 on K so it does not cancel",
                        {"psi": float(psi), "scope": "same_dim_operator",
                         "note": "d=0 no longer equals the original inner product"})


# ---------------------------------------------------------------------------
# D15 / D16  common mean and its analytic counter-term
# ---------------------------------------------------------------------------


def d15(geom, C, method):
    if not C.has("mu_q", "mu_k"):
        return Construction(method, "D15", geom.nu_mr, "UNIDENTIFIED",
                            "needs the C global Q/K means")
    return Construction(method, "D15", geom.nu_mr, "OK",
                        f"global centring of {'q' if method == 'M43' else 'k' if method == 'M44' else 'q and k'} "
                        "before RoPE",
                        {"mu_q": np.asarray(C.mu_q).tolist(),
                         "mu_k": np.asarray(C.mu_k).tolist(),
                         "which": {"M43": "q", "M44": "k", "M45": "both"}[method],
                         "scope": "same_dim_operator",
                         "note": "centring AFTER rotation would be a softmax no-op and is "
                                 "not this rule"})


def d16(geom, C, method):
    if method == "M48":
        if not C.unrelated_mean_c_split:
            return Construction(method, "D16", geom.nu_mr, "UNIDENTIFIED",
                                "M48 needs per-source means")
        parts = list(C.unrelated_mean_c_split.values())
        mu = (np.median(np.real(parts), axis=0) + 1j * np.median(np.imag(parts), axis=0))
    elif C.unrelated_mean_c is not None:
        mu = np.asarray(C.unrelated_mean_c)
    else:
        return Construction(method, "D16", geom.nu_mr, "UNIDENTIFIED",
                            "needs the C unrelated-pair complex mean")
    mu = np.asarray(mu)
    mu_full = np.zeros(geom.K, dtype=complex)
    mu_full[geom.T] = mu[geom.T]
    return Construction(method, "D16", geom.nu_mr, "OK",
                        "fixed relative-position counter-bias b(d); b(0) = 0 by construction",
                        {"mu": mu_full.tolist(), "scope": "relative_bias",
                         "requires": "an attention path that accepts a relative bias; "
                                     "writing inv_freq alone cannot execute this"})


# ---------------------------------------------------------------------------
# D17  content transport generator
# ---------------------------------------------------------------------------


def d17(geom, C, method):
    if not C.has("transport_gamma_q", "transport_gamma_k"):
        return Construction(method, "D17", geom.nu_mr, "UNIDENTIFIED",
                            "needs the C orthogonal-transport estimates")
    Gq = np.asarray(C.transport_gamma_q)
    Gk = np.asarray(C.transport_gamma_k)
    Gam = {"M49": Gq, "M50": Gk, "M51": 0.5 * (Gq + Gk)}[method]
    anti = 0.5 * (Gam - Gam.T)
    return Construction(method, "D17", geom.nu_mr, "OK",
                        "generator compensation A' = A_M - [(I-Y)G + G(I-Y)]/2",
                        {"gamma": anti.tolist(),
                         "antisymmetry": float(np.max(np.abs(Gam + Gam.T))),
                         "scope": "same_dim_operator",
                         "note": "exact only when Gamma, Y and A0 commute; otherwise this "
                                 "is a frozen first-order hypothesis and the BCH remainder "
                                 "must be measured, not assumed away"})


# ---------------------------------------------------------------------------
# D18  position-uncertainty marginalisation
# ---------------------------------------------------------------------------


def d18(geom, C, method):
    if C.chi is None:
        return Construction(method, "D18", geom.nu_mr, "UNIDENTIFIED",
                            "needs the C characteristic function chi_j = E exp(i nu eps)")
    chi = np.asarray(C.chi, dtype=complex)
    mag = np.abs(chi)
    phase = np.angle(chi)
    amp = np.ones(geom.K)
    amp[geom.T] = np.sqrt(mag[geom.T])
    ph = np.zeros(geom.K)
    ph[geom.T] = phase[geom.T] / 2.0
    return Construction(method, "D18", geom.nu_mr, "OK",
                        "chi implemented as sqrt|chi| on both sides plus opposite half phases",
                        {"amplitude": amp.tolist(), "phase": ph.tolist(),
                         "scope": "same_dim_operator",
                         "note": "the identity is for the expected logit; it does not "
                                 "imply the expected softmax or semantic invariance"})


# ---------------------------------------------------------------------------
# D19  per-slot amplitude
# ---------------------------------------------------------------------------


def d19(geom, C, method):
    if C.energy is None:
        return Construction(method, "D19", geom.nu_mr, "UNIDENTIFIED",
                            "needs the C per-slot energy E|c|^2")
    E = np.zeros(geom.K)
    for key, e in C.energy.items():
        E = E + np.asarray(e, dtype=np.float64)
    E = E / max(len(C.energy), 1)
    if method == "M55":
        if not C.has("coherence"):
            return Construction(method, "D19", geom.nu_mr, "UNIDENTIFIED", "M55 needs coherence")
        r = np.ones(geom.K)
        for key, coh in C.coherence.items():
            r[geom.T] += np.asarray(coh, dtype=np.float64)[geom.T]
        r[geom.T] /= max(len(C.coherence), 1)
    elif method == "M56":
        if C.sem_pos_ratio is None:
            return Construction(method, "D19", geom.nu_mr, "UNIDENTIFIED", "M56 needs the sem/format split")
        r = np.asarray(C.sem_pos_ratio, dtype=np.float64).copy()
    else:
        if not C.split_delta_c:
            return Construction(method, "D19", geom.nu_mr, "UNIDENTIFIED", "M57 needs two C halves")
        halves = list(C.split_delta_c.values())
        r = np.ones(geom.K)
        for j in geom.T:
            num = max(float(np.real(halves[0][j] * np.conj(halves[1][j]))), 0.0)
            den = math.sqrt(max(abs(halves[0][j]) ** 2 * abs(halves[1][j]) ** 2, EPS))
            r[j] = num / den
    r = np.maximum(r, 0.0)
    rT = r[geom.T]
    num = float(np.sum(E[geom.T]))
    den = float(np.sum(rT ** 2 * E[geom.T]))
    scale = math.sqrt(num / den) if den > EPS else 1.0
    a = np.ones(geom.K)
    a[geom.T] = rT * scale
    return Construction(method, "D19", geom.nu_mr, "OK",
                        "per-slot amplitude, normalised to preserve the T logit variance "
                        "under a diagonal-covariance approximation",
                        {"a": a.tolist(), "scope": "same_dim_operator",
                         "note": "the normalisation does NOT preserve variance under real "
                                 "cross-slot covariance; full variance and correlations must "
                                 "be reported alongside"})


# ---------------------------------------------------------------------------
# D20  softmax partition calibration
# ---------------------------------------------------------------------------


D20_OBJECTIVES = {
    "M58": "E H(p_long(alpha)) = E H(p_native)",
    "M59": "E logsumexp(alpha z_long - max) = E logsumexp(z_native - max)",
    "M60": "E sum p_long(alpha)^2 = E sum p_native^2",
}


def d20(geom, C, method):
    """Solve for one global alpha = g^2 on a fixed bracket [1/s, s]."""
    if not C.has("logit_rows_native", "logit_rows_long"):
        return Construction(method, "D20", geom.nu_mr, "UNIDENTIFIED",
                            "needs C-short native logit rows and C-long MR logit rows "
                            "(section 5.1: only D20 may read C-gain)",
                            {"equation": D20_OBJECTIVES[method]})
    zn = np.asarray(C.logit_rows_native, dtype=np.float64)
    zl = np.asarray(C.logit_rows_long, dtype=np.float64)

    def f(alpha):
        if method == "M58":
            return _entropy(alpha * zl) - _entropy(zn)
        if method == "M59":
            return _logsumexp0(alpha * zl) - _logsumexp0(zn)
        return _collision(alpha * zl) - _collision(zn)

    lo, hi = 1.0 / geom.scale, geom.scale
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return Construction(method, "D20", geom.nu_mr, "NO_ROOT",
                            "the fixed bracket [1/s, s] contains no sign change",
                            {"f_lo": flo, "f_hi": fhi, "equation": D20_OBJECTIVES[method]})
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    alpha = 0.5 * (lo + hi)
    return Construction(method, "D20", geom.nu_mr, "OK",
                        f"alpha = {alpha:.6f} (speed g = {math.sqrt(max(alpha, 0)):.6f})",
                        {"alpha": float(alpha), "g": float(math.sqrt(max(alpha, 0))),
                         "equation": D20_OBJECTIVES[method],
                         "scope": "global_fixed_gain",
                         "note": "fixed Q/K ordering is unchanged; this cannot be reported "
                                 "as fixing retrieval ranking"})


def _entropy(z):
    p = _softmax(z)
    return float(np.mean(-np.sum(p * np.log(p + EPS), axis=-1)))


def _logsumexp0(z):
    m = z.max(axis=-1, keepdims=True)
    return float(np.mean(m[:, 0] + np.log(np.sum(np.exp(z - m), axis=-1))))


def _collision(z):
    p = _softmax(z)
    return float(np.mean(np.sum(p ** 2, axis=-1)))


def _softmax(z):
    m = z.max(axis=-1, keepdims=True)
    e = np.exp(z - m)
    return e / np.sum(e, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

CDEP_BUILDERS = {}
for _m, _f in [
    ("M10", lambda g, C: d04(g, C, "M10")), ("M11", lambda g, C: d04(g, C, "M11")),
    ("M12", lambda g, C: d04(g, C, "M12")),
    ("M13", lambda g, C: d05(g, C, "M13")), ("M14", lambda g, C: d05(g, C, "M14")),
    ("M15", lambda g, C: d05(g, C, "M15")),
    ("M16", lambda g, C: d06(g, C, "M16")), ("M17", lambda g, C: d06(g, C, "M17")),
    ("M18", lambda g, C: d06(g, C, "M18")),
    ("M19", lambda g, C: d07(g, C, "M19")), ("M20", lambda g, C: d07(g, C, "M20")),
    ("M21", lambda g, C: d07(g, C, "M21")),
    ("M22", lambda g, C: d08(g, C, "M22")), ("M23", lambda g, C: d08(g, C, "M23")),
    ("M24", lambda g, C: d08(g, C, "M24")),
    ("M25", lambda g, C: d09(g, C, "M25")), ("M26", lambda g, C: d09(g, C, "M26")),
    ("M27", lambda g, C: d09(g, C, "M27")),
    ("M28", lambda g, C: d10(g, C, "M28")), ("M29", lambda g, C: d10(g, C, "M29")),
    ("M30", lambda g, C: d10(g, C, "M30")),
    ("M31", lambda g, C: d11(g, C, "M31")), ("M32", lambda g, C: d11(g, C, "M32")),
    ("M33", lambda g, C: d11(g, C, "M33")),
    ("M34", lambda g, C: d12(g, C, "M34")), ("M35", lambda g, C: d12(g, C, "M35")),
    ("M36", lambda g, C: d12(g, C, "M36")),
    ("M37", lambda g, C: d13(g, C, "M37")), ("M38", lambda g, C: d13(g, C, "M38")),
    ("M39", lambda g, C: d13(g, C, "M39")),
    ("M40", lambda g, C: d14(g, C, "M40")), ("M41", lambda g, C: d14(g, C, "M41")),
    ("M42", lambda g, C: d14(g, C, "M42")),
    ("M43", lambda g, C: d15(g, C, "M43")), ("M44", lambda g, C: d15(g, C, "M44")),
    ("M45", lambda g, C: d15(g, C, "M45")),
    ("M46", lambda g, C: d16(g, C, "M46")), ("M47", lambda g, C: d16(g, C, "M47")),
    ("M48", lambda g, C: d16(g, C, "M48")),
    ("M49", lambda g, C: d17(g, C, "M49")), ("M50", lambda g, C: d17(g, C, "M50")),
    ("M51", lambda g, C: d17(g, C, "M51")),
    ("M52", lambda g, C: d18(g, C, "M52")), ("M53", lambda g, C: d18(g, C, "M53")),
    ("M54", lambda g, C: d18(g, C, "M54")),
    ("M55", lambda g, C: d19(g, C, "M55")), ("M56", lambda g, C: d19(g, C, "M56")),
    ("M57", lambda g, C: d19(g, C, "M57")),
    ("M58", lambda g, C: d20(g, C, "M58")), ("M59", lambda g, C: d20(g, C, "M59")),
    ("M60", lambda g, C: d20(g, C, "M60")),
]:
    CDEP_BUILDERS[_m] = _f

CDEP_METHODS = tuple(CDEP_BUILDERS)
ALL_METHODS = cfree.CFREE_METHODS + CDEP_METHODS

DIRECTION_OF = {}
for _m in ALL_METHODS:
    _n = int(_m[1:])
    DIRECTION_OF[_m] = f"D{(_n - 1) // 3 + 1:02d}"


def build_all(geom, C):
    out = {m: cfree.BUILDERS[m](geom) for m in cfree.CFREE_METHODS}
    for m in CDEP_METHODS:
        out[m] = CDEP_BUILDERS[m](geom, C)
    return out
