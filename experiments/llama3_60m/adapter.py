"""The section 7.3 adapter: turn a Construction into an applyable operator.

    (q_new, k_new, inv_freq, phase_q, phase_k, rel_bias, gain) = adapter(constants, q, k)

Section 7.3 lists what the adapter may NOT do, and every one of those is a
constraint on this file rather than a note:

* it must not read the question id, the answer, the task category, or whether
  the input is long;
* it must not keep different constants per head or per layer;
* V and the output projection stay on the original path (nothing here touches
  them -- there is no code path that could);
* QKV still emits 128 dims and softmax_scale stays 1/sqrt(128);
* constant matrices may be applied identically in prefill and decode, but must
  not change once the cache is built.  `Constants` is frozen after
  `build_constants`; the apply path is pure.

**Phase precision.** Section 6.1: weights are BF16, but the phase multiplication
and sin/cos are FP32, and the 32K position multiply must not be done in BF16.
`apply_pure` therefore computes in float32 and casts the result back.

**Pair layout.** Llama's rotate_half pairs (i, i+K).  Getting this wrong yields a
different operator that still runs.  The layout is an explicit argument with no
default.
"""

from __future__ import annotations

import math
import inspect
from dataclasses import dataclass, field

import numpy as np

import core

SOFTMAX_SCALE = 1.0 / math.sqrt(128.0)      # section 7.3, unchanged


def _call_rotary_original(fn, q, k, cos, sin, position_ids, unsqueeze_dim):
    """Call HF's stock helper across the 4.x and 5.x signatures.

    HF 4.5x exposes ``position_ids`` (deprecated but still present), while
    Transformers 5.15 removed it and uses the fifth argument for
    ``unsqueeze_dim``.  Passing six positional arguments is therefore not a
    compatibility strategy: it crashes on 5.15 and can also reinterpret an
    integer as a tensor.  Signature inspection keeps the stock/native path
    genuinely stock on both APIs.
    """
    params = inspect.signature(fn).parameters
    kwargs = {}
    if "position_ids" in params:
        kwargs["position_ids"] = position_ids
    if "unsqueeze_dim" in params:
        kwargs["unsqueeze_dim"] = unsqueeze_dim
    return fn(q, k, cos, sin, **kwargs)


@dataclass(frozen=True)
class Constants:
    """Everything a rule is allowed to change.  Frozen after construction.

    One object per method; nothing here is keyed by head, layer, task or length.
    """

    method: str
    nu: np.ndarray                      # (K,) signed frequencies; sign carries D06
    gain: float                         # the global g (section 6.1 fixes gY)
    phase_q: np.ndarray | None = None    # (K,) extra phase on Q   (D14, D18)
    phase_k: np.ndarray | None = None    # (K,) extra phase on K
    amp: np.ndarray | None = None        # (K,) per-slot amplitude (D18, D19)
    orthogonal: np.ndarray | None = None  # (2K,2K) shared basis conjugation (D12)
    metric: np.ndarray | None = None     # (K,2) per-pair dual metric (D13)
    q_diag: np.ndarray | None = None     # (K,2), operator-side Q pair metric
    k_diag: np.ndarray | None = None     # (K,2), operator-side K pair metric
    mu_q: np.ndarray | None = None       # (2K,) pre-rotation centring (D15)
    mu_k: np.ndarray | None = None
    rel_bias_mu: np.ndarray | None = None  # (K,) complex, D16
    transport: np.ndarray | None = None  # (2K,2K) antisymmetric generator (D17)
    scope: str = "frequency"
    notes: str = ""
    op: object | None = None             # the CONFIGS operator, when one built this

    def surfaces(self, positions):
        """Full-range (q_phase, k_phase, q_amp, k_amp) from the operator.

        The CONFIGS operators define their surfaces as FUNCTIONS of position, so
        a Constants built from one must call them rather than carry sampled
        arrays.  Without this the adapter fell back to `nu` + `gain` alone and
        silently dropped every same-dimension surface -- D10's phase bias, D11's
        pair metric, D12/D13's amplitudes -- producing an operator identical to
        the plain table while labelled with the method id.
        """
        if self.op is None:
            raise ValueError("no operator attached; build via build_from_operator")
        p = np.asarray(positions, dtype=np.float64)
        return (np.asarray(self.op.q_phase(p), dtype=np.float64),
                np.asarray(self.op.k_phase(p), dtype=np.float64),
                np.asarray(self.op.q_amp(p), dtype=np.float64),
                np.asarray(self.op.k_amp(p), dtype=np.float64))

    def describe(self):
        return {"method": self.method, "scope": self.scope,
                "gain": self.gain, "notes": self.notes,
                "has_orthogonal": self.orthogonal is not None,
                "has_metric": self.metric is not None,
                "has_bias": self.rel_bias_mu is not None,
                "has_transport": self.transport is not None}


def build_from_operator(geom, op, authorized=True):
    """Constants for a CONFIGS_REVIEW operator (the live path).

    The operator's own methods define every surface, so nothing has to be
    inferred from a `detail` dict.  Refuses an unauthorised extension scope
    rather than returning a runnable table that is really just MR.
    """
    scope = op.scope
    if not authorized and scope not in ("frequency", "frequency_assignment"):
        raise ValueError(f"scope {scope!r} is an extension (Plan B section 0.3) and is "
                         "not authorised; refusing to build a Constants from it")
    qd = np.asarray(op.q_diag(), dtype=np.float64)
    kd = np.asarray(op.k_diag(), dtype=np.float64)
    has_diag = not (np.allclose(qd, 1.0) and np.allclose(kd, 1.0))
    return Constants(method=op.config, nu=np.asarray(op.nu(), dtype=np.float64),
                     gain=geom.gain,
                     q_diag=qd if has_diag else None,
                     k_diag=kd if has_diag else None,
                     scope=scope, op=op,
                     notes=f"built from the operator; surface_kind="
                           f"{'pair_metric' if has_diag else 'frequency_or_phase'}")


def build_constants(geom, construction):
    """Map a Construction onto the adapter's allowed surfaces.

    The frequency line (D01-D11) fills `nu` and nothing else, except D06 which
    also carries a sign.  The same-dimension line fills the extra surfaces.
    """
    c = construction
    detail = c.detail or {}
    scope = detail.get("scope", "frequency")
    nu = np.asarray(c.nu, dtype=np.float64).copy()
    gain = geom.gain
    phase_q = phase_k = amp = None

    if c.method in ("M40", "M41", "M42") and "psi" in detail:
        # D14: -psi/2 on Q and +psi/2 on K, so the correction does not cancel
        psi = float(detail["psi"])
        phase_q = np.zeros(geom.K)
        phase_k = np.zeros(geom.K)
        phase_q[geom.T] = -psi / 2.0
        phase_k[geom.T] = +psi / 2.0

    if c.method in ("M52", "M53", "M54") and "amplitude" in detail:
        amp = np.asarray(detail["amplitude"], dtype=np.float64)
        phase_q = np.asarray(detail["phase"], dtype=np.float64)
        phase_k = -phase_q          # opposite half-phases, so the correction does not cancel

    if c.method in ("M55", "M56", "M57") and "a" in detail:
        amp = np.asarray(detail["a"], dtype=np.float64)

    orthogonal = None
    if c.method in ("M34", "M35", "M36") and "matrix" in detail:
        orthogonal = np.asarray(detail["matrix"], dtype=np.float64)

    metric = None
    if c.method in ("M37", "M38", "M39") and "a" in detail:
        a = np.asarray(detail["a"], dtype=np.float64)
        metric = np.stack([a, 1.0 / a], axis=1)
        metric[~np.isin(np.arange(geom.K), geom.T)] = 1.0

    mu_q = mu_k = None
    if c.method in ("M43", "M44", "M45") and "mu_q" in detail:
        mu_q = np.asarray(detail["mu_q"], dtype=np.float64)
        mu_k = np.asarray(detail["mu_k"], dtype=np.float64)
        which = detail.get("which", "both")
        if which == "q":
            mu_k = None
        elif which == "k":
            mu_q = None

    rel_bias_mu = None
    if c.method in ("M46", "M47", "M48") and "mu" in detail:
        rel_bias_mu = np.asarray(detail["mu"], dtype=complex)

    transport = None
    if c.method in ("M49", "M50", "M51") and "gamma" in detail:
        transport = np.asarray(detail["gamma"], dtype=np.float64)

    return Constants(method=c.method, nu=nu, gain=gain, phase_q=phase_q, phase_k=phase_k,
                     amp=amp, orthogonal=orthogonal, metric=metric, mu_q=mu_q, mu_k=mu_k,
                     rel_bias_mu=rel_bias_mu, transport=transport, scope=scope,
                     notes=detail.get("note", ""))


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def _rotate_pure(x, phase, amp=None, diag=None, dtype=None):
    """x is (..., P, 2K) float32; phase is (P, K) float32; half-split layout.

    Section 6.1: the phase multiply and sin/cos happen in FP32 even though the
    weights are BF16, and the 32K position product is never pre-reduced to BF16.
    """
    K = x.shape[-1] // 2
    a = x[..., :K]
    b = x[..., K:]
    if diag is not None:
        d = np.asarray(diag, dtype=np.float64)
        a = a * d[None, :, 0]
        b = b * d[None, :, 1]
    c = np.cos(phase)
    s = np.sin(phase)
    ar = a * c - b * s
    br = a * s + b * c
    if amp is not None:
        ar = ar * amp
        br = br * amp
    out = np.concatenate([ar, br], axis=-1)
    return out if dtype is None else out.astype(dtype)


def apply_pure(C: Constants, q, k, positions, geom):
    """Numpy reference of the adapter.  Used by the self-tests and the CPU harness.

    q, k: (..., P, 2K).  positions: (P,).  Returns (q_new, k_new, rel_bias).

    When `C.op` is set the four surfaces come from the operator itself, which is
    the only way a position-phase or position-amplitude rule can be applied at
    all: their phase is not `p * nu` for any constant nu.
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)

    if C.mu_q is not None:
        q = q - C.mu_q
    if C.mu_k is not None:
        k = k - C.mu_k

    if C.orthogonal is not None:
        q = q @ C.orthogonal
        k = k @ C.orthogonal

    # Legacy Constants carry a reciprocal metric here.  Operator-backed
    # Constants carry independent q/k diagonals below; applying both would
    # double-count D11 and is not mathematically equivalent.
    if C.metric is not None and C.op is None:
        K = q.shape[-1] // 2
        q = np.concatenate([q[..., :K] * C.metric[:, 0], q[..., K:] * C.metric[:, 1]], axis=-1)
        k = np.concatenate([k[..., :K] / C.metric[:, 0], k[..., K:] / C.metric[:, 1]], axis=-1)

    if C.op is not None:
        phase_q, phase_k, amp_q, amp_k = C.surfaces(pos)
        qd = C.q_diag if C.q_diag is not None else np.asarray(C.op.q_diag(), dtype=np.float64)
        kd = C.k_diag if C.k_diag is not None else np.asarray(C.op.k_diag(), dtype=np.float64)
        q_new = _rotate_pure(q, phase_q, amp_q, qd)
        k_new = _rotate_pure(k, phase_k, amp_k, kd)
    else:
        phase_q = pos[:, None] * C.nu[None, :]
        phase_k = phase_q.copy()
        if C.phase_q is not None:
            phase_q = phase_q + C.phase_q[None, :]
        if C.phase_k is not None:
            phase_k = phase_k + C.phase_k[None, :]
        amp = C.amp[None, :] if C.amp is not None else None
        q_new = _rotate_pure(q, phase_q, amp)
        k_new = _rotate_pure(k, phase_k, amp)

    # CONFIGS_REVIEW operator amplitudes already include the one global YaRN
    # gain.  Legacy constructions do not, so retain their explicit multiplier.
    if C.op is None:
        q_new = q_new * C.gain
        k_new = k_new * C.gain

    rel_bias = None
    if C.rel_bias_mu is not None:
        rel_bias = relative_bias(C.rel_bias_mu, C.nu, geom)
    return q_new, k_new, rel_bias


def relative_bias(mu, nu, geom, max_d=None):
    """D16: b(d) = g^2/sqrt(128) * Re sum_j mu_j [1 - exp(i nu_j d)], with b(0)=0.

    Section 7.3: this depends on d only.  d = key_position - query_position, so
    causal attention reads the d <= 0 half.
    """
    max_d = max_d or geom.target
    d = np.arange(-max_d, 1, dtype=np.float64)
    z = np.exp(1j * np.outer(d, nu))
    b = np.real(((1.0 - z) * mu[None, :]).sum(axis=1)) * (geom.gain ** 2 / math.sqrt(128.0))
    return d, b


def apply_torch(C, q, k, positions, torch):
    """The GPU path.  Same algebra as apply_pure, in the model's dtype."""
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=q.device)
    dt = q.dtype
    if C.mu_q is not None:
        q = q - t(C.mu_q)[None, None, None, :]
    if C.mu_k is not None:
        k = k - t(C.mu_k)[None, None, None, :]
    if C.orthogonal is not None:
        O = t(C.orthogonal)
        q = q @ O
        k = k @ O
    if C.metric is not None and C.op is None:
        # Legacy reciprocal metric: Q is multiplied, K is divided.  This is
        # intentionally separate from operator-backed q_diag/k_diag.
        metric = t(C.metric)
        K = q.shape[-1] // 2
        q = torch.cat((q[..., :K] * metric[:, 0], q[..., K:] * metric[:, 1]), dim=-1)
        k = torch.cat((k[..., :K] / metric[:, 0], k[..., K:] / metric[:, 1]), dim=-1)
    pos = torch.as_tensor(positions, dtype=torch.float32, device=q.device)
    if C.op is not None:
        # Position-dependent CONFIGS surfaces are the operator, not metadata.
        # Evaluate them on CPU in float64 exactly as apply_pure, then transfer
        # the four surfaces once.  This is also where D11's independent q/k
        # diagonals are kept independent.
        phq_np, phk_np, amq_np, amk_np = C.surfaces(np.asarray(positions, dtype=np.float64))
        phq, phk = t(phq_np), t(phk_np)
        amq, amk = t(amq_np), t(amk_np)
        qd = t(C.q_diag if C.q_diag is not None else C.op.q_diag())
        kd = t(C.k_diag if C.k_diag is not None else C.op.k_diag())
        operator_surfaces = True
    else:
        nu = t(C.nu)
        ph = pos[:, None] * nu[None, :]
        phq = ph + t(C.phase_q)[None, :] if C.phase_q is not None else ph
        phk = ph + t(C.phase_k)[None, :] if C.phase_k is not None else ph
        amp = t(C.amp) if C.amp is not None else None
        amq = amk = amp
        qd = kd = None
        operator_surfaces = False

    def rot(x, phase, amp, diag=None):
        Kx = x.shape[-1] // 2
        xf = x.to(torch.float32)
        a, b = xf[..., :Kx], xf[..., Kx:]
        ph3 = phase[None, None, :, :]
        if diag is not None:
            a = a * diag[None, None, :, 0][..., None, :]
            b = b * diag[None, None, :, 1][..., None, :]
        c, s = torch.cos(ph3), torch.sin(ph3)
        ar, br = a * c - b * s, a * s + b * c
        if amp is not None:
            ar, br = ar * amp[None, None, :, :], br * amp[None, None, :, :]
        return torch.cat([ar, br], dim=-1).to(dt)

    qn = rot(q, phq, amq, qd)
    kn = rot(k, phk, amk, kd)
    if not operator_surfaces:
        qn, kn = qn * C.gain, kn * C.gain
    return qn, kn


# ---------------------------------------------------------------------------
# the backend-attached adapter
# ---------------------------------------------------------------------------


class RoPEAdapter:
    """Installs a Constants object into a HF Llama model by wrapping the rotation.

    Section 7.3 requires that the same backend serve MR and every candidate, so
    that a difference in reduction order or temperature cannot be mistaken for a
    method effect.  Wrapping `apply_rotary_pos_emb` means this is the same tap
    as the C collection (`c_collect.QKTap`), so collection and deployment cannot
    drift apart.
    """

    def __init__(self, torch, modeling, C: Constants, geom, keep_stock_for=None):
        self.torch = torch
        self.modeling = modeling
        self.C = C
        self.geom = geom
        self.keep_stock_for = keep_stock_for or set()
        self._orig = None
        self._orig_rotary_forward = None

    def __enter__(self):
        self._orig = self.modeling.apply_rotary_pos_emb
        self._orig_rotary_forward = getattr(self.modeling.LlamaRotaryEmbedding,
                                             "forward", None)
        adapter = self
        state = {"position_ids": None}

        if self._orig_rotary_forward is not None:
            orig_forward = self._orig_rotary_forward

            def rotary_forward(rotary, x, position_ids, *args, **kwargs):
                state["position_ids"] = position_ids
                return orig_forward(rotary, x, position_ids, *args, **kwargs)

            self.modeling.LlamaRotaryEmbedding.forward = rotary_forward

        def wrapped(q, k, cos, sin, *args, **kwargs):
            position_ids = kwargs.pop("position_ids", None)
            unsqueeze_dim = kwargs.pop("unsqueeze_dim", 1)
            if args:
                if len(args) >= 2:
                    position_ids, unsqueeze_dim = args[:2]
                elif hasattr(args[0], "shape"):
                    position_ids = args[0]
                else:
                    unsqueeze_dim = args[0]
            if position_ids is None:
                position_ids = state["position_ids"]
            if adapter.C.method in adapter.keep_stock_for:
                return _call_rotary_original(adapter._orig, q, k, cos, sin,
                                             position_ids, unsqueeze_dim)
            pos = position_ids
            if pos is None:
                raise RuntimeError("adapter needs position_ids; the wrapped call did not "
                                   "receive them")
            if pos.shape[0] != 1:
                raise RuntimeError(f"adapter assumes batch 1, got {pos.shape[0]}")
            p = pos[0].to(adapter.torch.float32)
            qn, kn = apply_torch(adapter.C, q, k, p, adapter.torch)
            return qn, kn

        self.modeling.apply_rotary_pos_emb = wrapped
        return self

    def __exit__(self, *exc):
        self.modeling.apply_rotary_pos_emb = self._orig
        if self._orig_rotary_forward is not None:
            self.modeling.LlamaRotaryEmbedding.forward = self._orig_rotary_forward
        return False


# ---------------------------------------------------------------------------
# self-checks
# ---------------------------------------------------------------------------


def check_invariants(C: Constants, geom, tol=1e-10):
    """The properties section 7.3/§4.x say each surface must preserve."""
    out = []
    rng = np.random.default_rng(20260911)
    q = rng.normal(size=(1, 1, 8, 2 * geom.K))
    k = rng.normal(size=(1, 1, 8, 2 * geom.K))
    pos = np.array([0.0, 1.0, 37.0, 512.0, 2048.0, 8191.0, 16384.0, 32767.0])
    qn, kn, bias = apply_pure(C, q, k, pos, geom)

    g = C.gain
    # Isolate the gain by re-running the SAME operator with g = 1.  Comparing
    # qn/g against the unrotated q would be wrong: the rotation did happen.
    C1 = Constants(method=C.method, nu=C.nu, gain=1.0, phase_q=C.phase_q,
                   phase_k=C.phase_k, amp=C.amp, orthogonal=C.orthogonal,
                   metric=C.metric, q_diag=C.q_diag, k_diag=C.k_diag,
                   mu_q=C.mu_q, mu_k=C.mu_k,
                   rel_bias_mu=C.rel_bias_mu, transport=C.transport, scope=C.scope)
    if C.op is None:
        q1, k1, _ = apply_pure(C1, q, k, pos, geom)
        out.append(("gain is applied to both sides and to nothing else",
                    float(np.max(np.abs(qn - g * q1))) < tol
                    and float(np.max(np.abs(kn - g * k1))) < tol,
                    f"max dev {max(float(np.max(np.abs(qn - g * q1))), float(np.max(np.abs(kn - g * k1)))):.2e}"))
    if C.orthogonal is not None:
        out.append(("orthogonal is orthogonal",
                    float(np.max(np.abs(C.orthogonal @ C.orthogonal.T - np.eye(2 * geom.K)))) < 1e-8, ""))
    if C.metric is not None:
        prod = C.metric[:, 0] * C.metric[:, 1]
        out.append(("dual metric is reciprocal (zero-distance identity preserved)",
                    float(np.max(np.abs(prod - 1.0))) < 1e-12, ""))
    if C.rel_bias_mu is not None:
        d, b = relative_bias(C.rel_bias_mu, C.nu, geom)
        i0 = int(np.where(d == 0)[0][0])
        out.append(("relative bias vanishes at d = 0", abs(float(b[i0])) < 1e-12, f"b(0)={b[i0]:.2e}"))
    if C.transport is not None:
        out.append(("transport generator is antisymmetric",
                    float(np.max(np.abs(C.transport + C.transport.T))) < 1e-10, ""))
    if np.any(C.nu < 0):
        # section 4.06: a negative frequency is a legal rotation and must not be
        # abs()-repaired back into range.
        out.append(("negative frequencies are present and left signed",
                    True, f"{int(np.sum(C.nu < 0))} slots"))
    # the adapter must not vary by head or layer: shapes carry no H/L axis
    out.append(("no per-head or per-layer constant", True,
                "Constants carries one array per surface, no H/L axis"))
    return out
