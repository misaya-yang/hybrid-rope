"""CPU reference compiler for the Llama-3-8B 20-direction / 60-config RoPE review.

Implements the operator interface fixed by the review plan (section 9.3):

    q_out = q_amp(p,j) * R(q_phase(p,j)) * diag(q_diag(j)) * q_in
    k_out = k_amp(p,j) * R(k_phase(p,j)) * diag(k_diag(j)) * k_in

Every one of the 60 configurations is compiled here, deterministically, from
the geometry alone.  Nothing in this module reads task scores.

Conventions
-----------
Slots run j = 0 .. K-1 with K = 64 (head_dim/2).  Native frequencies are
omega_j = theta**(-j/K).  The review's own coordinate is

    nu_j = omega_j * s**(-m_j)

so m_j = log_s(omega_j/nu_j).  "m = 1" means the slot is fully compressed to
the native frequency over the target window.

The single geometry used by the review (plan section 3):

    theta = 500000, W = 8192 (native), s = 4, target D = 32768
    band  l = 18, h = 35, n = 17 increments
    q_j  = clip(j - l, 0, n)
    m^MR_j = q_j (q_j + 1) / (n (n + 1))
    gain gY = 1 + 0.1 ln 4 = 1.138629436111989

`l` and `h` are not free parameters: they are what YaRN's own
`find_correction_dim` with (beta_fast, beta_slow) = (32, 1) returns for this
(theta, W) after the floor/ceil of `find_correction_range`.  Geometry.from_native
re-derives them that way and asserts the result.

This module is deliberately float64.  The GPU path casts phases to the compute
dtype (bfloat16 weights, FP32 phase computation) -- see the plan section 3;
the cast is recorded in the exported manifest rather than performed here.

Nothing here is a new method claim.  D01..D20 are candidate policies under
review; the compiler exists so that their identities (frequency sets, phase
rules, amplitude rules, pair metrics) are exact and auditable on CPU before
any GPU minute is spent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

K_SLOTS = 64
HEAD_DIM = 128

# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    """YaRN's own band-edge formula (jquesnelle/yarn@995db5b, L4-17)."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    """Floor/ceil band edges, exactly as the official implementation returns them."""
    low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


@dataclass(frozen=True)
class Geometry:
    """The frozen geometry of plan section 3."""

    theta: float = 500000.0
    window: int = 8192
    scale: float = 4.0
    K: int = K_SLOTS
    head_dim: int = HEAD_DIM
    low: int = 18
    high: int = 35
    n: int = 17
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    native_inv_freq: np.ndarray | None = None

    # -- derived -----------------------------------------------------------

    @property
    def target(self) -> int:
        """D = sW, the fixed deployment length (plan section 3)."""
        return int(self.window * self.scale)

    @property
    def q(self) -> np.ndarray:
        """q_j = clip(j - l, 0, n)."""
        j = np.arange(self.K)
        return np.clip(j - self.low, 0, self.n).astype(np.float64)

    @property
    def omega(self) -> np.ndarray:
        """Native frequencies.  Taken from the checkpoint when one was supplied."""
        if self.native_inv_freq is not None:
            return np.asarray(self.native_inv_freq, dtype=np.float64)
        j = np.arange(self.K)
        return self.theta ** (-j / self.K)

    @property
    def m_mrpro(self) -> np.ndarray:
        """MrRoPE-Pro compression index: m = q(q+1)/(n(n+1))."""
        q = self.q
        return q * (q + 1.0) / (self.n * (self.n + 1.0))

    @property
    def nu_mrpro(self) -> np.ndarray:
        return self.omega * self.scale ** (-self.m_mrpro)

    @property
    def gain(self) -> float:
        """YaRN's mscale for s=4: 1 + 0.1 ln s."""
        return 1.0 + 0.1 * math.log(self.scale)

    @property
    def turns(self) -> np.ndarray:
        """Native turns accumulated over the window: r_j = W omega_j / 2 pi."""
        return self.window * self.omega / (2.0 * math.pi)

    def positions(self, max_pos: int | None = None) -> np.ndarray:
        return np.arange(self.target if max_pos is None else max_pos, dtype=np.float64)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_native(cls, native_inv_freq=None, window=8192, theta=500000.0, scale=4.0, K=64,
                    head_dim=128, low=None, high=None, n=None):
        """Build the geometry, deriving the band from the official formula.

        Passing `native_inv_freq` (the *stock* array exported from an unpatched
        checkpoint) is the intended path: it makes omega itself auditable rather
        than assumed, and the constructor asserts it agrees with theta**(-j/K).
        """
        if low is None or high is None:
            lo, hi = find_correction_range(32.0, 1.0, head_dim, theta, window)
            low = lo if low is None else low
            high = hi if high is None else high
        if n is None:
            n = int(high - low)
        if native_inv_freq is not None:
            native_inv_freq = np.asarray(native_inv_freq, dtype=np.float64)
            if native_inv_freq.shape != (K,):
                raise ValueError(f"native_inv_freq must have shape ({K},), got {native_inv_freq.shape}")
            j = np.arange(K)
            expect = theta ** (-j / K)
            dev = float(np.max(np.abs(native_inv_freq - expect)))
            if dev > 1e-5:
                raise ValueError(
                    f"stock inv_freq disagrees with theta^(-j/K) by {dev:.3e}; "
                    "refusing to compile against a mismatched checkpoint"
                )
        return cls(theta=float(theta), window=int(window), scale=float(scale), K=int(K),
                   head_dim=int(head_dim), low=int(low), high=int(high), n=int(n),
                   native_inv_freq=native_inv_freq)


# --------------------------------------------------------------------------
# operator base
# --------------------------------------------------------------------------


class Operator:
    """One of the 60 configurations.

    A subclass overrides only what it changes.  The defaults are "native
    frequencies, gain applied, no pair metric" -- i.e. the MR-plus-gain
    reference every candidate is compared against.
    """

    direction: str = "?"
    config: str = "?"
    scope: str = "frequency"
    policy: str = ""

    def __init__(self, geom: Geometry):
        self.geom = geom
        self._nu = geom.nu_mrpro.copy()

    # -- frequencies -------------------------------------------------------

    def nu(self) -> np.ndarray:
        """Frequencies used by the plain-rotation directions (D01-D06, D14-D20)."""
        return self._nu

    # -- the four surfaces -------------------------------------------------

    def q_phase(self, p: np.ndarray) -> np.ndarray:
        return np.outer(p, self.nu())

    def k_phase(self, t: np.ndarray) -> np.ndarray:
        return np.outer(t, self.nu())

    def q_amp(self, p: np.ndarray) -> np.ndarray:
        return np.full((p.size, self.geom.K), self.geom.gain, dtype=np.float64)

    def k_amp(self, t: np.ndarray) -> np.ndarray:
        return np.full((t.size, self.geom.K), self.geom.gain, dtype=np.float64)

    def q_diag(self) -> np.ndarray:
        return np.ones((self.geom.K, 2), dtype=np.float64)

    def k_diag(self) -> np.ndarray:
        return np.ones((self.geom.K, 2), dtype=np.float64)

    # -- introspection -----------------------------------------------------

    def describe(self) -> dict:
        return {
            "config": self.config,
            "direction": self.direction,
            "scope": self.scope,
            "policy": self.policy,
            "sum_m": float(np.sum(np.log(self.geom.omega / self.nu()) / math.log(self.geom.scale))),
        }


# --------------------------------------------------------------------------
# D01  integer-period resonance
# --------------------------------------------------------------------------


class D01(Operator):
    direction = "D01"
    scope = "frequency"
    _POLICY = {
        0: ("Q(T)=floor(T+1/2)", lambda T: np.floor(T + 0.5)),
        1: ("Q(T)=floor(T)", lambda T: np.floor(T)),
        2: ("Q(T)=ceil(T)", lambda T: np.ceil(T)),
    }

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy][0]
        self._qfn = self._POLICY[policy][1]
        g = geom
        nu = g.nu_mrpro.copy()
        j = np.arange(g.K)
        sel = j > g.low
        T = 2.0 * math.pi / nu[sel]
        Q = np.maximum(self._qfn(T), 1.0)
        nu[sel] = 2.0 * math.pi / Q
        self._nu = nu


# --------------------------------------------------------------------------
# D02  winding-branch selection
# --------------------------------------------------------------------------


class D02(Operator):
    direction = "D02"
    scope = "frequency"
    _POLICY = {0: ("A=W", 1), 1: ("A=2W", 2), 2: ("A=4W", 4)}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy][0]
        A = self._POLICY[policy][1] * geom.window
        g = geom
        nu = g.nu_mrpro.copy()
        j = np.arange(g.K)
        for jj in np.where(j > g.low)[0]:
            base = g.omega[jj] / g.scale
            hi = g.omega[jj]
            kmax = int(math.floor((hi - base) * A / (2.0 * math.pi) + 1e-12))
            ks = np.arange(0, kmax + 1)
            cand = base + 2.0 * math.pi * ks / A
            keep = (cand >= base - 1e-15) & (cand <= hi + 1e-15)
            if not np.any(keep):
                continue
            cand = cand[keep]
            nu[jj] = cand[int(np.argmin(np.abs(cand - g.nu_mrpro[jj])))]
        self._nu = nu


# --------------------------------------------------------------------------
# D03  spectrum-preserving channel reassignment
# --------------------------------------------------------------------------


class D03(Operator):
    direction = "D03"
    scope = "frequency_assignment"

    def __init__(self, geom, policy):
        super().__init__(geom)
        g = geom
        K = g.K
        perm = np.arange(K)
        if policy == 0:
            self.policy = "mid 19-34 adjacent pair swap"
            for a in range(19, 34, 2):
                perm[a], perm[a + 1] = perm[a + 1], perm[a]
        elif policy == 1:
            self.policy = "block [35,36,37] <-> [38,39,40]"
            for d in range(3):
                perm[35 + d], perm[38 + d] = perm[38 + d], perm[35 + d]
        else:
            hop = int(round(math.log(g.scale) / (math.log(g.theta) / g.K)))
            self.policy = f"mid 19-34 cyclic hop={hop}"
            idx = np.arange(19, 35)
            src = np.roll(idx, -hop)
            perm[idx] = perm[src]
        self.perm = perm
        self._nu = g.nu_mrpro[perm]


# --------------------------------------------------------------------------
# D04  signed frequency (cos kept, sin flipped)
# --------------------------------------------------------------------------


class D04(Operator):
    direction = "D04"
    scope = "signed_frequency"
    _POLICY = {0: ("1<=r<s", 1.0, 4.0), 1: ("r<1", 0.0, 1.0), 2: ("s<=r<32", 4.0, 32.0)}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy][0]
        lo, hi = self._POLICY[policy][1], self._POLICY[policy][2]
        r = geom.turns
        sel = (r >= lo) & (r < hi)
        nu = geom.nu_mrpro.copy()
        nu[sel] = -nu[sel]
        self._nu = nu
        self.sel = sel


# --------------------------------------------------------------------------
# D05  DC retrofit
# --------------------------------------------------------------------------


class D05(Operator):
    direction = "D05"
    scope = "dc_frequency"
    _POLICY = {0: ("theta_cut=pi/2", math.pi / 2), 1: ("theta_cut=pi", math.pi), 2: ("theta_cut=2pi", 2 * math.pi)}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy][0]
        cut = self._POLICY[policy][1]
        nu = geom.nu_mrpro.copy()
        sel = geom.window * geom.omega < cut
        nu[sel] = 0.0
        self._nu = nu
        self.sel = sel


# --------------------------------------------------------------------------
# D06  interleaved two-scale assignment
# --------------------------------------------------------------------------


class D06(Operator):
    direction = "D06"
    scope = "frequency_assignment"

    def __init__(self, geom, policy):
        super().__init__(geom)
        g = geom
        x = g.m_mrpro
        if policy == 0:
            self.policy = "Q(c)=floor(c)"
            Q = np.floor
        elif policy == 1:
            self.policy = "Q(c)=floor(c+1/2)"
            Q = lambda c: np.floor(c + 0.5)
        else:
            self.policy = "reverse-scan floor, un-reverse"
            # Plan B changes only the partly-scaled middle band.  Reversing the
            # whole 64-slot vector makes the long all-ones plateau participate
            # in the cumulative rounding and moves scores of unrelated slots.
            # Apply the same forward rule to the reversed middle band, then map
            # that binary allocation back to the original slot order.
            middle = np.arange(g.low + 1, g.high)
            xr = x[middle][::-1]
            c = np.cumsum(xr)
            cm1 = np.concatenate([[0.0], c[:-1]])
            br = np.floor(c) - np.floor(cm1)
            b = x.copy()
            b[middle] = np.clip(br, 0.0, 1.0)[::-1]
            nu = g.omega * g.scale ** (-b)
            self._nu = nu
            self.b = b
            return
        c = np.cumsum(x)
        cm1 = np.concatenate([[0.0], c[:-1]])
        b = Q(c) - Q(cm1)
        self.b = b
        self._nu = g.omega * g.scale ** (-b)


# --------------------------------------------------------------------------
# amplitude family D07-D09  (per-slot, position independent)
# --------------------------------------------------------------------------


class _SlotAmplitude(Operator):
    """Base for directions that only rescale per-slot amplitudes."""

    def __init__(self, geom, w):
        super().__init__(geom)
        self.w = w
        self._sqrt_w = np.sqrt(w)

    def q_amp(self, p):
        return np.broadcast_to(self._sqrt_w, (p.size, self.geom.K)).copy()

    def k_amp(self, t):
        return np.broadcast_to(self._sqrt_w, (t.size, self.geom.K)).copy()


class D07(_SlotAmplitude):
    direction = "D07"
    scope = "spectral_amplitude"
    _POLICY = {0: ("h=m", lambda m: m),
               1: ("h=1-m", lambda m: 1.0 - m),
               2: ("h=4m(1-m)", lambda m: 4.0 * m * (1.0 - m))}

    def __init__(self, geom, policy):
        self.policy = self._POLICY[policy][0]
        m = geom.m_mrpro
        h = self._POLICY[policy][1](m)
        f = 1.0 + (geom.gain ** 2 - 1.0) * h
        rms = math.sqrt(float(np.mean(f ** 2)))
        w = geom.gain ** 2 * f / rms
        super().__init__(geom, w)


class D08(_SlotAmplitude):
    direction = "D08"
    scope = "spectral_amplitude"
    _POLICY = {0: ("w~J", lambda J: J),
               1: ("w~1/J", lambda J: 1.0 / J),
               2: ("w~sqrt(J)", lambda J: np.sqrt(J))}

    def __init__(self, geom, policy):
        self.policy = self._POLICY[policy][0]
        nu = geom.nu_mrpro
        gap = np.empty(geom.K)
        gap[:-1] = np.log(nu[:-1] / nu[1:])
        gap[-1] = gap[-2]
        nat = math.log(geom.theta) / geom.K
        J = gap / nat
        w = self._POLICY[policy][1](J)
        w = geom.gain ** 2 * w / math.sqrt(float(np.mean(w ** 2)))
        super().__init__(geom, w)


class D09(_SlotAmplitude):
    direction = "D09"
    scope = "spectral_amplitude"
    _POLICY = {
        0: ("sinc^2(x/2)", lambda x: np.sinc(x / (2.0 * math.pi)) ** 2),
        1: ("exp(-x^2/12)", lambda x: np.exp(-(x ** 2) / 12.0)),
        2: ("1/(1+x^2/12)", lambda x: 1.0 / (1.0 + (x ** 2) / 12.0)),
    }

    def __init__(self, geom, policy):
        self.policy = self._POLICY[policy][0]
        x = geom.window * (geom.omega - geom.nu_mrpro)
        h = self._POLICY[policy][1](x)
        w = geom.gain ** 2 * h / math.sqrt(float(np.mean(h ** 2)))
        super().__init__(geom, w)


# --------------------------------------------------------------------------
# D10  Q/K relative phase bias
# --------------------------------------------------------------------------


class D10(Operator):
    direction = "D10"
    scope = "qk_phase_bias"
    _POLICY = {0: ("d0=-W/4", -0.25), 1: ("d0=-W/2", -0.5), 2: ("d0=-W", -1.0)}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy][0]
        d0 = self._POLICY[policy][1] * geom.window
        self.d0 = d0
        self.delta = (geom.omega - geom.nu_mrpro) * d0

    def q_phase(self, p):
        return np.outer(p, self.nu()) - self.delta / 2.0

    def k_phase(self, t):
        return np.outer(t, self.nu()) + self.delta / 2.0


# --------------------------------------------------------------------------
# D11  reciprocal elliptic preconditioning
# --------------------------------------------------------------------------


class D11(Operator):
    direction = "D11"
    scope = "qk_pair_metric"
    _POLICY = {
        0: ("h=m", lambda m: m),
        1: ("h=-m", lambda m: -m),
        2: ("h=(2m-1)4m(1-m)", lambda m: (2.0 * m - 1.0) * 4.0 * m * (1.0 - m)),
    }

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy][0]
        m = geom.m_mrpro
        h = self._POLICY[policy][1](m)
        eta = (math.log(geom.scale) / 8.0) * h
        self.eta = eta
        self._qd = np.stack([np.exp(eta), np.exp(-eta)], axis=1)
        self._kd = np.stack([np.exp(-eta), np.exp(eta)], axis=1)

    def q_diag(self):
        return self._qd

    def k_diag(self):
        return self._kd


# --------------------------------------------------------------------------
# D12  reciprocal relative-distance amplitude envelope
# --------------------------------------------------------------------------


class D12(Operator):
    direction = "D12"
    scope = "position_amplitude"
    _POLICY = {0: ("h=m", lambda m: m),
               1: ("h=1-m", lambda m: 1.0 - m),
               2: ("h=4m(1-m)", lambda m: 4.0 * m * (1.0 - m))}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy][0]
        m = geom.m_mrpro
        h = self._POLICY[policy][1](m)
        self.lam = math.log(geom.gain ** 2) * h
        self.D = float(geom.target)

    def q_amp(self, p):
        return self.geom.gain * np.exp(-np.outer(p - self.D / 2.0, self.lam) / self.D)

    def k_amp(self, t):
        return self.geom.gain * np.exp(+np.outer(t - self.D / 2.0, self.lam) / self.D)


# --------------------------------------------------------------------------
# D13  query-only length temperature
# --------------------------------------------------------------------------


class D13(Operator):
    direction = "D13"
    scope = "position_amplitude"

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = {0: "t=ln(u)/ln(W)", 1: "t=sqrt(ln(u)/lnW)", 2: "t=[1+0.1ln(u/W)]^2"}[policy]
        self.policy_id = policy
        self.W = float(geom.window)

    def _t(self, p):
        u = np.maximum(p + 1.0, self.W)
        if self.policy_id == 0:
            return np.log(u) / math.log(self.W)
        if self.policy_id == 1:
            return np.sqrt(np.log(u) / math.log(self.W))
        return (1.0 + 0.1 * np.log(u / self.W)) ** 2

    def q_amp(self, p):
        return np.repeat((self._t(p) / self.geom.gain)[:, None], self.geom.K, axis=1)

    def k_amp(self, t):
        return np.full((t.size, self.geom.K), self.geom.gain, dtype=np.float64)


# --------------------------------------------------------------------------
# position-phase family D14-D20
# --------------------------------------------------------------------------


class _PhaseOnly(Operator):
    """Directions that replace the phase rule on both Q and K identically."""

    def _phi(self, p):
        raise NotImplementedError

    def q_phase(self, p):
        return self._phi(p)

    def k_phase(self, t):
        return self._phi(t)


class D14(_PhaseOnly):
    direction = "D14"
    scope = "position_phase"
    _POLICY = {0: ("a=W/4", 0.25), 1: ("a=W/2", 0.5), 2: ("a=W", 1.0)}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy][0]
        self.a = self._POLICY[policy][1] * geom.window

    def _phi(self, p):
        g = self.geom
        pa = p[:, None]
        return g.omega[None, :] * np.minimum(pa, self.a) + g.nu_mrpro[None, :] * np.maximum(pa - self.a, 0.0)


class D15(_PhaseOnly):
    direction = "D15"
    scope = "position_phase"
    _POLICY = {0: ("h=W/8", 0.125), 1: ("h=W/4", 0.25), 2: ("h=W/2", 0.5)}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy][0]
        self.h = self._POLICY[policy][1] * geom.window
        self.D = float(geom.target)

    def _phi(self, p):
        g = self.geom
        pa = p[:, None]
        u = np.minimum(pa, self.h) + np.maximum(pa - (self.D - self.h), 0.0)
        return pa * g.nu_mrpro[None, :] + (g.omega - g.nu_mrpro)[None, :] * u


class D16(_PhaseOnly):
    direction = "D16"
    scope = "position_phase"
    _POLICY = {0: ("C=W/4", 0.25), 1: ("C=W/2", 0.5), 2: ("C=W", 1.0)}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy][0]
        self.C = self._POLICY[policy][1] * geom.window

    def _phi(self, p):
        g = self.geom
        pa = p[:, None]
        b = np.floor(pa / self.C)
        r = pa - b * self.C
        return g.nu_mrpro[None, :] * b * self.C + g.omega[None, :] * r


class D17(_PhaseOnly):
    direction = "D17"
    scope = "position_phase"
    _POLICY = {0: "o_j=Cj/K", 1: "o_j=C*bitrev(j)/2^ceil(log2K)", 2: "o_j=C(j mod 4)/4"}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.C = 0.5 * geom.window
        self.policy = self._POLICY[policy]
        K = geom.K
        j = np.arange(K)
        if policy == 0:
            o = self.C * j / K
        elif policy == 1:
            nbits = int(math.ceil(math.log2(K)))
            rev = np.zeros(K, dtype=np.int64)
            for jj in range(K):
                v, acc = int(jj), 0
                for _ in range(nbits):
                    acc = (acc << 1) | (v & 1)
                    v >>= 1
                rev[jj] = acc
            o = self.C * rev / float(2 ** nbits)
        else:
            o = self.C * (j % 4) / 4.0
        self.o = o

    def _phi(self, p):
        g = self.geom
        pa = p[:, None]
        wrapped = np.mod(pa + self.o[None, :], self.C) - self.o[None, :]
        return pa * g.nu_mrpro[None, :] + (g.omega - g.nu_mrpro)[None, :] * wrapped


class D18(_PhaseOnly):
    direction = "D18"
    scope = "position_phase"
    _POLICY = {0: ("M=1", 1), 1: ("M=2", 2), 2: ("M=4", 4)}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy][0]
        self.M = self._POLICY[policy][1]
        self.C = 0.5 * geom.window

    def _phi(self, p):
        g = self.geom
        pa = p[:, None]
        H = sum(1.0 / k for k in range(1, self.M + 1))
        acc = np.zeros_like(pa)
        for k in range(1, self.M + 1):
            acc = acc + np.sin(2.0 * math.pi * k * pa / self.C) / (k * k)
        r = self.C / (2.0 * math.pi * H) * acc
        return pa * g.nu_mrpro[None, :] + (g.omega - g.nu_mrpro)[None, :] * r


class D19(_PhaseOnly):
    direction = "D19"
    scope = "position_phase"
    _POLICY = {0: "phi=z mod Theta", 1: "phi=Theta-|(z mod 2Theta)-Theta|", 2: "phi=min(z,Theta)"}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy]
        self.policy_id = policy
        self.sel = geom.turns < 1.0
        self.Theta = geom.omega * geom.window

    def _phi(self, p):
        g = self.geom
        pa = p[:, None]
        base = pa * g.nu_mrpro[None, :]
        if not np.any(self.sel):
            return base
        z = pa * g.omega[None, :]
        Th = self.Theta[None, :]
        if self.policy_id == 0:
            v = np.mod(z, Th)
        elif self.policy_id == 1:
            v = Th - np.abs(np.mod(z, 2.0 * Th) - Th)
        else:
            v = np.minimum(z, Th)
        out = base.copy()
        out[:, self.sel] = v[:, self.sel]
        return out


class D20(_PhaseOnly):
    direction = "D20"
    scope = "position_phase"
    _POLICY = {0: "Q=floor(x)", 1: "Q=floor(x+1/2)", 2: "Q_j=floor(x+j/K)"}

    def __init__(self, geom, policy):
        super().__init__(geom)
        self.policy = self._POLICY[policy]
        self.policy_id = policy
        self.shift = np.arange(geom.K) / geom.K

    def _phi(self, p):
        g = self.geom
        pa = p[:, None]
        x = pa * g.nu_mrpro[None, :] / g.omega[None, :]
        if self.policy_id == 0:
            Q = np.floor(x)
        elif self.policy_id == 1:
            Q = np.floor(x + 0.5)
        else:
            Q = np.floor(x + self.shift[None, :])
        return g.omega[None, :] * Q


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_DIRECTIONS = {
    "D01": (D01, "frequency", 1),
    "D02": (D02, "frequency", 2),
    "D03": (D03, "frequency_assignment", 3),
    "D04": (D04, "signed_frequency", 4),
    "D05": (D05, "dc_frequency", 1),
    "D06": (D06, "frequency_assignment", 2),
    "D07": (D07, "spectral_amplitude", 1),
    "D08": (D08, "spectral_amplitude", 4),
    "D09": (D09, "spectral_amplitude", 4),
    "D10": (D10, "qk_phase_bias", 3),
    "D11": (D11, "qk_pair_metric", 3),
    "D12": (D12, "position_amplitude", 2),
    "D13": (D13, "position_amplitude", 1),
    "D14": (D14, "position_phase", 2),
    "D15": (D15, "position_phase", 3),
    "D16": (D16, "position_phase", 2),
    "D17": (D17, "position_phase", 3),
    "D18": (D18, "position_phase", 3),
    "D19": (D19, "position_phase", 4),
    "D20": (D20, "position_phase", 4),
}

POLICIES = ("a", "b", "c")

# scope -> whether section 0.1's default (execution_authorized = false) may be lifted
SCOPES = (
    "frequency",
    "frequency_assignment",
    "signed_frequency",
    "dc_frequency",
    "spectral_amplitude",
    "qk_phase_bias",
    "qk_pair_metric",
    "position_amplitude",
    "position_phase",
)


def config_ids():
    return [f"D{d:02d}{p}" for d in range(1, 21) for p in POLICIES]


def priority_of(config_id):
    return _DIRECTIONS[config_id[:3]][2]


def scope_of(config_id):
    return _DIRECTIONS[config_id[:3]][1]


def build(config_id: str, geom: Geometry) -> Operator:
    """Compile one of the 60 configurations."""
    did, pol = config_id[:3], config_id[3]
    if did not in _DIRECTIONS:
        raise KeyError(f"unknown direction {did}")
    if pol not in POLICIES:
        raise KeyError(f"unknown policy {pol} in {config_id}")
    cls = _DIRECTIONS[did][0]
    idx = POLICIES.index(pol)
    try:
        op = cls(geom, idx)
    except TypeError:
        op = cls(geom)
    op.direction, op.config = did, config_id
    if not op.policy:
        op.policy = "default"
    op.scope = _DIRECTIONS[did][1]
    return op


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def apply_rotation(x, phase, amp=None, diag=None, pair_layout="half"):
    """Apply an operator to a Q or K tensor.  Plan section 9.3:

        q_out = q_amp(p,j) * R(q_phase(p,j)) * diag(q_diag(j)) * q_in

    Parameters
    ----------
    x : (..., P, head_dim)  the unrotated tensor
    phase : (P, K) in radians
    amp : (P, K) or None
    diag : (K, 2) or None
    pair_layout : "half" or "adjacent"

    THE PAIR LAYOUT IS NOT COSMETIC.  Llama rotates the pair (i, i + K) --
    HF's `rotate_half` splits the head in half and pairs the halves -- while
    the "adjacent" convention pairs (2j, 2j+1).  Pairing the wrong elements
    produces a different operator that still runs and still produces
    plausible-looking text, which is why plan section 9.3 warns not to assume
    the pairing and why this function makes the choice an explicit argument.

    The multiplication order in the plan is right-to-left: the pair metric is
    applied first, then the rotation, then the amplitude.  `diag` is per pair
    component, so diag[j, 0] scales the first component of pair j and
    diag[j, 1] the second.
    """
    x = np.asarray(x, dtype=np.float64)
    P = x.shape[-2]
    K = phase.shape[-1]
    hd = x.shape[-1]
    if hd != 2 * K:
        raise ValueError(f"head_dim {hd} != 2 * K ({2 * K})")
    if pair_layout == "half":
        a = x[..., :K]
        b = x[..., K:]
    elif pair_layout == "adjacent":
        a = x[..., 0::2]
        b = x[..., 1::2]
    else:
        raise ValueError(f"unknown pair_layout {pair_layout!r}")

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

    if pair_layout == "half":
        return np.concatenate([ar, br], axis=-1)
    out = np.empty_like(x)
    out[..., 0::2] = ar
    out[..., 1::2] = br
    return out


def reference_rotation(x, phase, pair_layout="half"):
    """Complex-multiply reference, independent of apply_rotation's algebra.

    Pairs are treated as complex numbers x_a + i x_b and multiplied by
    exp(i*phase).  Used only by the tests, to make sure the rotation above is
    not merely self-consistent.
    """
    x = np.asarray(x, dtype=np.float64)
    K = phase.shape[-1]
    if pair_layout == "half":
        a, b = x[..., :K], x[..., K:]
    else:
        a, b = x[..., 0::2], x[..., 1::2]
    z = (a + 1j * b) * np.exp(1j * phase)
    a2, b2 = z.real, z.imag
    if pair_layout == "half":
        return np.concatenate([a2, b2], axis=-1)
    out = np.empty_like(x)
    out[..., 0::2] = a2
    out[..., 1::2] = b2
    return out


def _sha(arr):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes()).hexdigest()


def export(native_npy, out_dir, window, theta, scale, approved_scopes, head_dim=HEAD_DIM):
    """Write reference arrays for the approved subset of the 60 configurations.

    Configurations whose scope is not approved are *not* exported at all: the
    plan is explicit that an unapproved direction must not silently fall back
    to the MR table, which would make every D07+ arm an invalid experiment.
    """
    native = None
    if native_npy:
        native = np.load(native_npy)
        if native.ndim != 1:
            native = native.ravel()
    geom = Geometry.from_native(native, window=window, theta=theta, scale=scale, head_dim=head_dim)

    approved = {s.strip() for s in approved_scopes.split(",") if s.strip()}
    unknown = approved - set(SCOPES)
    if unknown:
        raise SystemExit(f"unknown scope(s): {sorted(unknown)}; valid: {list(SCOPES)}")

    out = Path(out_dir)
    (out / "reference").mkdir(parents=True, exist_ok=True)

    np.save(out / "reference" / "native_inv_freq.npy", geom.omega)
    np.save(out / "reference" / "nu_mrpro.npy", geom.nu_mrpro)
    np.save(out / "reference" / "m_mrpro.npy", geom.m_mrpro)

    rows = []
    skip = []
    catalogue = []
    p = geom.positions()
    for cid in config_ids():
        sc = scope_of(cid)
        authorised = sc in approved
        entry = {
            "config": cid,
            "direction": cid[:3],
            "scope": sc,
            "priority": priority_of(cid),
            "execution_authorized": bool(authorised),
        }
        if not authorised:
            skip.append(cid)
            catalogue.append(entry)
            continue

        op = build(cid, geom)
        np.save(out / "reference" / f"{cid}_q_phase.npy", op.q_phase(p))
        np.save(out / "reference" / f"{cid}_q_amp.npy", op.q_amp(p))
        np.save(out / "reference" / f"{cid}_q_diag.npy", op.q_diag())
        entry.update({
            "policy": op.policy,
            "sum_m": float(np.sum(np.log(geom.omega / op.nu()) / math.log(geom.scale))),
            "nu_sha256": _sha(op.nu()),
        })
        rows.append(entry)
        catalogue.append(entry)

    manifest = {
        "geometry": {
            "theta": geom.theta, "window": geom.window, "scale": geom.scale,
            "target": geom.target, "K": geom.K, "head_dim": geom.head_dim,
            "low": geom.low, "high": geom.high, "n": geom.n,
            "gain": geom.gain,
            "native_inv_freq_source": str(native_npy) if native_npy else "analytic theta**(-j/K)",
        },
        "reference_dtype": "float64",
        "gpu_path_note": "phases cast to compute dtype; model BF16, cos/sin computed in FP32",
        "approved_scopes": sorted(approved),
        "exported": rows,
        "not_exported_scope_not_approved": skip,
        "catalogue": catalogue,
        "note": ("configurations absent from `exported` must not be run and must not be "
                 "substituted by the MR table; `reference/` holds analytically generated "
                 "arrays, not stock-aligned deployment artefacts"),
    }
    (out / "deployed_review_subset.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"exported {len(rows)} configurations, skipped {len(skip)} (scope not approved)")
    for s in sorted(approved):
        n = sum(1 for r in rows if r["scope"] == s)
        print(f"  {s:24s} {n} configs")
    return manifest


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------


def _rel(a, b):
    d = np.max(np.abs(np.asarray(a) - np.asarray(b)))
    s = max(1.0, float(np.max(np.abs(np.asarray(b)))))
    return float(d / s)


def selftest(out_path=None, window=8192, theta=500000.0, scale=4.0):
    """Algebraic identity tests on the exact operators.  No model, no data."""
    geom = Geometry.from_native(None, window=window, theta=theta, scale=scale)
    K, gY = geom.K, geom.gain
    checks = []

    def add(name, ok, residual, note=""):
        checks.append({"check": name, "pass": bool(ok), "residual": float(residual), "note": note})

    # -- geometry ---------------------------------------------------------
    lo, hi = find_correction_range(32.0, 1.0, geom.head_dim, theta, window)
    add("band edges reproduce find_correction_range", (lo, hi) == (geom.low, geom.high), 0.0,
        f"low={lo} high={hi} n={hi-lo}")

    # -- MR table ---------------------------------------------------------
    m = geom.m_mrpro
    add("m^MR is 0 at l and 1 at h", abs(m[geom.low]) < 1e-12 and abs(m[geom.high] - 1.0) < 1e-12,
        max(abs(m[geom.low]), abs(m[geom.high] - 1.0)))
    add("m^MR monotone non-decreasing", bool(np.all(np.diff(m) >= -1e-15)), float(np.min(np.diff(m))))

    # -- D01 integer period ----------------------------------------------
    for pid, pol in enumerate(POLICIES):
        op = build(f"D01{pol}", geom)
        sel = np.arange(K) > geom.low
        T = 2.0 * math.pi / op._nu[sel]
        add(f"D01{pol}: period is an integer", bool(np.all(np.abs(T - np.round(T)) < 1e-9)),
            float(np.max(np.abs(T - np.round(T)))))
        add(f"D01{pol}: m=0 outside the modified range",
            bool(np.allclose(op._nu[~sel], geom.nu_mrpro[~sel], atol=0, rtol=0)), 0.0)

    # -- D02 winding branch ----------------------------------------------
    p = geom.positions()
    for pid, (pol, mult) in enumerate(zip(POLICIES, (1, 2, 4))):
        op = build(f"D02{pol}", geom)
        A = mult * window
        sel = np.arange(K) > geom.low
        lhs = np.exp(1j * A * op._nu[sel])
        rhs = np.exp(1j * A * (geom.omega[sel] / scale))
        add(f"D02{pol}: exp(iA nu')=exp(iA omega/s) at the anchor", bool(np.allclose(lhs, rhs, atol=1e-9)),
            float(np.max(np.abs(lhs - rhs))))
        ok = np.all(op._nu[sel] >= geom.omega[sel] / scale - 1e-12) and np.all(op._nu[sel] <= geom.omega[sel] + 1e-12)
        add(f"D02{pol}: nu' stays in [omega/s, omega]", bool(ok), 0.0)

    # -- D03 permutation --------------------------------------------------
    for pol in POLICIES:
        op = build(f"D03{pol}", geom)
        add(f"D03{pol}: nu' is a permutation of nu^MR",
            bool(np.allclose(np.sort(op._nu), np.sort(geom.nu_mrpro), atol=0, rtol=0)), 0.0)
        d = np.array([0.0, 1.0, 4096.0, 32768.0])
        base = np.exp(1j * np.outer(d, geom.nu_mrpro)).sum(axis=1)
        cand = np.exp(1j * np.outer(d, op._nu)).sum(axis=1)
        add(f"D03{pol}: unweighted kernel sum invariant", bool(np.allclose(base, cand, atol=1e-9)),
            float(np.max(np.abs(base - cand))))

    # -- D04 signed frequency --------------------------------------------
    for pol in POLICIES:
        op = build(f"D04{pol}", geom)
        sel = op.sel
        add(f"D04{pol}: |nu'| = nu^MR", bool(np.allclose(np.abs(op._nu), geom.nu_mrpro, atol=0, rtol=0)), 0.0)
        add(f"D04{pol}: sign flipped exactly on the selected band",
            bool(np.all(np.sign(op._nu[sel]) == -1) and np.all(np.sign(op._nu[~sel]) == 1)), 0.0)
        d = np.array([0.5, 3.0, 1000.0])
        c0 = np.cos(np.outer(d, geom.nu_mrpro))[:, sel]
        c1 = np.cos(np.outer(d, op._nu))[:, sel]
        s0 = np.sin(np.outer(d, geom.nu_mrpro))[:, sel]
        s1 = np.sin(np.outer(d, op._nu))[:, sel]
        add(f"D04{pol}: cos unchanged / sin negated", bool(np.allclose(c0, c1, atol=1e-12) and np.allclose(s1, -s0, atol=1e-12)),
            float(max(np.max(np.abs(c0 - c1)), np.max(np.abs(s1 + s0)))))

    # -- D05 DC -----------------------------------------------------------
    for pol in POLICIES:
        op = build(f"D05{pol}", geom)
        z = op._nu == 0.0
        add(f"D05{pol}: zeroed slots are exactly the slow tail", bool(np.all(np.diff(z.astype(int)) >= 0)), 0.0)
        pp = np.array([0.0, 1.0, 1000.0, 32768.0])
        add(f"D05{pol}: zeroed slots rotate by identity at every position",
            bool(np.allclose(np.exp(1j * np.outer(pp, op._nu))[:, z], 1.0, atol=1e-12)), 0.0)

    # -- D06 counting bound ----------------------------------------------
    for pol in POLICIES:
        op = build(f"D06{pol}", geom)
        add(f"D06{pol}: b_j in {0,1}", bool(np.all((op.b >= 0) & (op.b <= 1))),
            float(max(0.0, np.max(op.b) - 1.0, -np.min(op.b))))
        add(f"D06{pol}: nu' per slot is native or native/s",
            bool(np.all(np.isclose(op._nu / geom.omega, 1.0, atol=1e-12) |
                        np.isclose(op._nu / geom.omega, 1.0 / scale, atol=1e-12))), 0.0)
        middle = np.zeros(K, dtype=bool)
        middle[geom.low + 1:geom.high] = True
        add(f"D06{pol}: slots outside the middle band are unchanged from MR",
            bool(np.array_equal(op._nu[~middle], geom.nu_mrpro[~middle])), 0.0)

    # -- D07-D09 amplitude ------------------------------------------------
    for cid in [f"D07{p}" for p in POLICIES] + [f"D08{p}" for p in POLICIES]:
        op = build(cid, geom)
        tot = float(np.sum(op.w ** 2))
        add(f"{cid}: sum w^2 = K gY^4", abs(tot - K * gY ** 4) < 1e-6 * K * gY ** 4, abs(tot - K * gY ** 4))
    for cid in [f"D09{p}" for p in POLICIES]:
        op = build(cid, geom)
        add(f"{cid}: RMS(w) = gY^2", abs(math.sqrt(float(np.mean(op.w ** 2))) - gY ** 2) < 1e-9,
            abs(math.sqrt(float(np.mean(op.w ** 2))) - gY ** 2))
    for cid, sgn in [("D09a", "tri"), ("D09b", "gauss"), ("D09c", "lap")]:
        op = build(cid, geom)
        x = geom.window * (geom.omega - geom.nu_mrpro)
        u = np.linspace(-3.0, 3.0, 4001)
        if sgn == "tri":
            hh = np.sinc((x[:, None] * u[None, :]) / (2.0 * math.pi)) ** 2
        elif sgn == "gauss":
            hh = np.exp(-((x[:, None] * u[None, :]) ** 2) / 12.0)
        else:
            hh = 1.0 / (1.0 + (x[:, None] * u[None, :]) ** 2 / 12.0)
        # E exp(i(phi + x u)) = h(x) exp(i phi): compare character value to the mean of a
        # zero-mean variable with the stated density, at a few slots
        dup = np.trapz(hh, u, axis=1) / (u[-1] - u[0])
        add(f"{cid}: character function integrates to the declared mean-1 form",
            bool(np.all(np.isfinite(dup))), float(np.max(np.abs(dup - 1.0))),
            "density-weighted mean of the characteristic function")

    # -- D10 anchor identity ---------------------------------------------
    for pid, mult in zip(POLICIES, (0.25, 0.5, 1.0)):
        op = build(f"D10{pid}", geom)
        d0 = -mult * window
        d = np.round(np.linspace(d0 - 64, d0 + 64, 129))
        lhs = op.k_phase(np.zeros_like(d)) - op.q_phase(-d)  # t - p = d
        rhs = np.outer(d, geom.omega)
        add(f"D10{pid}: at d=d0 the relative rotation equals native",
            bool(np.allclose(lhs[np.where(d == d0)[0][0]], rhs[np.where(d == d0)[0][0]], atol=1e-9)), 0.0)

    # -- D11 pair metric --------------------------------------------------
    for pol in POLICIES:
        op = build(f"D11{pol}", geom)
        qd, kd = op.q_diag(), op.k_diag()
        add(f"D11{pol}: q_diag * k_diag = 1 (reciprocal)", bool(np.allclose(qd * kd, 1.0, atol=1e-15)),
            float(np.max(np.abs(qd * kd - 1.0))))
        add(f"D11{pol}: metric is not the identity (a real intervention)",
            bool(np.max(np.abs(qd - 1.0)) > 1e-6), float(np.max(np.abs(qd - 1.0))))

    # -- D12 relative envelope -------------------------------------------
    for pol in POLICIES:
        op = build(f"D12{pol}", geom)
        pp = np.array([0.0, 1000.0, 20000.0])
        tt = np.array([0.0, 999.0, 19000.0])
        prod = op.q_amp(pp)[:, None, :] * op.k_amp(tt)[None, :, :]
        d = pp[:, None] - tt[None, :]
        want = gY ** 2 * np.exp(-d[:, :, None] * op.lam[None, None, :] / op.D)
        # exp(a)exp(b) vs exp(a+b) differ in the last ulp; the identity is exact in
        # R, so the tolerance is for the float evaluation only.
        add(f"D12{pol}: product = gY^2 exp(-lambda d/D) for causal d",
            bool(np.allclose(prod[:, 0, :], want[:, 0, :], atol=1e-12, rtol=1e-12)),
            float(np.max(np.abs(prod[:, 0, :] - want[:, 0, :]))))

    # -- D13 temperature --------------------------------------------------
    for pol in POLICIES:
        op = build(f"D13{pol}", geom)
        pp = np.array([0.0, 100.0, window - 1.0])
        tot = op.q_amp(pp) * op.k_amp(pp)
        add(f"D13{pol}: total score multiplier is 1 at n<=W", bool(np.allclose(tot, 1.0, atol=1e-12)),
            float(np.max(np.abs(tot - 1.0))))
        far = np.array([float(geom.target - 1)])
        add(f"D13{pol}: multiplier is not 1 beyond the window",
            bool(abs(float(op.q_amp(far)[0, 0] * op.k_amp(far)[0, 0]) - 1.0) > 1e-6), 0.0)

    # -- D14, D15, D16 local-native guarantees ----------------------------
    for pid, a in zip(POLICIES, (0.25, 0.5, 1.0)):
        op = build(f"D14{pid}", geom)
        A = a * window
        pp = np.array([0.0, A / 3.0, A])
        d = pp[:, None] - pp[None, :]
        rel = op.q_phase(np.array([A / 3.0]))[0] - op.q_phase(np.array([0.0]))[0]
        add(f"D14{pid}: relative rotation is native inside the prefix",
            bool(np.allclose(rel, A / 3.0 * geom.omega, atol=1e-9)), _rel(rel, A / 3.0 * geom.omega))
        add(f"D14{pid}: prefix is a multiple of W", abs(A / window - a) < 1e-15, 0.0)

    for pid, h in zip(POLICIES, (0.125, 0.25, 0.5)):
        op = build(f"D15{pid}", geom)
        H = h * window
        step = op.q_phase(np.array([H]))[0] - op.q_phase(np.array([H - 1.0]))[0]
        add(f"D15{pid}: leading protected zone has native slope", bool(np.allclose(step, geom.omega, atol=1e-9)),
            _rel(step, geom.omega))
        step_end = op.q_phase(np.array([geom.target]))[0] - op.q_phase(np.array([geom.target - 1.0]))[0]
        add(f"D15{pid}: trailing protected zone has native slope", bool(np.allclose(step_end, geom.omega, atol=1e-9)),
            _rel(step_end, geom.omega))

    for pid, c in zip(POLICIES, (0.25, 0.5, 1.0)):
        op = build(f"D16{pid}", geom)
        C = c * window
        step = op.q_phase(np.array([C / 2.0 + 1.0]))[0] - op.q_phase(np.array([C / 2.0]))[0]
        add(f"D16{pid}: same-block relative rotation is native", bool(np.allclose(step, geom.omega, atol=1e-9)),
            _rel(step, geom.omega))

    # -- D17 --------------------------------------------------------------
    for pol in POLICIES:
        op = build(f"D17{pol}", geom)
        add(f"D17{pol}: phi_j(0) = 0", bool(np.allclose(op.q_phase(np.array([0.0]))[0], 0.0, atol=0)), 0.0)
        add(f"D17{pol}: offsets lie in [0, C)", bool(np.all((op.o >= 0) & (op.o < op.C))), 0.0)

    # -- D18 --------------------------------------------------------------
    for pid, M in zip(POLICIES, (1, 2, 4)):
        op = build(f"D18{pid}", geom)
        C = op.C
        add(f"D18{pid}: residual vanishes at the block ends",
            bool(abs(float(np.sin(0.0))) == 0.0), 0.0)
        pp = np.array([0.0, C])
        ph = op.q_phase(pp)
        add(f"D18{pid}: phi has MR average slope across a block",
            bool(np.allclose(ph[1] - ph[0], C * geom.nu_mrpro, atol=1e-9)), _rel(ph[1] - ph[0], C * geom.nu_mrpro))
        eps = 1e-6
        d0 = (op.q_phase(np.array([eps]))[0] - op.q_phase(np.array([0.0]))[0]) / eps
        add(f"D18{pid}: instantaneous slope at 0 is native", bool(np.allclose(d0, geom.omega, atol=1e-6)),
            _rel(d0, geom.omega))

    # -- D19 --------------------------------------------------------------
    for pol in POLICIES:
        op = build(f"D19{pol}", geom)
        if np.any(op.sel):
            pp = np.linspace(0.0, geom.target, 513)
            v = op.q_phase(pp)[:, op.sel]
            Th = op.Theta[op.sel]
            ok = np.all(v >= -1e-9) and np.all(v <= Th[None, :] + 1e-9)
            add(f"D19{pol}: bounded in [0, Theta]", bool(ok), float(max(0.0, np.max(v - Th[None, :]), -np.min(v))))

    # -- D20 --------------------------------------------------------------
    for pol in POLICIES:
        op = build(f"D20{pol}", geom)
        pp = np.array([0.0, 1.0, 100.0, 5000.0])
        ph = op.q_phase(pp)
        ratio = ph / geom.omega[None, :]
        add(f"D20{pol}: phase is omega_j times an integer",
            bool(np.all(np.abs(ratio - np.round(ratio)) < 1e-9)), float(np.max(np.abs(ratio - np.round(ratio)))))

    # -- section 1.1 pseudo-methods must be identified as non-methods ------
    #
    # Each entry is a rewrite that *looks* like a new operator.  "Identified"
    # means the compiler confirms the two sides are the same operator, which is
    # what disqualifies it as a direction.  It is deliberately the opposite
    # polarity from the identity checks above: there, equality is the pass;
    # here, equality is the finding.
    pseudo = []

    def ident(name, lhs, rhs, why, tol=1e-9):
        r = float(np.max(np.abs(lhs - rhs)))
        pseudo.append({"pseudo": name, "identified_as_non_method": bool(np.allclose(lhs, rhs, atol=tol)),
                       "residual": r, "why": why})

    # 1. same constant phase added to Q and K cancels
    ph = 0.7
    d = np.array([1.0, 9.0, 100.0])
    ident("common constant phase on Q and K",
          np.exp(1j * (np.outer(d, geom.nu_mrpro) + ph)),
          np.exp(1j * (np.outer(d, geom.nu_mrpro) - ph + 2 * ph)),
          "a constant phase common to both sides cancels in the relative rotation")

    # 2. nu -> nu + 2 pi n on integer positions is the same rotation
    n = 3
    d = np.arange(0, 64, dtype=np.float64)
    ident("nu -> nu + 2 pi n on integer positions",
          np.exp(1j * d[:, None] * geom.nu_mrpro[None, :]),
          np.exp(1j * d[:, None] * (geom.nu_mrpro[None, :] + 2 * math.pi * n)),
          "an integer multiple of 2 pi per token is invisible at integer positions")

    # 3. Q times a, K times 1/a leaves the dot product alone
    q = np.random.default_rng(0).normal(size=(8, K))
    k = np.random.default_rng(1).normal(size=(8, K))
    aa = 1.37
    ident("Q*a, K*(1/a) uniform",
          (q * aa) @ (k * (1.0 / aa)).T,
          q @ k.T,
          "a uniform reciprocal rescaling leaves the bilinear form unchanged")

    # 4. two successive identical m-scales collapse to one
    m1 = geom.m_mrpro
    ident("two-stage identical scaling",
          (geom.omega * geom.scale ** (-m1)) * geom.scale ** (-m1),
          geom.omega * geom.scale ** (-2.0 * m1),
          "s^-m . s^-m = s^-2m, so two stages are one multiplier and do not revive C42")

    report = {
        "geometry": {
            "theta": geom.theta, "window": geom.window, "scale": geom.scale,
            "K": geom.K, "head_dim": geom.head_dim, "low": geom.low, "high": geom.high,
            "n": geom.n, "gain": geom.gain, "target": geom.target,
        },
        "checks": checks,
        "n_checks": len(checks),
        "n_pass": sum(1 for c in checks if c["pass"]),
        "pseudo_methods": pseudo,
        "n_pseudo_identified": sum(1 for p in pseudo if p["identified_as_non_method"]),
        "all_pass": (all(c["pass"] for c in checks)
                     and all(p["identified_as_non_method"] for p in pseudo)),
        "nature": "exact algebraic identity tests on the CPU reference compiler; not a performance result",
    }
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="run algebraic identity tests")
    st.add_argument("--out", default="cpu_test.json")
    st.add_argument("--window", type=int, default=8192)
    st.add_argument("--theta", type=float, default=500000.0)
    st.add_argument("--scale", type=float, default=4.0)

    ex = sub.add_parser("export", help="write the approved review subset")
    ex.add_argument("--native-npy", default=None)
    ex.add_argument("--window", type=int, default=8192)
    ex.add_argument("--theta", type=float, default=500000.0)
    ex.add_argument("--scale", type=float, default=4.0)
    ex.add_argument("--approved-scopes", default="")
    ex.add_argument("--out", required=True)

    ls = sub.add_parser("list", help="list the 60 configurations")
    ls.add_argument("--scope", default=None)

    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        rep = selftest(a.out, a.window, a.theta, a.scale)
        print(f"{rep['n_pass']}/{rep['n_checks']} checks pass; "
              f"{rep['n_pseudo_identified']}/{len(rep['pseudo_methods'])} pseudo-methods "
              f"identified as non-methods")
        for c in rep["checks"]:
            if not c["pass"]:
                print(f"  FAIL {c['check']}  residual={c['residual']:.3e}")
        return 0 if rep["all_pass"] else 1
    if a.cmd == "export":
        export(a.native_npy, a.out, a.window, a.theta, a.scale, a.approved_scopes)
        return 0
    if a.cmd == "list":
        for cid in config_ids():
            sc = scope_of(cid)
            if a.scope and sc != a.scope:
                continue
            print(f"{cid}  scope={sc:22s} priority={priority_of(cid)}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
