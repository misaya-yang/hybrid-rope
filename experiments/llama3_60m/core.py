"""Geometry, conventions and the no-op self-checks.

Implements the reference-package part the plan names in section 7.3:

    "参考包只实现无C频率构造与关键数学原语/自检"

Three things live here, in order of how badly they can go wrong:

1. **Geometry, read from the checkpoint and asserted** (section 6.1).  theta,
   head_dim, K, the training window and the absence of any extra rope_scaling
   are all checked against the actual config rather than assumed.

2. **Conventions, stated once and used everywhere.**  Section 2.1:
   `d = key_position - query_position`, so causal attention sees d <= 0.  The
   plan is emphatic that signed/carrier/phase constructions must not mix a
   positive-r convention into some places and negative-d into others.

3. **The seven pseudo-methods of section 2.4**, each as an executable check
   that the two sides really are the same operator.  A rewrite that collapses
   is not a method; the plan counts these as CPU controls only.  They are
   checked rather than asserted because "mathematically equivalent" is exactly
   the kind of claim that is wrong in the sign or in the pairing.

The complex-coefficient convention is section 5.3:

    c_j = conj(q_complex) * k_complex           (per slot j)

with the pair treated as a complex number.  Section 5.3 also forbids mixing
E|c|^2 with |Ec|^2, which `mean_energy` and `coherent_energy` keep separate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

S = 4.0                      # the deployment scale fixed by section 6.1
LN_S = math.log(S)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Frozen geometry of the main model.  Built from, not assumed about, config."""

    theta: float
    window: int
    head_dim: int
    K: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    scale: float = S
    low: int = 18
    high: int = 35
    n: int = 17
    native_inv_freq: np.ndarray | None = None

    @property
    def target(self) -> int:
        return int(self.window * self.scale)

    @property
    def omega(self) -> np.ndarray:
        if self.native_inv_freq is not None:
            return np.asarray(self.native_inv_freq, dtype=np.float64)
        j = np.arange(self.K)
        return self.theta ** (-j / self.K)

    @property
    def q(self) -> np.ndarray:
        """q_j = clip(j - 18, 0, 17); T = {19..34} are the partly-scaled slots."""
        return np.clip(np.arange(self.K) - self.low, 0, self.n).astype(np.float64)

    @property
    def T(self) -> np.ndarray:
        """The 16 slots that receive partial scaling (plan section 2.2)."""
        return np.arange(self.low + 1, self.high)

    @property
    def m_mr(self) -> np.ndarray:
        q = self.q
        return q * (q + 1.0) / (self.n * (self.n + 1.0))

    @property
    def nu_mr(self) -> np.ndarray:
        return self.omega * self.scale ** (-self.m_mr)

    # Aliases under the names the verified CONFIGS_REVIEW operator package uses
    # (`nu_mrpro` / `m_mrpro`).  Keeping ONE operator library is worth two
    # property names: a second implementation is a second source of truth, and
    # the first one is the version that was checked bit-identical against the
    # campaign's own tables.
    @property
    def m_mrpro(self) -> np.ndarray:
        return self.m_mr

    @property
    def nu_mrpro(self) -> np.ndarray:
        return self.nu_mr

    @property
    def gain(self) -> float:
        return gain_for(self.scale)

    # -- Plan B section 2.1/2.2: amplitude and the two coordinates ----------
    #
    #     nu_j = omega_j * s^(-m_j),      m_j = a * r_j(B)
    #
    # `a` is an INDEPENDENT first-class parameter -- the extra compression
    # amplitude in the fixed-s coordinate, with a = 1 for the standard three-band
    # table.  The historical log4 coordinate is a different quantity:
    #
    #     nu_j = omega_j * 4^(-m_tilde_j),     m_tilde_j = (log_4 s) * m_j
    #
    # Conflating them is what makes "raising the log4 amplitude from 1 to 1.5 at
    # 8x" look like a new method when it is the same profile deployed at 8x
    # instead of 4x (section 2.2).

    def r_of(self, profile="mr") -> np.ndarray:
        """The normalised profile r_j, high-frequency end 0 and plateau 1."""
        return PROFILES[profile](self)

    def m_of(self, a=1.0, profile="mr") -> np.ndarray:
        return a * self.r_of(profile)

    def nu_of(self, a=1.0, profile="mr") -> np.ndarray:
        return self.omega * self.scale ** (-self.m_of(a, profile))

    def m_tilde(self, a=1.0, profile="mr") -> np.ndarray:
        """The historical log4 coordinate m_tilde = (log_4 s) * m."""
        return (math.log(self.scale) / math.log(4.0)) * self.m_of(a, profile)

    def nu_from_m_tilde(self, m_tilde):
        return self.omega * 4.0 ** (-np.asarray(m_tilde, dtype=np.float64))

    # -- companion quantities the coverage theory uses ----------------------

    @property
    def turns(self) -> np.ndarray:
        """t_j(W) = W omega_j / 2 pi: turns the slot accumulates over the window."""
        return self.window * self.omega / (2.0 * math.pi)

    def kappa(self, delta=0.25):
        """kappa_j = log_s(t_j(W)/delta): the slot's signal margin."""
        return np.log(self.turns / delta) / LN_S

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, model_dir, scale=S, low=None, high=None, n=None):
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        return cls.from_mapping(cfg, scale=scale, low=low, high=high, n=n,
                                source=str(model_dir))

    @classmethod
    def from_mapping(cls, cfg, scale=S, low=None, high=None, n=None, source="<dict>"):
        theta = cfg.get("rope_theta", 10000.0)
        hd = cfg["hidden_size"] // cfg["num_attention_heads"]
        K = hd // 2
        window = cfg["max_position_embeddings"]
        # section 6.1: the stock checkpoint must carry no extra rope_scaling
        if cfg.get("rope_scaling") not in (None, {}, {"type": None}):
            raise ValueError(f"{source}: checkpoint carries rope_scaling "
                             f"{cfg.get('rope_scaling')!r}; section 6.1 requires the stock 8K model")
        lo, hi = find_correction_range(32.0, 1.0, hd, theta, window)
        return cls(theta=float(theta), window=int(window), head_dim=int(hd), K=int(K),
                   n_layers=int(cfg["num_hidden_layers"]),
                   n_heads=int(cfg["num_attention_heads"]),
                   n_kv_heads=int(cfg["num_key_value_heads"]),
                   scale=float(scale),
                   low=int(lo if low is None else low),
                   high=int(hi if high is None else high),
                   n=int((hi - lo) if n is None else n))

    def assert_expected(self):
        """Section 6.1's expected identity: 32 layers, 32/8 heads, theta 5e5, W 8192."""
        want = dict(theta=500000.0, window=8192, head_dim=128, K=64,
                    n_layers=32, n_heads=32, n_kv_heads=8)
        bad = {k: (getattr(self, k), v) for k, v in want.items() if getattr(self, k) != v}
        if bad:
            raise ValueError(f"checkpoint identity mismatch (got, want): {bad}")
        return True

    def m_to_nu(self, m):
        return self.omega * self.scale ** (-np.asarray(m, dtype=np.float64))

    def nu_to_m(self, nu):
        """m = log_s(omega/nu).  Negative m means the slot was sped up."""
        return -np.log(np.asarray(nu, dtype=np.float64) / self.omega) / LN_S


def gain_for(scale):
    """Plan B section 2.3: g_s = 1 + 0.1 ln s, and g = 1 for s <= 1.

    The official author code multiplies this factor into cos/sin, so it is
    applied once per side and the score multiplier is g^2.  Section 2.3 warns
    explicitly: do not stack a second g^2 on top.
    """
    return 1.0 + 0.1 * math.log(scale) if scale > 1.0 else 1.0


# ---------------------------------------------------------------------------
# Plan B section 2.3: the four named profiles, in the r-coordinate
# ---------------------------------------------------------------------------


def r_mr(g):
    """MrRoPE-Pro: r_j = q(q+1)/(n(n+1))."""
    q = g.q
    return q * (q + 1.0) / (float(g.n) * (g.n + 1.0))


def r_uni(g):
    """MrRoPE-Uni: r_j = q/n."""
    return g.q / float(g.n)


def r_bm(g):
    """BM: eps_i = i(n+1-i) / sum_k k(n+1-k), and r is its running sum."""
    n = g.n
    i = np.arange(1, n + 1, dtype=np.float64)
    eps = i * (n + 1 - i)
    eps = eps / eps.sum()
    r = np.zeros(g.K)
    r[g.low + 1:g.low + 1 + n] = np.concatenate([[0.0], np.cumsum(eps)])[1:]
    r[g.low + n + 1:] = 1.0
    return r


def r_yarn(g, scale=None):
    """Official index YaRN, as a profile IN THE GIVEN s.

    Section 2.3 is explicit that "官方 YaRN 的 profile 会随 s 变化" -- the s=4
    profile must not be reused at s=8 and still be called official YaRN.  This
    function therefore takes the scale rather than reading it off the geometry.
    """
    s = g.scale if scale is None else float(scale)
    u = g.q / float(g.n)
    return -np.log((1.0 - u) + u / s) / math.log(s)


def r_native(g):
    return np.zeros(g.K)


PROFILES = {"mr": r_mr, "uni": r_uni, "bm": r_bm, "yarn": r_yarn, "native": r_native}


def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base))


def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    lo = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
    hi = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(lo, 0), min(hi, dim - 1)


# ---------------------------------------------------------------------------
# exact operator, and the pair layout
# ---------------------------------------------------------------------------


def rotate_half(x):
    """HF's layout: pairs are (i, i + K), NOT adjacent (2j, 2j+1)."""
    K = x.shape[-1] // 2
    return np.concatenate([-x[..., K:], x[..., :K]], axis=-1)


def apply_rope(x, cos, sin):
    """The stock HF rotation, on the half-split layout."""
    return x * cos + rotate_half(x) * sin


def to_complex(x, layout="half"):
    """View a 2K-vector as K complex pairs.  Section 5.2: half-split for Llama."""
    x = np.asarray(x, dtype=np.float64)
    K = x.shape[-1] // 2
    if layout == "half":
        return x[..., :K] + 1j * x[..., K:]
    return x[..., 0::2] + 1j * x[..., 1::2]


def from_complex(z, layout="half"):
    z = np.asarray(z)
    if layout == "half":
        return np.concatenate([z.real, z.imag], axis=-1)
    out = np.empty(z.shape[:-1] + (2 * z.shape[-1],))
    out[..., 0::2] = z.real
    out[..., 1::2] = z.imag
    return out


def relative_kernel(nu, d):
    """K(d) = sum_j exp(i nu_j d).  d = key_position - query_position (section 2.1)."""
    return np.exp(1j * np.outer(np.asarray(d, dtype=np.float64), np.asarray(nu, dtype=np.float64)))


def coefficient(q, k, layout="half"):
    """c_j = conj(q_j) * k_j per slot (section 5.3)."""
    return np.conj(to_complex(q, layout)) * to_complex(k, layout)


def mean_energy(coefficients):
    """E|c|^2 -- an ENERGY, averaged before squaring (section 5.3)."""
    return np.mean(np.abs(coefficients) ** 2, axis=0)


def coherent_energy(coefficients):
    """|E c|^2 -- the coherent part.  Section 5.3 forbids conflating the two."""
    return np.abs(np.mean(coefficients, axis=0)) ** 2


def complex_mean(coefficients):
    return np.mean(coefficients, axis=0)


# ---------------------------------------------------------------------------
# section 2.4: the pseudo-methods, as executable checks
# ---------------------------------------------------------------------------

PSEUDO = (
    "same position-independent 2D phase added to both Q and K",
    "simultaneous permutation of frequencies AND the Q/K channels",
    "common orthogonal rotation inside an existing 2D block",
    "nu -> nu + 2 pi k on integer positions",
    "one key-independent logit constant added to every key",
    "a few ulp of difference from a different rounding path",
    "splitting one m into two offline multiplications",
)


def check_pseudo_methods(geom, rng=None, d=None):
    """Confirm each section-2.4 rewrite collapses to the same operator.

    Returns a list of {pseudo, collapses, residual} entries.  `collapses=True`
    is the finding: it is what disqualifies the rewrite as a method.
    """
    rng = rng or np.random.default_rng(20260911)
    K = geom.K
    nu = geom.nu_mr
    d = np.array([-4096.0, -1024.0, -37.0, -1.0, 0.0]) if d is None else np.asarray(d, float)
    out = []

    def add(name, a, b, tol=1e-12):
        r = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        out.append({"pseudo": name, "collapses": bool(np.allclose(a, b, atol=tol, rtol=tol)),
                    "residual": r})

    # 1. a common position-independent phase added to both sides cancels.
    #    The check must actually ADD the phase to both sides -- comparing an
    #    expression with itself is vacuously true and proves nothing.
    q = rng.normal(size=2 * K)
    k = rng.normal(size=2 * K)
    phi = 0.713
    rot_phase = np.exp(1j * phi)
    add(PSEUDO[0],
        coefficient(q, k),
        coefficient(from_complex(to_complex(q) * rot_phase),
                    from_complex(to_complex(k) * rot_phase)))

    # 2. permuting frequencies AND the Q/K channels together is a renaming:
    #    sum_j c_perm(j) e^{i nu_perm(j) d} = sum_j c_j e^{i nu_j d}.
    perm = rng.permutation(K)
    add(PSEUDO[1],
        (coefficient(q, k)[None, :] * relative_kernel(nu, d)).sum(axis=1),
        (coefficient(q, k)[perm][None, :] * relative_kernel(nu[perm], d)).sum(axis=1))

    # 3. a common rotation inside an existing 2D block commutes with RoPE
    th = 0.7
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    qa = from_complex(to_complex(q) * (R[0, 0] + 1j * R[1, 0]))
    ka = from_complex(to_complex(k) * (R[0, 0] + 1j * R[1, 0]))
    add(PSEUDO[2], coefficient(q, k), coefficient(qa, ka))

    # 4. nu -> nu + 2 pi k on integer positions
    n = np.arange(0, 64, dtype=np.float64)
    add(PSEUDO[3], relative_kernel(nu, n), relative_kernel(nu + 2 * math.pi * 3, n), tol=1e-9)

    # 5. one key-independent logit constant: softmax invariant
    z = rng.normal(size=64) * 3.0
    add(PSEUDO[4], np.exp(z - z.max()) / np.exp(z - z.max()).sum(),
        np.exp(z + 5.0 - (z + 5.0).max()) / np.exp(z + 5.0 - (z + 5.0).max()).sum())

    # 6. ulp-level differences are an implementation control, not a new method
    add(PSEUDO[5], nu.astype(np.float32).astype(np.float64), nu, tol=1e-5)

    # 7. splitting one m into two offline multiplications is still one table
    m = geom.m_mr
    add(PSEUDO[6], geom.m_to_nu(m), geom.omega * S ** (-m / 2) * S ** (-m / 2))

    return out
