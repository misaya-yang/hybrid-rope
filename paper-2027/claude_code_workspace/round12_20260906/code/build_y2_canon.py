#!/usr/bin/env python3
"""Build Y2: CANONICAL unfine-tuned YaRN s=4 for OLMo (L0=4096, base=500000).

Two fidelity fixes relative to the frozen round-12 Y arm (which stays frozen):
 1. Piecewise interpolation uses the paper's rotation rule with a SMOOTHSTEP
    ramp in frequency space (HF convention): dims with wavelength < L0/beta_fast
    are untouched, dims with wavelength > L0/beta_slow are fully interpolated,
    middle dims blend via smoothstep. The frozen Y arm is a hard cutoff
    (dims 0-14 kept, 15-63 all /4, zero ramp) — faithful only to the
    rotation-count boundaries, not to the blend.
 2. Attention temperature: the paper multiplies ATTENTION LOGITS by
    t = 0.1*ln(s)+1 (equivalently scales the RoPE embeddings by sqrt(t)).
    The engine applies rotary_amplitude to cos/sin, so the faithful amplitude
    is sqrt(t); the frozen Y arm installed t itself, giving logits x t^2
    (over-sharpened by a factor t).

Outputs tables_canon/{Y2.npy, manifest_round12.json}; the manifest reuses the
frozen N/Z/Y/M entries byte-for-byte so existing shas still verify.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

B12 = Path("/root/autodl-tmp/claude_round12_20260906")
FROZEN = B12 / "tables" / "manifest_round12.json"
OUT = B12 / "tables_canon"

K = 64
DIM = 128
BASE = 500000.0
L0 = 4096.0
S = 4.0
BETA_FAST = 32.0   # rotations-at-training-length threshold (paper alpha/beta)
BETA_SLOW = 1.0


def main():
    OUT.mkdir(exist_ok=True)
    omega = BASE ** (-np.arange(0, DIM, 2, dtype=np.float64) / DIM)
    wavelen = 2.0 * np.pi / omega
    high_wl = L0 / BETA_FAST          # 128: shorter wavelength => untouched
    low_wl = L0 / BETA_SLOW           # 4096: longer wavelength => full /S
    u = np.clip((wavelen - high_wl) / (low_wl - high_wl), 0.0, 1.0)
    gamma = u * u * (3.0 - 2.0 * u)   # smoothstep ramp, HF convention
    omega_y2 = omega * (1.0 - gamma) + (omega / S) * gamma

    t = 0.1 * math.log(S) + 1.0
    gain = math.sqrt(t)               # cos/sin multiplier => logits x t

    y2 = omega_y2.astype(np.float32)
    assert y2.dtype == np.float32 and y2.shape == (K,)
    assert np.all(y2 > 0) and np.all(y2[:-1] > y2[1:]), "must be positive ordered"
    np.save(OUT / "Y2.npy", y2)  # real .npy: track_a_eval loads via np.load

    man = json.loads(FROZEN.read_text())
    assert man["status"] == "ROUND12_STATIC_TABLES_FROZEN_V1"
    # Reuse frozen N/Z/Y/M entries verbatim; copy their files for completeness.
    for arm, entry in man["arms"].items():
        src = B12 / "tables" / entry["path"]
        dst = OUT / entry["path"]
        if src.exists():
            dst.write_bytes(src.read_bytes())
        elif arm == "N":
            # Native table is computed on the fly; materialize from the frozen
            # geometry (base 500000, head_dim 128) and verify against manifest.
            np.save(dst, omega.astype(np.float32))
        else:
            raise SystemExit(f"frozen table file missing for arm {arm}")
        blob = np.load(dst, allow_pickle=False)
        assert blob.dtype == np.float32
        assert hashlib.sha256(np.ascontiguousarray(blob).tobytes()).hexdigest() \
            == entry["float32_sha256"], f"{arm} copy sha mismatch"
    man["arms"]["Y2"] = {
        "path": "Y2.npy",
        "rotary_amplitude": gain,
        "float32_sha256": hashlib.sha256(np.ascontiguousarray(y2).tobytes()).hexdigest(),
        "source": ("canonical YaRN s=4 (Peng et al. 2309.00071): NTK-by-parts with "
                   "smoothstep ramp, beta_fast=32/beta_slow=1 rotations at L0=4096; "
                   "amplitude sqrt(t), t=0.1*ln(4)+1, so attention logits scale by t"),
        "temperature": {"t": t, "sqrt_t_gain": gain, "effective_logit_multiplier": t},
        "piecewise": {"untouched_dims": int((gamma == 0).sum()),
                      "full_interp_dims": int((gamma == 1).sum()),
                      "ramp_dims": int(((gamma > 0) & (gamma < 1)).sum())},
    }
    man["status"] = "ROUND12_STATIC_TABLES_FROZEN_V1"
    man["canon_addendum"] = ("Y2 added 2026-09-06 as the faithful unfine-tuned YaRN "
                             "control; frozen Y/M entries unchanged")
    (OUT / "manifest_round12.json").write_text(json.dumps(man, indent=1))
    print("Y2 gain=%.12f t=%.12f logits_mult=%.12f" % (gain, t, gain * gain))
    print("untouched/full/ramp dims:", man["arms"]["Y2"]["piecewise"])
    print("wavelength top/bottom:", 2 * np.pi / y2[0], 2 * np.pi / y2[-1])
    print("Y2_MANIFEST_WRITTEN", OUT / "manifest_round12.json")


if __name__ == "__main__":
    main()
