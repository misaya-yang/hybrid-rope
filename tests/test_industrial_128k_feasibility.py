"""Focused tests for scripts/analysis/industrial_128k_feasibility.py.

Locks the two things that make the plan trustworthy: (1) the EVQ-Cosh coverage
diagnostics reproduce the registered 8B/8K anchors, so the tau study uses the
project's real frequency definition; (2) the VRAM model and tau recommender have
the qualitative behavior the plan relies on.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_MOD = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "industrial_128k_feasibility.py"
spec = importlib.util.spec_from_file_location("industrial_128k_feasibility", _MOD)
fz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fz)

HD = 128
FINE_GRID = [round(0.5 + 0.1 * i, 3) for i in range(20)]  # 0.5 .. 2.4


def test_dormant_matches_registered_8b_anchors():
    geo = fz.geo_inv_freq(HD, 500000.0)
    mid = fz.evq_cosh_inv_freq(HD, 500000.0, 1e-6)      # tau->0 == midpoint geo
    evq = fz.evq_cosh_inv_freq(HD, 500000.0, 1.414)
    assert fz.dormant_count(geo, 8192) == 20
    assert fz.dormant_count(mid, 8192) == 20
    assert fz.dormant_count(evq, 8192) == 15


def test_entropy_rank_matches_registered_8b_anchors():
    geo = fz.geo_inv_freq(HD, 500000.0)
    evq = fz.evq_cosh_inv_freq(HD, 500000.0, 1.414)
    assert fz.entropy_effective_rank(geo, 8192) == pytest.approx(23.60, abs=0.1)
    assert fz.entropy_effective_rank(evq, 8192) == pytest.approx(36.22, abs=0.2)


def test_evq_tau_zero_limit_is_midpoint_geometric():
    lo = fz.evq_cosh_inv_freq(HD, 1e6, 1e-6)
    K = HD // 2
    u = (2 * np.arange(1, K + 1) - 1) / (2 * K)
    assert np.allclose(lo, (1e6) ** (-u), atol=1e-6)


def test_raising_tau_reduces_dormant_and_lifts_erank():
    L = 16384
    dprev, eprev = 999, -1.0
    for tau in [0.6, 1.0, 1.4, 1.8]:
        f = fz.evq_cosh_inv_freq(HD, 1e6, tau)
        d, e = fz.dormant_count(f, L), fz.entropy_effective_rank(f, L)
        assert d <= dprev and e >= eprev
        dprev, eprev = d, e


def test_recommend_tau_reproduces_anchor_and_decreases_with_length():
    short, _ = fz.recommend_tau(HD, 500000.0, 8192, 32768, FINE_GRID)
    long, _ = fz.recommend_tau(HD, 500000.0, 16384, 65536, FINE_GRID)
    assert short["dormant"] == 15 and 1.3 <= short["tau"] <= 1.5   # matches 8B anchor
    assert long["dormant"] == 15
    assert long["tau"] < short["tau"]                              # tau falls as L grows


def test_yarn_keeps_high_freq_and_interpolates_low_freq():
    geo = fz.geo_inv_freq(HD, 1e6)
    yarn, attn = fz.yarn_inv_freq(HD, 1e6, factor=4.0, orig_ctx=32768)
    assert attn == pytest.approx(0.1 * math.log(4.0) + 1.0, abs=1e-9)
    assert yarn[0] == pytest.approx(geo[0], rel=1e-6)              # highest freq kept
    assert yarn[-1] == pytest.approx(geo[-1] / 4.0, rel=1e-6)     # lowest freq interpolated


def test_vram_train_single_card_verdict():
    llama = fz.MODELS["Llama-3.1-8B"]
    qwen14 = fz.MODELS["Qwen2.5-14B"]
    assert fz.vram_train_gb(llama, 131072, 1, 64) < 96            # 8B trains at 128K
    assert fz.vram_train_gb(qwen14, 65536, 1, 64) < 96           # 14B trains at 64K
    assert fz.vram_train_gb(qwen14, 131072, 1, 64) > 96          # 14B OOMs at 128K
    assert fz.vram_eval_gen_gb(qwen14, 131072) < 96             # 14B evals 128K


def test_vram_monotone_in_length():
    m = fz.MODELS["Qwen2.5-14B"]
    vals = [fz.vram_train_gb(m, L, 1, 64) for L in (8192, 16384, 32768, 65536)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
