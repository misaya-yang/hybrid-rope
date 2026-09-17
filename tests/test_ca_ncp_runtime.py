from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.ca_ncp_native_20260917.runtime import (
    install_alignment,
    load_alignment,
    transform_post_norm,
)


torch = pytest.importorskip("torch")


def alignment_file(path: Path, *, identity=False) -> Path:
    layers, groups, active = 2, 2, np.asarray([1, 2, 4], dtype=np.int64)
    a = np.ones((layers, groups), dtype=np.float64)
    b = np.zeros_like(a)
    vr = np.zeros((layers, groups, len(active)), dtype=np.float64)
    vi = np.zeros_like(vr)
    if not identity:
        a[0, 0] = 0.8
        b[0, 0] = 0.6
        vr[0, 0, 0] = 1.0
    np.savez_compressed(
        path,
        a=a, b=b, v_real=vr, v_imag=vi,
        u_real=np.zeros_like(vr), u_imag=np.zeros_like(vi),
        identity=(b == 0), active_indices=active, carrier_local=np.asarray(1),
        method_receipt_sha256=np.asarray("m"), statistics_receipt_sha256=np.asarray("s"),
    )
    return path


class Norm(torch.nn.Module):
    def forward(self, value):
        return value


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm = Norm()
        self.k_norm = Norm()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([Block(), Block()]))
        self.config = SimpleNamespace(
            num_attention_heads=4, num_key_value_heads=2, hidden_size=48, head_dim=12,
        )


def test_load_alignment_validates_shapes(tmp_path):
    path = alignment_file(tmp_path / "a.npz")
    data = load_alignment(path)
    assert data["a"].shape == (2, 2)
    broken = tmp_path / "broken.npz"
    np.savez(broken, a=np.ones((2, 2)))
    with pytest.raises(ValueError):
        load_alignment(broken)


def test_identity_transform_returns_original_object():
    x = torch.randn(1, 5, 32, dtype=torch.bfloat16)
    y = transform_post_norm(
        x, heads=4, head_dim=8, active=[1, 2, 3], carrier_local=1,
        a=[1, 1], b=[0, 0], v_real=np.zeros((2, 3)), v_imag=np.zeros((2, 3)),
        head_to_group=[0, 0, 1, 1],
    )
    assert y is x


@pytest.mark.parametrize("layout", ["flat", "bthd", "bhtd"])
def test_layout_roundtrip_and_shape(layout):
    base = torch.randn(1, 5, 4, 8)
    value = base.reshape(1, 5, 32) if layout == "flat" else base if layout == "bthd" else base.transpose(1, 2)
    y = transform_post_norm(
        value, heads=4, head_dim=8, active=[1, 2, 3], carrier_local=1,
        a=[0.8, 1], b=[0.6, 0],
        v_real=np.asarray([[1, 0, 0], [0, 0, 0]]), v_imag=np.zeros((2, 3)),
        head_to_group=[0, 0, 1, 1],
    )
    assert y.shape == value.shape
    assert y.dtype == value.dtype


def test_install_alignment_adds_buffers_and_shared_group_hooks(tmp_path):
    model = FakeModel()
    path = alignment_file(tmp_path / "a.npz")
    handles, receipt = install_alignment(model, path)
    assert len(handles) == 4
    assert receipt["nonidentity_planes"] == 1
    assert not list(model.model.layers[0].self_attn.ca_ncp_alignment.parameters())
    buffers = dict(model.model.layers[0].self_attn.ca_ncp_alignment.named_buffers())
    assert {"active", "a", "b", "v_real", "v_imag"}.issubset(buffers)
    q = torch.randn(1, 3, 48)
    k = torch.randn(1, 3, 24)
    q_out = model.model.layers[0].self_attn.q_norm(q)
    k_out = model.model.layers[0].self_attn.k_norm(k)
    assert not torch.equal(q_out, q)
    assert not torch.equal(k_out, k)
    for handle in handles:
        handle.remove()


def test_identity_install_is_bit_exact(tmp_path):
    model = FakeModel()
    path = alignment_file(tmp_path / "identity.npz", identity=True)
    handles, receipt = install_alignment(model, path)
    q = torch.randn(1, 3, 48, dtype=torch.bfloat16)
    k = torch.randn(1, 3, 24, dtype=torch.bfloat16)
    assert model.model.layers[0].self_attn.q_norm(q) is q
    assert model.model.layers[0].self_attn.k_norm(k) is k
    assert receipt["nonidentity_planes"] == 0
    for handle in handles:
        handle.remove()


def test_second_install_refused(tmp_path):
    model = FakeModel()
    path = alignment_file(tmp_path / "a.npz")
    handles, _ = install_alignment(model, path)
    with pytest.raises(RuntimeError):
        install_alignment(model, path)
    for handle in handles:
        handle.remove()
