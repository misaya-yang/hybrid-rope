"""CPU integration checks for unchanged author scorers and policy boundaries."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import os
import sys

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding, apply_rotary_pos_emb

from experiments.pm_keep import baselines

SOURCE = Path(os.environ.get('PM_KEEP_KVPRESS_ROOT', '/tmp/hybrid-kvpress-20260909')).resolve()
pytestmark = pytest.mark.skipif(not SOURCE.is_dir(), reason='Pinned author checkout is not present')


@pytest.fixture
def prefix():
    torch.manual_seed(104)
    config = Qwen2Config(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1, max_position_embeddings=4096)
    module = Qwen2Attention(config, layer_idx=0).eval()
    module.rotary_emb = Qwen2RotaryEmbedding(config=config)
    hidden = torch.randn(1, 13, 16)
    q = module.q_proj(hidden).view(1, 13, 4, 4).transpose(1, 2)
    k = module.k_proj(hidden).view(1, 13, 2, 4).transpose(1, 2)
    v = module.v_proj(hidden).view(1, 13, 2, 4).transpose(1, 2)
    cos, sin = module.rotary_emb(hidden, torch.arange(13).unsqueeze(0))
    _, k = apply_rotary_pos_emb(q, k, cos, sin)
    return SimpleNamespace(attention_module=module, hidden_states=hidden,
                           keys=k.detach(), values=v.detach(), prefix_length=13)


def test_exact_source_and_default_contract(prefix):
    receipt = baselines.source_receipt(SOURCE)
    assert receipt['pinned_commit'] == '71640b4f9061054a7630c5049bb9ee659a01523c'
    assert receipt['ea_actual_defaults']['n_future_positions'] == 512
    assert receipt['ea_actual_defaults']['use_vnorm'] is True
    assert receipt['ea_actual_defaults']['epsilon'] == 0.0
    cls = baselines.load_author_class('ExpectedAttentionPress', SOURCE)
    assert Path(sys.modules[cls.__module__].__file__).resolve() == SOURCE / 'kvpress/presses/expected_attention_press.py'
    wrapped = baselines.ea_prefix_scores(prefix, source_root=SOURCE)
    direct = cls().score(prefix.attention_module, prefix.hidden_states, prefix.keys, prefix.values, None, {})[0]
    torch.testing.assert_close(wrapped, direct, rtol=0, atol=0)
    assert wrapped.shape == (2, 13)
    assert torch.all(wrapped[:, :4] > wrapped[:, 4:].max())
    assert 'kvpress.attention_patch' not in sys.modules
    assert 'kvpress.pipeline' not in sys.modules
    assert 'kvpress.presses.fastkvzip_press' not in sys.modules


def test_author_value_norm_is_retained(prefix):
    ea = baselines.ea_prefix_scores(prefix, source_root=SOURCE)
    modified = SimpleNamespace(**vars(prefix))
    modified.values = prefix.values.clone()
    modified.values[:, :, 7] *= 3
    rescaled = baselines.ea_prefix_scores(modified, source_root=SOURCE)
    torch.testing.assert_close(rescaled[:, 7], ea[:, 7] * 3)
    torch.testing.assert_close(rescaled[:, 4:7], ea[:, 4:7])
    torch.testing.assert_close(rescaled[:, 8:], ea[:, 8:])


def test_full_prefix_hidden_is_required(prefix):
    partial = SimpleNamespace(**vars(prefix))
    partial.hidden_states = prefix.hidden_states[:, -5:]
    with pytest.raises(ValueError, match='complete prefix'):
        baselines.ea_prefix_scores(partial, source_root=SOURCE)
    leaked = SimpleNamespace(**vars(prefix))
    leaked.hidden_states = torch.cat((prefix.hidden_states, prefix.hidden_states[:, :1]), dim=1)
    with pytest.raises(ValueError, match='complete prefix'):
        baselines.ea_prefix_scores(leaked, source_root=SOURCE)


def test_ea_uses_real_future_rotary_module(prefix):
    calls = []
    rotary = prefix.attention_module.rotary_emb
    hook = rotary.register_forward_pre_hook(lambda module, args: calls.append(args[1].detach().clone()))
    try:
        baselines.ea_prefix_scores(prefix, source_root=SOURCE)
    finally:
        hook.remove()
    assert len(calls) == 1
    assert calls[0].tolist() == [list(range(13, 13 + 512))]


def test_author_allocation_does_not_force_recent_tokens():
    scores = torch.tensor([[10., 10., 10., 10., 9., 8., 7., 0., 0.]])
    selected = baselines.ea_official_keep_indices(scores, 6)
    assert selected.tolist() == [[0, 1, 2, 3, 4, 5]]
    assert 8 not in selected
    assert baselines.ea_official_keep_indices(scores, 9).tolist() == [list(range(9))]
    with pytest.raises(ValueError):
        baselines.ea_official_keep_indices(scores, 10)


def test_keydiff_calls_pinned_source_and_is_named_as_adaptation(prefix):
    cls = baselines.load_author_class('KeyDiffPress', SOURCE)
    observed = baselines.keydiff_prefix_scores(prefix, source_root=SOURCE)
    direct = cls().score(prefix.attention_module, prefix.hidden_states, prefix.keys, prefix.values, None, {})[0]
    torch.testing.assert_close(observed, direct, rtol=0, atol=0)
    status = baselines.strong_baseline_status()
    assert not status['KeyDiff_score']['paper_reproduction']
    assert status['KVzip']['status'] == 'deferred_not_runtime_validated'
    assert status['TriAttention']['status'] == 'unsupported'


def test_modified_author_source_is_rejected(tmp_path):
    for relative in baselines.PINNED_SHA256:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / relative, target)
    victim = tmp_path / 'kvpress/presses/expected_attention_press.py'
    victim.write_text(victim.read_text().replace('epsilon: float = 0.0', 'epsilon: float = 1e-6'))
    with pytest.raises(baselines.UnsupportedBaseline, match='SHA256 mismatch'):
        baselines.source_receipt(tmp_path)
