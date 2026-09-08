"""Scalar or slotwise rotary amplitudes for explicitly declared experiments."""
import numpy as np
import torch

from scripts.experiments.cross_audit.tables import install_static, verify_static, tensor_sha


def gain_vector(table, device):
    values = np.asarray(table['gain_by_slot'], dtype=np.float32)
    if values.shape != (len(table['values_float32']),) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('invalid slotwise rotary amplitudes')
    # OLMo uses rotate_half: each rotary pair occupies coordinates j and j+K.
    return torch.from_numpy(np.concatenate((values, values))).to(device)


def install(model, table):
    values = np.asarray(table['values_float32'], dtype=np.float32)
    rotary = install_static(model, values, table['gain'])
    if 'gain_by_slot' in table:
        rotary.attention_scaling = gain_vector(table, rotary.inv_freq.device)


def verify(model, table):
    values = np.asarray(table['values_float32'], dtype=np.float32)
    if 'gain_by_slot' not in table:
        return verify_static(model, values, table['gain'])
    rotary = model.model.rotary_emb
    if rotary.inv_freq.dtype != torch.float32 or tensor_sha(rotary.inv_freq.detach().cpu().numpy()) != table['tensor_sha256']:
        raise RuntimeError('runtime frequency drift')
    expected = gain_vector(table, rotary.inv_freq.device)
    if not torch.equal(rotary.attention_scaling, expected):
        raise RuntimeError('runtime slotwise amplitude drift')
