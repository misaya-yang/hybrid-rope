from types import SimpleNamespace

import numpy as np
import torch

from experiments.fixed_rope_three_interfaces_20260913.tables import default_band, model_geometry
from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config
from scripts.experiments.cross_audit.tables import install_static, native_table


def glm_config_dict():
    return {
        "model_type": "glm4", "hidden_size": 4096, "num_attention_heads": 32,
        "head_dim": 128, "partial_rotary_factor": 0.5,
        "max_position_embeddings": 32768,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    }


def test_glm_geometry_uses_only_partial_rotary_dimension():
    geometry = model_geometry(glm_config_dict())
    assert geometry["attention_head_dim"] == 128
    assert geometry["head_dim"] == 64 and geometry["pairs"] == 32
    assert default_band(geometry) == (17, 30)


def test_recovery_native_table_has_32_glm_pairs():
    cfg = SimpleNamespace(**glm_config_dict(), num_hidden_layers=40, num_key_value_heads=2)
    table = table_for_config(cfg, "Native")
    assert len(table["values_float32"]) == 32


def test_static_installer_validates_against_runtime_rotary_shape():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(()))
            rotary = SimpleNamespace(
                inv_freq=torch.from_numpy(native_table(64, 10000).copy()),
                rope_type="default", attention_scaling=1.0,
            )
            self.model = SimpleNamespace(rotary_emb=rotary)
            self.config = SimpleNamespace(hidden_size=4096, num_attention_heads=32)

    model = Model(); values = native_table(64, 10000).astype(np.float32) / 2
    install_static(model, values, 1.2)
    assert model.model.rotary_emb.inv_freq.shape == (32,)
    assert model.model.rotary_emb.attention_scaling == 1.2
