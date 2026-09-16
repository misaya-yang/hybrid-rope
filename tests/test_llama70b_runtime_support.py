import json
from types import SimpleNamespace

from experiments.olmo_recovery_20260912.recovery_v2_eval import checkpoint_precision_identity
from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config


def test_matching_llama70b_geometry_is_supported():
    config = SimpleNamespace(
        model_type="llama", hidden_size=8192, num_attention_heads=64,
        num_key_value_heads=8, num_hidden_layers=80,
        max_position_embeddings=8192, rope_theta=500000.0,
    )
    assert len(table_for_config(config, "Native")["values_float32"]) == 64
    assert len(table_for_config(config, "MrPro_g4")["values_float32"]) == 64


def test_nf4_precision_identity_is_explicit(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True,
    }}))
    value = checkpoint_precision_identity(tmp_path)
    assert value["model_dtype"] == "bitsandbytes_4bit_nf4_compute_bfloat16"
    assert value["checkpoint_quantization"]["bnb_4bit_use_double_quant"] is True
