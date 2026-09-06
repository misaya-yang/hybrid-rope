import hashlib
import json

from scripts.lib.checkpoint_identity import safetensors_weight_set_sha256


def test_generic_weight_identity_preserves_single_file_hash(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"single")
    expected = hashlib.sha256(b"single").hexdigest()
    assert safetensors_weight_set_sha256(tmp_path) == expected


def test_generic_weight_identity_binds_all_shards(tmp_path):
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    identity = [
        {"name": first.name, "bytes": 5, "sha256": hashlib.sha256(b"first").hexdigest()},
        {"name": second.name, "bytes": 6, "sha256": hashlib.sha256(b"second").hexdigest()},
    ]
    expected = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert safetensors_weight_set_sha256(tmp_path) == expected
