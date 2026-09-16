from experiments.olmo_recovery_20260912.recovery_v2_eval import normalize_existing_identity


def test_legacy_batch1_contract_infers_panel_order():
    legacy = {"batch_size": 1, "row_ids": ["a", "b"]}
    assert normalize_existing_identity(legacy) == {
        "batch_size": 1, "row_ids": ["a", "b"], "generation_order": "panel_order_v1",
    }
    assert "generation_order" not in legacy


def test_batched_contract_never_infers_an_unknown_order():
    batched = {"batch_size": 2, "row_ids": ["a", "b"]}
    assert normalize_existing_identity(batched) == batched
