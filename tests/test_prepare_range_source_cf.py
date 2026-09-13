from experiments.olmo_recovery_20260912.prepare_range_source_cf import split_for_seed
from experiments.rope_fast_5090_20260912.source_counterfactual import SEEDS


def test_source_seed_split_is_eight_four_four():
    observed = [split_for_seed(seed) for seed in SEEDS]
    assert observed == [*("fit" for _ in range(8)), *("select" for _ in range(4)), *("internal_confirm" for _ in range(4))]
