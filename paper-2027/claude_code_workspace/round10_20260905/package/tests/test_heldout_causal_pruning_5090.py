from rebuttal.rebuttal_0723.experiments.heldout_causal_pruning_5090 import choose_masks


def test_choose_masks_separates_mean_from_ci() -> None:
    rows = [
        {"delta_vs_full_rope": -0.2, "paired_anchor_bootstrap_95ci": [-0.3, -0.1]},
        {"delta_vs_full_rope": -0.1, "paired_anchor_bootstrap_95ci": [-0.2, 0.1]},
        {"delta_vs_full_rope": 0.1, "paired_anchor_bootstrap_95ci": [0.0, 0.2]},
    ]
    assert choose_masks(rows) == {"negative_mean": [0, 1], "negative_ci": [0]}
