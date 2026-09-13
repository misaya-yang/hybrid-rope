from experiments.olmo_recovery_20260912.summarize_llama_minimal_band_screen import dominated


def arm(s8, s16, n8, n16):
    return {
        "by_length": {"8192": {"task_macro_official": s8}, "16384": {"task_macro_official": s16}},
        "lm": {"8192": {"whole_nll": n8}, "16384": {"whole_nll": n16}},
    }


def test_descriptive_dominance_requires_all_four_metrics():
    arms = {
        "best": arm(1.0, 0.9, 2.0, 2.1),
        "worse": arm(0.9, 0.8, 2.1, 2.2),
        "tradeoff": arm(0.8, 1.0, 2.0, 2.1),
    }
    assert dominated("worse", arms, [8192, 16384]) == ["best"]
    assert dominated("tradeoff", arms, [8192, 16384]) == []
