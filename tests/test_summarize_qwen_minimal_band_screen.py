from experiments.olmo_recovery_20260912.summarize_qwen_minimal_band_screen import (
    LENGTHS,
    TASKS,
    dominators,
    summarize_arm,
)


def arm(official, nll):
    metrics = {}
    for length, score in zip(LENGTHS, official):
        for task in TASKS:
            metrics[f"suite/{task}/{length}"] = {
                "rows": 2 if length == 32768 else 4,
                "ruler_official_score": score,
                "exact_plus_eos": 0.5,
            }
    ppl = {"by_length": {
        str(length): {"documents": 2, "mean_tail_nll": loss, "tail_ppl": 2.0}
        for length, loss in zip(LENGTHS, nll)
    }}
    return summarize_arm({"generation_metrics": metrics}, ppl)


def test_qwen_summary_and_dominance():
    arms = {
        "strong": arm((1.0, 0.75), (2.0, 2.0)),
        "weak": arm((1.0, 0.5), (2.1, 2.2)),
        "tradeoff": arm((0.75, 1.0), (1.9, 2.1)),
    }
    assert arms["strong"]["by_length"]["65536"]["task_macro_official"] == 0.75
    assert dominators("weak", arms) == ["strong"]
    assert dominators("tradeoff", arms) == []
