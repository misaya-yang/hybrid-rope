import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "train_log_p2_phase_transfer_lora",
    Path(__file__).parents[1] / "scripts/train/train_log_p2_phase_transfer_lora.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_schedule_exposes_only_physical_1x_2x_4x() -> None:
    observed = [MODULE.family_for_step(step) for step in range(1, 11)]
    assert observed == list(MODULE.FAMILY_PATTERN) * 2
    assert observed.count("near_2x") == 4
    assert observed.count("far_4x") == 4
    assert observed.count("replay_1x") == 2
