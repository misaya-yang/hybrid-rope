"""Reproduce the recorded query-offset and realized-gap schedules without a model.
Pure functions extracted from the SHA-bound historical training implementations.
"""
import hashlib
import math
import json
from pathlib import Path
import numpy as np
LENGTH=4096

def _deterministic_band_values(
    *,
    low: int,
    high: int,
    count: int,
    seed: int,
    label: str,
) -> np.ndarray:
    if not 0 <= low <= high or count < 0:
        raise ValueError("invalid deterministic offset band")
    width = int(high - low + 1)
    digest = hashlib.sha256(
        f"evq-query-gap-v1\0{int(seed)}\0{label}".encode("ascii")
    ).digest()
    start = int.from_bytes(digest[:8], "little") % width
    stride = 1 + int.from_bytes(digest[8:16], "little") % width
    while math.gcd(stride, width) != 1:
        stride = 1 if stride == width else stride + 1
    values = (
        low
        + (
            start
            + stride * np.arange(int(count), dtype=np.int64)
        )
        % width
    )
    if len(np.unique(values)) != min(int(count), width):
        raise RuntimeError("deterministic offset band coverage drift")
    return values.astype("<i8", copy=False)

def deterministic_query_offset_stream(
    *,
    seed: int,
    routing_steps: int,
) -> np.ndarray:
    """Return four row-independent offsets per routing optimizer step."""

    routing_steps = int(routing_steps)
    if routing_steps <= 0:
        raise ValueError("routing steps must be positive")
    transition_count = routing_steps // 2
    transition = _deterministic_band_values(
        low=1,
        high=LENGTH,
        count=transition_count,
        seed=seed,
        label="transition",
    )
    middle = _deterministic_band_values(
        low=LENGTH + 1,
        high=2 * LENGTH,
        count=routing_steps,
        seed=seed,
        label="middle",
    )
    far = _deterministic_band_values(
        low=2 * LENGTH + 1,
        high=3 * LENGTH + 1,
        count=2 * routing_steps,
        seed=seed,
        label="far",
    )
    stream: list[int] = []
    transition_cursor = 0
    for routing_ordinal in range(routing_steps):
        low_offset = 0
        if routing_ordinal % 2:
            low_offset = int(transition[transition_cursor])
            transition_cursor += 1
        local = [
            low_offset,
            int(middle[routing_ordinal]),
            int(far[2 * routing_ordinal]),
            int(far[2 * routing_ordinal + 1]),
        ]
        digest = hashlib.sha256(
            (
                "evq-query-gap-order-v1\0"
                f"{int(seed)}\0{routing_ordinal}"
            ).encode("ascii")
        ).digest()
        order = sorted(range(4), key=lambda index: (digest[index], index))
        stream.extend(local[index] for index in order)
    if transition_cursor != transition_count:
        raise RuntimeError("deterministic transition cursor drift")
    values = np.asarray(stream, dtype="<i8")
    expected = {
        "contiguous": (routing_steps + 1) // 2,
        "transition": routing_steps // 2,
        "middle": routing_steps,
        "far": 2 * routing_steps,
    }
    actual = {
        "contiguous": int((values == 0).sum()),
        "transition": int(
            ((values >= 1) & (values <= LENGTH)).sum()
        ),
        "middle": int(
            (
                (values >= LENGTH + 1)
                & (values <= 2 * LENGTH)
            ).sum()
        ),
        "far": int(
            (
                (values >= 2 * LENGTH + 1)
                & (values <= 3 * LENGTH + 1)
            ).sum()
        ),
    }
    if actual != expected or len(values) != 4 * routing_steps:
        raise RuntimeError("deterministic query-offset quota drift")
    return values

def deterministic_realized_gap_target_stream(
    *,
    seed: int,
    routing_steps: int,
) -> np.ndarray:
    """Return a 1:1:2:4 stream of realized source-to-answer gap targets."""

    routing_steps = int(routing_steps)
    if routing_steps <= 0:
        raise ValueError("routing steps must be positive")
    transition_count = routing_steps // 2
    transition = _deterministic_band_values(
        low=LENGTH,
        high=2 * LENGTH - 1,
        count=transition_count,
        seed=seed,
        label="realized-transition",
    )
    middle = _deterministic_band_values(
        low=2 * LENGTH,
        high=3 * LENGTH - 1,
        count=routing_steps,
        seed=seed,
        label="realized-middle",
    )
    far = _deterministic_band_values(
        low=3 * LENGTH,
        high=4 * LENGTH - 305,
        count=2 * routing_steps,
        seed=seed,
        label="realized-far",
    )
    stream: list[int] = []
    transition_cursor = 0
    for routing_ordinal in range(routing_steps):
        low_target = -1
        if routing_ordinal % 2:
            low_target = int(transition[transition_cursor])
            transition_cursor += 1
        local = [
            low_target,
            int(middle[routing_ordinal]),
            int(far[2 * routing_ordinal]),
            int(far[2 * routing_ordinal + 1]),
        ]
        digest = hashlib.sha256(
            (
                "evq-realized-gap-order-v1\0"
                f"{int(seed)}\0{routing_ordinal}"
            ).encode("ascii")
        ).digest()
        order = sorted(range(4), key=lambda index: (digest[index], index))
        stream.extend(local[index] for index in order)
    if transition_cursor != transition_count:
        raise RuntimeError("deterministic transition cursor drift")
    values = np.asarray(stream, dtype="<i8")
    expected = {
        "contiguous": (routing_steps + 1) // 2,
        "transition": routing_steps // 2,
        "middle": routing_steps,
        "far": 2 * routing_steps,
    }
    actual = {
        "contiguous": int((values == -1).sum()),
        "transition": int(
            ((values >= LENGTH) & (values < 2 * LENGTH)).sum()
        ),
        "middle": int(
            ((values >= 2 * LENGTH) & (values < 3 * LENGTH)).sum()
        ),
        "far": int(
            ((values >= 3 * LENGTH) & (values < 4 * LENGTH)).sum()
        ),
    }
    if actual != expected or len(values) != 4 * routing_steps:
        raise RuntimeError("deterministic realized-gap quota drift")
    return values

if __name__ == '__main__':
    packet=json.loads(Path(__file__).with_name('routing_protocol_receipts.json').read_text())
    checks=[('query_gap_100',deterministic_query_offset_stream(seed=20260728,routing_steps=67),'query_offset_stream_sha256'),
            ('answer_eos_32',deterministic_realized_gap_target_stream(seed=20260728,routing_steps=22),'gap_target_stream_sha256')]
    for name,stream,key in checks:
        actual=hashlib.sha256(stream.astype('<i8').tobytes()).hexdigest()
        expected=packet['runs'][name]['training'][key]
        assert actual==expected,(name,actual,expected)
    print('Two historical position schedules reproduced exactly against run-receipt SHA256 values.')
