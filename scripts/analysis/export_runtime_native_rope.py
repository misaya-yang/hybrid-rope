#!/usr/bin/env python3
"""Export the installed HF Native initializer on CPU, without model weights."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.export_frozen_coupling_transport import config_identity, sha256_file, sha256_bytes
from scripts.analysis.export_static_rope_baselines import resolve_native_initializer, tensor_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import transformers
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    identity = config_identity(args.config)
    raw = json.loads(args.config.read_text())
    model_type = raw.pop("model_type")
    config = AutoConfig.for_model(model_type, **raw)
    initializer = resolve_native_initializer(ROPE_INIT_FUNCTIONS, model_type)
    inv, gain = initializer(config, torch.device("cpu"))
    table = inv.detach().cpu().numpy()
    if (table.dtype != np.float32 or table.shape != (identity["pairs"],)
            or float(gain) != 1 or not np.isfinite(table).all()
            or not (table > 0).all() or not (np.diff(table) < 0).all()
            or tensor_hash(table) != args.expected_native_sha256):
        raise ValueError("Native initializer differs from the existing runtime receipt")
    receipt_path = args.output.with_suffix(".json")
    if args.output.exists() or receipt_path.exists():
        raise ValueError("refusing to overwrite an existing Native artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, table, allow_pickle=False)
    receipt = {"status": "CPU_NATIVE_INITIALIZER_EXPORTED", "checkpoint": identity,
        "native_sha256_float32": tensor_hash(table), "file_sha256": sha256_file(args.output),
        "initializer": f"{initializer.__module__}.{initializer.__qualname__}",
        "initializer_source_sha256": sha256_bytes(inspect.getsource(initializer).encode()),
        "exporter_sha256": sha256_file(Path(__file__)),
        "transformers": transformers.__version__, "torch": torch.__version__,
        "device": "cpu", "weights_loaded": False, "gain": float(gain)}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
