"""Measure production-shape updates without validation or checkpoint selection."""
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .model import Model
from .run import setup, autocast, load_data, acquire_gpu


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--config')
    a = p.parse_args(); lock = acquire_gpu(a.data); setup(901, 'cuda')
    x, y, _ = load_data(a.data, 'train', 'cuda')
    from .model import Config
    cfg = Config(**json.loads(Path(a.config).read_text())) if a.config else Config()
    model = Model(cfg, arm='tp').cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, fused=True)
    seconds = []
    for i in range(12):
        torch.cuda.synchronize(); start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            loss = F.cross_entropy(model(x[i*64:(i+1)*64]).float(), y[i*64:(i+1)*64])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); torch.cuda.synchronize()
        seconds.append(time.monotonic()-start)
    result = dict(model=model.identity(), gpu=torch.cuda.get_device_name(0),
        gpu_memory=torch.cuda.get_device_properties(0).total_memory,
        update_seconds=seconds, mean_last10=sum(seconds[2:])/10,
        peak_memory=torch.cuda.max_memory_allocated(), status='THROUGHPUT_ONLY_NO_EVAL')
    Path(a.out).write_text(json.dumps(result, indent=2)); print(json.dumps(result))


if __name__ == '__main__': main()
