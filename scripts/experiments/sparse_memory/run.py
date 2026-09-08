"""Train fixed-update paired arms; final checkpoint is never selected on eval."""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from .model import Config, Model

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n'); tmp.replace(path)


def acquire_gpu(data_root):
    """One phase lock shared by training, evaluation and throughput probes."""
    lock_path = Path(data_root).parent/'runs'/'gpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return lock


def load_data(root, split, device):
    path = Path(root)/f'{split}.npz'
    manifest = json.loads((Path(root)/'manifest.json').read_text())
    entry = next(s for s in manifest['splits'] if s['file'] == path.name)
    if sha(path) != entry['sha256']:
        raise ValueError('frozen input changed')
    data = np.load(path)
    return (torch.tensor(data['x'].astype(np.int64), device=device),
            torch.tensor(data['y'].astype(np.int64), device=device), entry)


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()


@torch.inference_mode()
def evaluate(model, x, y, batch=96):
    model.eval()
    preds, margins, losses = [], [], []
    for start in range(0, len(x), batch):
        with autocast(x.device.type):
            logits = model(x[start:start+batch]).float()
        yy = y[start:start+batch]
        pred = logits.argmax(-1)  # full vocabulary, no answer whitelist
        gold = logits.gather(1, yy[:, None]).squeeze(1)
        competitor = logits.clone().scatter_(1, yy[:, None], -float('inf')).max(-1).values
        preds.extend(pred.cpu().tolist()); margins.extend((gold-competitor).cpu().tolist())
        losses.extend(F.cross_entropy(logits, yy, reduction='none').cpu().tolist())
    truth = y.cpu().numpy()
    p = np.asarray(preds)
    ok = p == truth
    pair = ok[0::3] & ok[1::3]
    metrics = dict(pair_em=float(pair.mean()), relation_em=float(np.concatenate((ok[0::3], ok[1::3])).mean()),
                   content_em=float(ok[2::3].mean()), nll=float(np.mean(losses)),
                   mean_margin=float(np.mean(margins)), pairs=len(pair))
    rows = [dict(row=i, pair_id=i//3, task='content' if i%3 == 2 else 'relation',
                 world=i%3, target=int(truth[i]), prediction=int(p[i]), correct=bool(ok[i]),
                 margin=margins[i], nll=losses[i]) for i in range(len(p))]
    return metrics, rows


def setup(seed, device):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but absent')
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True


def train(args):
    setup(args.seed, args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    lock = acquire_gpu(args.data)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    if time.time() >= args.deadline:
        raise RuntimeError('phase deadline already passed')
    cfg = Config(**json.loads(Path(args.config).read_text())) if args.config else Config()
    model = Model(cfg, args.arm).to(args.device)
    initial_sha = state_sha(model)
    x, y, train_identity = load_data(args.data, 'train', args.device)
    dx, dy, dev_identity = load_data(args.data, 'development', args.device)
    # Independent RNG gives identical batches for every paired arm.
    rng = torch.Generator(device='cpu').manual_seed(args.seed+100000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                 betas=(0.9, 0.95), fused=args.device == 'cuda')
    source = {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    identity = dict(args=vars(args), model=model.identity(), initial_state_sha256=initial_sha,
                    train=train_identity, development=dev_identity, code=source,
                    torch=torch.__version__, start_unix=time.time(), pid=os.getpid(),
                    gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                    note='single-seed synthetic development; fixed final step; no checkpoint selection')
    atomic(out/'identity.json', identity)
    atomic(out.parent/'active.json', dict(pid=os.getpid(), out=str(out), deadline=args.deadline))
    start = time.monotonic()
    status = 'RUNNING'
    step = 0
    with (out/'progress.jsonl').open('x') as log:
        while step < args.steps:
            if STOP or time.time() >= args.deadline:
                status = 'INTERRUPTED_NONFINAL'; break
            model.train()
            idx = torch.randint(len(x), (args.batch,), generator=rng).to(args.device)
            warm = min((step+1)/100, 1.0)
            cosine = 0.1+0.9*0.5*(1+math.cos(math.pi*step/args.steps))
            for group in optimizer.param_groups:
                group['lr'] = args.lr*warm*cosine
            optimizer.zero_grad(set_to_none=True)
            with autocast(args.device):
                logits = model(x[idx])
                loss = F.cross_entropy(logits.float(), y[idx])
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); step += 1
            if step == 1 or step % args.log_every == 0:
                if args.device == 'cuda': torch.cuda.synchronize()
                row = dict(step=step, loss=float(loss), grad_norm=float(norm),
                           elapsed=time.monotonic()-start, lr=optimizer.param_groups[0]['lr'])
                log.write(json.dumps(row)+'\n'); log.flush()
                atomic(out/'status.json', dict(status=status, **row))
                print(json.dumps(row), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics, _ = evaluate(model, dx, dy)
                row = dict(step=step, development=metrics, elapsed=time.monotonic()-start)
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
    final = step == args.steps
    status = 'COMPLETE' if final else status
    checkpoint = out/('final.pt' if final else 'recovery_nonfinal.pt')
    torch.save(dict(model=model.state_dict(), config=vars(cfg), arm=args.arm, step=step,
                    final=final, optimizer=optimizer.state_dict() if not final else None,
                    batch_rng=rng.get_state(), identity=identity), checkpoint)
    metrics, rows = evaluate(model, dx, dy)
    with (out/'development_rows.jsonl').open('x') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
    receipt = dict(status=status, step=step, wall_seconds=time.monotonic()-start,
                   checkpoint=checkpoint.name, checkpoint_sha256=sha(checkpoint),
                   development=metrics, prediction_tokens=step*args.batch,
                   input_tokens=step*args.batch*x.shape[1],
                   max_memory_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
    atomic(out/'result.json', receipt); atomic(out/'status.json', receipt)
    print(json.dumps(receipt), flush=True)
    if not final: raise SystemExit(2)


def eval_saved(args):
    lock = acquire_gpu(args.data)
    setup(0, args.device)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not payload['final']:
        raise ValueError('recovery snapshot cannot serve as final evaluation')
    model = Model(Config(**payload['config']), payload['arm']).to(args.device)
    model.load_state_dict(payload['model'])
    x, y, entry = load_data(args.data, args.split, args.device)
    metrics, rows = evaluate(model, x, y)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    with (out/'rows.jsonl').open('x') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
    atomic(out/'result.json', dict(metrics=metrics, data=entry,
        checkpoint_sha256=sha(args.checkpoint), raw_sha256=sha(out/'rows.jsonl')))
    print(json.dumps(metrics))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'eval'])
    p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--arm', choices=['baseline', 'tp', 'marginal'], default='baseline')
    p.add_argument('--config', help='frozen model config JSON; omitted for v1')
    p.add_argument('--seed', type=int, default=137)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--deadline', type=float, default=0)
    p.add_argument('--checkpoint'); p.add_argument('--split', default='test', choices=['development','test'])
    a = p.parse_args()
    train(a) if a.mode == 'train' else eval_saved(a)


if __name__ == '__main__':
    main()
