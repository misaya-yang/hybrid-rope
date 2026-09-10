"""Refresh the equivalent binary-weight factorization before E10 evaluation."""
import importlib
import torch
from .worker import save,sha


def run(worker,job):
    from . import operators
    importlib.reload(operators)
    table=worker.tables['MrPro']
    worker.apply({'table':table})
    tokens=torch.tensor([worker.screen[0]['prompt_ids'][:512]],device='cuda')
    with torch.inference_mode():
        reference=worker.model(tokens,use_cache=False,logits_to_keep=8).logits.float()
        worker.apply(dict(operator='dual_frequency',table=table,second_table=table))
        got=worker.model(tokens,use_cache=False,logits_to_keep=8).logits.float()
        rms=float((got-reference).square().mean().sqrt())
        relative=float((got-reference).square().mean().sqrt()/reference.square().mean().sqrt())
        argmax_matches=int((got.argmax(-1)==reference.argmax(-1)).sum())
        assert relative<.03,relative
    worker.apply({'table':table})
    result=dict(status='PASS',rms=rms,relative_rms=relative,argmax_matches_of_8=argmax_matches,
        factorization='Q weighted by 1/2, K unweighted; same bilinear kernel as two sqrt(1/2) factors with less rounding',
        source_sha256=sha(operators.__file__))
    save(worker.root/'dual_precision_check.json',result)
    return result
