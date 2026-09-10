"""Repair the E7 budget to use the same local support as its response matrix."""
import importlib
import json
import numpy as np

from .worker import save


def run(worker,job):
    from . import select, project
    importlib.reload(select)
    importlib.reload(project)
    result=project.run(worker,job)
    result['budget_correction']='Both finite control and Jacobian now modify the same local key support. Initial full-support budget was inconsistent; no E7 model evaluation used it.'
    values=np.asarray(result['spec']['table']['values_float32'],dtype=np.float32)
    bm=np.asarray(worker.tables['MrProBM']['values_float32'],dtype=np.float32)
    if np.array_equal(values,bm):
        result['status']='IDENTICAL_TO_HISTORICAL_BM'
        save(worker.root/'done/019_E7_local_projection.json',dict(status='REUSED_IDENTICAL_BASELINE',
            result='The corrected local constraint is inactive; the exact resulting table is historical BM. Reuse its NLL and 36-row RULER results instead of rerunning.',
            source='/root/autodl-tmp/bm_transfer_20260908/run_qwen3_01/MrProBM.jsonl'))
    save(worker.root/'selection/E7_projection.json',result)
    return result
