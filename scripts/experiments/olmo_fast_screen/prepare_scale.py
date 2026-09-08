"""Transport the fixed MrPro and BM formulas to another static extension scale."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.analysis.build_boundary_matched_mrpro import build
from scripts.experiments.cross_audit.tables import transform, tensor_sha
from .prepare import sha_file, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--scale', required=True, type=float)
    args = p.parse_args()
    old, out = args.prepared.resolve(), args.out.resolve()
    manifest = json.loads((old/'manifest.json').read_text())
    for name, expected in manifest['prepared_files'].items():
        if sha_file(old/name) != expected:raise ValueError('source changed: '+name)
    tables = json.loads((old/'tables.json').read_text())
    native = np.asarray(tables['Native']['values_float32'],dtype=np.float32)
    mr,gain,meta = transform(native,dim=manifest['head_dim'],base=manifest['base'],
                            reference_length=manifest['native_length'],scale=args.scale,method='mrpro')
    bm = build(native.tolist(),mr.tolist(),gain,args.scale,meta['low'],meta['high'])
    tables['MrPro'] = dict(values_float32=mr.tolist(),tensor_sha256=tensor_sha(mr),gain=gain)
    tables['MrProBM'] = {k:bm[k] for k in ('values_float32','tensor_sha256','gain')}
    out.mkdir(parents=True,exist_ok=False)
    for name in manifest['prepared_files']:shutil.copyfile(old/name,out/name)
    write(out/'tables.json',tables)
    root = Path(__file__).resolve().parents[3]
    deps=set(manifest['code_files'])|{'scripts/experiments/olmo_fast_screen/runtime.py',
        'scripts/experiments/olmo_fast_screen/prepare_scale.py',
        'scripts/analysis/build_boundary_matched_mrpro.py','scripts/analysis/project_mrpro_transition.py'}
    manifest.update(static_scale=args.scale,status='TABLES_READY_INPUT_GENERATION_REQUIRED',
                    source_manifest_sha256=sha_file(old/'manifest.json'),
                    code_files={name:sha_file(root/name) for name in sorted(deps)})
    manifest['prepared_files']={name:sha_file(out/name) for name in manifest['prepared_files']}
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(scale=args.scale,low=meta['low'],high=meta['high'],gain=gain,bm_sha=bm['tensor_sha256'])))


if __name__=='__main__':main()
