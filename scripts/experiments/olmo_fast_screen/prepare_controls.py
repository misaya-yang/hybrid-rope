"""Matched existing-method controls for interpreting the BM improvement."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scripts.experiments.cross_audit.tables import transform, tensor_sha
from .prepare import sha_file, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();old=args.prepared.resolve();out=args.out.resolve()
    manifest=json.loads((old/'manifest.json').read_text())
    for name,h in manifest['prepared_files'].items():
        if sha_file(old/name)!=h:raise ValueError('source drift: '+name)
    tables=json.loads((old/'tables.json').read_text())
    native=np.array(tables['Native']['values_float32'],dtype=np.float32)
    for name,method in [('MrUni','mruni'),('OfficialYaRN','yarn')]:
        values,gain,meta=transform(native,dim=manifest['head_dim'],base=manifest['base'],
                                  reference_length=manifest['native_length'],scale=manifest['static_scale'],method=method)
        tables[name]=dict(values_float32=values.tolist(),gain=gain,tensor_sha256=tensor_sha(values),construction=meta)
    out.mkdir(parents=True,exist_ok=False)
    for name in manifest['prepared_files']:shutil.copyfile(old/name,out/name)
    write(out/'tables.json',tables)
    write(out/'queue.json',dict(max_candidates=10,ordered_candidates=[dict(
        id='OfficialYaRN',eligible=True,review_status='REVIEWED_FOR_GPU',
        definition='Existing official linear-ramp YaRN with amplitude 1+0.1*ln(S)',
        hypothesis='Existing-method reference, not a new proposed method; compare both controls to previously fixed BM.',
        failure_rule='Report complete scores; no outcome-based changes or claim of a new method.')]))
    root=Path(__file__).resolve().parents[3]
    deps=set(manifest['code_files'])|{'scripts/experiments/olmo_fast_screen/runtime.py',
                                   'scripts/experiments/olmo_fast_screen/prepare_controls.py'}
    manifest.update(reference_arm='MrUni',complete_candidate_queue=True,status='EXISTING_CONTROLS_READY',
        source_manifest_sha256=sha_file(old/'manifest.json'),
        comparison_roles='MrUni and OfficialYaRN are existing controls; compare paired rows against frozen BM and MrPro separately.',
        code_files={name:sha_file(root/name) for name in deps})
    manifest['prepared_files']={name:sha_file(out/name) for name in manifest['prepared_files']}
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(rows=manifest['screen_rows'],reference='MrUni',control='OfficialYaRN')))


if __name__=='__main__':main()
