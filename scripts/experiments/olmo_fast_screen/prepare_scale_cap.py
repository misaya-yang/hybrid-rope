"""Test a previously successful BM deployment at the failed longer context."""
import argparse
import json
from pathlib import Path
import shutil

from .prepare import sha_file, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--long-prepared',required=True,type=Path)
    p.add_argument('--successful-prepared',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();old=args.long_prepared.resolve();out=args.out.resolve()
    manifest=json.loads((old/'manifest.json').read_text())
    successful=json.loads((args.successful_prepared/'manifest.json').read_text())
    for directory,m in [(old,manifest),(args.successful_prepared,successful)]:
        for name,h in m['prepared_files'].items():
            if sha_file(directory/name)!=h:raise ValueError('source drift: '+name)
    for key in ('model_id','revision','weight_sha256','native_length','base','head_dim'):
        if manifest[key]!=successful[key]:raise ValueError('source model mismatch: '+key)
    tables=json.loads((old/'tables.json').read_text())
    source=json.loads((args.successful_prepared/'tables.json').read_text())['MrProBM']
    full=tables['MrProBM']
    tables['BMCappedS4']=source
    tables['BMFreq8Gain4']=dict(full,gain=source['gain'])
    out.mkdir(parents=True,exist_ok=False)
    for name in manifest['prepared_files']:shutil.copyfile(old/name,out/name)
    write(out/'tables.json',tables)
    write(out/'queue.json',dict(max_candidates=10,ordered_candidates=[dict(
        id='BMFreq8Gain4',eligible=True,review_status='REVIEWED_FOR_GPU',
        definition='Original BM S8 frequencies with the previously tested S4 amplitude',
        hypothesis='Control if reducing the scalar amplitude alone explains recovery, compared with S4 frequency and amplitude.',
        failure_rule='Compare full unchanged 32K/4K panel against saved full-S8 and capped-S4; no coefficient tuning.')]))
    root=Path(__file__).resolve().parents[3]
    deps=set(manifest['code_files'])|{'scripts/experiments/olmo_fast_screen/prepare_scale_cap.py'}
    manifest.update(reference_arm='BMCappedS4',complete_candidate_queue=True,status='SCALE_CAP_DEVELOPMENT_READY',
        source_manifest_sha256=sha_file(old/'manifest.json'),
        successful_source_manifest_sha256=sha_file(args.successful_prepared/'manifest.json'),
        deployment_scale_by_arm={'BMCappedS4':dict(frequency=4,gain=4),'BMFreq8Gain4':dict(frequency=8,gain=4),'saved_MrProBM':dict(frequency=8,gain=8)},
        selection='32K macro gain over saved S8 BM and short recovery; S4 cap is empirical and is not an independently established universal scale law.',
        code_files={name:sha_file(root/name) for name in deps})
    manifest['prepared_files']={name:sha_file(out/name) for name in manifest['prepared_files']}
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(rows=manifest['screen_rows'],reference='BMCappedS4',control='BMFreq8Gain4')))


if __name__=='__main__':main()
