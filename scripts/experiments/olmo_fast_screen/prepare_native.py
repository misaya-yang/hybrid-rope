"""Reuse the exact short rows for a native-frequency, native-amplitude reference."""
import argparse
import json
from pathlib import Path
import shutil

from .bench import digest
from .prepare import sha_file, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();old=args.prepared.resolve();out=args.out.resolve()
    manifest=json.loads((old/'manifest.json').read_text())
    for name,h in manifest['prepared_files'].items():
        if sha_file(old/name)!=h:raise ValueError('source drift: '+name)
    rows=[json.loads(line) for line in (old/'screen.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['length_cap']==4096]
    if not rows:raise ValueError('no native-length rows')
    out.mkdir(parents=True,exist_ok=False)
    for name in manifest['prepared_files']:shutil.copyfile(old/name,out/name)
    (out/'screen.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'qualification.jsonl').write_text('')
    ids={r['row_id'] for r in rows}
    prompts=[json.loads(line) for line in (old/'prompts.jsonl').read_text().splitlines()]
    (out/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prompts if r['row_id'] in ids))
    write(out/'queue.json',dict(max_candidates=10,ordered_candidates=[]))
    root=Path(__file__).resolve().parents[3]
    deps=set(manifest['code_files'])|{'scripts/experiments/olmo_fast_screen/runtime.py',
                                   'scripts/experiments/olmo_fast_screen/prepare_native.py'}
    manifest.update(reference_arm='Native',static_scale=1,status='NATIVE_SHORT_REFERENCE_READY',
        source_manifest_sha256=sha_file(old/'manifest.json'),physical_caps=[4096],screen_rows=len(rows),
        screen_input_tokens=sum(r['input_tokens'] for r in rows),
        screen_min_max_tokens=[min(r['input_tokens'] for r in rows),max(r['input_tokens'] for r in rows)],
        row_order=[r['row_id'] for r in rows],prompt_collection_sha256=digest([r['prompt_ids'] for r in rows]),
        code_files={name:sha_file(root/name) for name in deps})
    manifest['prepared_files']={name:sha_file(out/name) for name in manifest['prepared_files']}
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(rows=len(rows),reference='Native')))


if __name__=='__main__':main()
