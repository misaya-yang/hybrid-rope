"""Download a pinned public checkpoint through a mirror, checking official SHAs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'
from huggingface_hub import hf_hub_download


def main():
    root = Path('/root/autodl-tmp/bm_transfer_qwen7b_20260908')
    metadata = Path('/root/autodl-tmp/qwen7b_metadata')
    spec = json.loads((metadata/'asset_source.json').read_text())
    model = root/'model'
    model.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < sum(x['size'] for x in spec['weights'].values())+2_000_000_000:
        raise RuntimeError('insufficient free space; no cleanup performed')
    shutil.copyfile(metadata/'asset_source.json', root/'asset_source.json')
    for name, expected in spec['metadata_sha256'].items():
        data = (metadata/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('official metadata transfer mismatch')
        (model/name).write_bytes(data)

    def download(name):
        path = Path(hf_hub_download(repo_id=spec['model_id'], filename=name,
            revision=spec['revision'], local_dir=model, endpoint='https://hf-mirror.com',
            token=False, etag_timeout=15))
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                sha.update(chunk)
        if sha.hexdigest() != spec['weights'][name]['sha256'] or path.stat().st_size != spec['weights'][name]['size']:
            raise ValueError('mirror weights differ from pinned official identity')
        print(json.dumps(dict(file=name, status='OFFICIAL_SHA_MATCHED')), flush=True)

    (root/'download_status.json').write_text(json.dumps(dict(status='DOWNLOADING', pid=os.getpid())))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, spec['weights']))
    except Exception as exc:
        (root/'download_status.json').write_text(json.dumps(dict(status='FAILED', error_type=type(exc).__name__)))
        raise
    (root/'download_status.json').write_text(json.dumps(dict(status='COMPLETE', revision=spec['revision'])))


if __name__ == '__main__':
    main()
