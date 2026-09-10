"""Download public preparation sources; stream files and verify available identities."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import http.client
from pathlib import Path
import time
import urllib.request
import xml.etree.ElementTree as ET

LONGALIGN_REV = '12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc'
PG_BUCKET = 'https://storage.googleapis.com/deepmind-gutenberg/'


def file_hash(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def download(url, path, *, md5=None, size=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (size is None or path.stat().st_size == size):
        if md5 is None or file_hash(path, 'md5') == md5:
            return dict(path=str(path), url=url, bytes=path.stat().st_size, sha256=file_hash(path))
    temporary = path.with_suffix(path.suffix + '.part')
    failures = 0
    total = size
    while True:
        offset = temporary.stat().st_size if temporary.exists() else 0
        try:
            headers = {'User-Agent': 'EVQ-research-data-preparation', 'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                content_range = response.headers.get('Content-Range')
                if offset and (response.status != 206 or not content_range or
                               not content_range.startswith(f'bytes {offset}-')):
                    raise ValueError(f'server did not honor continuation offset {offset}: {content_range}')
                if content_range:
                    total = int(content_range.rsplit('/',1)[1])
                elif response.headers.get('Content-Length'):
                    total = int(response.headers['Content-Length'])
                with temporary.open('ab' if offset else 'wb') as target:
                    while True:
                        try:
                            block = response.read(128 * 1024)
                        except http.client.IncompleteRead as error:
                            target.write(error.partial)
                            break
                        if not block:
                            break
                        target.write(block)
            if total is not None and temporary.stat().st_size < total:
                if temporary.stat().st_size > offset:
                    failures = 0
                    continue
                raise ValueError('incomplete HTTP response without progress')
            if size is not None and temporary.stat().st_size != size:
                raise ValueError('source size mismatch')
            if md5 and file_hash(temporary, 'md5') != md5:
                raise ValueError('source MD5 mismatch')
            temporary.replace(path)
            return dict(path=str(path), url=url, bytes=path.stat().st_size, sha256=file_hash(path))
        except Exception:
            failures += 1
            if failures == 4:
                raise
            time.sleep(2 ** (failures-1))


def list_books(split, count=24):
    request = PG_BUCKET + '?prefix=' + split + '/&max-keys=1000'
    raw = urllib.request.urlopen(request, timeout=45).read()
    root = ET.fromstring(raw)
    ns = {'s': 'http://doc.s3.amazonaws.com/2006-03-01'}
    books = []
    for item in root.findall('s:Contents', ns):
        key = item.findtext('s:Key', namespaces=ns)
        size = int(item.findtext('s:Size', namespaces=ns))
        if key.endswith('.txt') and size >= 180000:
            books.append(dict(key=key, bytes=size, etag=item.findtext('s:ETag', namespaces=ns).strip('"')))
    if len(books) < count:
        raise ValueError(f'not enough long {split} books in fixed listing')
    return books[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pg19-selection', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    selected = json.loads(args.pg19_selection.read_text())['books']
    selected = [{**row, 'split': 'train'} for row in selected]
    for split in ('validation', 'test'):
        selected.extend({**row, 'split': split} for row in list_books(split))
    (args.root / 'pg19_books.json').write_text(json.dumps(selected, indent=2) + '\n')

    jobs = [(PG_BUCKET + r['key'], args.root / 'pg19' / r['key'], r['etag'], r['bytes']) for r in selected]
    jobs.extend([
        (f'https://huggingface.co/datasets/zai-org/LongAlign-10k/resolve/{LONGALIGN_REV}/long.jsonl',
         args.root / 'longalign.jsonl', None, None),
        ('https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz',
         args.root / 'qasper-train-dev.tgz', None, None),
        ('https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz',
         args.root / 'qasper-test.tgz', None, None),
    ])

    def get(job):
        url, target, md5, size = job
        return download(url, target, md5=md5, size=size)

    receipts = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(get, jobs):
            receipts.append(item)
            if len(receipts) % 16 == 0 or len(receipts) > len(selected):
                print(json.dumps({'downloaded': len(receipts), 'total': len(jobs), 'last': item['path']}), flush=True)
    (args.root / 'acquisition.json').write_text(json.dumps(dict(
        status='PUBLIC_SOURCES_DOWNLOADED', longalign_revision=LONGALIGN_REV,
        files=receipts, pg19_selection='Existing 128 public train books; first 24 sufficiently long official validation/test books.'
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
