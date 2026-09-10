"""CPU-only held-out PG19 and Proof-Pile sources; no concatenation or scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET


def sha(data):return hashlib.sha256(data).hexdigest()


def main(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    source=root/'proofpile_test.jsonl.gz'
    expected='b1bc923aa34b2b03db08e2f451d8442d9ca7aad1c857a8835382c94a8bb1d835'
    if sha(source.read_bytes())!=expected:raise ValueError('official Proof-Pile archive hash mismatch')
    docs=[]
    with gzip.open(source,'rt') as stream:
        for index,line in enumerate(stream):
            record=json.loads(line);text=record['text']
            if len(text)<300000 or record.get('meta',{}).get('config')!='arxiv':continue
            prefix=text[:1000000];name=f'proofpile_test_{index:06d}.txt'
            raw=prefix.encode();(root/name).write_bytes(raw)
            docs.append(dict(dataset='proofpile',split='test',file=name,sha256=sha(raw),source_row=index,
                source_archive_sha256=expected,full_text_sha256=sha(text.encode()),full_text_chars=len(text),
                source_metadata={k:v for k,v in record.items() if k!='text'},
                source_url='https://huggingface.co/datasets/hoskinson-center/proof-pile/resolve/main/test/proofpile_test.jsonl.gz'))
            if len(docs)==32:break
    print(json.dumps(dict(proofpile_eligible=len(docs))),flush=True)
    listing_url='https://storage.googleapis.com/deepmind-gutenberg?prefix=test%2F&max-keys=1000'
    listing=urllib.request.urlopen(listing_url,timeout=30).read();(root/'pg19_listing.xml').write_bytes(listing)
    tree=ET.fromstring(listing);ns={'g':'http://doc.s3.amazonaws.com/2006-03-01'}
    entries=[]
    for node in tree.findall('g:Contents',ns):
        key=node.findtext('g:Key',namespaces=ns);size=int(node.findtext('g:Size',namespaces=ns))
        if key.endswith('.txt') and size>=700000:
            entries.append(dict(key=key,full_size=size,etag=node.findtext('g:ETag',namespaces=ns)))
    entries=sorted(entries,key=lambda x:x['key'])[:16]
    def download(entry):
        url='https://storage.googleapis.com/deepmind-gutenberg/'+entry['key']
        req=urllib.request.Request(url,headers={'Range':'bytes=0-1048575'})
        with urllib.request.urlopen(req,timeout=40) as response:
            raw=response.read(1048576);headers=dict(response.headers);status=response.status
        name='pg19_test_'+Path(entry['key']).name
        (root/name).write_bytes(raw)
        return dict(dataset='pg19',split='test',file=name,sha256=sha(raw),source_url=url,
            listing_sha256=sha(listing),source=entry,read_prefix_bytes=len(raw),http_status=status,
            content_range=headers.get('Content-Range'),note='Byte prefix of one genuine book; incomplete trailing UTF-8 codepoint ignored at tokenization')
    with ThreadPoolExecutor(max_workers=4) as pool:docs.extend(pool.map(download,entries))
    receipt=dict(status='SOURCES_READY_FOR_TOKEN_LENGTH_CHECK',docs=docs,
        selection='First 32 Proof-Pile arxiv test rows with >=300000 chars; first 16 lexical PG19 test books with >=700000 bytes. Model-independent eligibility; later retain first 8 per source with >=131073 tokens.',
        no_packing=True,proofpile_archive_sha256=expected)
    (root/'sources.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(source_documents=len(docs))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
