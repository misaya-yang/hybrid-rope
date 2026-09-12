#!/usr/bin/env python3
"""Validate current indexes, document relocations and paper sources.

Historical snapshots are deliberately outside the active-link contract. This
checks local availability/identity, not scientific correctness or model runs.
Only --refresh-inventory writes a new Git-visible file inventory.
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
import argparse
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MAINTENANCE = ROOT / 'docs/maintenance'


def refresh_inventory() -> None:
    path = MAINTENANCE / 'repository_inventory.json'
    inventory = json.loads(path.read_text())
    moves = json.loads((MAINTENANCE / 'relocations.json').read_text())['moves']
    aliases = {move['old']: move['new'] for move in moves}
    listed = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
        cwd=ROOT, text=True).splitlines()
    files = sorted({aliases.get(name, name) for name in listed if (ROOT / name).is_file()})
    records = []
    for name in files:
        file = Path(name)
        if file.name.lower() in {'index.md', 'readme.md', 'agents.md'}:
            category = 'navigation'
        elif file.suffix == '.md':
            category = 'document'
        elif file.suffix in {'.py', '.sh'}:
            category = 'code'
        elif file.suffix in {'.json', '.jsonl', '.csv', '.tsv'}:
            category = 'data-or-receipt'
        elif name.startswith('paper-2027/'):
            category = 'manuscript'
        else:
            category = 'other'
        records.append({'path': name, 'area': name.split('/')[0] if '/' in name else 'root',
                        'category': category})
    inventory['files'] = records
    path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')


def check(local_evidence: bool = False) -> dict:
    inventory = json.loads((MAINTENANCE / 'repository_inventory.json').read_text())
    visible = {entry['path'] for entry in inventory['files']}
    errors = []
    links = 0
    for relative in inventory['managed_documents']:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f'Missing managed document: {relative}')
            continue
        text = path.read_text()
        # Keep fenced historical commands/code out of the navigation contract.
        text = re.sub(r'```.*?```', '', text, flags=re.S)
        for raw in re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)', text):
            target = raw.strip().strip('<>')
            if target.startswith(('https://', 'http://', 'mailto:', '#')):
                continue
            if Path(target).is_absolute() or re.match(r'^[A-Za-z]:', target):
                errors.append(f'Machine-specific link: {relative} -> {raw}')
                continue
            target = unquote(target.split('#', 1)[0])
            if not target:
                continue
            links += 1
            destination = Path(target) if target.startswith('/') else path.parent / target
            if not destination.exists():
                errors.append(f'Broken link: {relative} -> {raw}')
            try:
                repo_path = destination.resolve().relative_to(ROOT.resolve()).as_posix()
            except ValueError:
                errors.append(f'Link outside repository: {relative} -> {raw}')
                continue
            # Exact Git spelling also catches case errors hidden by macOS.
            if repo_path not in visible and not any(f.startswith(repo_path + '/') for f in visible):
                errors.append(f'Link not distributed through Git: {relative} -> {raw}')
    registry = json.loads((ROOT / 'paper-2027/research/evidence/asset_registry.json').read_text())
    sources = 0
    local_sources = 0
    for asset in registry['assets']:
        for source in asset['sources']:
            if Path(source['path']).is_absolute() or '..' in Path(source['path']).parts:
                errors.append(f'Non-repository source path: {source["path"]}')
                continue
            if source['availability'] == 'local-ignored':
                local_sources += 1
                if not local_evidence:
                    continue
            sources += 1
            path = ROOT / source['path']
            if source['availability'] == 'missing':
                continue
            if not path.is_file():
                errors.append(f'Missing source: {asset["id"]} -> {source["path"]}')
            elif hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
                errors.append(f'Source changed since registry snapshot: {source["path"]}')
    relocations = json.loads((MAINTENANCE / 'relocations.json').read_text())
    for move in relocations['moves']:
        if not (ROOT / move['new']).is_file():
            errors.append(f'Missing relocation target: {move["new"]}')
    result = {'managed_documents': len(inventory['managed_documents']),
              'local_links_checked': links, 'assets': len(registry['assets']),
              'source_occurrences_checked': sources, 'relocations': len(relocations['moves']),
              'local_source_occurrences': local_sources,
              'mode': 'local-evidence' if local_evidence else 'git-portable',
              'errors': errors, 'model_execution': False}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-inventory', action='store_true',
                        help='Refresh the repository-relative Git-visible file inventory before checking.')
    parser.add_argument('--local-evidence', action='store_true',
                        help='Also require ignored raw source mirrors to be present and hash-matched.')
    args = parser.parse_args()
    if args.refresh_inventory:
        refresh_inventory()
    result = check(local_evidence=args.local_evidence)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(bool(result['errors']))
