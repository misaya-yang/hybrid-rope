import json
import statistics
from pathlib import Path

ROOT = Path('/root/autodl-tmp/dfrope/hybrid-rope/results/evidence_chain_50m_3cfg3seed')
INP = ROOT / 'results.json'
OUT = ROOT / 'summary.md'

if not INP.exists():
    raise SystemExit(f'missing {INP}')

obj = json.loads(INP.read_text())
res = obj['results']

lines = []
lines.append('# 50M 3cfg x 3seed Summary')
lines.append('')
lines.append('| Config | PPL@2048 (mean±std) | PPL@16384 (mean±std) |')
lines.append('|---|---:|---:|')

for cfg, seedmap in res.items():
    p2 = [seedmap[s]['2048'] for s in sorted(seedmap.keys(), key=int)]
    p16 = [seedmap[s]['16384'] for s in sorted(seedmap.keys(), key=int)]
    m2, s2 = statistics.mean(p2), statistics.pstdev(p2)
    m16, s16 = statistics.mean(p16), statistics.pstdev(p16)
    lines.append(f'| {cfg} | {m2:.3f} ± {s2:.3f} | {m16:.3f} ± {s16:.3f} |')

# per-seed table
lines.append('')
lines.append('| Config | Seed | PPL@2048 | PPL@16384 |')
lines.append('|---|---:|---:|---:|')
for cfg, seedmap in res.items():
    for seed in sorted(seedmap.keys(), key=int):
        lines.append(f"| {cfg} | {seed} | {seedmap[seed]['2048']:.3f} | {seedmap[seed]['16384']:.3f} |")

OUT.write_text('\n'.join(lines) + '\n')
print(f'wrote {OUT}')
