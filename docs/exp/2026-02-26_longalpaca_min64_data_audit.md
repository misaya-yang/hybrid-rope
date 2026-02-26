# LongAlpaca Min64 Data Audit (2026-02-26)

## Audited dataset
- Path: `/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl`
- Source: `LongAlpaca-12k.jsonl` exported from ModelScope mirror
- Filter rule: keep samples with non-empty `instruction/output` and `assistant_tokens >= 64`

## Quality summary
- Input rows: `12000`
- Kept rows: `9526`
- Dropped (`assistant_tokens < 64`): `2472`
- Dropped (missing required fields): `2`
- Continuation-like ratio (post-filter): `0.0`

## Token stats (assistant/output)
- min: `64`
- p50: `211`
- p90: `278`
- mean: `206.87`
- max: `634`

## Integrity
- SHA256: `a9e86ac088aae843556a7d88f97d8369bf05e668a5e2d09e59af2784ba476587`
- Size: `469.33 MB`

## Reproduction check command
```bash
/root/miniconda3/bin/python - <<'PY'
import json, hashlib
p='/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl'
h=hashlib.sha256()
for line in open(p,'rb'):
    h.update(line)
print(h.hexdigest())
PY
```
