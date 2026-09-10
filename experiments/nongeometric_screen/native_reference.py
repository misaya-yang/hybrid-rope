"""Fill only the missing native short-context generation reference once."""
import json
from .worker import read_rows,save,sha


def run(worker,job):
    from scripts.experiments.olmo_fast_screen.ruler_bench import summarize
    folder=worker.root/'native_reference';folder.mkdir(exist_ok=True)
    save(folder/'contract.json',dict(table=worker.tables['Native'],generation_config_sha256=sha(worker.prepared/'generation_config.json')))
    raw=folder/'ruler.jsonl';existing={r['row_id']:r for r in read_rows(raw)}
    worker.apply({'table':worker.tables['Native']})
    for row in [r for r in worker.screen if r['length_cap']==32768]:
        if row['row_id'] in existing:continue
        r=worker.generate(row);existing[r['row_id']]=r
        with raw.open('a') as stream:stream.write(json.dumps(r)+'\n')
        save(worker.root/'live.json',dict(job=job['id'],phase='native_short_reference',completed=len(existing),requested=12))
    result=dict(status='COMPLETE',summary=summarize(list(existing.values())),
        existing_native_nll=str(worker.history/'run_nll_01/Native.jsonl'),
        scope='Native table, S=1, gain=1, original twelve short development inputs. Existing Native NLL is reused, not recomputed.')
    save(folder/'summary.json',result);worker.apply({'table':worker.tables['MrPro']})
    return result
