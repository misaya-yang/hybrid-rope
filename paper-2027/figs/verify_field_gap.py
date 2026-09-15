"""Recompute the clean panel point estimates and generate its appendix table."""
from pathlib import Path
import json
import numpy as np
HERE = Path(__file__).resolve().parent

def main():
    data=json.loads((HERE/'field_gap_inputs.json').read_text())
    clean=data['clean'];pairs=clean['pairs'];tasks=sorted(clean['tasks'])
    assert len(pairs)==2600 and len({r['prompt_sha256']for r in pairs})==2600
    means={t:{a:float(np.mean([r[a]for r in pairs if r['task']==t]))for a in ['tailspline','mrpro']}for t in tasks}
    assert all(sum(r['task']==t for r in pairs)==200 for t in tasks)
    for arm in ['tailspline','mrpro']:
        assert abs(np.mean([means[t][arm]for t in tasks])-clean['point'][arm])<1e-12
    deltas={t:means[t]['tailspline']-means[t]['mrpro']for t in tasks}
    for t in tasks:
        assert abs(deltas[t]-clean['task_deltas'][t])<1e-12
        assert abs(np.mean([v for k,v in deltas.items()if k!=t])-clean['leave_one_task_out_delta'][t])<1e-12
    # Independent stratified paired bootstrap, fixed rows/tasks; Monte Carlo agreement.
    rng=np.random.default_rng(20260915);draws=np.zeros(20000)
    for task in tasks:
        values=np.array([r['tailspline']-r['mrpro']for r in pairs if r['task']==task])
        draws+=values[rng.integers(200,size=(20000,200))].mean(1)/13
    ci=np.quantile(draws,[.025,.975]);reported=clean['contrast']['bootstrap']['delta_log_auc']['interval95']
    assert max(abs(ci-np.array(reported)))<.0015
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Clean RULER-200: all task means and output health. Scores include capped and empty responses.}',r'\label{tab:clean-full13}',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'Task & TailSpline (\%) & MrPro (\%) & Difference (pp) \\',r'\midrule']
    for task in tasks:
        label=task.replace('_',r'\_')
        lines.append(f'{label} & {100*means[task]["tailspline"]:.2f} & {100*means[task]["mrpro"]:.2f} & {100*deltas[task]:+.2f} '+r'\\')
    lines += [r'\midrule',r'Task-equal mean & 68.27 & 56.54 & +11.72 \\',r'\midrule']
    for key,label in [('eos','EOS count'),('cap','Cap-hit count'),('empty','Empty count'),('generated_tokens','Generated tokens')]:
        lines.append(f'{label} & {clean["health"]["tailspline"][key]} & {clean["health"]["mrpro"][key]} & --- '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (HERE.parent/'tables/table_clean_full13.tex').write_text('\n'.join(lines)+'\n')
    assert data['e1']['runtime_match']=='QUALIFIED_ONLY'
    assert data['e0']['changed_score_rows']==0 and data['e0']['changed_generation_rows']==6
    clean16=data.get('clean16k')
    if clean16:
        assert clean16['paired_prompts']==650
        for arm in ['tailspline','mrpro']:
            block=clean16['summaries'][arm]['by_length']['16384']
            values=[block['tasks'][t]['official']for t in clean16['tasks']]
            assert all(block['tasks'][t]['rows']==50 for t in clean16['tasks'])
            assert abs(float(np.mean(values))-block['task_macro_official'])<1e-12
        delta=clean16['summaries']['tailspline']['log_length_auc']-clean16['summaries']['mrpro']['log_length_auc']
        assert abs(delta-clean16['contrasts']['mrpro']['delta_by_length']['16384'])<1e-12
        print(json.dumps({'clean16k_report':'PASS','paired_rows':650,'reported_delta_pp':delta*100}))
    natural=data.get('naturalqa')
    if natural:
        qa=natural['scored_pairs'];qa_tasks=natural['tasks']
        assert len(qa)==631 and len({r['row_id']for r in qa})==631
        assert len({r['document_cluster_id']for r in qa})==524
        def macro(values, arm):
            return float(np.mean([np.mean([r[arm]for r in values if r['task']==t])for t in qa_tasks]))
        def cluster_ci(values, seed):
            rng=np.random.default_rng(seed);draws=np.zeros(20000)
            for task in qa_tasks:
                clusters={}
                for row in values:
                    if row['task']==task:clusters.setdefault(row['document_cluster_id'],[]).append(row['tailspline']-row['mrpro'])
                sums=np.array([sum(v)for v in clusters.values()]);counts=np.array([len(v)for v in clusters.values()])
                sampled=rng.integers(len(sums),size=(20000,len(sums)))
                draws+=sums[sampled].sum(1)/counts[sampled].sum(1)/len(qa_tasks)
            return np.quantile(draws,[.025,.975])
        groups=[('All (primary)',qa,natural['candidate_minus_baseline'],20260914),
                (r'$\le8$K',[r for r in qa if r['input_tokens']<=8192],natural['llama_length_audit']['within_native_effect']['candidate_minus_baseline'],20260916),
                (r'$>8$K',[r for r in qa if r['input_tokens']>8192],natural['llama_length_audit']['extended_effect']['candidate_minus_baseline'],20260915)]
        lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{\textbf{TailSpline natural QA.} Task-equal F1 with source-context cluster intervals. Both arms use extension tables, including within the native window.}',r'\label{tab:tailspline-natural-main}',r'\begin{tabular}{@{}lrrrl@{}}',r'\toprule',r'Input stratum & $N$ & T (\%) & P (\%) & T--P (pp), $95\%$ CI \\',r'\midrule']
        for label,values,result,seed in groups:
            ts,mp=macro(values,'tailspline'),macro(values,'mrpro')
            assert abs(ts-mp-result['estimate'])<1e-12
            assert np.allclose(cluster_ci(values,seed),result['ci95'],rtol=0,atol=1e-12)
            lo,hi=np.array(result['ci95'])*100
            lines.append(f'{label} & {len(values)} & {100*ts:.2f} & {100*mp:.2f} & {100*(ts-mp):+.2f} [{lo:+.2f}, {hi:+.2f}] '+r'\\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        (HERE.parent/'tables/table_tailspline_natural_main.tex').write_text('\n'.join(lines)+'\n')
        lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Natural-QA631 task means and complete-output health.}',r'\label{tab:tailspline-natural-tasks}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Task & $N$ & TailSpline (\%) & MrPro (\%) & Difference (pp) \\',r'\midrule']
        for task in qa_tasks:
            values=[r for r in qa if r['task']==task]
            ts,mp=[float(np.mean([r[a]for r in values]))for a in ['tailspline','mrpro']]
            label=task.replace('_',r'\_')
            lines.append(f'{label} & {len(values)} & {100*ts:.2f} & {100*mp:.2f} & {100*(ts-mp):+.2f} '+r'\\')
        lines += [r'\midrule']
        for key,label in [('ended_eos','EOS'),('hit_cap','Cap-hit'),('empty','Empty'),('generated_tokens','Generated tokens')]:
            lines.append(f"{label} & --- & {natural['output_health']['tailspline'][key]} & {natural['output_health']['mrpro'][key]} & --- "+r'\\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        (HERE.parent/'tables/table_tailspline_natural_tasks.tex').write_text('\n'.join(lines)+'\n')
        print(json.dumps({'natural_qa':'PASS','paired_questions':631,'source_clusters':524,'primary_and_both_stratum_intervals':'reproduced exactly'}))
    print(json.dumps({'status':'PASS','paired_rows':2600,'independent_ci95':ci.tolist(),'reported_ci95':reported,'wins':sum(v>0 for v in deltas.values())}))

if __name__=='__main__':main()
