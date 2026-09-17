"""Verify recorded aggregates and render the September 17 evidence tables.

This script does not load models or rescore generated text. Paired score rows
are checked where present; remaining observations retain report-level identity.
"""
from pathlib import Path
import json
import math
from statistics import mean
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / 'revision_evidence_inputs.json').read_text())['reports']
ARMS = ('tailspline', 'mrpro', 'yarn')
MODELS = [('llama', 'Llama', '32'), ('olmo', 'OLMo', '16'),
          ('qwen', 'Qwen', '128'), ('glm', 'GLM', '128')]


def close(a, b):
    assert abs(a - b) < 1e-10, (a, b)


def normalized(key):
    d = D[key + ('_direct' if key in ('llama', 'olmo') else '_quick')]
    if key == 'qwen':
        tasks = {t: {a: d['arms'][a]['by_task'][t] for a in ARMS}
                 for t in d['arms']['tailspline']['by_task']}
        return ({a:d['arms'][a]['niah8_macro'] for a in ARMS},
                {a:d['arms'][a]['ppl'] for a in ARMS}, tasks,
                {b:(d['contrasts']['tailspline_minus_'+b]['niah8_macro'],
                    d['contrasts']['tailspline_minus_'+b]['niah8_ci95'],
                    d['contrasts']['tailspline_minus_'+b]['delta_nll'],
                    d['contrasts']['tailspline_minus_'+b]['delta_nll_ci95']) for b in ('mrpro','yarn')})
    return (d['niah']['task_equal_macro'], {a:d['ppl']['pooled'][a]['ppl'] for a in ARMS},
            d['niah']['by_task'],
            {b:(d['comparisons']['tailspline_minus_'+b]['niah_macro_delta'],
                d['comparisons']['tailspline_minus_'+b]['niah_task_stratified_bootstrap_ci95'],
                d['comparisons']['tailspline_minus_'+b]['mean_document_nll_delta'],
                d['comparisons']['tailspline_minus_'+b]['document_bootstrap_ci95']) for b in ('mrpro','yarn')})


def full_cell(key, arm):
    return next(iter(D[key+'_full13']['summaries'][arm]['by_length'].values()))


def full_score(key, arm):
    return full_cell(key, arm)['task_macro_official']


def full_ci(key, base):
    b=D[key+'_full13']['contrasts'][base]['bootstrap']
    return b['delta_task_macro_interval95'] if key!='llama70' else b['delta_by_length_interval95']['32768']


def verify():
    for key, _, _ in MODELS:
        macro, ppl, tasks, contrasts = normalized(key)
        assert len(tasks)==8
        for arm in ARMS:
            close(mean(v[arm] for v in tasks.values()),macro[arm])
        if key!='qwen':
            d=D[key+('_direct' if key in ('llama','olmo') else '_quick')]
            rows=d['niah']['paired_rows'];docs=d['ppl']['paired_documents']
            n=200 if key in ('llama','olmo') else 5
            assert len(rows)==len({r['prompt_sha256'] for r in rows})==8*n
            assert len(docs)==(46 if n==200 else 5)
            assert len({d['ppl']['pooled'][a]['target_tokens'] for a in ARMS})==1
            for arm in ARMS:
                for task in tasks:
                    vals=[r['scores'][arm] for r in rows if r['task']==task]
                    assert len(vals)==n
                    close(mean(vals),tasks[task][arm])
                close(math.exp(d['ppl']['pooled'][arm]['nll']),ppl[arm])
            for base in ('mrpro','yarn'):
                close(mean(r['mean_nll']['tailspline']-r['mean_nll'][base] for r in docs),contrasts[base][2])
        for base in ('mrpro','yarn'):
            close(macro['tailspline']-macro[base],contrasts[base][0])
    for key in ('llama','olmo','qwen','glm','llama70'):
        assert D[key+'_full13']['paired_prompts']==130
        arms=ARMS if key!='llama70' else ARMS[:2]
        for arm in arms:
            cell=full_cell(key,arm)
            assert len(cell['tasks'])==13 and all(t['rows']==10 for t in cell['tasks'].values())
            close(mean(t['official'] for t in cell['tasks'].values()),full_score(key,arm))
        for base in arms[1:]:
            c=D[key+'_full13']['contrasts'][base]
            close(full_score(key,'tailspline')-full_score(key,base),
                  c['delta_task_macro_official'] if key!='llama70' else c['delta_log_length_auc'])
    for key in ('qwen_qa','glm_qa','glm_qa_second','llama_qa','olmo_qa'):
        d=D[key];metric='macro_f1' if key in ('llama_qa','olmo_qa') else 'qa_f1'
        for base in ('mrpro','yarn'):
            close(d['arms']['tailspline'][metric]-d['arms'][base][metric],
                  d['contrasts']['tailspline_minus_'+base]['estimate'])
    for key in ('llama70_ppl32','llama70_ppl128'):
        for arm in ARMS[:2]:close(math.exp(D[key]['arms'][arm]['nll']),D[key]['arms'][arm]['ppl'])
        close(D[key]['arms']['tailspline']['nll']-D[key]['arms']['mrpro']['nll'],D[key]['delta_nll_tailspline_minus_mrpro'])
    qa=D['llama70_qa']
    close(qa['arms']['tailspline']['macro_f1']-qa['arms']['mrpro']['macro_f1'],qa['candidate_minus_baseline']['estimate'])
    c=D['ncp_confirm_arms']['formal_fixed_panel_scores']
    close(c['C0']-c['N0'],D['ncp_confirmation']['ruler']['ncp_minus_native']['estimate'])
    close(mean(D['ncp_confirmation']['ruler']['ncp_minus_native']['by_task'].values()),c['C0']-c['N0'])
    lm=D['ncp_lm']['metrics']
    close(lm['ncp_full']['estimate']-lm['native_full']['estimate'],lm['delta_full']['estimate'])
    close(lm['use_ncp']['estimate']-lm['use_native']['estimate'],lm['delta_use']['estimate'])
    close(lm['delta_recent']['estimate']-lm['delta_full']['estimate'],lm['delta_use']['estimate'])
    for key in ('llama_extreme_qa','llama_extreme_dia'):
        d=D[key]['overall'];close(d['tailspline']-d['mrpro'],d['delta_tailspline_minus_mrpro'])


def table(name, caption, label, columns, header, rows):
    text = '\n'.join([r'\begin{table}[ht]', r'\centering\small',
        '\\caption{' + caption + '}', '\\label{' + label + '}',
        '\\begin{tabular}{@{}' + columns + '@{}}', r'\toprule',
        header + r' \\', r'\midrule', *rows,
        r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    (HERE.parent / 'tables' / name).write_text(text)


def interval(v, scale=100, digits=2):
    return f'[{scale*v[0]:.{digits}f}, {scale*v[1]:.{digits}f}]'


def verify_connections():
    def lse(x):
        m = np.max(x)
        return m + np.log(np.exp(x-m).sum())
    rng = np.random.default_rng(20260916)
    error = 0.
    for _ in range(1000):
        logits = rng.normal(size=12)*3
        eta = rng.normal(size=12)*2
        direct = lse(logits[:4]+eta[:4])-lse(logits[4:]+eta[4:])-lse(logits[:4])+lse(logits[4:])
        rhs = lse(logits[:4]-lse(logits[:4])+eta[:4])-lse(logits[4:]-lse(logits[4:])+eta[4:])
        error = max(error, abs(direct-rhs))
    assert error < 1e-12
    for base, pairs, length in [(500000,64,8192),(500000,64,4096),(1000000,64,32768),(10000,32,32768)]:
        omega=base**(-np.arange(pairs,dtype=float)/pairs)
        turns=length*omega/(2*np.pi)
        l=np.where(turns>32)[0][-1];h=np.where(turns<1)[0][0]
        c=np.log(base)/pairs;n=h-l;u=np.arange(n+1)/n
        ep=np.log(length*omega[l]/(64*np.pi));em=np.log(2*np.pi/(length*omega[h]))
        close(n*c,np.log(32)+ep+em)
        assert np.allclose(length*omega[l:h+1],64*np.pi*32**(-u)*np.exp((1-u)*ep-u*em),rtol=1e-12)
    mature=json.loads((HERE/'revision_evidence_inputs.json').read_text())['cosh_mature']['llama_lora']
    temporal=json.loads((HERE/'llama_temporal_summary.json').read_text())['lengths']
    for i,length in enumerate(('8K','16K','32K')):
        for arm,values in [('geo_lora',mature['native_ppl']),('evq_lora',mature['cosh_ppl'])]:
            assert abs(math.exp(temporal[length]['domain_macro_nll'][arm])-values[i])<.00051
        if length!='8K':
            assert len(temporal[length]['paired_pack_deltas'])==24
            assert all(v<0 for v in temporal[length]['paired_pack_deltas'])
    return float(error)


def fmt(v, scale=100, digits=2):
    return format(Decimal(str(v*scale)).quantize(Decimal(10)**-digits, rounding=ROUND_HALF_UP),'f')


def scores(vals, scale=100, digits=2, lower=False):
    best=(min if lower else max)(v for v in vals if v is not None)
    return ' & '.join('--' if v is None else (r'\textbf{'+fmt(v,scale,digits)+'}' if v==best else fmt(v,scale,digits)) for v in vals)


def native_analysis():
    raw=json.loads((HERE/'revision_evidence_inputs.json').read_text())['native_control_score_rows']
    result={}
    for suite in ('ruler','qa'):
        rows=raw[suite];tasks=sorted({r['task'] for r in rows})
        point={a:mean(mean(r['scores'][a] for r in rows if r['task']==t) for t in tasks) for a in ('native','ncp')}
        rng=np.random.default_rng(20260917 if suite=='qa' else 20260916);draws=[]
        for t in tasks:
            selected=[r for r in rows if r['task']==t];clusters=list(dict.fromkeys(r['cluster'] for r in selected))
            counts=np.array([sum(r['cluster']==c for r in selected) for c in clusters])
            ix=rng.integers(len(clusters),size=(20000,len(clusters)))
            sums=np.array([sum(r['scores']['ncp']-r['scores']['native'] for r in selected if r['cluster']==c) for c in clusters])
            draws.append(sums[ix].sum(axis=1)/counts[ix].sum(axis=1))
        result[suite]={'scores':point,'estimate':point['ncp']-point['native'],
                       'ci95':np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist(),
                       'by_task':{t:{a:mean(r['scores'][a] for r in rows if r['task']==t) for a in point} for t in tasks}}
    close(result['ruler']['estimate'],D['ncp_confirmation']['ruler']['ncp_minus_native']['estimate'])
    lm=raw['lm'];docs=list(dict.fromkeys(r['document_id'] for r in lm))
    means={a+'_'+c:mean(mean(r['nll'] for r in lm if r['arm']==a and r['context']==c and r['document_id']==doc) for doc in docs) for a in ('native','ncp') for c in ('full','recent')}
    for k,v in means.items():close(v,D['ncp_lm']['metrics'][k]['estimate'])
    result['lm']={'means':means,'documents':len(docs),'windows':len({r['pair_id'] for r in lm})}
    import sys
    root=HERE.parent
    sys.path.insert(0,str(root/'runtime' if (root/'runtime/experiments').is_dir() else root.parent))
    from experiments.native_contrastive_proximal_20260915.tables import reference_fourier
    _,coefficients=reference_fourier();absolute_sum=float(np.sum(abs(coefficients)))
    assert absolute_sum<0.554403
    result['ncp_finite_series_convexity']={'modes':len(coefficients),'sum_absolute_coefficients':absolute_sum,'scalar_curvature_lower_bound':2.25-4*absolute_sum,'bound':'abs(log-curvature sinc-squared kernel)<=4, by the two analytic ranges in Appendix E.1'}
    result['scope']='Paired stored RULER scores and rescored native QA; question-weighted task means with source-cluster resampling; document-equal same-target NLL.'
    (HERE/'native_control_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def generate():
    rows=[r'\multicolumn{7}{l}{\emph{(a) Large-panel confirmation across the deployment window}} \\',
          r'Llama-8B & $8$K & 50 & 85.16 & 82.39 & -- & [1.29, 4.25] \\',
          r' & $16$K & 50 & 86.09 & 82.71 & -- & [1.53, 5.34] \\',
          r' & $32$K & 200 & 68.27 & 56.54 & -- & [10.32, 13.11] \\',
          r'OLMo & $16$K & 200 & 50.65 & 9.23 & -- & [39.99, 42.81] \\',
          r'\midrule',r'\multicolumn{7}{l}{\emph{(b) Matched direct baselines and scale transfer}} \\']
    for key,label,length in MODELS+[('llama70','Llama-70B NF4','32')]:
        vals=[full_score(key,a) for a in ARMS[:2]]+[full_score(key,'yarn') if key!='llama70' else None]
        rows.append(f'{label} & ${length}$K & 10 & '+scores(vals)+' & '+interval(full_ci(key,'mrpro'))+r' \\')
    table('table_clean_length_main.tex',
          r'\textbf{Quality across lengths, families and scale.} Clean Full-13 scores (\%) with one static $s=4$ table per model. '
          r'T/P/Y: TailSpline/MrPro/static YaRN. Scores are paired within each row; (a)/(b) retain their stated sample sizes, with small Llama/OLMo panels drawn as subsets. '
          r'Original RoPE scores $90.03\%$ at Llama $8$K. Intervals are paired $95\%$ T--P contrasts (pp).',
          'tab:clean-length-main','llrrrrl',r'Model & Length & $N$/task & T & P & Y & T--P interval',rows)
    cases=[('llama_qa','Llama-8B','Natural-QA',631,'macro_f1'),('olmo_qa','OLMo','Natural-QA',631,'macro_f1'),
           ('llama70_qa','Llama-70B NF4','Natural-QA',631,'macro_f1'),
           ('qwen_qa','Qwen','Book QA (I)',35,'qa_f1'),('glm_qa','GLM','Book QA (I)',35,'qa_f1'),
           ('glm_qa_second','GLM','Book QA (II)',77,'qa_f1')]
    rows=[]
    for key,label,bench,n,metric in cases:
        d=D[key];vals=[d['arms'][a][metric] if a in d['arms'] else None for a in ARMS]
        ci=(d['candidate_minus_baseline'] if key=='llama70_qa' else d['contrasts']['tailspline_minus_mrpro'])['ci95']
        rows.append(f'{label} & {bench} & {n} & '+scores(vals)+' & '+interval(ci)+r' \\')
        if key=='llama70_qa':
            rows.append(r'Llama-8B & LongBench v2 & 117 & \textbf{35.04} & 30.77 & -- & [$-3.39$, 11.97] \\')
    table('table_natural_transfer_main.tex',
          r'\textbf{Natural-task evaluation.} Scores (\%): five-task macro F1 for Natural-QA, accuracy for LongBench v2 and official F1 for InfiniteBench book QA. '
          r'I/II are separate book pools. Paired $95\%$ T--P intervals use source contexts (pp); T--Y and simultaneous intervals are in Appendix~\ref{sec:infinite-qa-new}.',
          'tab:natural-transfer-main','llrrrrl',r'Model & Benchmark & $N$ & T & P & Y & T--P interval',rows)
    nat=native_analysis();lm=D['ncp_lm']['metrics']
    rows=[]
    for key,label,length in MODELS:
        macro,ppl,_,_=normalized(key)
        if key in ('qwen','glm'):
            macro={a:mean(v['official'] for t,v in full_cell(key,a)['tasks'].items() if t.startswith('niah_')) for a in ARMS}
        rows.append(label+f' & ${length}$K & '+scores([macro[a] for a in ARMS])+' & '+scores([ppl[a] for a in ARMS],1,4,True)+r' \\')
    table('table_three_method_main.tex',
          r'\textbf{Retrieval and language modeling with static $s=4$ tables.} NIAH uses $200$/task for Llama/OLMo and the $10$/task Full-13 retrieval subset for Qwen/GLM. '
          r'PPL uses $46$ matched documents for Llama/OLMo and $5$ for Qwen/GLM; corpora and tokenizers remain model-specific. T/P/Y are defined in Table~\ref{tab:clean-length-main}.',
          'tab:three-method-main','llrrrrrr',r'Model & Length & \multicolumn{3}{c}{NIAH $\uparrow$ (\%)} & \multicolumn{3}{c}{PPL $\downarrow$} \\'+'\n'+r' & & T & P & Y & T & P & Y',rows)
    rows=[]
    for key,label,_ in MODELS:
        _,_,_,contrasts=normalized(key)
        for b,pretty in [('mrpro','P'),('yarn','Y')]:
            point,ci,nll,nllci=contrasts[b]
            ni=(f'{100*point:+.2f} & {interval(ci)}') if key in ('llama','olmo') else '-- & --'
            rows.append(f'{label} & T--{pretty} & {ni} & {nll:+.5f} & {interval(nllci,1,5)}'+r' \\')
    table('table_three_method_intervals.tex',
          r'Paired marginal $95\%$ contrasts. NIAH uses $200$/task; Qwen/GLM task uncertainty is reported for their primary Full-13 endpoint in Table~\ref{tab:full13-intervals}. '
          r'NLL intervals resample matched documents ($46$ for Llama/OLMo, $5$ for Qwen/GLM) and concern the mean document NLL difference.',
          'tab:three-method-intervals','llrlrl',r'Model & Contrast & NIAH (pp) & Interval & $\Delta$NLL & Interval',rows)
    tasks=list(normalized('llama')[2]);rows=[]
    for t in tasks:
        rows.append(t.replace('niah_','').replace('_',r'\_')+' & '+' & '.join(fmt(normalized(k)[2][t][a]) for k in ('llama','olmo') for a in ARMS)+r' \\')
    table('table_three_method_tasks_lo.tex',r'Complete large-panel NIAH scores (\%, $200$/task). Passkey is single\_1, already included in NIAH-8.',
          'tab:three-method-tasks-lo','lrrrrrr',r'Task & \multicolumn{3}{c}{Llama-8B} & \multicolumn{3}{c}{OLMo} \\'+'\n'+r' & T & P & Y & T & P & Y',rows)
    # Separate paired panels are not combined across different sample sizes.
    for keys,suffix in [(('llama','olmo'),'lo'),(('qwen','glm','llama70'),'qg')]:
        labels={'llama':'Llama-8B','olmo':'OLMo','qwen':'Qwen','glm':'GLM','llama70':'70B NF4'};rows=[]
        arms={k:ARMS if k!='llama70' else ARMS[:2] for k in keys}
        for t in sorted(full_cell(keys[0],'tailspline')['tasks']):
            vals=[full_cell(k,a)['tasks'][t]['official'] for k in keys for a in arms[k]]
            rows.append(t.replace('niah_','').replace('_',r'\_')+' & '+' & '.join(fmt(v) for v in vals)+r' \\')
        table('table_full13_tasks_'+suffix+'.tex',r'Complete paired Full-13 scores (\%, $10$/task). These are the direct-baseline panels in Table~\ref{tab:clean-length-main}.',
              'tab:full13-tasks-'+suffix,'l'+'r'*sum(map(len,arms.values())),
              r'Task & '+' & '.join(r'\multicolumn{'+str(len(arms[k]))+r'}{c}{'+labels[k]+'}' for k in keys)+r' \\'+'\n'+r' & '+' & '.join('T & P & Y' if k!='llama70' else 'T & P' for k in keys),rows)
    rows=[]
    for key,label,_ in MODELS+[('llama70','Llama-70B NF4','32')]:
        for b,pretty in [('mrpro','P')]+([('yarn','Y')] if key!='llama70' else []):
            rows.append(f'{label} & T--{pretty} & {100*(full_score(key,"tailspline")-full_score(key,b)):+.2f} & '+interval(full_ci(key,b))+r' \\')
    table('table_full13_intervals.tex',r'Full-13 paired task-stratified marginal $95\%$ intervals (pp), $10$/task. Each contrast retains its own frozen panel.',
          'tab:full13-intervals','llrl',r'Model & Contrast & Difference & Interval',rows)
    rows=[]
    for key,label,_,_,_ in cases:
        for b,pretty in [('mrpro','P')]+([('yarn','Y')] if key!='llama70_qa' else []):
            c=D[key]['candidate_minus_baseline'] if key=='llama70_qa' else D[key]['contrasts']['tailspline_minus_'+b]
            rows.append(f'{label}'+(' II' if key.endswith('second') else '')+f' & T--{pretty} & {100*c["estimate"]:+.2f} & '+interval(c['ci95'])+' & '+interval(c['familywise_ci95_bonferroni'])+r' \\')
    table('table_infinite_qa_intervals.tex',
          r'Natural-QA and book-QA contrasts (pp). Paired source-context bootstrap uses $20{,}000$ draws. '
          r'The last column gives Bonferroni simultaneous intervals for the two comparisons within each three-arm panel; the 70B panel has one comparison. II identifies GLM\textquotesingle s second book pool.',
          'tab:infinite-qa-intervals','llrll',r'Model & Contrast & Difference & Marginal $95\%$ & Simultaneous $95\%$',rows)
    rows=[]
    for name,label in [('recent','Recent history'),('full','Full history')]:
        vals=[lm[a+'_'+name]['estimate'] for a in ('native','ncp')];c=lm['delta_'+name]
        rows.append(label+' & '+' & '.join(fmt(v,1,6) for v in vals)+f' & {c["estimate"]:+.6f} & '+interval(c['ci95'],1,6)+r' \\')
    rows.append(r'Context benefit & '+fmt(lm['use_native']['estimate'],1,6)+' & '+fmt(lm['use_ncp']['estimate'],1,6)+f' & {lm["delta_use"]["estimate"]:+.6f} & '+interval(lm['delta_use']['ci95'],1,6)+r' \\')
    table('table_ncp_context.tex',r'Same-target NLL (nat/token), document-equal across $103$ documents and $128$ windows. Context benefit is recent-history minus full-history NLL; positive benefit differences favor NCP.',
          'tab:ncp-context','lrrrl',r'Context & Native & NCP & NCP--Native & $95\%$ interval',rows)
    rows=[]
    old=json.loads((HERE/'completed_evidence_inputs.json').read_text())['reports']['olmo_native_ncp']
    for t in nat['ruler']['by_task']:
        vals=[old['arm_task_scores'][a][t] for a in ('native','ncp')]+[nat['ruler']['by_task'][t][a] for a in ('native','ncp')]
        rows.append(t.replace('niah_','').replace('_',r'\_')+' & '+' & '.join(fmt(v) for v in vals)+r' \\')
    table('table_ncp_tasks_new.tex',r'Complete native-window Full-13 scores (\%). Original/new OLMo panels use $60/10$ inputs per task at $4$K. Every task stays in its primary mean.',
          'tab:ncp-tasks-new','lrrrr',r'Task & \multicolumn{2}{c}{Original ($60$/task)} & \multicolumn{2}{c}{New ($10$/task)} \\'+'\n'+r' & Native & NCP & Native & NCP',rows)
    rows=[]
    for t,values in nat['qa']['by_task'].items():
        rows.append(t.replace('_',r'\_')+' & '+' & '.join(fmt(values[a]) for a in ('native','ncp'))+r' \\')
    table('table_native_qa_tasks.tex',r'Native-QA99 task F1 (\%). The primary mean weights questions within task and the three tasks equally.',
          'tab:native-qa-tasks','lrr',r'Task & Native & NCP',rows)
    rows=[]
    for task in sorted(D['llama_qa']['arms']['tailspline']['by_task']):
        vals=[D[k]['arms'][a]['by_task'][task] for k in ('llama_qa','olmo_qa') for a in ARMS]
        rows.append(task.replace('_',r'\_')+' & '+' & '.join(fmt(v) for v in vals)+r' \\')
    table('table_natural_three_tasks.tex',r'Complete Natural-QA631 task scores (F1, \%). Each model uses the same question pool across T/P/Y.',
          'tab:natural-three-tasks','lrrrrrr',r'Task & \multicolumn{3}{c}{Llama-8B} & \multicolumn{3}{c}{OLMo} \\'+'\n'+r' & T & P & Y & T & P & Y',rows)


if __name__ == '__main__':
    verify()
    error=verify_connections()
    generate()
    result={'status':'PASS','paired_score_rows_reaggregated':3240,
            'document_nll_rows_reaggregated':97,
            'native_paired_prompt_rows':229,'native_lm_score_rows':512,
            'other_results':'report-backed; aggregate arithmetic checked; intervals retained',
            'model_execution':False,'raw_text_rescoring':'native QA99 only; other reports retain their original scoring',
            'fixed_state_log_odds_max_error':error,'turn_identity_public_grids':4,
            'llama_lora_ppl':'recomputed from domain NLL; 24/24 long-prefix packs favor Cosh'}
    (HERE/'revision_evidence_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
