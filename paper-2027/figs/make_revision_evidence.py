"""Verify recorded aggregates and render the September 16 evidence tables.

This script does not load models or rescore generated text. Paired score rows
are checked where present; remaining observations retain report-level identity.
"""
from pathlib import Path
import json
import math
from statistics import mean

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / 'revision_evidence_inputs.json').read_text())['reports']
ARMS = ('tailspline', 'mrpro', 'yarn')
MODELS = [('llama', 'Llama', '32'), ('olmo', 'OLMo', '16'),
          ('qwen', 'Qwen', '128'), ('glm', 'GLM', '128')]


def close(a, b):
    assert abs(a - b) < 1e-10, (a, b)


def normalized(key):
    d = D[key + '_quick']
    if key == 'qwen':
        by_task = {t: {a: d['arms'][a]['by_task'][t] for a in ARMS}
                   for t in d['arms']['tailspline']['by_task']}
        return ({a: d['arms'][a]['niah8_macro'] for a in ARMS},
                {a: d['arms'][a]['ppl'] for a in ARMS}, by_task,
                {b: (d['contrasts']['tailspline_minus_' + b]['niah8_macro'],
                     d['contrasts']['tailspline_minus_' + b]['niah8_ci95'],
                     d['contrasts']['tailspline_minus_' + b]['delta_nll'],
                     d['contrasts']['tailspline_minus_' + b]['delta_nll_ci95'])
                 for b in ('mrpro', 'yarn')})
    return (d['niah']['task_equal_macro'],
            {a: d['ppl']['pooled'][a]['ppl'] for a in ARMS},
            d['niah']['by_task'],
            {b: (d['comparisons']['tailspline_minus_' + b]['niah_macro_delta'],
                 d['comparisons']['tailspline_minus_' + b]['niah_task_stratified_bootstrap_ci95'],
                 d['comparisons']['tailspline_minus_' + b]['mean_document_nll_delta'],
                 d['comparisons']['tailspline_minus_' + b]['document_bootstrap_ci95'])
             for b in ('mrpro', 'yarn')})


def verify():
    for key, _, _ in MODELS:
        macro, ppl, tasks, contrasts = normalized(key)
        assert len(tasks) == 8
        for arm in ARMS:
            close(mean(v[arm] for v in tasks.values()), macro[arm])
        if key != 'qwen':
            d = D[key + '_quick']
            rows = d['niah']['paired_rows']
            assert len(rows) == len({r['prompt_sha256'] for r in rows}) == 40
            docs = d['ppl']['paired_documents']
            assert len(docs) == 5
            assert len({d['ppl']['pooled'][a]['target_tokens'] for a in ARMS}) == 1
            for arm in ARMS:
                for task in tasks:
                    vals = [r['scores'][arm] for r in rows if r['task'] == task]
                    assert len(vals) == 5
                    close(mean(vals), tasks[task][arm])
                close(math.exp(d['ppl']['pooled'][arm]['nll']), ppl[arm])
            for base in ('mrpro', 'yarn'):
                close(mean(r['mean_nll']['tailspline'] - r['mean_nll'][base]
                           for r in docs), contrasts[base][2])
        for base in ('mrpro', 'yarn'):
            close(macro['tailspline'] - macro[base], contrasts[base][0])
    glm = D['glm_full13']
    assert glm['paired_prompts'] == 65
    for arm in ('tailspline', 'mrpro'):
        d = glm['summaries'][arm]['by_length']['131072']
        assert all(t['rows'] == 5 for t in d['tasks'].values())
        close(mean(t['official'] for t in d['tasks'].values()), d['task_macro_official'])
    for key in ('qwen_qa', 'glm_qa'):
        d = D[key]
        assert d['rows_per_arm'] == 35 and d['source_context_clusters'] == 7
        for base in ('mrpro', 'yarn'):
            close(d['arms']['tailspline']['qa_f1'] - d['arms'][base]['qa_f1'],
                  d['contrasts']['tailspline_minus_' + base]['estimate'])
    for key in ('llama_extreme_qa', 'llama_extreme_dia'):
        d = D[key]['overall']
        close(d['tailspline'] - d['mrpro'], d['delta_tailspline_minus_mrpro'])


def table(name, caption, label, columns, header, rows):
    text = '\n'.join([r'\begin{table}[ht]', r'\centering\small',
        '\\caption{' + caption + '}', '\\label{' + label + '}',
        '\\begin{tabular}{@{}' + columns + '@{}}', r'\toprule',
        header + r' \\', r'\midrule', *rows,
        r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    (HERE.parent / 'tables' / name).write_text(text)


def interval(v, scale=100, digits=2):
    return f'[{scale*v[0]:.{digits}f}, {scale*v[1]:.{digits}f}]'


def generate():
    rows = []
    for key, label, length in MODELS:
        macro, ppl, _, _ = normalized(key)
        values = []
        for vals, factor, digits, best in [(macro, 100, 2, max(macro.values())),
                                            (ppl, 1, 4, min(ppl.values()))]:
            for a in ARMS:
                s = f'{factor*vals[a]:.{digits}f}'
                values.append('\\textbf{' + s + '}' if vals[a] == best else s)
        rows.append(label + ' & $' + length + r'$K & ' + ' & '.join(values) + r' \\')
    table('table_three_method_main.tex',
          r'\textbf{Direct training-free comparison.} Static $s=4$ tables; '
          r'NIAH is the eight-task macro score (\%, $5$/task), and PPL uses '
          r'$5$ matched documents per model. T/P/Y: TailSpline/MrPro/official static YaRN. '
          r'PPL corpora differ across models; compare methods within each row. '
          r'Intervals and task scores are in Appendix~\ref{sec:three-method-complete}.',
          'tab:three-method-main', 'llrrrrrr',
          r'Model & Length & \multicolumn{3}{c}{NIAH $\uparrow$} & \multicolumn{3}{c}{PPL $\downarrow$} \\'
          '\n' + r' & & T & P & Y & T & P & Y', rows)
    rows = []
    for key, label, _ in MODELS:
        _, _, _, contrasts = normalized(key)
        for b, pretty in [('mrpro', 'P'), ('yarn', 'Y')]:
            point, ci, nll, nllci = contrasts[b]
            rows.append(f'{label} & T--{pretty} & {100*point:+.2f} & {interval(ci)} & '
                        f'{nll:+.5f} & {interval(nllci,1,5)}' + r' \\')
    table('table_three_method_intervals.tex',
          r'Paired contrasts for Table~\ref{tab:three-method-main}. NIAH intervals '
          r'resample within task; NLL intervals resample the five documents. '
          r'All intervals are marginal $95\%$ intervals.',
          'tab:three-method-intervals', 'llrlrl',
          r'Model & Contrast & NIAH (pp) & Interval & $\Delta$NLL & Interval', rows)
    tasks = list(normalized('llama')[2])
    for keys, suffix in [(('llama','olmo'),'lo'), (('qwen','glm'),'qg')]:
        rows=[]
        for t in tasks:
            vals=[normalized(k)[2][t][a]*100 for k in keys for a in ARMS]
            rows.append(t.replace('niah_','').replace('_',r'\_')+' & '+
                        ' & '.join(f'{v:.2f}' for v in vals)+r' \\')
        table('table_three_method_tasks_'+suffix+'.tex',
              r'Complete NIAH task scores (\%) for '+keys[0].capitalize()+' and '+keys[1].capitalize()+'. '
              r'Passkey is single\_1, already included in the eight-task mean.',
              'tab:three-method-tasks-'+suffix, 'lrrrrrr',
              r'Task & \multicolumn{3}{c}{'+keys[0].capitalize()+r'} & \multicolumn{3}{c}{'+keys[1].capitalize()+r'} \\'+'\n'+r' & T & P & Y & T & P & Y', rows)
    g=D['glm_full13']['summaries'];rows=[]
    for t in D['glm_full13']['tasks']:
        vals=[g[a]['by_length']['131072']['tasks'][t]['official']*100 for a in ('tailspline','mrpro')]
        rows.append(t.replace('_',r'\_')+' & '+f'{vals[0]:.2f} & {vals[1]:.2f} & {vals[0]-vals[1]:+.2f}'+r' \\')
    table('table_glm_full13.tex','GLM Full-13 at $128$K, $5$ inputs/task. Scores are percentages.',
          'tab:glm-full13','lrrr',r'Task & T & P & Difference (pp)', rows)
    rows=[]
    for key,label in [('qwen_qa','Qwen'),('glm_qa','GLM')]:
        d=D[key]
        for b,pretty in [('mrpro','P'),('yarn','Y')]:
            c=d['contrasts']['tailspline_minus_'+b]
            rows.append(f'{label} & T--{pretty} & {100*c["estimate"]:+.2f} & '+interval(c['ci95'])+' & '+interval(c['familywise_ci95_bonferroni'])+r' \\')
    table('table_infinite_qa_intervals.tex',
          r'InfiniteBench En.QA contrasts (pp). Paired source-context bootstrap '
          r'uses $20{,}000$ draws; the final column controls the two T--P/T--Y comparisons '
          r'within each model by Bonferroni adjustment.',
          'tab:infinite-qa-intervals','llrll',
          r'Model & Contrast & Difference & Marginal $95\%$ & Simultaneous $95\%$',rows)


if __name__ == '__main__':
    verify()
    generate()
    result={'status':'PASS','paired_score_rows_reaggregated':120,
            'document_nll_rows_reaggregated':15,
            'other_results':'report-backed; aggregate arithmetic checked',
            'model_execution':False,'raw_text_rescoring':False}
    (HERE/'revision_evidence_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
