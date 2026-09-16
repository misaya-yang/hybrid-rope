import json,numpy as np,collections,math
from pathlib import Path
root=Path(__file__).resolve().parent;d=json.loads((root/'inputs.json').read_text());out={'conditions':{},'comparison_notes':[]}
for name,v in d.items():
 if name=='configs':continue
 a=v['arms'];T=a['tailspline']['table'];P=a['mrpro']['table'];geom=T['model_geometry'];s=T['scale'];L=geom['native_length'];H=int(L*s)
 vals=lambda t:np.array(t.get('table',t)['values_float32'])
 t,p=vals(T),vals(P);mt,mp=np.array(T['exponents']),np.array(P['exponents']);native=t*s**mt
 active=np.flatnonzero(t!=p);l,h=T['band_envelope'];expectedq=np.arange(geom['pairs']);q=np.clip(expectedq-l,0,h-l);n=h-l
 emt=q*(3*n*n+3*n+1-q*q)/(n*(n+1)*(2*n+1));emp=q*(q+1)/(n*(n+1));assert np.max(abs(mt-emt))<2e-6;assert np.max(abs(mp-emp))<2e-6
 for arm in ['tailspline','mrpro']:
  assert np.array_equal(vals(a[arm]['table']),np.array(a[arm]['contract']['static_table']['values_float32']))
 rows={arm:{x['row_id']:x for x in a[arm]['rows'] if x['task'].startswith('niah_')} for arm in ['tailspline','mrpro']};ids=list(rows['tailspline']);assert set(ids)==set(rows['mrpro'])
 pairs=[]
 for i in ids:
  x,y=rows['tailspline'][i],rows['mrpro'][i];assert all(x[k]==y[k] for k in ['task','prompt_sha256','references','input_tokens'])
  pairs.append((x,y,x['ruler_official_score']-y['ruler_official_score']))
 tasks=sorted({x['task'] for x,_,_ in pairs});tb={task:{arm:float(np.mean([r['ruler_official_score'] for r in rows[arm].values() if r['task']==task])) for arm in rows} for task in tasks}
 health={arm:{f:sum(bool(z[f]) for z in rows[arm].values()) for f in ['empty','ended_eos','hit_cap']} for arm in rows}
 infos={'scale':s,'native_length':L,'target':H,'base':geom['base'],'band':[l,h],'transition_width':n,'different_TP_slots':len(active),'fixed_high_pairs':l+1,'fully_scaled_low_pairs':geom['pairs']-h,'max_TP_log_ratio':float(np.log(p/t).max()),'max_TP_wavelength_ratio':float((p/t).max()),'sum_m_difference':float((mt-mp).sum()),'max_target_TP_phase_gap':float(((H-1)*abs(t-p)).max()),'native_phase_span_at_changed_edges':list(map(float,(L-1)*native[active[[0,-1]]])),'rows_NIAH':len(pairs),'T_wins':sum(z>0 for _,_,z in pairs),'P_wins':sum(z<0 for _,_,z in pairs),'score_ties':sum(z==0 for _,_,z in pairs),'same_output_text':sum(x['output_sha256']==y['output_sha256'] for x,y,_ in pairs),'task_scores':tb,'score_T':float(np.mean([v['tailspline'] for v in tb.values()])),'score_P':float(np.mean([v['mrpro'] for v in tb.values()])),'health':health}
 infos['per_task_contribution_pp']={k:100*(v['tailspline']-v['mrpro'])/8 for k,v in tb.items()}
 out['conditions'][name]=infos
 print(name,json.dumps({k:v for k,v in infos.items() if k not in ['task_scores','health']},ensure_ascii=False));print('TASKS',tb);print('HEALTH',health)
# Same-scale geometry from public native grids, not additional model runs.
out['same_scale_geometry']={}
for cfg in ['llama','olmo','qwen3b','qwen1p5b']:
 c=d['configs'][cfg];K=c['hidden_size']//c['num_attention_heads']//2;L=c['max_position_embeddings'];w=c['rope_theta']**(-np.arange(K)/K);turn=w*L/(2*np.pi);l=np.flatnonzero(turn>32)[-1];h=np.flatnonzero(turn<1)[0];n=h-l;q=np.clip(np.arange(K)-l,0,n);mt=q*(3*n*n+3*n+1-q*q)/(n*(n+1)*(2*n+1));mp=q*(q+1)/(n*(n+1));t=w*4**(-mt);p=w*4**(-mp)
 out['same_scale_geometry'][cfg]={'band':[int(l),int(h)],'S4_max_wavelength_ratio':float((p/t).max()),'S4_max_target_phase_difference':float((L*4*abs(t-p)).max()),'S4_sum_m_difference':float((mt-mp).sum())}
print('SAME_SCALE',json.dumps(out['same_scale_geometry']))
# Qwen1.5B and3B share exact T and P grids at S4.
out['qwen_s4_same_installed_tables']={arm:a['table']['table_sha256_float32']==d['qwen25_1p5b_128k']['arms'][arm]['table']['table_sha256_float32'] for arm,a in d['qwen25_3b_128k']['arms'].items()}
# Fixable arithmetic counterfactual is diagnostic only, never a new reported metric.
q=out['conditions']['qwen25_3b_128k'];out['qwen3b_s4_one_binary_row_weight_pp']=100/40
(root/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
