"""Declarative paper-to-experiment mapping. Missing external baselines stay visible."""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
from rebunet.models.registry import CLASSICAL,EXTERNAL
from rebunet.utils import merge,write_json
from rebunet.training.runner import run


def variants():
    specs=[]
    def add(label,changes,tables):specs.append({'label':label,'changes':changes,'tables':tables})
    add('full',{},['T3','T4','T5','T6','T7','T8','T11','D2','D3','sparsity','burstiness'])
    for name in CLASSICAL+['dlinear','patchtst','itransformer','deepar','robust_rnn','transformer_idf','deep_renewal','switch_hurdle','deep_state','score_hurdle_nb']:
        tables=['T4'] if name in ['deep_state','score_hurdle_nb'] else ['T3']
        if name in ['deepar','robust_rnn','deep_renewal','switch_hurdle']:tables+=['T4']
        add(name,{'model':{'name':name}},tables)
    for state in ['direct','grs','ead','cad']:
        add('state_'+state,{'model':{'state':state}},['T5','T8']+(['T7'] if state=='direct' else ['T11'] if state=='ead' else ['sparsity'] if state=='grs' else []))
    for mark in ['single','unordered','raw_ordered','shifted_nb','ztpoisson','dlognormal']:
        tables=['T6'] if mark in ['single','unordered','raw_ordered'] else []
        if mark in ['single','raw_ordered','shifted_nb']:tables+=['T7']
        if mark in ['single','ztpoisson','dlognormal']:tables+=['D2']
        if mark=='single':tables+=['burstiness']
        if mark=='shifted_nb':tables+=['T11']
        add('mark_'+mark,{'model':{'mark':mark}},tables)
    add('no_event_stream',{'model':{'use_events':False}},['T7'])
    for l in [14,28,56,84,112]:
        for state in ['scalar','grs']:add(f'history_{l}_{state}',{'model':{'lookback':l,'state':state}},['T10'])
    for cap in [32,64,128]:add(f'cap_{cap}',{'model':{'state':'cad','cap':cap}},['D3'])
    for state in ['fixed_age','no_feedback','detached']:add('extra_'+state,{'model':{'state':state}},['EXTRA-mechanism'])
    return specs


def plan(cfg,seeds,scope='paper'):
    jobs=[]
    for spec in variants():
        if scope=='paper' and spec['label'].startswith('extra_'):continue
        if scope=='smoke' and spec['label'] not in ['full','state_direct','state_grs','state_ead','state_cad','mark_single','mark_unordered','mark_raw_ordered','mark_shifted_nb','no_event_stream']:continue
        for seed in ([seeds[0]] if spec['label'] in CLASSICAL else seeds):
            job=merge(cfg,spec['changes']);job['train']['seed']=seed
            jobs.append({'label':spec['label'],'seed':seed,'tables':spec['tables'],'config':job,
                         'requires_external':job['model']['name'] in EXTERNAL and not job.get('external',{}).get(job['model']['name'],{}).get('factory')})
    return jobs


def suite(cfg,out,seeds=(0,1,2,3,4),scope='paper',execute=False,allow_missing_external=False,labels=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);jobs=plan(cfg,seeds,scope)
    if labels:
        unknown=set(labels)-{j['label'] for j in jobs}
        if unknown:raise ValueError('Unknown labels: '+','.join(sorted(unknown)))
        jobs=[j for j in jobs if j['label'] in labels]
    write_json(out/'plan.json',jobs)
    missing=[f"{x['label']}/seed_{x['seed']}" for x in jobs if x['requires_external'] and not (out/x['label']/f"seed_{x['seed']}"/'metrics.json').exists()]
    write_json(out/'coverage.json',{'jobs':len(jobs),'missing_external':missing,'complete':False,'mode':scope})
    if not execute:return {'jobs':len(jobs),'missing_external':missing,'plan':str(out/'plan.json')}
    if missing and not allow_missing_external:raise RuntimeError('External baseline outputs/adapters missing. See coverage.json. --allow-missing-external explicitly runs a PARTIAL suite.')
    from rebunet.data.panel import Panel
    current_fingerprint=Panel(cfg['data']['path'],verify=True).fingerprint
    statuses=[]
    for job in jobs:
        path=out/job['label']/f"seed_{job['seed']}"
        if (path/'metrics.json').exists():
            existing=json.loads((path/'config.json').read_text())
            if existing!=job['config']:raise ValueError(f'Stale config at {path}; use a new output directory')
            provenance=json.loads((path/'provenance.json').read_text())
            if provenance.get('data_sha256')!=current_fingerprint:raise ValueError(f'Stale dataset at {path}')
            status='complete_existing'
        elif job['requires_external']:status='missing_external'
        else:
            run(job['config'],path,resume=(path/'last.pt').exists());status='complete'
        if status.startswith('complete') and (path/'best.pt').exists():
            from rebunet.experiments.diagnostics import profile,propagation
            import torch
            requested=job['config']['train']['device']
            device=('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested
            if ('T8' in job['tables'] or 'D3' in job['tables']) and not (path/'profile.json').exists():
                profile(path/'best.pt',path/'profile.json',device,repeats=10 if scope=='smoke' else 30,
                        batch_size=job['config']['train']['batch_size'])
            if job['label']=='full' and not (path/'propagation.json').exists():
                propagation(path/'best.pt',path,device)
        statuses.append({'label':job['label'],'seed':job['seed'],'status':status})
        write_json(out/'coverage.json',{'jobs':len(jobs),'completed':statuses,'missing_external':missing,
                                      'complete':len(statuses)==len(jobs) and all(s['status'].startswith('complete') for s in statuses),
                                      'author_exact_reproduction':False})
    return summarize(out)


def summarize(out):
    out=Path(out);rows=[]
    jobs=json.loads((out/'plan.json').read_text())
    for job in jobs:
        path=out/job['label']/f"seed_{job['seed']}";metric=path/'metrics.json'
        if metric.exists():
            values=json.loads(metric.read_text());row={'variant':job['label'],'seed':job['seed'],'tables':','.join(job['tables'])}
            row.update({k:v for k,v in values.items() if not isinstance(v,(dict,list))});rows.append(row)
    if not rows:return {'completed':0}
    df=pd.DataFrame(rows);df.to_csv(out/'all_runs.csv',index=False)
    columns=[c for c in ['MAE','RMSSE','bWRMSSE','NLL','DRPS','AUPRC','Burst_MAE','Brier'] if c in df]
    summary=df.groupby('variant')[columns].agg(['mean','std','count']);summary.to_csv(out/'summary.csv')
    efficiency=[];propagation_rows=[]
    for job in jobs:
        folder=out/job['label']/f"seed_{job['seed']}"
        for filename,container in [('profile.json',efficiency),('propagation.json',propagation_rows)]:
            if (folder/filename).exists():
                values=json.loads((folder/filename).read_text())
                container.append({'variant':job['label'],'seed':job['seed'],**{k:v for k,v in values.items() if not isinstance(v,dict)}})
    if efficiency:pd.DataFrame(efficiency).to_csv(out/'T8_efficiency.csv',index=False)
    if propagation_rows:pd.DataFrame(propagation_rows).to_csv(out/'T11_propagation.csv',index=False)
    write_json(out/'T9_configurations.json',[{'variant':j['label'],'seed':j['seed'],'config':j['config']} for j in jobs])
    for table in sorted(set(t for j in jobs for t in j['tables'])):
        selected=[j['label'] for j in jobs if table in j['tables']]
        df[df.variant.isin(selected)].to_csv(out/(table+'_runs.csv'),index=False)
    # Training-defined strata comparisons with explicit baseline and denominator.
    strata=[]
    for job in jobs:
        if job['label'] not in ['full','state_grs','mark_single']:continue
        path=out/job['label']/f"seed_{job['seed']}"/'strata.csv'
        if path.exists():
            a=pd.read_csv(path);a['variant']=job['label'];a['seed']=job['seed'];strata.append(a)
    if strata:
        strata_df=pd.concat(strata,ignore_index=True)
        strata_df.to_csv(out/'strata_runs.csv',index=False)
        gains=[]
        for kind,baseline in [('adi_quartile','state_grs'),('burst_quartile','mark_single')]:
            keys=['seed','stratification','quartile']
            full=strata_df[(strata_df.variant=='full')&(strata_df.stratification==kind)]
            ref=strata_df[(strata_df.variant==baseline)&(strata_df.stratification==kind)]
            matched=full.merge(ref,on=keys,suffixes=('_full','_baseline'),validate='one_to_one')
            if len(matched):
                if not (matched.targets_full==matched.targets_baseline).all():
                    raise ValueError('Stratified comparison has unequal target coverage')
                matched['relative_NLL_improvement_percent']=np.where(matched.NLL_baseline>0,
                    100*(matched.NLL_baseline-matched.NLL_full)/matched.NLL_baseline,np.nan)
                matched['baseline']=baseline;gains.append(matched)
        if gains:
            gain_df=pd.concat(gains,ignore_index=True)
            gain_df.to_csv(out/'strata_relative_improvement.csv',index=False)
            gain_df.groupby(['stratification','quartile','baseline']).relative_NLL_improvement_percent.agg(['mean','std','count']).to_csv(out/'strata_improvement_summary.csv')
    write_json(out/'summary_status.json',{'completed_runs':len(df),'expected_runs':len(jobs),'numerically_matches_paper':'not asserted'})
    return {'completed':len(df),'expected':len(jobs),'summary':str(out/'summary.csv')}
