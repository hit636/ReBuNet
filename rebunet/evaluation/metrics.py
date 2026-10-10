from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from rebunet.evaluation.probability import CountMixture
from rebunet.utils import write_json


def evaluate_shards(paths,stats,cfg,out):
    """Accumulate per-series errors before RMSSE; zero-scale exclusions are explicit."""
    n=len(stats);cnt=np.zeros(n);abs_error=np.zeros(n);square=np.zeros(n)
    nll_sum=np.zeros(n);drps_sum=np.zeros(n);prob_count=np.zeros(n)
    burst_sum=np.zeros(n);burst_count=np.zeros(n)
    labels=[];probabilities=[];lead={};calibration={};max_bound=0.;max_support=0
    probabilistic=None
    for path in paths:
        a=np.load(path,allow_pickle=False);y=a['y'];mask=a['mask'].astype(bool);point=a['point']
        if not np.isfinite(point[mask]).all() or (point[mask]<0).any():raise ValueError('Invalid point forecasts')
        series=np.broadcast_to(a['series'][:,None],y.shape);horizon=np.broadcast_to(np.arange(1,y.shape[1]+1),y.shape)
        ids=series[mask];truth=y[mask];pred=point[mask];err=pred-truth
        np.add.at(cnt,ids,1);np.add.at(abs_error,ids,np.abs(err));np.add.at(square,ids,err**2)
        large=(truth>stats.burst_threshold.to_numpy()[ids])&(truth>0)
        np.add.at(burst_sum,ids[large],np.abs(err[large]));np.add.at(burst_count,ids[large],1)
        has_dist='w' in a
        if probabilistic is not None and probabilistic!=has_dist:raise ValueError('Mixed point-only/distribution shards')
        probabilistic=has_dist
        if has_dist and mask.any():
            mix=CountMixture(a['p'][mask],a['w'][mask],a['mu'][mask],a['k'][mask],str(a['kind']))
            logp=mix.log_prob(truth)
            if not np.isfinite(logp).all():raise FloatingPointError('Nonfinite observed marginal log probability')
            drps,bound,support=mix.drps(truth,cfg['metrics']['tail_tolerance'],cfg['metrics']['max_support'],cfg['metrics']['support_chunk'])
            occurrence=1-mix.cdf(np.zeros(len(truth)))
            labels.append((truth>0).astype('int8'));probabilities.append(occurrence)
            np.add.at(nll_sum,ids,-logp);np.add.at(drps_sum,ids,drps);np.add.at(prob_count,ids,1)
            max_bound=max(max_bound,float(bound.max()));max_support=max(max_support,support)
            for hh in np.unique(horizon[mask]):
                selected=horizon[mask]==hh;row=lead.setdefault(int(hh),np.zeros(4))
                row+=np.array([selected.sum(),-logp[selected].sum(),drps[selected].sum(),np.abs(err[selected]).sum()])
            for level in cfg['metrics'].get('intervals',[]):
                low=mix.quantile((1-level)/2);high=mix.quantile((1+level)/2)
                row=calibration.setdefault(str(level),np.zeros(3))
                row+=np.array([len(truth),((truth>=low)&(truth<=high)).sum(),(high-low).sum()])
    if cnt.sum()==0:raise ValueError('No usable evaluation targets')
    valid=cnt>0;scale=stats.scale2.to_numpy();scaled=valid&np.isfinite(scale)&(scale>0)
    mae=np.divide(abs_error,cnt,out=np.full(n,np.nan),where=valid)
    rmsse=np.sqrt(np.divide(square,cnt*scale,out=np.full(n,np.nan),where=scaled))
    weights=stats.weight.to_numpy()[scaled]
    result={'MAE':float(np.nanmean(mae)),'RMSSE':float(np.nanmean(rmsse)),
            'bWRMSSE':float(np.dot(weights,rmsse[scaled])/weights.sum()) if weights.sum()>0 else None,
            'Burst_MAE':float(burst_sum.sum()/burst_count.sum()) if burst_count.sum() else None,
            'usable_targets':int(cnt.sum()),'series_evaluated':int(valid.sum()),'scale_excluded_series':int((valid&~scaled).sum()),
            'burst_targets':int(burst_count.sum()),'probabilistic':bool(probabilistic)}
    if probabilistic:
        target=np.concatenate(labels);prob=np.concatenate(probabilities)
        result.update(NLL=float(nll_sum.sum()/prob_count.sum()),DRPS=float(drps_sum.sum()/prob_count.sum()),
                      AUPRC=float(average_precision_score(target,prob)) if target.any() else None,
                      Brier=float(np.mean((prob-target)**2)),DRPS_max_omitted_bound=max_bound,DRPS_max_support=max_support)
        bins=np.linspace(0,1,11);rows=[]
        for j in range(10):
            selected=(prob>=bins[j])&((prob<bins[j+1]) if j<9 else (prob<=1))
            rows.append({'bin':j,'count':int(selected.sum()),'predicted':float(prob[selected].mean()) if selected.any() else np.nan,
                         'observed':float(target[selected].mean()) if selected.any() else np.nan})
        pd.DataFrame(rows).to_csv(Path(out)/'reliability.csv',index=False)
        result['intervals']={k:{'coverage':float(v[1]/v[0]),'mean_width':float(v[2]/v[0])} for k,v in calibration.items()}
    per_series=stats.copy()
    for key,value in [('n',cnt),('MAE',mae),('RMSSE',rmsse),('NLL_sum',nll_sum),('DRPS_sum',drps_sum),('prob_count',prob_count),('Burst_abs_sum',burst_sum),('Burst_count',burst_count)]:per_series[key]=value
    per_series.to_csv(Path(out)/'per_series.csv',index=False)
    groups=[]
    for col in ['adi_quartile','burst_quartile']:
        for group in range(1,5):
            selected=(stats[col].to_numpy()==group)&valid;pc=prob_count[selected].sum()
            groups.append({'stratification':col,'quartile':group,'series':int(selected.sum()),'targets':int(cnt[selected].sum()),
                          'NLL':float(nll_sum[selected].sum()/pc) if pc else None,
                          'MAE':float(np.nanmean(mae[selected])) if selected.any() else None})
    pd.DataFrame(groups).to_csv(Path(out)/'strata.csv',index=False)
    pd.DataFrame([{'horizon':h,'count':v[0],'NLL':v[1]/v[0],'DRPS':v[2]/v[0],'MAE':v[3]/v[0]} for h,v in sorted(lead.items())],columns=['horizon','count','NLL','DRPS','MAE']).to_csv(Path(out)/'by_horizon.csv',index=False)
    write_json(Path(out)/'metrics.json',result)
    return result
