from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_run(run,out):
    run,out=Path(run),Path(out);out.mkdir(parents=True,exist_ok=True);files=[]
    if (run/'reliability.csv').exists():
        data=pd.read_csv(run/'reliability.csv');fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
        ax.plot([0,1],[0,1],'--',color='gray');ax.plot(data.predicted,data.observed,'o-')
        ax.set(xlabel='Predicted occurrence probability',ylabel='Observed positive frequency',title='Occurrence calibration')
        path=out/'reliability.png';fig.savefig(path,dpi=180);plt.close(fig);files.append(str(path))
    if (run/'by_horizon.csv').exists():
        data=pd.read_csv(run/'by_horizon.csv')
        if len(data):
            fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
            for ax,col in zip(axes,['NLL','DRPS']):ax.plot(data.horizon,data[col]);ax.set(xlabel='Forecast horizon',ylabel=col)
            path=out/'horizon_scores.png';fig.savefig(path,dpi=180);plt.close(fig);files.append(str(path))
    if (run/'propagation_by_horizon.csv').exists():
        data=pd.read_csv(run/'propagation_by_horizon.csv');fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
        ax.plot(data.horizon,data.mean_occurrence_gap);ax.set(xlabel='Horizon',ylabel='Mean absolute occurrence gap')
        path=out/'propagation_gap.png';fig.savefig(path,dpi=180);plt.close(fig);files.append(str(path))
    shards=sorted((run/'predictions').glob('part_*.npz'))
    if shards:
        import numpy as np
        from rebunet.evaluation.probability import CountMixture
        a=np.load(shards[0],allow_pickle=False)
        H=a['y'].shape[1];days=np.arange(1,H+1)
        fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
        observed=np.where(a['mask'][0],a['y'][0],np.nan)
        ax.plot(days,observed,'o-',label='Observed usable count');ax.plot(days,a['point'][0],label='Predictive mean')
        if 'w' in a:
            mix=CountMixture(a['p'][0],a['w'][0],a['mu'][0],a['k'][0],str(a['kind']))
            ax.fill_between(days,mix.quantile(.1),mix.quantile(.9),alpha=.2,label='80% marginal interval')
        ax.set(xlabel='Forecast horizon',ylabel='Count',title='One forecast origin (not cumulative uncertainty)');ax.legend()
        path=out/'forecast_example.png';fig.savefig(path,dpi=180);plt.close(fig);files.append(str(path))
        if 'w' in a:
            mix=CountMixture(a['p'][0,:1],a['w'][0,:1],a['mu'][0,:1],a['k'][0,:1],str(a['kind']))
            grid=np.arange(min(100,int(mix.quantile(.999)[0]))+1)
            cdf=mix.cdf(grid[:,None])[:,0];pmf=np.diff(np.r_[0,cdf])
            fig,ax=plt.subplots(figsize=(6,3.5),layout='constrained');ax.bar(grid,pmf)
            ax.set(xlabel='Count',ylabel='Probability',title='First-horizon predictive marginal')
            path=out/'marginal_example.png';fig.savefig(path,dpi=180);plt.close(fig);files.append(str(path))
    return {'plots':files}
