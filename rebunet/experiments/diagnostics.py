import copy
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rebunet.models.registry import build
from rebunet.data.panel import Panel,Windows
from rebunet.training.runner import loader
from rebunet.utils import move,write_json,environment


def load_model(checkpoint,device='cpu'):
    old=torch.load(checkpoint,map_location=device,weights_only=False)
    cfg=old['config'];panel=Panel(cfg['data']['path'])
    if panel.fingerprint!=old['dataset_fingerprint']:raise ValueError('Checkpoint/dataset mismatch')
    model=build(cfg,panel.dimensions()).to(device);model.load_state_dict(old['model']);model.eval()
    if hasattr(model,'warmup'):model.warmup(False)
    return model,panel,cfg


def propagation(checkpoint,out,device='cpu'):
    model,panel,cfg=load_model(checkpoint,device)
    if cfg['model']['name']!='rebunet' or cfg['model']['state'] not in ['scalar','ead','cad']:raise ValueError('Need a compatible ReBuNet checkpoint')
    torch.set_num_threads(cfg['train'].get('threads',4))
    gaps=[];equivalence=[];rows=[];H=cfg['model']['horizon']
    # A common sufficient cap, including the M5 maximum reachable age.
    model.cap=cfg['model']['lookback']+H-1
    with torch.no_grad():
        for raw in loader(Windows(panel,cfg,'test'),cfg):
            b=move(raw,device);scalar,rs=model(b,'scalar',True);exact,re=model(b,'ead',True);capped,rc=model(b,'cad',True)
            diff=(scalar.p-exact.p).abs().cpu().numpy();gaps.append(diff)
            equivalence.append(float((exact.p-capped.p).abs().max()))
            if not torch.allclose(exact.p,capped.p,atol=2e-7,rtol=2e-7):raise AssertionError('CAD/EAD common-weight equivalence failed')
            if not torch.allclose(exact.mean(),capped.mean(),atol=2e-6,rtol=2e-6):raise AssertionError('CAD/EAD mark equality failed')
            rows.append((rs-re).abs().cpu().numpy())
    diff=np.concatenate(gaps);age=np.concatenate(rows)
    Path(out).mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'horizon':np.arange(1,H+1),'mean_occurrence_gap':diff.mean(0),'max_occurrence_gap':diff.max(0),
                  'mean_age_gap':age.mean(0)}).to_csv(Path(out)/'propagation_by_horizon.csv',index=False)
    result={'fixed_parameter_mean_absolute_occurrence_gap':float(diff.mean()),'max_gap':float(diff.max()),
            'CAD_EAD_max_gap':max(equivalence),'equivalence_tolerance_fp32_hazard':2e-7,'common_cap':model.cap,'parameter_source':str(checkpoint),
            'averaging':'all eligible series, origins and horizons; no target mask used'}
    write_json(Path(out)/'propagation.json',result);return result


def profile(checkpoint,out,device='cpu',repeats=30,warmup=5,batch_size=256):
    model,panel,cfg=load_model(checkpoint,device);cfg=copy.deepcopy(cfg);cfg['train']['batch_size']=batch_size
    torch.set_num_threads(cfg['train'].get('threads',4));raw=next(iter(loader(Windows(panel,cfg,'test'),cfg)))
    # Repeat an observed batch only for speed measurements, never for accuracy metrics.
    n=raw['history'].shape[0]
    raw={k:v.repeat((int(np.ceil(batch_size/n)),)+(1,)*(v.ndim-1))[:batch_size] for k,v in raw.items()}
    b=move(raw,device);is_cuda=str(device).startswith('cuda')
    def sync():
        if is_cuda:torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):model(b)
        sync()
        if is_cuda:torch.cuda.reset_peak_memory_stats()
        elapsed=[]
        for _ in range(repeats):
            sync();start=time.perf_counter();model(b);sync();elapsed.append(1000*(time.perf_counter()-start))
    result={'params':sum(p.numel() for p in model.parameters()),'median_latency_ms':float(np.median(elapsed)),
            'mean_latency_ms':float(np.mean(elapsed)),'p95_latency_ms':float(np.quantile(elapsed,.95)),
            'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9 if is_cuda else None,
            'peak_reserved_GB':torch.cuda.max_memory_reserved()/1e9 if is_cuda else None,
            'batch_size':batch_size,'repeats':repeats,'warmup':warmup,'mode':'eval_no_grad_model_only',
            'environment':environment(),'includes_numerical_inverse':True,'includes_data_loading':False}
    write_json(out,result);return result
