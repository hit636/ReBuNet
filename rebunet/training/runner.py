from __future__ import annotations
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,RandomSampler
from rebunet.data.panel import Panel,Windows,training_statistics,resolve_split
from rebunet.models.registry import build,CLASSICAL
from rebunet.models.distributions import masked_nll
from rebunet.baselines.classical import forecast as classical_forecast
from rebunet.evaluation.metrics import evaluate_shards
from rebunet.utils import write_json,environment,seed_all,move,sha256


def loader(ds,cfg,training=False,epoch=0):
    c=cfg['train'];generator=torch.Generator().manual_seed(c['seed']+epoch*100003)
    sampler=RandomSampler(ds,replacement=True,num_samples=c['steps_per_epoch']*c['batch_size'],generator=generator) if training and c.get('steps_per_epoch') else None
    return DataLoader(ds,batch_size=c['batch_size'],shuffle=training and sampler is None,sampler=sampler,
                      num_workers=c.get('num_workers',0),generator=generator,pin_memory=torch.cuda.is_available())


def validation(model,dl,device,stats,cfg):
    model.eval();numer=0.;denom=0.;sse=np.zeros(len(stats));counts=np.zeros(len(stats))
    # Repeatable validation paths for autoregressive Monte Carlo baselines.
    devices=[torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg['train']['seed']+99991)
        with torch.no_grad():
            for batch in dl:
                b=move(batch,device);out=model(b);m=b['mask']
                if getattr(model,'probabilistic',True):
                    numer+=float((-out.log_prob(b['y']))[m].sum());denom+=int(m.sum())
                else:
                    error=((out-b['y'])**2*m).sum(1).cpu().numpy();ids=b['series'].cpu().numpy()
                    np.add.at(sse,ids,error);np.add.at(counts,ids,m.sum(1).cpu().numpy())
    if getattr(model,'probabilistic',True):return numer/max(denom,1)
    scale=stats.scale2.to_numpy();use=(counts>0)&np.isfinite(scale)&(scale>0)
    scores=np.sqrt(sse[use]/counts[use]/scale[use]);weights=stats.validation_weight.to_numpy()[use]
    return float(np.average(scores,weights=weights)) if cfg['data'].get('remove_leading_zeros') else float(np.mean(scores))


def save_checkpoint(path,model,opt,epoch,best,stale,cfg,fingerprint):
    value={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'stale':stale,
           'config':cfg,'dataset_fingerprint':fingerprint,'rng_torch':torch.get_rng_state(),
           'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
           'rng_numpy':np.random.get_state(),'rng_python':random.getstate()}
    temporary=Path(str(path)+'.tmp');torch.save(value,temporary);temporary.replace(path)


def train_model(cfg,panel,out,stats,resume=False):
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if cfg['train']['device']=='auto' else cfg['train']['device'])
    model=build(cfg,panel.dimensions()).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['train']['learning_rate'])
    best=float('inf');stale=0;start=0;history=[]
    if resume and (out/'last.pt').exists():
        old=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if old['config']!=cfg or old['dataset_fingerprint']!=panel.fingerprint:raise ValueError('Resume config/data mismatch')
        model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);start=old['epoch']+1;best=old['best'];stale=old['stale']
        torch.set_rng_state(old['rng_torch'].cpu());np.random.set_state(old['rng_numpy']);random.setstate(old['rng_python'])
        if device.type=='cuda' and old['rng_cuda'] is not None:torch.cuda.set_rng_state_all(old['rng_cuda'])
        history=json.loads((out/'training.json').read_text()) if (out/'training.json').exists() else []
    train_ds=Windows(panel,cfg,'train');val_dl=loader(Windows(panel,cfg,'val'),cfg)
    for epoch in range(start,cfg['train']['epochs']):
        model.train();warm=epoch<cfg['train'].get('warmup_epochs',5)
        if hasattr(model,'warmup'):model.warmup(warm)
        if hasattr(model,'teacher_ratio'):model.teacher_ratio=max(0.,1-epoch/max(1,cfg['train']['epochs']//2))
        tic=time.perf_counter();loss_sum=0.;steps=0
        for raw in loader(train_ds,cfg,True,epoch):
            b=move(raw,device);opt.zero_grad(set_to_none=True);pred=model(b)
            if getattr(model,'probabilistic',True):loss=masked_nll(pred,b['y'],b['mask'])
            else:
                # Per-series scaled MSE training; checkpoint chosen by exact aggregate validation score.
                scale=torch.tensor(stats.scale2.fillna(1).to_numpy()[raw['series'].numpy()],device=device)[:,None]
                loss=(((pred-b['y'])**2)/scale.clamp_min(1e-8))[b['mask']].mean()
            if hasattr(model,'auxiliary_loss'):loss=loss+model.auxiliary_loss()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['gradient_clip'],error_if_nonfinite=True);opt.step()
            loss_sum+=float(loss.detach());steps+=1
        # Validate the deployed gate, including after a warm-up epoch.
        if hasattr(model,'warmup'):model.warmup(False)
        score=validation(model,val_dl,device,stats,cfg)
        improved=score<best
        if improved:best=score;stale=0
        else:stale+=1
        row={'epoch':epoch,'loss':loss_sum/max(steps,1),'validation':score,'seconds':time.perf_counter()-tic,'warmup':warm}
        history.append(row);write_json(out/'training.json',history);print(json.dumps(row),flush=True)
        save_checkpoint(out/'last.pt',model,opt,epoch,best,stale,cfg,panel.fingerprint)
        if improved:save_checkpoint(out/'best.pt',model,opt,epoch,best,stale,cfg,panel.fingerprint)
        if not warm and stale>=cfg['train']['patience']:break
    best_ckpt=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(best_ckpt['model']);model.eval()
    if hasattr(model,'warmup'):model.warmup(False)
    return model,device


def predict_model(model,device,panel,cfg,out):
    directory=out/'predictions';directory.mkdir(exist_ok=True)
    for old in directory.glob('part_*.npz'):old.unlink()
    with torch.no_grad():
        for j,raw in enumerate(loader(Windows(panel,cfg,'test'),cfg)):
            b=move(raw,device);pred=model(b)
            values={k:raw[k].numpy() for k in ['series','origin','y','mask']}
            if getattr(model,'probabilistic',True):values.update(pred.numpy());values['kind']=np.asarray(pred.kind);values['point']=pred.mean().cpu().numpy()
            else:values['point']=pred.cpu().numpy()
            np.savez_compressed(directory/f'part_{j:05}.npz',**values)
    return sorted(directory.glob('part_*.npz'))


def classical(cfg,panel,out,stats):
    name=cfg['model']['name'];grid=[(.05,.05),(.1,.1),(.2,.2),(.3,.1)]
    val=Windows(panel,cfg,'val');scale=stats.scale2.to_numpy();weights=stats.validation_weight.to_numpy()
    def score(alpha,beta,aggregation):
        se=np.zeros(panel.n);count=np.zeros(panel.n)
        for i in val.series:
            for origin in val.origins:
                pred=classical_forecast(panel.y[i,:origin],panel.mask[i,:origin],val.H,name,alpha,beta,aggregation)
                mask=panel.mask[i,origin:origin+val.H];se[i]+=(((pred-panel.y[i,origin:origin+val.H])**2)*mask).sum();count[i]+=mask.sum()
        use=(count>0)&np.isfinite(scale)&(scale>0)
        rms=np.sqrt(se[use]/count[use]/scale[use])
        return np.average(rms,weights=weights[use]) if cfg['data'].get('remove_leading_zeros') else rms.mean()
    trials=[]
    for a,b in grid:
        for aggregation in ([1,7,14] if name=='adida' else [7]):trials.append((score(a,b,aggregation),a,b,aggregation))
    _,a,b,aggregation=min(trials);write_json(out/'classical_selection.json',{'trials':trials,'selected':[a,b,aggregation]})
    folder=out/'predictions';folder.mkdir(exist_ok=True)
    for j,raw in enumerate(loader(Windows(panel,cfg,'test'),cfg)):
        predictions=[classical_forecast(panel.y[i,:o],panel.mask[i,:o],cfg['model']['horizon'],name,a,b,aggregation) for i,o in zip(raw['series'].numpy(),raw['origin'].numpy())]
        values={k:raw[k].numpy() for k in ['series','origin','y','mask']};values['point']=np.asarray(predictions)
        np.savez_compressed(folder/f'part_{j:05}.npz',**values)
    return sorted(folder.glob('part_*.npz'))


def run(cfg,out,resume=False):
    out=Path(out)
    if (out/'metrics.json').exists():raise FileExistsError(f'{out} already contains completed results; choose a new output')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cfg['train'].get('threads',4));seed_all(cfg['train']['seed'],cfg['train'].get('deterministic',True))
    panel=Panel(cfg['data']['path'],verify=True);stats=training_statistics(panel,cfg);stats.to_csv(out/'training_statistics.csv',index=False)
    splits=resolve_split(panel,cfg)
    write_json(out/'config.json',cfg)
    source=Path(__file__).resolve().parents[1]
    write_json(out/'provenance.json',{'environment':environment(),'data_sha256':panel.fingerprint,'split_indices':splits,
                                    'split_dates':{'train_last':str(panel.dates[splits[0]-1]),'val_last':str(panel.dates[splits[1]-1]),'test_last':str(panel.dates[-1])},
                                    'code_sha256':{str(p.relative_to(source)):sha256(p) for p in sorted(source.rglob('*.py'))},
                                    'implementation':'independent research reconstruction','protocol':cfg.get('protocol',{})})
    if cfg['model']['name'] in CLASSICAL:paths=classical(cfg,panel,out,stats)
    else:
        model,device=train_model(cfg,panel,out,stats,resume);paths=predict_model(model,device,panel,cfg,out)
    result=evaluate_shards(paths,stats,cfg,out);print(json.dumps(result),flush=True)
    return result
