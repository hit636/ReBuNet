"""Validation-only model selection; test predictions are generated once after selection."""
import copy
import itertools
import shutil
from pathlib import Path
import yaml
import torch
from rebunet.cli import overrides
from rebunet.data.panel import Panel,training_statistics
from rebunet.training.runner import train_model,predict_model
from rebunet.evaluation.metrics import evaluate_shards
from rebunet.utils import write_json,seed_all,environment


def tune(cfg,grid_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists():raise FileExistsError('Completed tuning output already exists')
    panel=Panel(cfg['data']['path']);stats=training_statistics(panel,cfg)
    grid=yaml.safe_load(Path(grid_path).read_text());keys=list(grid);trials=[]
    for index,values in enumerate(itertools.product(*(grid[k] for k in keys))):
        trial=overrides(copy.deepcopy(cfg),[k+'='+yaml.safe_dump(v).strip().split('\n...')[0] for k,v in zip(keys,values)])
        folder=out/f'trial_{index:03}';folder.mkdir(exist_ok=True)
        seed_all(trial['train']['seed'],trial['train'].get('deterministic',True));torch.set_num_threads(trial['train'].get('threads',4))
        model,device=train_model(trial,panel,folder,stats,resume=(folder/'last.pt').exists())
        ckpt=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        trials.append({'trial':index,'validation':ckpt['best'],'config':trial})
        write_json(out/'trials.json',trials)
    selected=min(trials,key=lambda x:x['validation']);source=out/f"trial_{selected['trial']:03}"/'best.pt';dest=out/'selected';dest.mkdir(exist_ok=True)
    shutil.copy2(source,dest/'best.pt')
    from rebunet.experiments.diagnostics import load_model
    device=('cuda' if torch.cuda.is_available() else 'cpu') if cfg['train']['device']=='auto' else cfg['train']['device']
    model,_,selected_cfg=load_model(dest/'best.pt',device)
    write_json(dest/'config.json',selected_cfg);write_json(dest/'provenance.json',{'data_sha256':panel.fingerprint,'environment':environment(),'selection':'validation_only'})
    write_json(out/'selection.json',selected)
    paths=predict_model(model,device,panel,selected_cfg,dest)
    return evaluate_shards(paths,stats,selected_cfg,dest)
