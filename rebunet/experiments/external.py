"""Verified import boundary for actual external baseline predictions."""
from pathlib import Path
import json
import shutil
import numpy as np
from rebunet.data.panel import Panel,Windows,training_statistics
from rebunet.evaluation.metrics import evaluate_shards
from rebunet.utils import write_json,sha256


def import_predictions(cfg,source,out):
    source,out=Path(source),Path(out);panel=Panel(cfg['data']['path']);windows=Windows(panel,cfg,'test')
    metadata=json.loads((source/'external_manifest.json').read_text())
    required=['method','implementation_url','commit_or_version','training_config','validation_selection','seed','data_fingerprint']
    if any(k not in metadata for k in required):raise ValueError('Incomplete external implementation provenance')
    if metadata['method']!=cfg['model']['name'] or metadata['seed']!=cfg['train']['seed'] or metadata['data_fingerprint']!=panel.fingerprint:
        raise ValueError('External method/seed/data mismatch')
    expected={(int(i),int(o)) for i in windows.series for o in windows.origins};seen=set();paths=sorted(source.glob('part_*.npz'))
    if not paths:raise ValueError('No external prediction shards')
    for path in paths:
        a=np.load(path,allow_pickle=False)
        for key in ['series','origin','y','mask','point']:
            if key not in a:raise ValueError('Missing shard field '+key)
        H=cfg['model']['horizon'];B=len(a['series'])
        if any(a[k].shape!=(B,H) for k in ['y','mask','point']):raise ValueError('External horizon/shape mismatch')
        for j,(i,o) in enumerate(zip(a['series'],a['origin'])):
            key=(int(i),int(o))
            if key not in expected or key in seen:raise ValueError('Unexpected or duplicate forecast origin')
            seen.add(key)
            if not np.array_equal(a['y'][j],panel.y[i,o:o+H]) or not np.array_equal(a['mask'][j],panel.mask[i,o:o+H]):
                raise ValueError('External targets or availability mask differ from canonical data')
        if 'w' in a:
            if any(k not in a for k in ['p','mu','k','kind']):raise ValueError('Incomplete marginal distribution')
        elif cfg['model']['name'] in ['robust_rnn','deep_renewal','deep_state','score_hurdle_nb']:
            raise ValueError('This Table 4 baseline needs full marginal distributions, not point predictions only')
    if seen!=expected:raise ValueError(f'Missing {len(expected-seen)} forecast origins')
    if (out/'metrics.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);folder=out/'predictions';folder.mkdir(exist_ok=True)
    for path in paths:shutil.copy2(path,folder/path.name)
    write_json(out/'config.json',cfg)
    write_json(out/'provenance.json',{'external':metadata,'input_checksums':{p.name:sha256(p) for p in paths},'data_sha256':panel.fingerprint})
    return evaluate_shards(sorted(folder.glob('part_*.npz')),training_statistics(panel,cfg),cfg,out)
