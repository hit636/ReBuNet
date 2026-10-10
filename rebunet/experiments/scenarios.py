"""Scenario orchestration and observed-only event/background diagnostics."""
from pathlib import Path
import copy
import json
import tempfile
import numpy as np
import pandas as pd
import yaml
from rebunet.data.scenarios import generate
from rebunet.evaluation.metrics import evaluate_shards
from rebunet.experiments.suite import suite
from rebunet.utils import write_json


def summarize_scenarios(out):
    from rebunet.evaluation.plots import plot_run
    out = Path(out); rows = []; event_rows = []
    for metric_file in sorted(out.glob('*/runs/*/seed_*/metrics.json')):
        folder = metric_file.parent
        identity = {'scenario': folder.parents[2].name, 'variant': folder.parent.name,
                    'seed': int(folder.name.split('_')[-1]), 'synthetic_only': True}
        metrics = json.loads(metric_file.read_text())
        rows.append({**identity, **{k:v for k,v in metrics.items() if not isinstance(v,(dict,list))}})
        event_file = folder / 'event_metrics.json'
        if event_file.exists():
            for group, values in json.loads(event_file.read_text()).items():
                event_rows.append({**identity, 'period': group,
                    **{k:v for k,v in values.items() if not isinstance(v,(dict,list))}})
        if identity['variant'] == 'full':
            plot_run(folder, folder/'figures')
    if rows:
        pd.DataFrame(rows).to_csv(out/'all_scenario_runs.csv', index=False)
    if event_rows:
        pd.DataFrame(event_rows).to_csv(out/'all_event_periods.csv', index=False)
    return {'runs': len(rows), 'event_period_rows': len(event_rows)}


def event_scores(run_dir, data_dir):
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    oracle = np.load(data_dir / 'oracle' / 'diagnostics.npz', allow_pickle=False)
    cfg = json.loads((run_dir / 'config.json').read_text())
    stats = pd.read_csv(run_dir / 'per_series.csv')
    shards = sorted((run_dir / 'predictions').glob('*.npz'))
    if not shards:
        raise ValueError('No prediction shards for scenario diagnostics')
    results = {}
    for group in ['event', 'background']:
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); files = []; targets = 0
            for j, shard in enumerate(shards):
                with np.load(shard, allow_pickle=False) as source:
                    arrays = {k: source[k] for k in source.files}
                idx = arrays['origin'][:, None] + np.arange(arrays['y'].shape[1])[None, :]
                selected = oracle['event'][arrays['series'][:, None], idx]
                if group == 'background':
                    selected = ~selected
                arrays['mask'] = arrays['mask'].astype(bool) & selected
                targets += int(arrays['mask'].sum())
                path = temp / f'{j:06}.npz'; np.savez(path, **arrays); files.append(path)
            results[group] = evaluate_shards(files, stats, cfg, temp) if targets else {'usable_targets': 0, 'status': 'no_observed_targets'}
    write_json(run_dir / 'event_metrics.json', results)
    return results


def scenario_suite(template, cfg, out, seeds=(0,), execute=False, labels=None):
    spec = yaml.safe_load(Path(template).read_text(encoding='utf-8'))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    labels = labels or ['full', 'state_direct', 'state_grs', 'mark_single']
    names = [s['name'] for s in spec['scenarios']]
    if len(names) != len(set(names)):
        raise ValueError('Scenario names must be unique')
    report = []
    for scenario in spec['scenarios']:
        name = scenario['name']; folder = out / name; data = folder / 'data'
        request = {'name': name, 'series': int(spec.get('series',48)), 'days': int(spec.get('days',224)),
                   'seed': int(spec.get('generator_seed',2026)), 'events': scenario.get('events')}
        request_file = folder / 'generator_request.json'
        if request_file.exists() and json.loads(request_file.read_text()) != request:
            raise ValueError('Generator settings changed; use a new output directory to preserve prior results')
        write_json(request_file, request)
        generate(data, name, int(spec.get('series', 48)), int(spec.get('days', 224)),
                 int(spec.get('generator_seed', 2026)), scenario.get('events'))
        config = copy.deepcopy(cfg); days = int(spec.get('days', 224))
        config['data'].update(path=str(data), train_end=days-56, val_end=days-28,
                              remove_leading_zeros=False)
        config['protocol'] = {'status': 'SYNTHETIC_STRESS_TEST_NOT_PAPER_REPRODUCTION',
                              'template': str(template), 'scenario': name}
        result = suite(config, folder / 'runs', seeds, 'paper', execute, False, labels)
        if execute:
            for label in labels:
                for seed in seeds:
                    run_dir = folder / 'runs' / label / f'seed_{seed}'
                    if (run_dir / 'metrics.json').exists():
                        event_scores(run_dir, data)
        report.append({'scenario': name, **result})
    write_json(out / 'scenario_status.json', {'synthetic_only': True, 'scenarios': report})
    if execute:
        summarize_scenarios(out)
    return report
