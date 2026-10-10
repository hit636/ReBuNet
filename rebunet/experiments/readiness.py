"""Machine-readable readiness and per-table evidence, never numerical claims."""
from pathlib import Path
import json
from rebunet.data.panel import Panel, resolve_split, training_statistics, Windows
from rebunet.experiments.suite import plan
from rebunet.utils import write_json


def preflight(cfg, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    report = {'data_path': cfg['data']['path'], 'ready_to_train': False,
              'author_exact_reproduction': False, 'blocking_issues': [], 'warnings': []}
    try:
        panel = Panel(cfg['data']['path'], verify=True)
        report.update(shape=[panel.n, panel.t], data_sha256=panel.fingerprint,
                      dataset=panel.meta['metadata'].get('dataset'))
        if panel.meta['metadata'].get('diagnostic_only'):
            report['blocking_issues'].append('New Beiyang observed-row diagnostic panel is not paper-ready')
        tr, va, start = resolve_split(panel, cfg)
        report['split_indices'] = {'train_end_exclusive': tr, 'val_end_exclusive': va, 'first_origin': start}
        report['usable_targets'] = {k: int(panel.mask[:, a:b].sum()) for k, a, b in [('train', 0, tr), ('val', tr, va), ('test', va, panel.t)]}
        if any(v == 0 for v in report['usable_targets'].values()):
            report['blocking_issues'].append('At least one split has no usable targets')
        report['windows'] = {k: len(Windows(panel, cfg, k)) for k in ['train', 'val', 'test']}
        stats = training_statistics(panel, cfg); stats.to_csv(out / 'training_statistics.csv', index=False)
        report['scale_eligible_series'] = int(stats.scale2.notna().sum())
        report['training_zero_ratio_mean'] = float(stats.zero_ratio.mean())
        if report['scale_eligible_series'] == 0:
            report['warnings'].append('No valid RMSSE scales; point-model validation cannot select a checkpoint')
    except (ValueError, FileNotFoundError, KeyError) as exc:
        report['blocking_issues'].append(str(exc))
    report['ready_to_train'] = not report['blocking_issues']
    write_json(out / 'preflight.json', report)
    return report


def paper_report(cfg, suite_dir, out, seeds=(0,1,2,3,4)):
    suite_dir, out = Path(suite_dir), Path(out)
    data = preflight(cfg, out / 'data')
    jobs = plan(cfg, list(seeds), 'paper')
    tables = {}
    for job in jobs:
        folder = suite_dir / job['label'] / f'seed_{job["seed"]}'
        metric_file = folder / 'metrics.json'
        status = 'external_implementation_missing' if job['requires_external'] else 'not_run'
        if metric_file.exists():
            provenance = folder / 'provenance.json'; conf = folder / 'config.json'
            if provenance.exists() and conf.exists() and json.loads(provenance.read_text()).get('data_sha256') == data.get('data_sha256') and json.loads(conf.read_text()) == job['config']:
                status = 'results_present'
            else:
                status = 'stale_or_unverifiable_result'
        for table in job['tables']:
            evidence = status
            if status == 'results_present' and table in ['T8','D3'] and not (folder/'profile.json').exists():
                evidence = 'missing_efficiency_profile'
            if status == 'results_present' and table == 'T11' and job['label']=='full' and not (folder/'propagation.json').exists():
                evidence = 'missing_fixed_parameter_propagation'
            tables.setdefault(table, []).append({'variant':job['label'], 'seed':job['seed'], 'status':evidence})
    report = {'data': data, 'tables': {}, 'author_exact_reproduction': False,
              'scope': 'paper configuration matrix; numerical equality to published results is unverified'}
    for table, items in sorted(tables.items()):
        report['tables'][table] = {'expected_runs': len(items),
            'results_present': sum(r['status']=='results_present' for r in items), 'runs':items}
    report['all_configured_runs_present'] = all(x['results_present']==x['expected_runs'] for x in report['tables'].values())
    write_json(out/'paper_coverage.json',report)
    lines=['# Paper experiment coverage\n', '| Paper section | Expected runs | Verified results |', '|---|---:|---:|']
    lines += [f'| {k} | {v["expected_runs"]} | {v["results_present"]} |' for k,v in report['tables'].items()]
    lines += ['', 'A configured task is not a completed experiment; results must match the configuration and dataset fingerprint.',
              'Table 2 uses data/preflight.json; Table 9 uses saved suite configurations.',
              'Exact numerical reproduction is not asserted without the original splits and baseline settings.']
    (out/'PAPER_COVERAGE.md').write_text('\n'.join(lines),encoding='utf-8')
    return report
