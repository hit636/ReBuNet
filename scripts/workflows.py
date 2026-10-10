"""Shell workflow orchestration over the public local-package API."""
import _bootstrap
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


def parser():
    p=argparse.ArgumentParser(description='ReBuNet local-source workflows; no project installation required')
    commands=p.add_subparsers(dest='workflow',required=True)
    smoke=commands.add_parser('smoke',help='Generate synthetic data, validate, train, evaluate and plot')
    smoke.add_argument('--config',default='configs/examples/synthetic.yaml')
    smoke.add_argument('--series',type=int,default=48)
    smoke.add_argument('--days',type=int,default=224)
    smoke.add_argument('--generator-seed',type=int,default=2026)
    smoke.add_argument('--resume',action='store_true')
    real=commands.add_parser('real',help='Prepare and train on M5 or New Beiyang data')
    real.add_argument('--dataset',choices=['m5','newbeiyang'],required=True)
    real.add_argument('--input',help='M5 raw directory or New Beiyang sales CSV')
    real.add_argument('--inventory',help='New Beiyang daily inventory CSV')
    real.add_argument('--sales-log-complete',action='store_true')
    real.add_argument('--config')
    real.add_argument('--data',help='Processed panel directory; defaults to data.path in the configuration')
    real.add_argument('--skip-prepare',action='store_true',help='Use an existing processed panel')
    real.add_argument('--prepare-only',action='store_true',help='Prepare and validate data without training')
    real.add_argument('--audit-only',action='store_true',help='Audit the New Beiyang CSV without requiring inventory')
    real.add_argument('--resume',action='store_true')
    exp=commands.add_parser('experiments',help='Paper experiment matrix or six synthetic stress scenarios')
    exp.add_argument('--kind',choices=['paper','scenarios'],default='paper')
    exp.add_argument('--dataset',choices=['m5','newbeiyang'],default='m5',help='Dataset for paper mode')
    exp.add_argument('--config')
    exp.add_argument('--template',default='configs/scenarios/templates.yaml')
    exp.add_argument('--seeds',type=int,nargs='+')
    exp.add_argument('--labels',nargs='+')
    exp.add_argument('--execute',action='store_true',help='Execute training; otherwise generate a plan only')
    exp.add_argument('--allow-missing-external',action='store_true',help='Paper mode: run implemented methods while reporting missing baselines')
    for command in [smoke,real,exp]:
        command.add_argument('--out',help='Output directory; defaults to a new timestamped directory')
        command.add_argument('--set',action='append',default=[],metavar='KEY=VALUE',help='Override configuration, e.g. train.device=cuda')
    return p


def validate_args(p,a):
    if getattr(a,'resume',False) and not a.out:
        p.error('--resume requires the original run directory via --out')
    if a.workflow=='real':
        if a.dataset=='m5' and (a.inventory or a.sales_log_complete or a.audit_only):
            p.error('Inventory arguments and --audit-only require newbeiyang')
        if a.skip_prepare and (a.input or a.inventory or a.sales_log_complete or a.audit_only):
            p.error('--skip-prepare cannot be combined with raw-data preparation arguments')
        if a.audit_only and (a.resume or a.prepare_only):
            p.error('--audit-only cannot be combined with --resume or --prepare-only')
    if a.workflow=='experiments':
        if a.seeds is not None and (not a.seeds or len(a.seeds)!=len(set(a.seeds))):
            p.error('--seeds must be nonempty and unique')
        if a.kind=='scenarios' and a.allow_missing_external:
            p.error('Scenario mode does not support --allow-missing-external')


def main():
    p=parser();a=p.parse_args();validate_args(p,a)
    # Lazy imports keep --help and argument validation usable before dependencies exist.
    from rebunet import api
    from rebunet.cli import overrides
    from rebunet.utils import write_json
    suffix=getattr(a,'kind',None) or getattr(a,'dataset',None) or 'synthetic'
    out=Path(a.out or f'results/{a.workflow}_{suffix}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    out.mkdir(parents=True,exist_ok=True)
    status={'workflow':a.workflow,'output':str(out),'state':'started'}
    write_json(out/'workflow_status.json',status)
    print(f'Output directory: {out}',flush=True)
    try:
        if a.workflow in ['smoke','real'] and (out/'model/metrics.json').exists():
            raise FileExistsError('Completed model results already exist; choose a new --out directory')
        if a.workflow=='smoke':
            cfg=overrides(api.load_config(a.config),a.set)
            # Each smoke run owns its data, preventing one run from changing another's panel.
            cfg['data']['path']=str(out/'data')
            request={'series':a.series,'days':a.days,'seed':a.generator_seed}
            request_file=out/'generator_request.json'
            if request_file.exists() and json.loads(request_file.read_text())!=request:
                raise ValueError('Generator parameters differ from the existing run; choose a new --out')
            write_json(request_file,request)
            api.prepare_data('synthetic',cfg['data']['path'],series=a.series,days=a.days,seed=a.generator_seed)
            ready=api.check_data(cfg,out/'preflight')
            if not ready['ready_to_train']:raise ValueError(ready['blocking_issues'])
            status['metrics']=api.train(cfg,out/'model',resume=a.resume)
            api.plot_results(out/'model',out/'figures')
            status['state']='completed_synthetic'
        elif a.workflow=='real':
            cfg=overrides(api.load_config(a.config or f'configs/datasets/{a.dataset}.yaml'),a.set)
            if a.data:cfg['data']['path']=a.data
            source=a.input or ('data/raw/m5' if a.dataset=='m5' else 'data/raw/newbeiyang/daily_sales.csv')
            if not a.skip_prepare:
                if not Path(source).exists():raise FileNotFoundError(f'Raw data does not exist: {source}')
                if a.dataset=='newbeiyang':
                    audit=api.audit_newbeiyang(source,out/'raw_audit',inventory=a.inventory)
                    if a.audit_only:
                        status.update(state='audit_only',audit=audit)
                        write_json(out/'workflow_status.json',status)
                        print(json.dumps(status,ensure_ascii=False,indent=2));return 0
                    inventory=a.inventory or 'data/raw/newbeiyang/inventory_daily.csv'
                    if not Path(inventory).is_file():
                        raise FileNotFoundError(f'Missing inventory logs: {inventory}. The raw-data audit was saved; availability and zero sales are not inferred.')
                    api.prepare_data('newbeiyang',cfg['data']['path'],input_path=source,
                                     inventory=inventory,sales_log_complete=a.sales_log_complete)
                else:
                    api.prepare_data('m5',cfg['data']['path'],input_path=source)
            ready=api.check_data(cfg,out/'preflight')
            if not ready['ready_to_train']:raise ValueError(ready['blocking_issues'])
            if a.prepare_only:status['state']='prepared_and_checked'
            else:
                status['metrics']=api.train(cfg,out/'model',resume=a.resume)
                api.plot_results(out/'model',out/'figures');status['state']='completed'
        else:
            cfg=overrides(api.load_config(a.config or ('configs/scenarios/quick.yaml' if a.kind=='scenarios' else f'configs/datasets/{a.dataset}.yaml')),a.set)
            seeds=a.seeds if a.seeds is not None else ([0] if a.kind=='scenarios' else [0,1,2,3,4])
            if a.kind=='scenarios':
                status['result']=api.run_scenarios(a.template,cfg,out/'scenarios',seeds=seeds,execute=a.execute,labels=a.labels)
            else:
                # Write the requested plan even when execution will be blocked by missing data.
                status['result']=api.run_suite(cfg,out/'suite',seeds=seeds,labels=a.labels)
                if a.execute:
                    ready=api.check_data(cfg,out/'preflight')
                    if not ready['ready_to_train']:raise ValueError(ready['blocking_issues'])
                    status['result']=api.run_suite(cfg,out/'suite',seeds=seeds,labels=a.labels,
                                                   execute=True,allow_missing_external=a.allow_missing_external)
                    coverage=json.loads((out/'suite/coverage.json').read_text())
                    if not coverage['complete']:
                        status['state']='partial_missing_external'
                        write_json(out/'workflow_status.json',status)
                        print('Implemented experiments completed, but external baselines are missing; see suite/coverage.json.',file=sys.stderr)
                        return 2
            status['state']='completed' if a.execute else 'planned_not_trained'
        write_json(out/'workflow_status.json',status)
        print(json.dumps(status,ensure_ascii=False,default=str,indent=2));return 0
    except Exception as exc:
        status.update(state='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(out/'workflow_status.json',status)
        print(f'Execution stopped: {exc}\nStatus file: {out / "workflow_status.json"}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
