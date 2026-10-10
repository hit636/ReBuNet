import argparse
import json
from pathlib import Path
import yaml
from rebunet.utils import read_config


def overrides(cfg,items):
    for item in items or []:
        key,value=item.split('=',1);parts=key.split('.');dest=cfg
        for p in parts[:-1]:dest=dest.setdefault(p,{})
        dest[parts[-1]]=yaml.safe_load(value)
    return cfg


def main():
    parser=argparse.ArgumentParser(description='ReBuNet research reconstruction; see README.md')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('dataset',choices=['synthetic','m5','industrial','newbeiyang']);p.add_argument('--input');p.add_argument('--out',required=True);p.add_argument('--series',type=int,default=48);p.add_argument('--days',type=int,default=224)
    p.add_argument('--inventory');p.add_argument('--sales-log-complete',action='store_true');p.add_argument('--observed-rows-only',action='store_true')
    p=sub.add_parser('audit-newbeiyang');p.add_argument('--input',required=True);p.add_argument('--inventory');p.add_argument('--out',required=True)
    p=sub.add_parser('scenario-suite');p.add_argument('--template',default='configs/scenarios/templates.yaml');p.add_argument('--config',default='configs/examples/synthetic.yaml');p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,nargs='+',default=[0]);p.add_argument('--labels',nargs='+');p.add_argument('--execute',action='store_true')
    p=sub.add_parser('preflight');p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('paper-report');p.add_argument('--config',required=True);p.add_argument('--suite-dir',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('score-m5-official');p.add_argument('--raw',required=True);p.add_argument('--forecasts',required=True);p.add_argument('--out',required=True);p.add_argument('--phase',choices=['validation','evaluation'],default='validation');p.add_argument('--actuals')
    p=sub.add_parser('export-m5');p.add_argument('--run',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--phase',choices=['validation','evaluation'],default='validation')
    p=sub.add_parser('train');p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--set',action='append');p.add_argument('--resume',action='store_true')
    p=sub.add_parser('suite');p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]);p.add_argument('--scope',choices=['paper','all','smoke'],default='paper');p.add_argument('--execute',action='store_true');p.add_argument('--allow-missing-external',action='store_true');p.add_argument('--labels',nargs='+');p.add_argument('--set',action='append')
    p=sub.add_parser('summarize');p.add_argument('--out',required=True)
    p=sub.add_parser('propagation');p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p=sub.add_parser('profile');p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=256);p.add_argument('--repeats',type=int,default=30)
    p=sub.add_parser('import-external');p.add_argument('--config',required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--set',action='append')
    p=sub.add_parser('validate-data');p.add_argument('--config',required=True)
    p=sub.add_parser('plot');p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('tune');p.add_argument('--config',required=True);p.add_argument('--grid',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        from rebunet.data import prepare
        if args.dataset=='synthetic':result=prepare.synthetic(args.out,args.series,args.days)
        else:
            if not args.input:parser.error('--input required for real data')
            if args.dataset=='newbeiyang':
                from rebunet.data.newbeiyang import prepare as prepare_newbeiyang
                result=prepare_newbeiyang(args.input,args.out,args.inventory,args.sales_log_complete,args.observed_rows_only)
            else:result=getattr(prepare,args.dataset)(args.input,args.out)
    elif args.command=='audit-newbeiyang':
        from rebunet.data.newbeiyang import audit
        result=audit(args.input,args.out,args.inventory)
    elif args.command=='scenario-suite':
        from rebunet.experiments.scenarios import scenario_suite
        result=scenario_suite(args.template,read_config(args.config),args.out,args.seeds,args.execute,args.labels)
    elif args.command=='preflight':
        from rebunet.experiments.readiness import preflight
        result=preflight(read_config(args.config),args.out)
        if not result['ready_to_train']:
            print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(2)
    elif args.command=='paper-report':
        from rebunet.experiments.readiness import paper_report
        result=paper_report(read_config(args.config),args.suite_dir,args.out)
    elif args.command=='score-m5-official':
        from rebunet.evaluation.m5_official import score_files
        result=score_files(args.raw,args.forecasts,args.out,args.phase,args.actuals)
    elif args.command=='export-m5':
        from rebunet.evaluation.m5_official import export_run
        result=export_run(args.run,args.data,args.out,args.phase)
    elif args.command=='train':
        from rebunet.training.runner import run
        result=run(overrides(read_config(args.config),args.set),args.out,args.resume)
    elif args.command=='suite':
        from rebunet.experiments.suite import suite
        result=suite(overrides(read_config(args.config),args.set),args.out,args.seeds,args.scope,args.execute,args.allow_missing_external,args.labels)
    elif args.command=='summarize':
        from rebunet.experiments.suite import summarize
        result=summarize(args.out)
    elif args.command in ['profile','propagation']:
        from rebunet.experiments.diagnostics import profile,propagation
        if args.command=='profile':result=profile(args.checkpoint,args.out,args.device,args.repeats,batch_size=args.batch_size)
        else:result=propagation(args.checkpoint,args.out,args.device)
    elif args.command=='import-external':
        from rebunet.experiments.external import import_predictions
        result=import_predictions(overrides(read_config(args.config),args.set),args.input,args.out)
    elif args.command=='validate-data':
        from rebunet.data.panel import Panel,training_statistics,resolve_split
        cfg=read_config(args.config);p=Panel(cfg['data']['path'],verify=True);tr,va,begin=resolve_split(p,cfg)
        result={'shape':[p.n,p.t],'splits':[tr,va,begin],'fingerprint':p.fingerprint,'statistics':training_statistics(p,cfg).describe().to_dict()}
    elif args.command=='plot':
        from rebunet.evaluation.plots import plot_run
        result=plot_run(args.run,args.out)
    elif args.command=='tune':
        from rebunet.training.tuning import tune
        result=tune(read_config(args.config),args.grid,args.out)
    print(json.dumps(result,ensure_ascii=False,default=str,indent=2))


if __name__=='__main__':main()
