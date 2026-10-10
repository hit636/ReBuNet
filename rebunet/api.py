"""Public Python API for an unpacked project. No package installation required.

Paths follow the calling process's working directory. Run notebooks/scripts in
the project root, or pass absolute config, data and output paths. Inputs are
copied before overrides are applied; these functions never change cwd.
"""
from pathlib import Path
import copy
from rebunet.utils import read_config, merge

__all__ = ['load_config', 'prepare_data', 'audit_newbeiyang', 'check_data',
           'train', 'run_suite', 'run_scenarios', 'tune', 'summarize',
           'plot_results', 'paper_report', 'score_m5', 'export_m5']


def load_config(config='configs/examples/synthetic.yaml', overrides=None):
    """Accept a YAML path or configuration dict; recursively apply dict overrides."""
    cfg = read_config(config) if isinstance(config, (str, Path)) else copy.deepcopy(config)
    if not isinstance(cfg, dict):
        raise TypeError('config must be a YAML path or dict')
    return merge(cfg, overrides or {})


def prepare_data(dataset, out, *, input_path=None, inventory=None,
                 sales_log_complete=False, observed_rows_only=False,
                 series=48, days=224, seed=2026):
    """Prepare synthetic/M5/industrial/New Beiyang inputs with the existing policies."""
    from rebunet.data import prepare
    if dataset not in ['synthetic', 'm5', 'industrial', 'newbeiyang']:
        raise ValueError('dataset must be synthetic, m5, industrial or newbeiyang')
    if dataset != 'newbeiyang' and (inventory or sales_log_complete or observed_rows_only):
        raise ValueError('Inventory/diagnostic options apply only to newbeiyang')
    if dataset == 'synthetic':
        return prepare.synthetic(out, series, days, seed)
    if input_path is None:
        raise ValueError('input_path is required for real datasets')
    if dataset == 'newbeiyang':
        from rebunet.data.newbeiyang import prepare as prepare_nby
        return prepare_nby(input_path, out, inventory, sales_log_complete, observed_rows_only)
    return getattr(prepare, dataset)(input_path, out)


def audit_newbeiyang(input_path, out, *, inventory=None):
    from rebunet.data.newbeiyang import audit
    return audit(input_path, out, inventory)


def check_data(config, out, *, overrides=None):
    """Return readiness evidence without starting training or exiting Python."""
    from rebunet.experiments.readiness import preflight
    return preflight(load_config(config, overrides), out)


def train(config, out, *, overrides=None, resume=False):
    """Train, select on validation, evaluate test data, save outputs; return metrics."""
    from rebunet.training.runner import run
    return run(load_config(config, overrides), out, resume=resume)


def run_suite(config, out, *, seeds=(0, 1, 2, 3, 4), scope='paper',
              execute=False, allow_missing_external=False, labels=None, overrides=None):
    """Generate the experiment matrix; set execute=True to run it."""
    from rebunet.experiments.suite import suite
    return suite(load_config(config, overrides), out, seeds, scope, execute,
                 allow_missing_external, labels)


def run_scenarios(template, config, out, *, seeds=(0,), execute=False, labels=None,
                  overrides=None):
    from rebunet.experiments.scenarios import scenario_suite
    return scenario_suite(template, load_config(config, overrides), out, seeds, execute, labels)


def tune(config, grid, out, *, overrides=None):
    from rebunet.training.tuning import tune as tune_model
    return tune_model(load_config(config, overrides), grid, out)


def summarize(out):
    from rebunet.experiments.suite import summarize as summarize_suite
    return summarize_suite(out)


def plot_results(run_dir, out):
    from rebunet.evaluation.plots import plot_run
    return plot_run(run_dir, out)


def paper_report(config, suite_dir, out, *, seeds=(0, 1, 2, 3, 4), overrides=None):
    from rebunet.experiments.readiness import paper_report as report
    return report(load_config(config, overrides), suite_dir, out, seeds)


def score_m5(raw, forecasts, out, *, phase='validation', actuals=None):
    from rebunet.evaluation.m5_official import score_files
    return score_files(raw, forecasts, out, phase, actuals)


def export_m5(run_dir, data_dir, out, *, phase='validation'):
    from rebunet.evaluation.m5_official import export_run
    return export_run(run_dir, data_dir, out, phase)
