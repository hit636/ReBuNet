"""Run data preparation, readiness checks, training, scoring, and plotting."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/api_example')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        parser.error('Choose a new output directory to preserve previous runs.')
    from rebunet.api import prepare_data, load_config, check_data, train, plot_results

    prepare_data('synthetic', out=str(out / 'data'), series=48, days=224, seed=2026)
    config = load_config(ROOT / 'configs/examples/synthetic.yaml',
                         overrides={'data': {'path': str(out / 'data')}})
    status = check_data(config, out=str(out / 'preflight'))
    if not status['ready_to_train']:
        raise RuntimeError('Data are not ready; inspect the preflight report.')
    metrics = train(config, out=str(out / 'model'))
    plot_results(str(out / 'model'), out=str(out / 'figures'))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
