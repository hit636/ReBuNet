"""Create a paper experiment plan without training or downloading datasets."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['m5', 'newbeiyang'], default='m5')
    parser.add_argument('--out', default='results/example_plan')
    args = parser.parse_args()
    from rebunet.api import run_suite
    out = Path(args.out).resolve()
    if out.exists():
        parser.error('Choose a new output directory to preserve previous plans.')
    run_suite(ROOT / f'configs/datasets/{args.dataset}.yaml', out=str(out))
    print(f'Experiment plan saved to {out}; no models were trained.')


if __name__ == '__main__':
    main()
