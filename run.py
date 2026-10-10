"""Run directly from a checkout: python run.py train --config ... --out ..."""
from pathlib import Path
import os


if __name__ == '__main__':
    # CLI paths consistently refer to the project; importing the API never changes cwd.
    os.chdir(Path(__file__).resolve().parent)
    from rebunet.cli import main
    main()
