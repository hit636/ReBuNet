#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# First accept M5 competition rules in your browser and configure Kaggle authentication.
# Credentials belong in Kaggle's standard user configuration, never this repository.
mkdir -p data/raw/m5
kaggle competitions download -c m5-forecasting-accuracy -p data/raw/m5
python - <<'PY'
from zipfile import ZipFile
from pathlib import Path
root=Path('data/raw/m5')
for path in root.glob('*.zip'):
    with ZipFile(path) as archive:
        for item in archive.infolist():
            destination=(root/item.filename).resolve()
            if not destination.is_relative_to(root.resolve()): raise ValueError('Unsafe ZIP path')
        archive.extractall(root)
PY
python run.py prepare m5 --input data/raw/m5 --out data/processed/m5
