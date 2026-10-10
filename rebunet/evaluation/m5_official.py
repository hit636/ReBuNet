"""Independent implementation of official 12-level M5 WRMSSE (not paper bWRMSSE)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from rebunet.data.m5_official import validate_frames, aligned_forecasts
from rebunet.utils import sha256, write_json

LEVELS = [[], ['state_id'], ['store_id'], ['cat_id'], ['dept_id'],
          ['state_id', 'cat_id'], ['state_id', 'dept_id'],
          ['store_id', 'cat_id'], ['store_id', 'dept_id'], ['item_id'],
          ['state_id', 'item_id'], ['item_id', 'store_id']]


def score_hierarchy(metadata, train, actual, forecast, revenue):
    """Score one 28-day origin, aggregate one hierarchy level at a time."""
    train = np.asarray(train); actual = np.asarray(actual); forecast = np.asarray(forecast)
    revenue = np.asarray(revenue, dtype=np.float64)
    n = len(metadata)
    if train.ndim != 2 or train.shape[0] != n or actual.shape != forecast.shape or actual.shape != (n, 28):
        raise ValueError('Expected train N,T and actual/forecast N,28')
    if revenue.shape != (n,) or not np.isfinite(revenue).all() or (revenue < 0).any() or revenue.sum() <= 0:
        raise ValueError('Need nonnegative pre-origin 28-day revenue per bottom series')
    for a in [train, actual, forecast]:
        if not np.isfinite(a).all() or (a < 0).any():
            raise ValueError('Invalid training counts, truth or forecasts')
    rows = []
    for level, keys in enumerate(LEVELS, 1):
        if keys:
            codes, unique = pd.factorize(pd.MultiIndex.from_frame(metadata[keys]), sort=True)
            labels = [json.dumps(list(v), ensure_ascii=False) for v in unique]
        else:
            codes = np.zeros(n, dtype=int); labels = ['Total']
        agg = csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(len(labels), n))
        history = agg @ train
        target = agg @ actual; predicted = agg @ forecast
        weights = np.asarray(agg @ revenue).ravel() / revenue.sum() / len(LEVELS)
        first = np.argmax(history != 0, axis=1)
        valid_pairs = np.arange(history.shape[1]-1)[None, :] >= first[:, None]
        denominator = valid_pairs.sum(1)
        scale = np.divide((np.diff(history, axis=1)**2 * valid_pairs).sum(1), denominator,
                          out=np.zeros(len(labels)), where=denominator > 0)
        if np.any((scale <= 0) & (weights > 0)):
            raise ValueError(f'Positive-weight series has undefined scale at hierarchy level {level}')
        rmsse = np.sqrt(np.divide(np.mean((target-predicted)**2, axis=1), scale,
                                 out=np.zeros(len(labels)), where=scale > 0))
        rows.extend({'level': level, 'group': label, 'scale2': float(s), 'weight': float(w),
                     'RMSSE': float(r), 'contribution': float(w*r)}
                    for label, s, w, r in zip(labels, scale, weights, rmsse))
    details = pd.DataFrame(rows)
    return {'WRMSSE': float(details.contribution.sum()), 'hierarchy_levels': 12,
            'aggregate_series': len(details), 'weight_sum': float(details.weight.sum()),
            'bottom_level_RMSSE_weighted': float(details.loc[details.level.eq(12), 'contribution'].sum()*12)}, details


def score_files(raw, forecasts, out, phase='validation', actuals=None, strict=True):
    raw, out = Path(raw), Path(out); out.mkdir(parents=True, exist_ok=True)
    sales_path = raw / 'sales_train_evaluation.csv'
    sales = pd.read_csv(sales_path); cal = pd.read_csv(raw / 'calendar.csv'); prices = pd.read_csv(raw / 'sell_prices.csv')
    validate_frames(sales, cal, prices, strict)
    end = {'validation': 1913, 'evaluation': 1941}[phase]
    train = sales[[f'd_{i}' for i in range(1, end+1)]].to_numpy(dtype=np.float32)
    if phase == 'validation':
        truth = sales[[f'd_{i}' for i in range(1914, 1942)]].to_numpy(dtype=np.float64)
    else:
        if not actuals:
            raise ValueError('Private evaluation requires released official truth --actuals with id,d_1942..d_1969')
        truth = aligned_forecasts(actuals, sales.id, phase, [f'd_{i}' for i in range(1942, 1970)])
        if not np.equal(truth, np.floor(truth)).all():
            raise ValueError('Actuals must be integer counts')
    forecast = aligned_forecasts(forecasts, sales.id, phase)
    weeks = cal.set_index('d').loc[[f'd_{i}' for i in range(end-27, end+1)], 'wm_yr_wk'].tolist()
    price = prices.pivot(index=['item_id', 'store_id'], columns='wm_yr_wk', values='sell_price')
    price = price.reindex(pd.MultiIndex.from_frame(sales[['item_id', 'store_id']])).reindex(columns=weeks).to_numpy()
    recent = train[:, -28:]
    if np.any((recent > 0) & ~np.isfinite(price)):
        raise ValueError('Positive sales lack prices in weight window')
    revenue = (recent * np.nan_to_num(price, nan=0.)).sum(1)
    report, details = score_hierarchy(sales, train, truth, forecast, revenue)
    report.update(phase=phase, training_end_day=end, horizon=28, implementation='independent_official_formula',
                  paper_metric=False, input_sha256={p.name: sha256(p) for p in [sales_path, raw/'calendar.csv', raw/'sell_prices.csv', Path(forecasts)]})
    if actuals:
        report['actuals_sha256'] = sha256(actuals)
    details.to_csv(out / 'per_hierarchy_series.csv', index=False)
    details.groupby('level').agg(series=('group', 'size'), weight=('weight', 'sum'), contribution=('contribution', 'sum')).to_csv(out / 'per_level.csv')
    write_json(out / 'official_m5_score.json', report)
    return report


def export_run(run_dir, data_dir, out, phase='validation'):
    """Convert a single-origin run's bottom forecasts to id,F1..F28 format."""
    from rebunet.data.panel import Panel
    run_dir = Path(run_dir); panel = Panel(data_dir, verify=True)
    provenance = json.loads((run_dir / 'provenance.json').read_text())
    if provenance.get('data_sha256') != panel.fingerprint:
        raise ValueError('Run and M5 panel fingerprints differ')
    expected_origin = {'validation': 1913, 'evaluation': 1941}[phase]
    prediction = np.full((panel.n, 28), np.nan); seen = set()
    for path in sorted((run_dir / 'predictions').glob('*.npz')):
        with np.load(path, allow_pickle=False) as a:
            if a['point'].shape[1] != 28 or not np.all(a['origin'] == expected_origin):
                raise ValueError('Official export needs a single official 28-day forecast origin')
            for i, pred in zip(a['series'], a['point']):
                if int(i) in seen:
                    raise ValueError('Duplicate bottom series')
                seen.add(int(i)); prediction[i] = pred
    if len(seen) != panel.n or not np.isfinite(prediction).all():
        raise ValueError('Incomplete M5 forecast coverage')
    ids = pd.Series(panel.ids).str.replace(r'_(validation|evaluation)$', '', regex=True) + '_' + phase
    df = pd.DataFrame(prediction, columns=[f'F{i}' for i in range(1, 29)]); df.insert(0, 'id', ids)
    Path(out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False)
    return {'rows': len(df), 'phase': phase, 'output': str(out), 'training_protocol': 'retains original run training boundaries'}
