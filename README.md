# ReBuNet

A local-source implementation of **ReBuNet: Renewal-Aware Burst Mixture Networks for Probabilistic Forecasting of Sparse Count Time Series**.

<img width="2400" height="1208" alt="image" src="https://github.com/user-attachments/assets/4d6dc3e7-6ba4-4d59-8f38-0e7ece00ebb0" />

# ReBuNet


The repository provides the model, data adapters, training and evaluation pipeline, runnable baselines, experiment configurations, and shell entrypoints. Import the local `rebunet` directory directly; installing this project as a Python package is not required.

**Repository scope.** The repository includes ReBuNet, nine local baselines, experiment configurations, and interfaces for five external baselines. The complete comparison matrix uses both local and external implementations. Industrial sales data are private; datasets, trained checkpoints, and generated experiment outputs are excluded from the repository. Dataset splits and model settings are specified in the configuration files. Numerical results depend on the dataset snapshot, configuration, and execution environment.

## Contents

- [Method and implementation](#method-and-implementation)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Dataset preparation](#dataset-preparation)
- [Training and configuration](#training-and-configuration)
- [Running the experiment matrix](#running-the-experiment-matrix)
- [Synthetic event scenarios](#synthetic-event-scenarios)
- [Outputs and interpretation](#outputs-and-interpretation)
- [Official M5 scoring](#official-m5-scoring)
- [External baselines](#external-baselines)
- [Python API](#python-api)
- [Repository structure](#repository-structure)

## Method and implementation

ReBuNet forecasts nonnegative integer counts by separating occurrence timing from positive-event magnitude.

1. A causal grid-time encoder represents the recent daily history, including observed availability and historical features.
2. An event-time encoder represents previous positive events and their calendar gaps.
3. A renewal-aware decoder propagates a continuous age state over the forecast horizon. The scalar update is `r_next = (1 - p) * (r + 1)`.
4. A hurdle distribution assigns probability `1 - p` to zero. Its positive component is an ordered mixture of two zero-truncated negative-binomial distributions (ZTNB).
5. Conditional positive means are parameterized in order and mapped to underlying NB means by a differentiable numerical inverse.
6. A masked marginal negative log likelihood excludes unavailable observations without deleting dates from the time axis.

Implemented controls include Direct, GRS, scalar renewal, reachable exact-age distribution (EAD), dense capped-age distribution (CAD), single-component marks, unordered and raw-mean-ordered mixtures, shifted NB, zero-truncated Poisson, discretized lognormal, and removal of the event stream. Fixed-age, no-feedback and detached-feedback controls are available through the extended suite.

Nine baselines are runnable in this repository: Croston, SBA, TSB, ADIDA, DLinear, PatchTST, iTransformer, DeepAR and Switch-Hurdle. Neural baselines are local implementations or adaptations. Their model and training settings are recorded in the experiment configurations.

## Installation

Use **Python 3.12** for the pinned dependency set. Activate an existing environment or create one:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate `.venv\Scripts\activate` using the appropriate shell. Shell workflows require **Bash**, available through Linux, WSL or Git Bash. Use `bash script.sh`, not `sh script.sh`.

`requirements.txt` is the single dependency file. It includes all runtime libraries and the Kaggle downloader. Select a PyTorch build suitable for your CPU/CUDA environment when provisioning the environment. All direct runtime dependencies are pinned with `==`. The dependency file specifies the repository environment; pip resolves transitive dependencies. The paper reports PyTorch 2.1/CUDA 12.1 on an RTX 4090. Record the software and hardware environment when reporting performance measurements.

For a CPU-only PyTorch installation, install its matching wheel before the requirements:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU execution, provision a compatible PyTorch 2.6.0 CUDA build and driver.

No `pip install -e .` step is needed. The scripts use the active Python environment. To select another interpreter explicitly:

```bash
PYTHON_BIN=/path/to/environment/bin/python bash scripts/01_train_synthetic.sh
```

The scripts do not automatically install dependencies, activate Conda, or download datasets. All relative paths in shell commands are resolved from the project root. Quote paths containing spaces.

## Quick start

Two standalone Python examples are included:

```bash
python examples/train_synthetic.py --out results/api_example
python examples/plan_experiments.py --dataset m5 --out results/example_plan
```

See [examples/README.md](examples/README.md) for usage details and
[docs/formulas.md](docs/formulas.md) for formulas linked to their implementations.

Run the complete small synthetic workflow:

```bash
bash scripts/01_train_synthetic.sh --out results/synthetic_example
```

This generates data, checks the splits, trains a small model, selects a checkpoint on validation data, evaluates the test split, and creates figures. Outputs appear under:

```text
results/synthetic_example/
  data/
  preflight/
  model/
  figures/
  workflow_status.json
```

The default synthetic configuration has 48 series, 224 days, a 28-day horizon, two training epochs and three sampled batches per epoch. Use this configuration to verify the end-to-end workflow; use the dataset configurations for benchmark training.

Predictions, trained weights, metrics, and four figure files are generated locally.

Use a new output directory for another completed run. To see all options:

```bash
bash scripts/01_train_synthetic.sh --help
bash scripts/02_train_data.sh --help
bash scripts/03_run_experiments.sh --help
python run.py --help
```

## Dataset preparation

Datasets are stored locally and excluded from the repository. Local `data/`, `results/`, `logs/`, credentials and checkpoints are ignored by Git.

### M5: official files

Obtain data from the [M5 Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). The [organizers' repository](https://github.com/Mcompetitions/M5-methods) provides additional official resources and released evaluation data.

Place these files in `data/raw/m5/`:

```text
sales_train_evaluation.csv
calendar.csv
sell_prices.csv
```

The standard adapter checks 30,490 bottom-level series, 1,941 sales days, the official date range, hierarchy counts, unique identifiers, calendar continuity, and prices. It retains all daily sales values as observations. Calendar events and SNAP are known covariates; realized prices are historical features only.

Prepare and validate the dataset without training:

```bash
bash scripts/02_train_data.sh --dataset m5 \
  --input data/raw/m5 --prepare-only --out results/m5_preparation
```

Alternatively, after accepting the competition rules and configuring your own Kaggle credentials:

```bash
bash scripts/download_m5.sh
```

The download script prepares the panel; run the preparation workflow with `--skip-prepare --prepare-only` if you also want the split-readiness report. Credentials must never be committed.

### New Beiyang: daily sales and availability

Provide a complete daily sales export and corresponding inventory logs. A sales row requires:

| Field             | Meaning                             |
| ----------------- | ----------------------------------- |
| `vem_id`          | Device identifier, read as a string |
| `merc_id`         | SKU identifier, read as a string    |
| `date`            | Calendar date                       |
| `daily_quantity`  | Nonnegative integer sales count     |
| `merc_sale_price` | Optional historical price           |
| `coupon_amount`   | Optional historical coupon amount   |

The inventory CSV requires `vem_id,merc_id,date,available`, where `available` is an explicit 0/1 indicator derived from inventory/device logs. `capacity` is **not** interpreted as availability. Changing scene/location metadata are not treated as permanent static attributes.

Example inventory schema:

```csv
vem_id,merc_id,date,available
machine_001,sku_001,2024-01-01,1
machine_001,sku_001,2024-01-02,0
machine_001,sku_001,2024-01-03,1
```

Prepare the real panel and check readiness:

```bash
bash scripts/02_train_data.sh --dataset newbeiyang \
  --input data/raw/newbeiyang/daily_sales.csv \
  --inventory data/raw/newbeiyang/inventory_daily.csv \
  --sales-log-complete --prepare-only --out results/newbeiyang_preparation
```

Use `--sales-log-complete` **only if the sales export is complete**. This permits zero filling when inventory confirms availability and the complete sales export has no transaction row. Without this declaration, available dates lacking sales rows are rejected. Dates with neither sales nor availability evidence remain masked. Duplicate device–SKU–date records must be aggregated explicitly before import.

Audit an export without training:

```bash
bash scripts/02_train_data.sh --dataset newbeiyang \
  --input data/raw/newbeiyang/daily_sales.csv --audit-only
```

The full history-length matrix requires at least 196 calendar days to construct one training, validation, and test window with the largest context. The paper uses an industrial snapshot containing 18,642 series and 1,826 days. Select chronological boundaries that match the supplied export and verify usable-window counts with `preflight`.

### Generic industrial panel

The generic adapter accepts a daily CSV with `series_id,date,y,available`. Optional numeric `hist_*` columns become historical covariates with past-only forward filling and missing indicators.

```bash
python run.py prepare industrial \
  --input data/raw/industrial_daily.csv --out data/processed/industrial
python run.py preflight --config configs/datasets/industrial.yaml --out results/industrial_preflight
```

Missing dates are unavailable, not observed zeros. Future realized availability is used only to mask the loss and evaluation; it is never a future model input.

## Training and configuration

### Split conventions

| Configuration                      | Training end (exclusive index) | Validation end (exclusive index) | Expected total days |
| ---------------------------------- | -----------------------------: | -------------------------------: | ------------------: |
| `configs/datasets/m5.yaml`         |                           1885 |                             1913 |                1941 |
| `configs/datasets/newbeiyang.yaml` |                           1770 |                             1798 |                1826 |
| `configs/datasets/industrial.yaml` |                           1770 |                             1798 |                1826 |
| `configs/examples/synthetic.yaml`  |                            168 |                              196 |                 224 |

These boundaries define the repository's default evaluation protocol. For a different data snapshot or split protocol, set the boundaries in the dataset configuration before training. Integer boundaries are right-exclusive; date-string boundaries include the specified date. Validation and test partitions must each contain a complete 28-day horizon.

The real model defaults are hidden dimensions 128/96 for M5/industrial data, lookbacks 112/84, batch size 256, learning rate 0.0005, and a 28-day forecast horizon. Probabilistic models use validation NLL for checkpoint selection. Deterministic point models use the corresponding scaled point-error criterion. M5 validation and test weights are computed separately using the 28 days before their respective forecast origins.

### Train on prepared data

```bash
bash scripts/02_train_data.sh --dataset m5 --skip-prepare \
  --out results/m5_full --set train.device=cuda

bash scripts/02_train_data.sh --dataset newbeiyang --skip-prepare \
  --out results/newbeiyang_full --set train.device=cuda
```

Omit `--skip-prepare` and supply the raw-data arguments to perform preparation and training in one command. `--data` overrides the processed panel directory. Use repeated `--set` arguments to override nested configuration fields:

```bash
bash scripts/02_train_data.sh --dataset m5 --skip-prepare \
  --out results/m5_seed1 --set train.seed=1 --set train.batch_size=128
```

Resume an interrupted run using the **same** output directory, configuration and data:

```bash
bash scripts/02_train_data.sh --dataset m5 --skip-prepare \
  --out results/m5_full --set train.device=cuda --resume
```

Completed runs are not overwritten. The trainer restores optimizer and random-number state from `last.pt`. The scripts record errors in `workflow_status.json`; `logs/` contains the console output. Pipeline failures retain their nonzero exit status through `tee`.

### Validation-only tuning

```bash
python run.py tune --config configs/datasets/m5.yaml \
  --grid configs/experiments/tuning_grid.yaml --out results/m5_tuning
```

The included grid defines a configurable search space. The selected architecture and training settings are saved in `selection.json`; apply them to the experiment configuration before running the five-seed suite. The tuning command does not automatically rewrite the dataset YAML files.

## Running the experiment matrix

Follow this order:

1. Obtain the full datasets and availability logs.
2. Prepare panels and resolve every preflight failure.
3. Set chronological boundaries, common forecast origins, and available features in the configurations.
4. Select model-specific settings on validation data and freeze them.
5. Configure external baseline integrations for the complete comparison, or select the local subset.
6. Execute the five-seed matrix for each dataset.
7. Inspect coverage, per-run metrics, aggregated tables, strata and diagnostics.

### Generate plans

```bash
bash scripts/03_run_experiments.sh --kind paper --dataset m5 --out results/m5_paper
bash scripts/03_run_experiments.sh --kind paper --dataset newbeiyang --out results/newbeiyang_paper
```

Each default dataset plan contains 179 runs. Neural methods use seeds 0–4; deterministic classical methods run once. Plan generation does not train models or require the full dataset to be present.

### Execute

```bash
bash scripts/03_run_experiments.sh --kind paper --dataset m5 \
  --out results/m5_paper --execute
bash scripts/03_run_experiments.sh --kind paper --dataset newbeiyang \
  --out results/newbeiyang_paper --execute
```

Complete-matrix execution requires the external baseline integrations. To execute only the available implementations and record coverage:

```bash
bash scripts/03_run_experiments.sh --kind paper --dataset m5 \
  --out results/m5_partial --execute --allow-missing-external
```

A partial matrix returns exit code **2** and status `partial_missing_external`, even if all implemented methods finish. See `suite/coverage.json` for the missing methods. This status identifies a subset run; `coverage.json` records its scope.

### Experiment mapping

| Paper experiment                            | Suite labels / operation                                     |
| ------------------------------------------- | ------------------------------------------------------------ |
| Table 2: dataset statistics                 | Data preparation and `preflight`                             |
| Tables 3–4: point/probabilistic comparisons | `full` and baseline labels                                   |
| Table 5: decoder states                     | `full`, `state_direct`, `state_grs`, `state_ead`, `state_cad` |
| Table 6: mark ordering                      | `full`, `mark_single`, `mark_unordered`, `mark_raw_ordered`  |
| Table 7: component ablation                 | State, mark and `no_event_stream` controls                   |
| Table 8: efficiency                         | Automatic profiles for designated variants                   |
| Table 9: configurations                     | Saved configuration snapshots                                |
| Table 10: history length                    | `history_{14,28,56,84,112}_{scalar,grs}`                     |
| Table 11: propagation and stability         | Full-model propagation, independently trained EAD, shifted NB, seed summaries |
| Sparsity groups                             | Full vs GRS, training-period ADI quartiles                   |
| Burstiness groups                           | Full vs single ZTNB, training-period burst quartiles         |
| Appendix D.2: positive distribution         | `full`, `mark_single`, `mark_ztpoisson`, `mark_dlognormal`   |
| Appendix D.3: age caps                      | `full`, `cap_32`, `cap_64`, `cap_128`                        |

Select individual comparisons:

```bash
bash scripts/03_run_experiments.sh --kind paper --dataset newbeiyang \
  --out results/newbeiyang_states --labels full state_direct state_grs state_ead state_cad \
  --seeds 0 1 2 3 4 --execute
```

The suite also makes history/cap extensions available on M5, although the paper emphasizes industrial-data sensitivity results. To include additional mechanism controls, use the underlying CLI:

```bash
python run.py suite --config configs/datasets/newbeiyang.yaml --out results/extra_controls \
  --scope all --labels extra_fixed_age extra_no_feedback extra_detached --execute
```

### Propagation and efficiency

Suite state variants are independently trained. Fixed-parameter propagation instead compares the **same checkpoint** under scalar, EAD and sufficiently large-cap CAD updates:

```bash
python run.py propagation --checkpoint results/m5_full/model/best.pt --out results/m5_full/propagation
python run.py profile --checkpoint results/m5_full/model/best.pt \
  --out results/m5_full/profile.json --device cuda --batch-size 256 --repeats 30
```

Profiles use evaluation mode without gradients and exclude data loading. GPU timings synchronize CUDA. GPU memory measurements are reported for CUDA runs. Compare latency measurements under matching hardware, batch sizes, and execution settings.

## Synthetic event scenarios

Six configurable scenarios supplement the real-data experiments:

| Scenario            | Purpose                                                    |
| ------------------- | ---------------------------------------------------------- |
| `renewal_reference` | Age-dependent occurrences and heterogeneous positive marks |
| `age_independent`   | Remove true age dependence                                 |
| `rare_large_burst`  | Rare, large positive-event magnitudes                      |
| `planned_promotion` | Activities announced before forecast origins               |
| `unannounced_shock` | Unobserved future shock labels excluded from model inputs  |
| `stockout`          | Random and sustained unavailable intervals                 |

Run the small scenario matrix:

```bash
bash scripts/03_run_experiments.sh --kind scenarios --out results/scenarios --execute
```

This runs six scenarios, four default controls (`full`, Direct, GRS and single ZTNB), and one seed. Omit `--execute` to generate data and plans only. Use the industrial model configuration and five seeds for larger experiments:

```bash
bash scripts/03_run_experiments.sh --kind scenarios --config configs/datasets/industrial.yaml \
  --seeds 0 1 2 3 4 --out results/scenarios_large --execute
```

Modify `configs/scenarios/templates.yaml` to change the number of series, duration, generator seed and event schedules. Event `start_day` is zero-based; `announced: true` means the schedule was known before all forecast origins. The model seed and data-generator seed are independent.

Generator truth is saved separately under `oracle/` and excluded from the training manifest. Event/background metrics use observed targets only. A stockout interval with no available observations reports zero evaluable targets rather than scoring hidden demand. Scenario results are reported separately from real-data benchmark results.

## Outputs and interpretation

No historical run directories are included. The following files are generated when you run the code:

| Output                                                     | Interpretation                                               |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| `workflow_status.json`                                     | Planned, completed, partial or failed workflow status        |
| `preflight/preflight.json`                                 | Shape, split boundaries, usable targets and readiness        |
| `model/config.json`                                        | Resolved run configuration                                   |
| `model/provenance.json`                                    | Environment, data fingerprint, code hashes and split dates   |
| `model/training.json`                                      | Epoch losses, validation values, timing and warm-up state    |
| `model/best.pt`, `model/last.pt`                           | Selected checkpoint and resumable training state             |
| `model/predictions/part_*.npz`                             | Forecast origins, observed targets/masks, means and distribution parameters |
| `model/metrics.json`                                       | Aggregate point and probabilistic metrics                    |
| `model/per_series.csv`, `strata.csv`                       | Per-series and training-defined group scores                 |
| `model/reliability.csv`, `by_horizon.csv`                  | Occurrence calibration and horizon-wise scores               |
| `figures/`                                                 | Calibration, horizon, predictive distribution and forecast plots |
| `suite/plan.json`, `coverage.json`                         | Requested jobs and execution completeness                    |
| `suite/summary.csv`, `T*_runs.csv`                         | Seed summaries and paper-table run selections                |
| `suite/strata_relative_improvement.csv`                    | Paired full-vs-control relative NLL improvements             |
| `scenarios/all_scenario_runs.csv`, `all_event_periods.csv` | Cross-scenario and event-period summaries                    |

For suite jobs, `model/` in the table is replaced by `suite/<variant>/seed_<seed>/`. Scenario suites use `scenarios/<scenario>/runs/<variant>/seed_<seed>/`.

### Metrics

- **MAE:** average absolute error within each eligible series, then across series.
- **RMSSE:** per-series root mean squared error divided by the training first-difference scale, then averaged. Unusable/zero scales are reported and excluded.
- **bWRMSSE:** revenue-weighted bottom-level RMSSE on M5, not the official full hierarchy score. Outside M5, the same implementation field uses uniform weights and should not be presented as an official M5 score.
- **NLL:** negative marginal log likelihood over available test tuples.
- **DRPS:** discrete ranked probability score, with adaptive support and a reported omitted-tail bound.
- **AUPRC:** occurrence average precision, computed on the same available targets.
- **Burst-MAE:** error for test positives strictly above the series' training positive-count 95th percentile.
- **Brier / interval coverage / width:** additional occurrence and marginal-distribution diagnostics.

Lower is better for error/loss scores; higher AUPRC is better. Interval coverage should be interpreted relative to its nominal level and width. Unavailable metrics are left empty. Seed standard deviations use `ddof=1`; one-run deterministic methods have no estimated seed standard deviation.

ADI and burst quartiles are defined from training data. Ties can leave empty groups. Relative NLL improvement is `(baseline - full) / baseline * 100%`, using matched seeds and target coverage.

Regenerate summaries and coverage:

```bash
python run.py summarize --out results/m5_paper/suite
python run.py paper-report --config configs/datasets/m5.yaml \
  --suite-dir results/m5_paper/suite --out results/m5_paper/coverage
python run.py plot --run results/m5_full/model --out results/m5_full/figures
```

`paper-report` evaluates the default full paper matrix. A deliberately restricted label/seed run will therefore remain incomplete in that report; use the selected suite's `coverage.json` to inspect its own scope. Pass the same edited configuration used for training when checking provenance.

## Official M5 scoring

The paper metric and official competition metric are separate. The official-style scorer implements all 12 hierarchy levels and aggregates one level at a time.

| Phase      | History for scaling | Weight window | Forecast target |
| ---------- | ------------------- | ------------- | --------------- |
| Validation | d1–d1913            | d1886–d1913   | d1914–d1941     |
| Evaluation | d1–d1941            | d1914–d1941   | d1942–d1969     |

Export predictions from the configured M5 holdout and score them:

```bash
python run.py export-m5 --run results/m5_full/model \
  --data data/processed/m5 --out results/m5_validation.csv
python run.py score-m5-official --raw data/raw/m5 \
  --forecasts results/m5_validation.csv --phase validation --out results/m5_official
```

Exporting does not retrain the model through d1913. The scorer uses official-style scaling and weights for the forecast interval, while the original run retains its own training boundaries. Do not overwrite the paper bWRMSSE with this score.

The evaluation phase requires released official actuals in `id,d_1942,...,d_1969` format and forecasts in `id,F1,...,F28` format:

```bash
python run.py score-m5-official --raw data/raw/m5 --phase evaluation \
  --forecasts private_predictions.csv --actuals official_actuals.csv --out results/m5_private_score
```

The standard 1,941-day training pipeline performs the paper-style holdout evaluation; automatic private-phase refitting/submission is not provided. The scorer is an independent formula implementation, not a bundled official scoring program.

## External baselines

The following methods connect through the external baseline interfaces:

| Registry name     | Method                 |
| ----------------- | ---------------------- |
| `robust_rnn`      | RobustRNN              |
| `transformer_idf` | Transformer-IDF        |
| `deep_renewal`    | Deep Renewal Process   |
| `deep_state`      | Deep State Space       |
| `score_hurdle_nb` | Score-driven hurdle-NB |

### Model factory

For methods compatible with this trainer, add a factory to the dataset configuration:

```yaml
external:
  robust_rnn:
    factory: my_project.robust_adapter:build_model
```

`build_model(model_config, dimensions)` returns a `torch.nn.Module`. Set `probabilistic` and implement `forward(batch)` returning a `Forecast` or a nonnegative `[B,H]` tensor. The batch contract is defined in `rebunet/data/panel.py`. Optional hooks are `auxiliary_loss()` and `warmup(active)`. Evaluation must not use future targets or realized future masks as features.

### Predictions from a separate trainer

Use this path when the original method needs its own optimizer, inference procedure or ensemble. Export `external_manifest.json` with the following provenance fields:

| Field                  | Required content                                           |
| ---------------------- | ---------------------------------------------------------- |
| `method`               | Registry name listed above                                 |
| `implementation_url`   | Source repository or implementation archive URL            |
| `commit_or_version`    | Revision used for the run                                  |
| `training_config`      | JSON object containing the resolved training configuration |
| `validation_selection` | Validation criterion and selected configuration            |
| `seed`                 | Integer run seed                                           |
| `data_fingerprint`     | SHA256 of the processed panel's `manifest.json`            |

Each prediction shard `part_XXXXX.npz` contains `series [B]`, `origin [B]`, `y [B,H]`, `mask [B,H]` and `point [B,H]`. Probabilistic methods additionally provide `p [B,H]`, `w/mu/k [B,H,C]` and scalar-string `kind` (`ztnb`, `nb`, `shifted_nb`, `ztpoisson`, or `dlognormal`). For ordinary NB, `p=1`; its zero mass comes from the NB distribution. Distribution parameters must come from the external model's predictive distribution. Point-only methods provide point forecasts.

```bash
python run.py import-external --config configs/datasets/m5.yaml \
  --set model.name=deep_renewal --set train.seed=0 \
  --input external_results/deep_renewal/seed_0 \
  --out results/m5_paper/suite/deep_renewal/seed_0
```

The importer checks identifiers, expected series-origin coverage, targets, masks, seed and fingerprint before rescoring. Repeat for each seed. Additional predictive distribution families require a corresponding extension to the forecasting and scoring interfaces.

## Python API

The shell workflows call the same local API available to user scripts:

```python
from rebunet.api import load_config, prepare_data, train, run_suite

prepare_data('synthetic', out='data/processed/synthetic')
cfg = load_config('configs/examples/synthetic.yaml')
metrics = train(cfg, out='results/api_example')
print(metrics)

run_suite('configs/datasets/m5.yaml', out='results/api_m5_plan')
```

Additional functions include `check_data`, `audit_newbeiyang`, `run_scenarios`, `tune`, `summarize`, `plot_results`, `paper_report`, `score_m5` and `export_m5`. Configuration arguments accept YAML paths or dictionaries; nested dictionary overrides are copied before application.

Unlike shell entrypoints, the API does not change the calling process's working directory. Run from the project root or use absolute configuration, dataset and output paths.

## Repository structure

| Path                               | Responsibility                                               |
| ---------------------------------- | ------------------------------------------------------------ |
| `rebunet/api.py`, `rebunet/cli.py` | Public function and CLI interfaces                           |
| `rebunet/data/`                    | Dataset preparation, observation masks, windows and generators |
| `rebunet/models/`                  | Encoders, renewal decoder and probability distributions      |
| `rebunet/baselines/`               | Runnable comparison models                                   |
| `rebunet/training/`                | Single-run optimization and validation-only tuning           |
| `rebunet/evaluation/`              | Metrics, M5 hierarchy scoring and visualization              |
| `rebunet/experiments/`             | Suite orchestration, coverage, diagnostics and external adapters |
| `configs/base.yaml`                | Shared model and training defaults                           |
| `configs/datasets/`                | M5 and industrial dataset configurations                     |
| `configs/examples/`                | Small executable configuration                               |
| `configs/experiments/`             | Hyperparameter search grid                                   |
| `configs/scenarios/`               | Six event templates and small scenario configuration         |
| `scripts/`                         | Bash entrypoints and shared API orchestration                |
| `examples/`                        | Standalone Python API examples                               |
| `docs/formulas.md`                 | Mathematical definitions and source mapping                  |
| `run.py`                           | Root CLI entrypoint                                          |
| `requirements.txt`                 | Single pinned runtime dependency list                        |

Only source, configuration, examples and documentation are distributed.
`results/`, `checkpoints/`, `debug/`, logs, private datasets, generated weights
and development test files are excluded. Running an example creates new local
outputs that are ignored by Git. Checkpoint selection and interrupted-run recovery are part of the training workflow.

The package remains at the project root to support direct imports without
installation. CLI and shell layers call the public API; training depends on
models and data; experiment orchestration composes training and evaluation.
Dataset terms and third-party implementation licenses are separate from the
repository's code license.

