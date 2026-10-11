# ReBuNet

A local-source implementation of **ReBuNet: Renewal-Aware Burst Mixture Networks for Probabilistic Forecasting of Sparse Count Time Series**.

<img width="2400" height="1208" alt="image" src="https://github.com/user-attachments/assets/4d6dc3e7-6ba4-4d59-8f38-0e7ece00ebb0" />


## Overview

Sparse count time series are commonly observed in:

- retail demand forecasting,
- industrial sales prediction,
- unattended retail systems,
- event occurrence modeling.

Compared with conventional time series, sparse count sequences contain two coupled uncertainties:

1. whether a positive event occurs;
2. how large the positive count will be.

ReBuNet introduces a structured probabilistic forecasting framework consisting of:

1. **Dual-time encoding**
   - calendar-time representation;
   - event-time representation.

2. **Renewal-aware occurrence modeling**
   - differentiable probability-age feedback;
   - compact renewal state propagation across forecasting horizons.

3. **Burst mixture positive-count modeling**
   - ordered two-component zero-truncated negative-binomial mixture;
   - low-mean and high-mean latent regimes.

4. **Marginal probabilistic forecasting**
   - horizon-specific predictive distributions;
   - masked marginal composite likelihood training.


---

# Architecture

The overall framework contains four stages:

```
Historical observations
          |
          v
+-------------------------+
| Dual-Time Encoder       |
|                         |
| Calendar Encoder        |
| Event-Time Encoder      |
+-------------------------+
          |
          v
 Context Representation
          |
          v
+-------------------------+
| Renewal-Aware Decoder   |
|                         |
| Occurrence Probability  |
| Age Feedback State      |
+-------------------------+
          |
          |
          v
+-------------------------+
| Burst Mixture Decoder   |
|                         |
| ZTNB Mixture            |
| Low/High Mean Components|
+-------------------------+
          |
          v
 Predictive Distribution
```
## Contents

- [Method and implementation](#method-and-implementation)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Dataset preparation](#dataset-preparation)
- [Training and configuration](#training-and-configuration)
- [Reproducing the experiment matrix](#reproducing-the-experiment-matrix)
- [Synthetic event scenarios](#synthetic-event-scenarios)
- [Outputs and interpretation](#outputs-and-interpretation)
- [Official M5 scoring](#official-m5-scoring)
- [External baselines](#external-baselines)
- [Python API](#python-api)
- [Repository structure](#repository-structure)


---

# Installation

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA recommended


Install dependencies:

```bash
pip install -r requirements.txt
```


---

# Quick Start

## 1. Data Preparation

The framework supports sparse count forecasting datasets organized in the following format:

```
data/
├── counts.npy
├── covariates.npy
├── static_features.npy
└── masks.npy
```

where:

- `counts.npy`: non-negative integer count observations;
- `covariates.npy`: historical and known future covariates;
- `static_features.npy`: optional series-level attributes;
- `masks.npy`: target availability mask.


Unavailable observations are handled through masking rather than being treated as zero counts.


---

# 2. Training

Example:

```bash
python train.py \
    --config configs/rebunet.yaml
```


The training pipeline includes:

1. dual-time feature encoding;
2. renewal-aware occurrence prediction;
3. ordered burst mixture estimation;
4. marginal probabilistic likelihood optimization.


---

# 3. Evaluation

Run evaluation:

```bash
python evaluate.py \
    --checkpoint checkpoints/rebunet.pt
```


The evaluation module supports:

- point forecasting metrics;
- probabilistic forecasting metrics;
- occurrence prediction evaluation;
- distributional analysis.


---

# Dataset

## M5 Forecasting Dataset

ReBuNet is evaluated on the M5 forecasting benchmark.

The dataset contains:

- large-scale retail count series;
- strong sparsity patterns;
- temporal covariates.


## Industrial-Vending Dataset

The Industrial-Vending dataset contains daily machine-SKU sales sequences from an unattended retail environment.

Characteristics:

- sparse daily demand;
- irregular positive events;
- heterogeneous demand magnitudes.

The implementation supports availability-aware training and evaluation to distinguish unavailable observations from genuine zero demand.


---

# Model Components


## Dual-Time Encoder

ReBuNet combines:

### Calendar-Time Stream

Captures:

- recent count dynamics;
- temporal patterns;
- calendar covariates.


Implemented using gated temporal convolution.


### Event-Time Stream

Captures:

- positive-event intervals;
- positive-event magnitudes.


Implemented using recurrent event-history encoding.


---

## Renewal-Aware Occurrence Decoder

The occurrence branch predicts:

\[
p_h=P(Y_{t+h}>0)
\]


The renewal state is propagated by:

\[
r_{h+1}=(1-p_h)(r_h+1)
\]


This enables probability-driven renewal state evolution across forecasting horizons.


---

## Burst Mixture Decoder

Positive counts are modeled using a two-component zero-truncated negative-binomial mixture:

\[
P(Y=y|Y>0)
=
(1-g)f_0(y)+gf_1(y)
\]


where:

- `f0`: routine low-mean component;
- `f1`: high-mean burst component;
- `g`: latent burst probability.


The mixture components are ordered according to positive conditional means to maintain stable interpretation.


---

# Repository Structure

```
ReBuNet/
│
├── configs/
│
├── datasets/
│
├── models/
│   ├── encoder.py
│   ├── renewal_decoder.py
│   ├── burst_mixture.py
│   └── rebunet.py
│
├── experiments/
│
├── utils/
│
├── train.py
├── evaluate.py
└── requirements.txt
```


---

# Experiments

The repository includes:

- ReBuNet model implementation;
- training pipeline;
- evaluation scripts;
- configuration files;
- baseline models;
- diagnostic tools.


The implementation follows the model formulation and experimental design described in the paper.


---

# Citation

If you use this repository in your research, please cite:


```bibtex
@article{li2026rebunet,
title={ReBuNet: Renewal-Aware Burst Mixture Networks for Probabilistic Forecasting of Sparse Count Time Series},
author={Li, Qianyang and Liu, Bingqing and Zhang, Xingjun and Yu, Ming and Zhao, Zhenjian and Wei, Jia},
journal={Information Sciences},
year={2026}
}
```

