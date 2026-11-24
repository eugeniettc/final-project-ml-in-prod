
# PredictMaint AI — Data Documentation

## 1. Data Source

- Dataset: AI4I 2020 Predictive Maintenance
- Provider: UCI Machine Learning Repository
- Dataset ID: 601
- URL: https://archive.ics.uci.edu/dataset/601/ai4i
- License: CC BY 4.0
- Dataset type: Synthetic industrial maintenance data
- Total observations: 10,000

## 2. Prediction Task

Binary classification of industrial machine failure.

Target: Machine failure

- 0: No machine failure
- 1: Machine failure

## 3. Input Features

- Type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

UID is retained for data traceability only.

Product ID and failure-mode indicators
(TWF, HDF, PWF, OSF, RNF) are excluded
from model inputs to avoid data leakage.

## 4. Data Quality

- Missing values: 0
- Duplicate rows: 0
- Duplicate feature combinations: 0
- Total failures: 339 (3.39%)

## 5. Data Splitting

Random state: 42
Method: Stratified random splitting

| Split | Rows | Failures |
|---|---:|---:|
| Train v1 | 3,000 | 102 |
| Train v2 | 6,000 | 203 |
| Validation | 2,000 | 68 |
| Test | 2,000 | 68 |

Validation and test sets remain fixed
between dataset versions.

Train v1 is a subset of Train v2.

## 6. Data Versions

### Version 1
- Git tag: data-v1
- Git commit: f21d45e
- Training records: 3,000
- DVC file: data/releases/v1/train.csv.dvc
- DVC hash: 8f6dcc591e175d2b969190e89b696bd8

### Version 2
- Git tag: data-v2
- Git commit: fc4ca34
- Training records: 6,000
- Change: 3,000 additional training records
- DVC file: data/releases/v2/train.csv.dvc
- DVC hash: c852112ede49666f3454963fd76d5276

## 7. Storage

DVC remote: myazure

Storage: Azure Blob Storage

Storage account: wut2026

Remote path: azure://dvc/predictmaint-ai

Data is stored in Azure using DVC.
Git tracks the corresponding DVC metadata.

## 8. Reproducibility

The datasets are generated using:

python -m ml.download_data
python -m ml.prepare_data

The DVC snapshots can be retrieved using
their Git tags and the configured Azure remote.
Authentication credentials are required.
