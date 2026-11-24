
# PredictMaint AI — ML Experiments

## 1. Objective

Predict industrial machine failures using the
AI4I 2020 Predictive Maintenance dataset.

The task is binary classification:

- 0: No machine failure
- 1: Machine failure

## 2. Experiment Tracking

- Platform: MLflow / DagsHub
- Repository: Abdel380/predictmaint-ai
- Experiment: predictmaint-ai-baselines
- Registered model: predictmaint-ai
- Initial champion version: 1
- Initial training dataset: data-v1
- Initial training Git SHA: c46a43fa9ff22a7a5768644dbd68df0faf198bf0
- Initial training DVC hash: 8f6dcc591e175d2b969190e89b696bd8

## 3. Initial Baseline Experiments

Both models were trained on 3,000 observations
and evaluated on the same fixed validation split
of 2,000 observations.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Average Precision | 0.3795 | 0.5559 |
| ROC-AUC | 0.8857 | 0.9626 |
| Precision | 0.1455 | 0.5278 |
| Recall | 0.8088 | 0.5588 |
| F1-score | 0.2466 | 0.5429 |

The Random Forest was selected as the initial
champion due to its higher Average Precision
and better precision-recall balance at threshold 0.5.

## 4. Primary Metric

Average Precision (AP) is the primary metric.

Machine failures represent approximately 3.39%
of the dataset, making accuracy unsuitable as
the main model-selection metric.

Higher Average Precision is better.

Secondary metrics:
- ROC-AUC
- Precision
- Recall
- F1-score

## 5. Initial Promotion Policy

The following quality gates are defined before
evaluating the v2 candidate on the held-out test set.

A candidate can replace the champion only if:

1. Candidate Average Precision >= 0.40
2. Candidate AP >= Champion AP + 0.02
3. Candidate Recall >= 0.50 at threshold 0.50
4. Evaluation uses the same held-out observations
5. The model can be loaded and serve predictions

The AP improvement of 0.02 is absolute,
not relative.

These are academic demonstration thresholds,
not validated industrial safety requirements.

Threshold changes must be documented before
running the corresponding promotion evaluation.

## 6. Evaluation Protocol

- Train v1: 3,000 observations
- Train v2: 6,000 observations
- Validation: 2,000 fixed observations
- Held-out test: 2,000 fixed observations

Validation is used for development and
model selection.

The held-out test set is reserved for the
predefined acceptance evaluation.

Champion and candidate must be evaluated on
the same held-out test observations.

The held-out test must not be used for fitting,
hyperparameter tuning or threshold optimization.

## 7. Promotion and Rejection

If all quality gates pass:
- Register the candidate as a new model version.
- Update the champion alias.
- Refresh or redeploy the inference service.
- Verify the model version and a live prediction.

If any gate fails:
- Keep the existing champion alias unchanged.
- Log the rejection decision and reason.
- Keep the candidate experiment for traceability.

## 8. Rollback

A previously approved model version can be
restored by moving the champion alias back
to that version.

The inference service must then reload the
approved model and verify its version.

## 9. Current Status

- Baseline training: completed
- Experiment tracking: completed
- Initial model registered: version 1
- Initial champion alias: version 1
- Registry loading smoke test: passed
- Held-out acceptance evaluation: pending
- Continuous training: pending
- Production deployment: pending