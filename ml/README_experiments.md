# PredictMaint AI — Machine Learning Experiments and Model Lifecycle

This document records the dataset versions, training experiments, evaluation protocol, model promotion policy, continuous-training evidence, deployment verification, and rollback demonstration for **PredictMaint AI**.

> **Evidence scope (25 November 2026):** Baseline training, an actual v2 retraining run, controlled promotion and rejection, a local rollback, and production inference with MLflow champion v2 have been demonstrated. The protected **CT Production Release** workflow and its automatic Render failure-recovery path are implemented but have not yet been demonstrated end to end on Render. The isolated demo registry and the production registry are intentionally separate.

## 1. Prediction task and dataset

Predict binary industrial machine failure using the [AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i) (UCI, CC BY 4.0).

- Target: `Machine failure` (`0`: no failure; `1`: failure).
- Model inputs: `Type`, `Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, and `Tool wear`.
- Data quality: 10,000 observations, including 339 failures (**3.39%**). `UID` is used for traceability, not prediction; product identifiers and failure-mode indicators are excluded to avoid leakage.
- Data versions are tracked with **DVC** and an **Azure Blob Storage** remote, not committed as raw CSV files to Git.

| Dataset release | Training observations | Git tag | DVC MD5 (training snapshot) |
| --- | ---: | --- | --- |
| v1 | 3,000 | `data-v1` | `8f6dcc591e175d2b969190e89b696bd8` |
| v2 | 6,000 | `data-v2` | `c852112ede49666f3454963fd76d5276` |

The v2 release adds **3,000 training observations** to v1. The validation and held-out test sets remain fixed across releases. The dataset release commits recorded in [`README_data.md`](README_data.md) are `f21d45e` (v1) and `fc4ca34` (v2).

See [`README_data.md`](README_data.md) for the source, license, split construction, DVC pointers, hashes, and reproducibility commands.

## 2. Tracking and registry

- Shared MLflow tracking server and registry: [PredictMaint AI on DagsHub](https://dagshub.com/Abdel380/predictmaint-ai.mlflow/).
- Baseline experiment: `predictmaint-ai-baselines`.
- Continuous-training experiment: `predictmaint-ai-continuous-training`.
- **Production registered model:** `predictmaint-ai` (alias: `champion`).
- **Isolated demonstration model:** `predictmaint-ai-ct-demo-henri-fixed` (independent alias: `champion`).
- Initial approved production model: `predictmaint-ai` **version 1**.
- Last verified online production champion as of 25 November 2026: `predictmaint-ai` **version 2**.

The inference service resolves `champion` to a **concrete model version at startup**, loads that version from MLflow, and reports the version via `/ready`, `/model-info`, and `/predict`. Updating the registry alias **does not hot-reload** an already-running FastAPI process: a restart/redeployment and a real prediction are required to verify any model change.

## 3. Baseline experiments (v1)

Two classifiers were trained using the **3,000-row v1 training split** and compared on the same **2,000-row validation split**:

| Validation metric | Logistic Regression | Random Forest |
| --- | ---: | ---: |
| Average Precision (AP) | 0.3795 | **0.5559** |
| ROC-AUC | 0.8857 | **0.9626** |
| Precision at 0.5 | 0.1455 | **0.5278** |
| Recall at 0.5 | **0.8088** | 0.5588 |
| F1 at 0.5 | 0.2466 | **0.5429** |

**Selection:** Random Forest was chosen as the baseline champion because it delivered better AP and a stronger overall precision–recall balance at the fixed 0.5 decision threshold. Logistic Regression had higher recall, but with substantially lower precision.

Initial model provenance, as recorded in the original experiment log:

- Initial registered version: **v1**.
- Training dataset: **data-v1**.
- Training Git SHA: `c46a43fa9ff22a7a5768644dbd68df0faf198bf0`.
- Training DVC hash: `8f6dcc591e175d2b969190e89b696bd8`.

**Do not compare** the validation AP of **0.5559** directly with the held-out test AP in Section 6; they were measured on different evaluation splits.

## 4. Primary metric and evaluation protocol

**Primary selection metric: Average Precision (AP); higher is better.** AP is suitable for this imbalanced binary classification task, where machine failures account for only 3.39% of observations. Accuracy alone could conceal poor failure detection.

Secondary metrics are ROC-AUC, precision, recall, F1, and the confusion matrix. Precision, recall, and F1 are calculated at a **0.5 probability threshold**.

| Partition | Rows | Positive labels |
| --- | ---: | ---: |
| v1 training | 3,000 | 102 |
| v2 training | 6,000 | 203 |
| Validation (fixed) | 2,000 | 68 |
| Held-out test (fixed) | 2,000 | 68 |

The data preparation uses a stratified split (`random_state=42`). Training, validation, and test partitions have disjoint `UID` values; `ml/retrain.py` validates the schema, missing values, binary labels, duplicate IDs, and split overlap. Preprocessing is fitted **only on the training split** as part of the scikit-learn pipeline.

Promotion comparisons evaluate the current champion and the candidate on **the same held-out test observations**. The held-out test is not used to fit the models or tune thresholds. This is an academic, randomly partitioned demonstration rather than a chronology-preserving industrial time-series validation.

## 5. Training configuration and quality gates

The continuous-training candidate uses a scikit-learn pipeline:

- Numeric variables: `StandardScaler`.
- `Type`: `OneHotEncoder(handle_unknown="ignore")`.
- Classifier: `RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)`.

The promotion rules are implemented in [`promotion_policy.py`](promotion_policy.py). **All three conditions must hold**:

| Quality gate | Requirement | Direction |
| --- | --- | --- |
| Candidate AP floor | `candidate_AP >= 0.40` | Higher is better |
| Minimum AP improvement | `candidate_AP - champion_AP >= 0.02` | Higher is better |
| Candidate recall at threshold 0.5 | `candidate_recall >= 0.50` | Higher is better |

The `0.02` AP improvement is **absolute**, not a relative percentage. Eligibility is assessed after data validation, consistent held-out evaluation, and model-loadability checks. A failing candidate **must not replace** the champion. These are **academic project thresholds**, not validated industrial safety guarantees.

## 6. Verified v2 retraining and controlled promotion

On **25 November 2026**, GitHub Actions executed a real retraining using the newer **v2 DVC snapshot** and the isolated demonstration model `predictmaint-ai-ct-demo-henri-fixed`.

1. The retraining gate detected **v2 > v1** and returned `TRAIN`.
2. The workflow fetched the data from Azure using DVC and trained on **6,000 observations**.
3. The current demonstration champion **v1** and v2 candidate were evaluated on the same **2,000-row held-out test split**.
4. The candidate satisfied all three quality gates.
5. The candidate was approved and promoted to **demo registry version 4** (`champion`), without changing the production registry.

| Held-out test metric | Demo champion v1 | Candidate trained on v2 |
| --- | ---: | ---: |
| Average Precision | 0.6186 | **0.6481** |
| ROC-AUC | 0.9604 | **0.9653** |
| Precision | **0.6316** | 0.5610 |
| Recall | 0.5294 | **0.6765** |
| F1 | 0.5760 | **0.6133** |
| True positives | 36 | **46** |
| False negatives | 32 | **22** |
| False positives | **21** | 36 |
| True negatives | **1,911** | 1,896 |

**Promotion decision:** `ELIGIBLE` (AP improvement **+0.0295**; recall **0.6765**). The v2 candidate detects more failures, with a trade-off in false positives. These figures describe the **isolated demo's comparison**, not a newly executed production CT release.

Direct evidence:

- [GitHub Actions — actual v2 training, quality-gate evaluation, and demo promotion](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45575717813).
- [MLflow CT run — `9dd7e7aa262e43ea835cf4bdf150aff8`](https://dagshub.com/Abdel380/predictmaint-ai.mlflow/#/experiments/2/runs/9dd7e7aa262e43ea835cf4bdf150aff8).
- Recorded output: `Decision: TRAIN`, `PROMOTION ELIGIBLE`, `PROMOTION SUCCESSFUL`, `New champion version: 4` (isolated demo registry).

## 7. Verified controlled rejection

A separate **historical CT run** was rechecked against the production champion **v2** on **24–25 November 2026**:

- Historical MLflow CT run: `de410647f3f94c14a56db7b1940d0977`.
- Current production champion at the check: **v2**.
- Champion AP: **0.6482**.
- Candidate AP: **0.6481**.
- Result: `REJECTED` because `minimum_improvement` failed.
- No alias modification was performed; the existing champion remained **v2**.

Evidence: [GitHub Actions — controlled rejection](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45506491721).

This establishes the **retention/rejection** path. It is distinct from the successful demonstration promotion above.

## 8. Continuous training and promotion safeguards

The workflow [`.github/workflows/continuous-training.yml`](../.github/workflows/continuous-training.yml) supports `workflow_dispatch` with a dataset input (`v1` or `v2`) and execution mode (`validate-only` or `train-and-check`). Its principal stages are:

1. Run automated ML tests.
2. Retrieve DVC snapshots from Azure and validate the data contract and split separation.
3. Resolve the MLflow champion and inspect dataset provenance (`dataset_version`, `dataset_dvc_md5`).
4. Return `SKIPPED` if the selected dataset is older or already represented by the champion; return `TRAIN` for a newer validated release.
5. Train a candidate, evaluate it against the champion, and log metrics, parameters, Git SHA, dataset/DVC identifiers, decision artifact, and candidate model to MLflow.
6. Perform the promotion decision in **read-only** mode by default.

The separate protected [CT production release workflow](../.github/workflows/ct-production-release.yml) accepts a CT run ID, checks the existing live application and candidate provenance, and can perform an explicitly authorized production promotion plus Render deployment in `release` mode. `ml/ct_release.py` implements fallback rollback and redeployment logic if a post-promotion release fails.

**Scope limitation:** The controlled training/promotion evidence above does **not** prove that the protected production CT release and its automatic failover have successfully run end to end. This remains a final integration-verification item. No production promotion should be executed solely to create documentation evidence.

## 9. Verified rollback and recovery behavior

The rollback script [`rollback.py`](rollback.py) has a **dry-run default** and requires `--execute` to change an MLflow alias. It checks the current champion version, `READY` status of the target, CT approval provenance, the recorded `previous_champion_version`, and a local inference precheck.

On **25 November 2026**, a controlled rollback of **demo model v4 → v1** was completed. After a FastAPI restart using the isolated demo model, `/model-info` reported **v1** and a live local `/predict` call returned:

- `failure_prediction = 0`
- `risk_score = 0.0058`
- `model_version = 1`

The audit event was recorded in [this MLflow rollback run](https://dagshub.com/Abdel380/predictmaint-ai.mlflow/#/experiments/2/runs/62ee72d6e93a41de99705dbedd304dcd).

**This was a real rollback in an isolated registry and local inference process, not a rollback of Render production.** The full repeatable procedure, commands, reloading requirements, and failure-to-become-ready runbook are in [`DEPLOYMENT.md`](../DEPLOYMENT.md#4-mlflow-model-rollback).

## 10. Production verification and current status

At the latest verified production deployment on **25 November 2026**:

- Registered production model: `predictmaint-ai`.
- MLflow alias: `champion`.
- Concrete model served: **v2**.
- The Render production deployment and live HTTP smoke tests passed, including model-version verification and a prediction.
- The production synchronization monitor also passed.

Evidence:

- [Production deployment — tests, Docker build, Render deployment, and HTTP smoke verification](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45695248713).
- [Production model synchronization monitor](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45696050322).
- [Production application](https://predictmaint-ai-production.onrender.com/) and [model information endpoint](https://predictmaint-ai-production.onrender.com/model-info).

A repeatable verification command is:

```bash
python -m ml.verify_deployment \
  --url https://predictmaint-ai-production.onrender.com \
  --expected-version 2
```

The initial baseline model **v1 was trained and registered**, but separate historical evidence of the **first online production service serving v1** has not been established by the deployment logs reviewed here; do not present the v2 production deployment as proof of that particular event.

### Completion summary

| Item | Recorded state |
| --- | --- |
| Baseline models and initial champion v1 | Completed |
| DVC dataset versions v1 and v2 | Completed |
| Real v2 continuous-training run | Demonstrated |
| Successful controlled promotion | Demonstrated in isolated registry |
| Controlled rejection retaining champion | Demonstrated |
| Rollback plus local HTTP inference | Demonstrated in isolated registry |
| Production Docker/Render inference with champion v2 | Verified |
| Protected production CT release, including recovery on Render | Implemented; end-to-end validation outstanding |
| First online deployment explicitly serving v1 | Historical evidence not yet established |

---

**Reproducibility note:** All numerical results above come from the project's documented baseline experiment table or the linked GitHub Actions logs. The tables distinguish **validation** from **held-out test** scores, and **demo** from **production** model registries.