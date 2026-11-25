# PredictMaint AI — Machine Learning Experiments and Model Lifecycle

This document records the dataset versions, training experiments, evaluation protocol, model promotion policy, continuous-training evidence, deployment verification, and rollback demonstration for **PredictMaint AI**.

> **Production result (25 November 2026).** A candidate trained on dataset v2 improved held-out Average Precision by **0.0295** over champion v1. After human approval, the release promoted the candidate as **model version 3** and deployed it on Render. The team then deliberately restored **champion v1** for the presentation and verified a live prediction.

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
- Last verified online production champion as of 25 November 2026: `predictmaint-ai` **version 1**, restored from **version 3**.

The Git tags `data-v1` and `data-v2` identify training snapshots tracked by DVC. MLflow assigns model versions within each registered model. Training on `data-v2` produced production **v3** in the rehearsal and **v4** in the earlier isolated exercise. Each registered model has its own `champion` alias.

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

The production registry links v1 to baseline run `35a09c77d6b0428ab32906ecdbd864fd`. [Random Forest baseline run](https://dagshub.com/Abdel380/predictmaint-ai.mlflow/#/experiments/1/runs/35a09c77d6b0428ab32906ecdbd864fd)

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

The `0.02` AP improvement is an **absolute difference**. Training logs the gate decision from the held-out metrics. During promotion, the release checks that the registered candidate is `READY` and loads successfully before moving the alias. A rejected candidate leaves the champion in place. The thresholds are specific to this academic demonstration.

## 6. Production Continuous Training and protected promotion

On **25 November 2026**, Render was serving `predictmaint-ai@champion` **v1**, trained on `data-v1`. Continuous Training detected the newer `data-v2` snapshot, trained a candidate and evaluated both models on the same **2,000 held-out observations**, including **68 failures**. [Starting production deployment](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45709532179) · [CT workflow](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45710611379)

The MLflow CT run `83942914196549e89ebb69a4dbb89844` records the comparison, dataset hash, training commit and `ELIGIBLE` decision. Its run name ends in `dry-run` because the promotion check leaves the alias in place. The run trained and logged the candidate model. [MLflow CT run](https://dagshub.com/Abdel380/predictmaint-ai.mlflow/#/experiments/2/runs/83942914196549e89ebb69a4dbb89844)

| Measure | Production champion v1 | Candidate promoted as production v3 |
| --- | ---: | ---: |
| Training samples | 3,000 | 6,000 |
| Average Precision | 0.6186 | **0.6481** |
| ROC-AUC | 0.9604 | **0.9653** |
| Precision | **0.6316** | 0.5610 |
| Recall | 0.5294 | **0.6765** |
| F1 | 0.5760 | **0.6133** |
| True positives | 36 | **46** |
| False negatives | 32 | **22** |
| False positives | **21** | 36 |
| True negatives | **1,911** | 1,896 |

**Decision.** All three quality gates passed. Candidate AP was **0.6481** (minimum 0.40), its absolute AP gain was **0.0295** (minimum 0.02), and recall was **0.6765** (minimum 0.50). An operator supplied the CT run ID to a separate Production Release. After human approval in `production-ct`, the workflow promoted v3, redeployed Render and verified inference. The following monitor confirmed synchronization. [Protected release](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45712120942) · [Post-promotion monitor](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45712976568)

**Precision and recall.** At threshold 0.5, v3 detected 46 of 68 failures, ten more than v1. Missed failures fell from 32 to 22, while false alarms rose from 21 to 36. This raised recall from 0.5294 to 0.6765 and reduced precision from 0.6316 to 0.5610. The policy accepts this trade-off on the basis of AP and minimum recall.

### Earlier isolated promotion

The earlier demonstration used the same training snapshot and evaluated the isolated model `predictmaint-ai-ct-demo-henri-fixed`. Its candidate passed the gates against demo champion v1 and became **demo registry version 4**.

- [GitHub Actions — actual v2 training, quality-gate evaluation, and demo promotion](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45575717813).
- [MLflow CT run — `9dd7e7aa262e43ea835cf4bdf150aff8`](https://dagshub.com/Abdel380/predictmaint-ai.mlflow/#/experiments/2/runs/9dd7e7aa262e43ea835cf4bdf150aff8).
- Recorded output: `Decision: TRAIN`, `PROMOTION ELIGIBLE`, `PROMOTION SUCCESSFUL`, `New champion version: 4` (isolated demo registry).

## 7. Verified controlled rejection

A separate **historical CT run** was rechecked against the production champion **v2** on **24–25 November 2026**:

- Historical MLflow CT run: [`de410647f3f94c14a56db7b1940d0977`](https://dagshub.com/Abdel380/predictmaint-ai.mlflow/#/experiments/2/runs/de410647f3f94c14a56db7b1940d0977).
- Current production champion at the check: **v2**.
- Champion AP: **0.6482**.
- Candidate AP: **0.6481**.
- Result: `REJECTED` because `minimum_improvement` failed.
- No alias modification was performed; the existing champion remained **v2**.

Evidence: [GitHub Actions — controlled rejection](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45506491721).

## 8. Continuous training and promotion safeguards

The CT workflow runs tests on PRs targeting `dev`. Pushes to `dev` validate both datasets and run the retraining gate for v2. Manual dispatch on `main` accepts a dataset (`v1` or `v2`) and a mode. The default `validate-only` stops after data and registry validation. The full `train-and-check` path proceeds as follows. [Continuous Training workflow](../.github/workflows/continuous-training.yml)

1. Run automated ML tests.
2. Retrieve DVC snapshots from Azure and validate the data contract and split separation.
3. Resolve the MLflow champion and inspect dataset provenance (`dataset_version`, `dataset_dvc_md5`).
4. Return `SKIPPED` for an older dataset or the same release with the same DVC hash. A changed hash for the same release fails validation. A newer validated release returns `TRAIN`.
5. Train a candidate, evaluate it against the champion, and log metrics, parameters, Git SHA, dataset/DVC identifiers, decision artifact, and candidate model to MLflow.
6. Recompute the promotion decision from the logged metrics in read-only mode. This workflow leaves the production alias in place.

Production Release requires a separate manual dispatch on `main`. Its default `preflight` mode verifies the serving application and recomputes the decision against the current champion. In `release` mode, the protected job checks the run's production provenance, promotes the approved candidate and redeploys Render. [CT Production Release workflow](../.github/workflows/ct-production-release.yml)

## 9. Verified rollback and recovery behavior

The rollback script [`rollback.py`](rollback.py) has a **dry-run default** and requires `--execute` to change an MLflow alias. It checks the current champion version, `READY` status of the target, CT approval provenance, the recorded `previous_champion_version`, and a local inference precheck.

**Production v3 → v1.** After a successful dry-run, the team ran the rollback script with `--execute` to restore the production alias for the presentation. The audit records previous version `3` and restored version `1`. A separate production redeployment loaded v1 and checked `/ready`, `/model-info` and `POST /predict` over HTTP. [Production rollback audit](https://dagshub.com/Abdel380/predictmaint-ai.mlflow/#/experiments/2/runs/87a607ce17f74f329995e2f4db0a085b) · [Post-rollback deployment](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45716691906)

**Historical isolated v4 → v1.** On **25 November 2026**, a controlled rollback of **demo model v4 → v1** was completed. After a FastAPI restart using the isolated demo model, `/model-info` reported **v1** and a live local `/predict` call returned:

- `failure_prediction = 0`
- `risk_score = 0.0058`
- `model_version = 1`

The audit event was recorded in [this MLflow rollback run](https://dagshub.com/Abdel380/predictmaint-ai.mlflow/#/experiments/2/runs/62ee72d6e93a41de99705dbedd304dcd).

The deployment runbook gives commands for both rollbacks and describes the automatic recovery path. Controlled tests cover recovery after a failed deployment. The team has not induced a deployment failure on Render. [Rollback and recovery runbook](../DEPLOYMENT.md#4-mlflow-model-rollback)

## 10. Production verification and current status

At the latest verified production deployment on **25 November 2026**:

- Registered production model: `predictmaint-ai`.
- MLflow alias: `champion`.
- Concrete model served: **v1**, deliberately restored for the presentation.
- The rehearsal recorded **46 passing automated tests**.
- The Render production deployment and live HTTP smoke tests passed, including model-version verification and a prediction.
- The production synchronization monitor also passed.

Evidence:

- [Final `main` deployment at `b2044c0` with HTTP verification](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45718935803).
- [Final production model synchronization monitor](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45719605125).
- [Production application](https://predictmaint-ai-production.onrender.com/) and [model information endpoint](https://predictmaint-ai-production.onrender.com/model-info).

A repeatable verification command is:

```bash
python -m ml.verify_deployment \
  --url https://predictmaint-ai-production.onrender.com \
  --expected-version 1
```

### Completion summary

| Item | Recorded state |
| --- | --- |
| Baseline models and initial champion v1 | Completed |
| DVC dataset versions v1 and v2 | Completed |
| Real CT on dataset v2 | Trained and evaluated against production champion v1 |
| Approved production release | Human approval, v3 promotion, Render deployment and HTTP verification demonstrated |
| Historical controlled rejection | Insufficient AP improvement retained the then-champion v2 |
| Historical isolated promotion and rollback | Demo model promoted to v4, restored to v1 and verified locally |
| Controlled production rollback | Production v3 restored to v1 and verified on Render |
| Automatic recovery following deployment failure | Covered by controlled tests. A real Render failure remains unexercised. |

---

**Reproducibility note.** Section 3 reports baseline validation scores. Section 6 reports the production CT run's held-out test scores. The MLflow run records the training Git SHA and DVC hash, which identify the code and the dataset used for this comparison.
