
from pathlib import Path
import re
import subprocess

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================
# 1. CONFIGURATION
# ============================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/releases/v1/train.csv"
VALIDATION_PATH = ROOT / "data/splits/validation.csv"

DVC_FILE = ROOT / "data/releases/v1/train.csv.dvc"

TARGET = "Machine failure"

NUMERICAL_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

CATEGORICAL_FEATURES = ["Type"]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

RANDOM_STATE = 42
DATASET_VERSION = "v1"
DATASET_GIT_TAG = "data-v1"


# ============================================
# 2. CONNECT TO DAGSHUB / MLFLOW
# ============================================

dagshub.init(
    repo_owner="Abdel380",
    repo_name="predictmaint-ai",
    mlflow=True,
)

mlflow.set_experiment("predictmaint-ai-baselines")

print("MLflow Tracking URI:", mlflow.get_tracking_uri())


# ============================================
# 3. REPRODUCIBILITY METADATA
# ============================================

# Exact Git commit containing the training code.
git_sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    text=True,
).strip()

# Hash stored in the DVC pointer for training v1.
dvc_content = DVC_FILE.read_text()

match = re.search(
    r"(?m)^\s*-?\s*md5:\s*([a-f0-9]{32})\s*$",
    dvc_content,
)

if match is None:
    raise ValueError("Cannot find the DVC MD5 hash.")

dvc_md5 = match.group(1)

print("Git commit:", git_sha)
print("Dataset version:", DATASET_VERSION)
print("DVC hash:", dvc_md5)


# ============================================
# 4. LOAD DATA
# ============================================

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

print("\n=== LOADING DATA ===")
print("Train samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================
# 5. PREPROCESSING
# ============================================

def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# ============================================
# 6. DEFINE MODELS
# ============================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ============================================
# 7. TRAIN, EVALUATE AND LOG TO MLFLOW
# ============================================

results = []

for name, estimator in models.items():

    print(f"\n=== TRAINING: {name} ===")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", estimator),
        ]
    )

    # Each algorithm gets its own MLflow run.
    run_name = "baseline-" + name.lower().replace(" ", "-")

    with mlflow.start_run(run_name=run_name) as run:

        # A. Record the exact code and dataset versions.
        mlflow.set_tags({
            "git_commit": git_sha,
            "dataset_version": DATASET_VERSION,
            "dataset_git_tag": DATASET_GIT_TAG,
            "dataset_dvc_md5": dvc_md5,
            "evaluation_split": "validation",
        })

        # B. Record hyperparameters.
        mlflow.log_params({
            f"model_{key}": str(value)
            for key, value in estimator.get_params().items()
        })

        mlflow.log_params({
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "decision_threshold": 0.5,
            "random_state": RANDOM_STATE,
        })

        # C. Train on v1 only.
        pipeline.fit(X_train, y_train)

        # D. Evaluate on the fixed validation split.
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "average_precision": average_precision_score(
                y_val, y_proba
            ),
            "roc_auc": roc_auc_score(y_val, y_proba),
            "precision": precision_score(
                y_val, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_val, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_val, y_pred, zero_division=0
            ),
        }

        # E. Upload evaluation metrics to DagsHub.
        mlflow.log_metrics(metrics)

        # F. Record the confusion matrix as an artifact.
        tn, fp, fn, tp = confusion_matrix(
            y_val, y_pred, labels=[0, 1]
        ).ravel()

        mlflow.log_dict(
            {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "threshold": 0.5,
            },
            "evaluation/confusion_matrix.json",
        )

        # G. Save the complete trained pipeline.
        # The logged model returns class probabilities.
        example = X_train.head(3)

        signature = infer_signature(
            example,
            pipeline.predict_proba(example),
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=example,
            signature=signature,
            pyfunc_predict_fn="predict_proba",
            serialization_format="cloudpickle",
        )

        print("\nMetrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        print("MLflow Run ID:", run.info.run_id)
        print("Logged Model URI:", model_info.model_uri)

        results.append({
            "model": name,
            **metrics,
        })


# ============================================
# 8. COMPARE RESULTS
# ============================================

results_df = pd.DataFrame(results).sort_values(
    by="average_precision",
    ascending=False,
)

print("\n=== MODEL COMPARISON ===")
print(results_df.round(4).to_string(index=False))

print(
    "\nBest model on validation:",
    results_df.iloc[0]["model"],
)

print("Held-out test set has NOT been used.")
print("Both runs have been logged to MLflow.")