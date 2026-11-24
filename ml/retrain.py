from pathlib import Path

import dagshub
import pandas as pd
from mlflow import MlflowClient


# ============================================
# 1. CONFIGURATION
# ============================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_V2_PATH = ROOT / "data/releases/v2/train.csv"
VALIDATION_PATH = ROOT / "data/splits/validation.csv"
TEST_PATH = ROOT / "data/splits/test.csv"

FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

TARGET = "Machine failure"
MODEL_NAME = "predictmaint-ai"


# ============================================
# 2. VALIDATE DATA
# ============================================

def load_and_validate(path, name):
    df = pd.read_csv(path)

    required = ["UID", *FEATURES, TARGET]
    missing_columns = set(required) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{name}: missing columns {missing_columns}"
        )

    if df[required].isna().any().any():
        raise ValueError(f"{name}: missing values found")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"{name}: invalid target values")

    if df["UID"].duplicated().any():
        raise ValueError(f"{name}: duplicate UIDs")

    print(
        f"{name}: {len(df)} rows, "
        f"{df[TARGET].sum()} failures"
    )

    return df


print("\n=== DATA VALIDATION ===")

train = load_and_validate(TRAIN_V2_PATH, "Train v2")
validation = load_and_validate(VALIDATION_PATH, "Validation")
test = load_and_validate(TEST_PATH, "Test")

# Verify no leakage between the splits.
train_ids = set(train["UID"])
val_ids = set(validation["UID"])
test_ids = set(test["UID"])

if (
    train_ids & val_ids
    or train_ids & test_ids
    or val_ids & test_ids
):
    raise ValueError("Data leakage: overlapping UIDs")

print("Data validation PASSED!")
print("No overlapping UIDs.")


# ============================================
# 3. CONNECT TO MLFLOW
# ============================================

dagshub.init(
    repo_owner="Abdel380",
    repo_name="predictmaint-ai",
    mlflow=True,
)

client = MlflowClient()


# ============================================
# 4. FIND CURRENT CHAMPION
# ============================================

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion",
)

if champion.status != "READY":
    raise RuntimeError("Champion model is not READY")

print("\n=== CURRENT CHAMPION ===")
print("Model:", champion.name)
print("Version:", champion.version)
print("Status:", champion.status)

print("\nPreflight checks PASSED!")

# ============================================
# 5. LOAD CHAMPION MODEL
# ============================================

import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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


print("\n=== LOADING CHAMPION ===")

champion_version = str(champion.version)

champion_model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}/{champion_version}"
)

print(f"Champion version {champion_version} loaded!")


# ============================================
# 6. PREPARE TRAINING AND TEST DATA
# ============================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# ============================================
# 7. TRAIN CANDIDATE ON DATASET V2
# ============================================

print("\n=== TRAINING CANDIDATE V2 ===")

numeric_features = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Type"],
        ),
    ]
)

candidate_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

candidate_model.fit(X_train, y_train)

print("Candidate trained successfully!")
print(f"Training observations: {len(X_train)}")


# ============================================
# 8. EVALUATION FUNCTION
# ============================================

def evaluate_model(model, X, y):
    """Evaluate a fitted model on the same held-out data."""

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "average_precision": average_precision_score(
            y, probabilities
        ),
        "roc_auc": roc_auc_score(y, probabilities),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


# ============================================
# 9. COMPARE CHAMPION AND CANDIDATE
# ============================================

print("\n=== EVALUATING ON HELD-OUT TEST ===")

champion_metrics = evaluate_model(
    champion_model, X_test, y_test
)

candidate_metrics = evaluate_model(
    candidate_model, X_test, y_test
)

comparison = pd.DataFrame(
    [
        {"model": "Champion v1", **champion_metrics},
        {"model": "Candidate v2", **candidate_metrics},
    ]
)

print(comparison.round(4).to_string(index=False))


# ============================================
# 10. APPLY PROMOTION RULES
# ============================================

# Thresholds defined in README_experiments.md
# BEFORE evaluating the v2 candidate.

MIN_AP = 0.40
MIN_AP_IMPROVEMENT = 0.02
MIN_RECALL = 0.50

champion_ap = champion_metrics["average_precision"]
candidate_ap = candidate_metrics["average_precision"]
candidate_recall = candidate_metrics["recall"]

ap_improvement = candidate_ap - champion_ap

checks = {
    "minimum_average_precision": candidate_ap >= MIN_AP,
    "minimum_improvement": ap_improvement >= MIN_AP_IMPROVEMENT,
    "minimum_recall": candidate_recall >= MIN_RECALL,
}

print("\n=== PROMOTION QUALITY GATES ===")
print(f"Champion AP: {champion_ap:.4f}")
print(f"Candidate AP: {candidate_ap:.4f}")
print(f"AP improvement: {ap_improvement:+.4f}")

for rule, passed in checks.items():
    print(f"{rule}: {'PASS' if passed else 'FAIL'}")

approved = all(checks.values())

print("\n=== DRY-RUN DECISION ===")

if approved:
    print("PROMOTION ELIGIBLE: candidate passes all gates.")
else:
    print("REJECT: candidate does not meet all gates.")
    print(
        "Failed gates:",
        [rule for rule, passed in checks.items() if not passed],
    )

print("Champion alias has NOT been modified.")


# ============================================
# 11. LOG CONTINUOUS TRAINING TO MLFLOW
# ============================================

import re
import subprocess

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


print("\n=== LOGGING CONTINUOUS TRAINING ===")

# Identify the exact dataset used.
dvc_file = ROOT / "data/releases/v2/train.csv.dvc"
dvc_content = dvc_file.read_text()

match = re.search(
    r"(?m)^\s*-?\s*md5:\s*([a-f0-9]{32})\s*$",
    dvc_content,
)

if match is None:
    raise ValueError("Cannot find v2 DVC hash.")

dvc_hash = match.group(1)

# Identify the Git commit used for this run.
git_sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    text=True,
).strip()

# Select our Continuous Training experiment.
mlflow.set_experiment(
    "predictmaint-ai-continuous-training"
)

with mlflow.start_run(
    run_name="ct-dataset-v2-dry-run"
) as run:

    # 1. Reproducibility metadata.
    mlflow.set_tags({
        "git_commit": git_sha,
        "dataset_version": "v2",
        "dataset_git_tag": "data-v2",
        "dataset_dvc_md5": dvc_hash,
        "champion_version": champion_version,
        "run_type": "continuous_training",
        "promotion_decision": (
            "ELIGIBLE" if approved else "REJECTED"
        ),
    })

    # 2. Training configuration and quality gates.
    mlflow.log_params({
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "algorithm": "RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "decision_threshold": 0.5,
        "min_average_precision": MIN_AP,
        "min_ap_improvement": MIN_AP_IMPROVEMENT,
        "min_recall": MIN_RECALL,
    })

    # 3. Champion and candidate metrics.
    metrics = {}

    for key, value in champion_metrics.items():
        metrics[f"champion_{key}"] = float(value)

    for key, value in candidate_metrics.items():
        metrics[f"candidate_{key}"] = float(value)

    metrics["ap_improvement"] = float(ap_improvement)

    mlflow.log_metrics(metrics)

    # 4. Promotion decision artifact.
    decision = {
        "dataset_version": "v2",
        "champion_version": champion_version,
        "candidate_status": (
            "ELIGIBLE" if approved else "REJECTED"
        ),
        "quality_gates": {
            key: bool(value)
            for key, value in checks.items()
        },
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "ap_improvement": float(ap_improvement),
        "alias_modified": False,
    }

    mlflow.log_dict(
        decision,
        "evaluation/promotion_decision.json",
    )

    # 5. Save the trained candidate pipeline.
    example = X_train.head(3)

    signature = infer_signature(
        example,
        candidate_model.predict_proba(example),
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=candidate_model,
        name="candidate_model",
        input_example=example,
        signature=signature,
        pyfunc_predict_fn="predict_proba",
        serialization_format="cloudpickle",
    )

    print("\nMLflow Run ID:", run.info.run_id)
    print("Candidate Model URI:", model_info.model_uri)
    print("Promotion eligible:", approved)

print("\nContinuous Training logged successfully!")
print("Champion alias remains unchanged.")
