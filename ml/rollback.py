import argparse
from pathlib import Path

import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient


MODEL_NAME = "predictmaint-ai"
ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


def validate_rollback_policy(champion, target):
    """Allow rollback only to the recorded previous champion."""

    if str(champion.version) == str(target.version):
        raise RuntimeError(
            "Rollback target is already the current champion."
        )

    if target.status != "READY":
        raise RuntimeError(
            "Rollback target model is not READY."
        )

    tags = champion.tags or {}

    if tags.get("approved_by") != "continuous_training_policy":
        raise RuntimeError(
            "Current champion has no approved CT provenance."
        )

    previous_version = tags.get("previous_champion_version")

    if previous_version != str(target.version):
        raise RuntimeError(
            "Rollback target is not the recorded "
            "previous champion."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Safely restore an approved MLflow model version."
    )

    parser.add_argument("--to-version", type=int, required=True)
    parser.add_argument("--expected-current", type=int, required=True)
    parser.add_argument("--execute", action="store_true")

    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="MLflow registered model name",
    )

    args = parser.parse_args()

    dagshub.init(
        repo_owner="Abdel380",
        repo_name="predictmaint-ai",
        mlflow=True,
    )

    client = MlflowClient()

    # 1. Check the current champion
    champion = client.get_model_version_by_alias(
        args.model_name, "champion"
    )

    current_version = int(champion.version)

    if current_version != args.expected_current:
        raise RuntimeError(
            f"Expected champion {args.expected_current}, "
            f"but found {current_version}."
        )

    # 2. Check the target model version
    target = client.get_model_version(
        args.model_name, str(args.to_version)
    )

    validate_rollback_policy(champion, target)

    # 3. Load and test the target BEFORE modifying the alias
    model = mlflow.sklearn.load_model(
        f"models:/{args.model_name}/{args.to_version}"
    )

    X = pd.read_csv(
        ROOT / "data/splits/validation.csv"
    )[FEATURES].head(3)

    scores = model.predict_proba(X)[:, 1]

    if not (
        np.isfinite(scores).all()
        and ((scores >= 0) & (scores <= 1)).all()
    ):
        raise RuntimeError("Target model inference failed.")

    print("\n=== ROLLBACK PRECHECK ===")
    print("Model name:", args.model_name)
    print("Current champion:", current_version)
    print("Target version:", args.to_version)
    print("Target status:", target.status)
    print("Inference: PASSED")

    # 4. Default behavior: no registry modification
    if not args.execute:
        print("\nDRY-RUN PASSED")
        print("No champion alias modification.")
        return

    # Verify that the champion has not changed
    # while the rollback prechecks were running.
    latest_champion = client.get_model_version_by_alias(
        args.model_name, "champion"
    )

    if int(latest_champion.version) != current_version:
        raise RuntimeError(
            "Champion changed during rollback validation."
        )
    # 5. Change the alias
    client.set_registered_model_alias(
        name=args.model_name,
        alias="champion",
        version=str(args.to_version),
    )

    # 6. Verify the alias
    new_champion = client.get_model_version_by_alias(
        args.model_name, "champion"
    )

    if int(new_champion.version) != args.to_version:
        raise RuntimeError(
            "Alias verification failed. Check registry state."
        )

    print("\nROLLBACK SUCCESSFUL")
    print("Previous version:", current_version)
    print("New champion:", new_champion.version)

    # 7. Enregistrer le rollback dans MLflow
    mlflow.set_experiment(
        "predictmaint-ai-continuous-training"
    )

    with mlflow.start_run(
        run_name=f"rollback-{current_version}-to-{args.to_version}"
    ):
        mlflow.log_params({
            "model_name": args.model_name,
            "previous_version": current_version,
            "restored_version": args.to_version,
        })

        mlflow.set_tags({
            "event": "model_rollback",
            "decision": "RESTORED",
            "verification": "PASSED",
        })

    print("Rollback audit recorded in MLflow.")


if __name__ == "__main__":
    main()
