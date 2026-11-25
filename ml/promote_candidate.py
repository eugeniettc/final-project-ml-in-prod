
"""Validate whether a completed CT run can be promoted.

This first version is read-only:
it NEVER changes the MLflow champion alias.
"""

import argparse

import dagshub
from mlflow import MlflowClient

from ml.promotion_policy import evaluate_promotion

import mlflow
import mlflow.sklearn

MODEL_NAME = "predictmaint-ai"


def main():
    parser = argparse.ArgumentParser(
        description="PredictMaint AI promotion preflight"
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="MLflow continuous training run ID",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute an approved promotion (default: dry-run)",
    )

    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Target MLflow registered model name",
    )

    args = parser.parse_args()

    # Connect to the shared registry.
    dagshub.init(
        repo_owner="Abdel380",
        repo_name="predictmaint-ai",
        mlflow=True,
    )

    client = MlflowClient()

    # Fetch the completed training run.
    run = client.get_run(args.run_id)

    recorded_model_name = run.data.tags.get(
        "registry_model_name"
    )

    if (
        recorded_model_name
        and recorded_model_name != args.model_name
    ):
        raise RuntimeError(
            "Candidate was evaluated for a different "
            "registered model."
        )
    if run.info.status != "FINISHED":
        raise RuntimeError("CT run is not finished successfully")

    if run.data.tags.get("run_type") != "continuous_training":
        raise RuntimeError("Run is not a continuous training run")

    # Ensure that no other workflow has replaced the champion
    # since the candidate was evaluated.
    champion = client.get_model_version_by_alias(
    args.model_name, "champion"
    )

    expected_version = run.data.tags.get("champion_version")

    if expected_version != str(champion.version):
        raise RuntimeError(
            "Champion changed since candidate evaluation. "
            "A new evaluation is required."
        )

    if champion.status != "READY":
        raise RuntimeError("Champion is not READY")

    # Retrieve the metrics actually logged by retrain.py.
    metrics = run.data.metrics

    required_metrics = [
        "champion_average_precision",
        "candidate_average_precision",
        "candidate_recall",
    ]

    missing = [
        key for key in required_metrics
        if key not in metrics
    ]

    if missing:
        raise RuntimeError(
            f"Missing MLflow metrics: {missing}"
        )

    # Recompute the decision instead of trusting
    # a manually written status.
    result = evaluate_promotion(
        champion_ap=metrics["champion_average_precision"],
        candidate_ap=metrics["candidate_average_precision"],
        candidate_recall=metrics["candidate_recall"],
    )

    decision = "ELIGIBLE" if result["approved"] else "REJECTED"

    # Check consistency with the original CT run.
    recorded = run.data.tags.get("promotion_decision")

    if recorded != decision:
        raise RuntimeError(
            "Promotion decision does not match logged results"
        )

    print("\n=== PROMOTION PREFLIGHT ===")
    print("CT Run:", args.run_id)
    print("Current champion:", champion.version)
    print(
        "Champion AP:",
        round(metrics["champion_average_precision"], 4),
    )
    print(
        "Candidate AP:",
        round(metrics["candidate_average_precision"], 4),
    )
    print(
        "AP improvement:",
        round(result["ap_improvement"], 4),
    )
    print("Decision:", decision)
    print("Failed gates:", result["failed_gates"])

    if not result["approved"]:
        print("\nREJECTED: existing champion retained.")
        print("No registry modification performed.")
        return

    print("\nCandidate is eligible for promotion.")

    if not args.execute:
        print("DRY-RUN: no registry modification performed.")
        return

    # Production promotion stays disabled until
    # application refresh and live smoke tests are ready.
    if args.model_name == MODEL_NAME:
        raise RuntimeError(
            "Production promotion is blocked until "
            "live application verification is configured."
        )

    # Locate the model logged by this exact CT run.
    logged_models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{args.run_id}'",
        output_format="list",
    )

    candidates = [
        model for model in logged_models
        if model.name == "candidate_model"
        and model.source_run_id == args.run_id
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one candidate_model "
            "associated with this CT run."
        )

    candidate_uri = f"models:/{candidates[0].model_id}"

    # Register the approved candidate.
    registered = mlflow.register_model(
        model_uri=candidate_uri,
        name=args.model_name,
        tags={
            "ct_run_id": args.run_id,
            "approved_by": "continuous_training_policy",
        },
    )

    if registered.status != "READY":
        raise RuntimeError(
            "Registered candidate is not READY."
        )

    # Verify that the registered model can be loaded.
    mlflow.sklearn.load_model(
        f"models:/{args.model_name}/{registered.version}"
    )

    # Prevent promotion if the champion changed.
    current = client.get_model_version_by_alias(
        args.model_name, "champion"
    )

    if str(current.version) != expected_version:
        raise RuntimeError(
            "Champion changed before promotion. Aborting."
        )

    # Move the alias only after all prior checks pass.
    client.set_registered_model_alias(
        args.model_name,
        "champion",
        registered.version,
    )

    # Verify the new registry alias.
    updated = client.get_model_version_by_alias(
        args.model_name, "champion"
    )

    if str(updated.version) != str(registered.version):
        raise RuntimeError(
            "Registry alias verification failed."
        )

    print("\nPROMOTION SUCCESSFUL")
    print("Model:", args.model_name)
    print("New champion version:", updated.version)

if __name__ == "__main__":
    main()
