"""Log a controlled rejection demonstration in MLflow.

No model is trained, registered or promoted.
"""

import subprocess
from pathlib import Path

import dagshub
import mlflow
from mlflow import MlflowClient

from ml.promotion_policy import (
    MIN_AP,
    MIN_AP_IMPROVEMENT,
    MIN_RECALL,
    evaluate_promotion,
)


MODEL_NAME = "predictmaint-ai"
ROOT = Path(__file__).resolve().parents[1]


def main():
    # 1. Connect to our shared MLflow server.
    dagshub.init(
        repo_owner="Abdel380",
        repo_name="predictmaint-ai",
        mlflow=True,
    )

    client = MlflowClient()

    # 2. Read the current champion.
    champion = client.get_model_version_by_alias(
        MODEL_NAME, "champion"
    )

    if str(champion.version) != "2":
        raise RuntimeError(
            f"Expected champion v2, got v{champion.version}"
        )

    # 3. Use simulated candidate metrics.
    champion_ap = 0.6482
    candidate_ap = 0.6550
    candidate_recall = 0.70

    result = evaluate_promotion(
        champion_ap=champion_ap,
        candidate_ap=candidate_ap,
        candidate_recall=candidate_recall,
    )

    if result["approved"]:
        raise RuntimeError(
            "Expected rejection, but candidate was approved."
        )

    print("\n=== CONTROLLED REJECTION ===")
    print("Current champion:", champion.version)
    print("Candidate AP:", candidate_ap)
    print("AP improvement:", result["ap_improvement"])
    print("Failed gates:", result["failed_gates"])
    print("Decision: REJECTED")

    # 4. Record the code revision.
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    # 5. Log the simulated decision.
    mlflow.set_experiment(
        "predictmaint-ai-continuous-training"
    )

    with mlflow.start_run(
        run_name="ct-controlled-rejection-demo"
    ) as run:
        mlflow.set_tags({
            "event": "controlled_rejection_test",
            "decision": "REJECTED",
            "simulation": "true",
            "model_trained": "false",
            "champion_version": str(champion.version),
            "git_commit": git_sha,
        })

        mlflow.log_params({
            "min_ap": MIN_AP,
            "min_ap_improvement": MIN_AP_IMPROVEMENT,
            "min_recall": MIN_RECALL,
        })

        mlflow.log_metrics({
            "champion_average_precision": champion_ap,
            "candidate_average_precision": candidate_ap,
            "candidate_recall": candidate_recall,
            "ap_improvement": result["ap_improvement"],
        })

        mlflow.log_dict(
            {
                "simulation": True,
                "decision": "REJECTED",
                "reason": "Insufficient AP improvement",
                "quality_gates": result["checks"],
                "failed_gates": result["failed_gates"],
                "alias_modified": False,
            },
            "evaluation/rejection_decision.json",
        )

        print("MLflow Run ID:", run.info.run_id)

    # 6. Verify the champion remains unchanged.
    champion_after = client.get_model_version_by_alias(
        MODEL_NAME, "champion"
    )

    if str(champion_after.version) != "2":
        raise RuntimeError(
            "Champion changed during the demonstration."
        )

    print("\nREJECTION DEMONSTRATION PASSED!")
    print("Champion remains version:", champion_after.version)


if __name__ == "__main__":
    main()
