
"""Decide whether a dataset release requires retraining.

"""

import argparse
import re
from pathlib import Path

import dagshub
from mlflow import MlflowClient
import os

import mlflow
ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = os.getenv("CT_MODEL_NAME", "predictmaint-ai")

def get_dvc_hash(dataset_version):
    dvc_file = (
        ROOT
        / "data"
        / "releases"
        / dataset_version
        / "train.csv.dvc"
    )

    content = dvc_file.read_text(encoding="utf-8")

    match = re.search(
        r"(?m)^\s*-?\s*md5:\s*([a-f0-9]{32})\s*$",
        content,
    )

    if not match:
        raise ValueError(
            f"Missing DVC hash for {dataset_version}"
        )

    return match.group(1)


def main():
    parser = argparse.ArgumentParser(
        description="PredictMaint AI retraining decision"
    )

    parser.add_argument(
        "--dataset-version",
        choices=["v1", "v2"],
        required=True,
    )

    args = parser.parse_args()

    selected_version = args.dataset_version
    selected_hash = get_dvc_hash(selected_version)

    dagshub.init(
        repo_owner="Abdel380",
        repo_name="predictmaint-ai",
        mlflow=True,
    )

    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion",
    )

    if champion.status != "READY":
        raise RuntimeError("Champion is not READY")

    if not champion.run_id:
        raise RuntimeError(
            "Champion training run is unknown"
        )

    run = client.get_run(champion.run_id)

    champion_dataset = run.data.tags.get(
        "dataset_version"
    )

    champion_hash = run.data.tags.get(
        "dataset_dvc_md5"
    )

    if not champion_dataset or not champion_hash:
        raise RuntimeError(
            "Champion dataset provenance is missing"
        )

    if not re.fullmatch(r"v[0-9]+", champion_dataset):
        raise RuntimeError(
            "Invalid champion dataset version"
        )

    selected_number = int(selected_version[1:])
    champion_number = int(champion_dataset[1:])

    if selected_number < champion_number:
        decision = "SKIPPED"
        reason = "Selected dataset is older than champion data"

    elif selected_number == champion_number:
        if selected_hash != champion_hash:
            raise RuntimeError(
                "DVC hash mismatch for the same dataset "
                "version. Check release integrity."
            )

        decision = "SKIPPED"
        reason = "Champion already uses this dataset release"

    else:
        decision = "TRAIN"
        reason = "Newer validated dataset release detected"

    print("\n=== RETRAINING DECISION ===")
    print("Champion model version:", champion.version)
    print("Champion dataset:", champion_dataset)
    print("Selected dataset:", selected_version)
    print("DVC hash matches champion:", selected_hash == champion_hash)
    print("Decision:", decision)
    print("Reason:", reason)
    print("No registry modification performed.")


    # Record skipped retraining decisions in MLflow.
    if decision == "SKIPPED":
        mlflow.set_experiment(
            "predictmaint-ai-continuous-training"
        )

        with mlflow.start_run(
            run_name=f"ct-{selected_version}-skipped"
        ) as audit_run:

            mlflow.set_tags({
                "run_type": "retraining_gate",
                "retraining_decision": "SKIPPED",
                "dataset_version": selected_version,
                "dataset_dvc_md5": selected_hash,
                "champion_version": str(champion.version),
                "champion_dataset_version": champion_dataset,
            })

            mlflow.log_dict(
                {
                    "decision": "SKIPPED",
                    "reason": reason,
                    "selected_dataset": selected_version,
                    "selected_dvc_hash": selected_hash,
                    "champion_version": str(champion.version),
                    "champion_run_id": champion.run_id,
                    "champion_dataset": champion_dataset,
                    "champion_dvc_hash": champion_hash,
                    "training_executed": False,
                    "registry_modified": False,
                },
                "evaluation/retraining_decision.json",
            )

            print(
                "MLflow audit Run ID:",
                audit_run.info.run_id,
            )

        print("SKIPPED decision recorded in MLflow.")


    github_output = os.environ.get("GITHUB_OUTPUT")

    if github_output:
        with open(
            github_output,
            "a",
            encoding="utf-8",
        ) as output:
            output.write(f"decision={decision}\n")
            output.write(f"reason={reason}\n")



if __name__ == "__main__":
    main()
