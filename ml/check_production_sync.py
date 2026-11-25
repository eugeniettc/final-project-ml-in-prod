"""Audit synchronization between MLflow and production Render API.

Read-only: never changes the registry or triggers deployment.
"""

import argparse

from mlflow import MlflowClient

from ml.verify_deployment import verify


MODEL_NAME = "predictmaint-ai"


def check_production_sync(url, client=None):
    if not url.startswith("https://"):
        raise ValueError("Production URL must use HTTPS")

    registry = client if client is not None else MlflowClient()

    champion_before = registry.get_model_version_by_alias(MODEL_NAME, "champion")

    if champion_before.status != "READY":
        raise RuntimeError("MLflow champion is not READY")

    expected_version = str(champion_before.version)

    print("MLflow champion version:", expected_version)

    try:
        prediction = verify(url, expected_version)
    except Exception as exc:
        raise RuntimeError(
            "PRODUCTION SYNC FAILED: Render is unavailable "
            "or is not serving MLflow champion "
            f"v{expected_version}"
        ) from exc

    champion_after = registry.get_model_version_by_alias(MODEL_NAME, "champion")

    if str(champion_after.version) != expected_version:
        raise RuntimeError("MLflow champion changed during verification")

    print("\n=== PRODUCTION SYNC VERIFIED ===")
    print("MLflow champion:", expected_version)
    print("Render loaded model:", expected_version)
    print("Prediction:", prediction["failure_prediction"])
    print("Status: SYNCHRONIZED")

    return expected_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    check_production_sync(args.url)


if __name__ == "__main__":
    main()
