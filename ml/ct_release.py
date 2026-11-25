"""Controlled CT production release with automatic rollback.

Read-only by default.
Production execution requires explicit GitHub authorization.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from mlflow import MlflowClient

from ml.promote_candidate import validate_production_release_context
from ml.render_redeploy import deploy_and_verify
from ml.verify_deployment import verify

MODEL_NAME = "predictmaint-ai"


def promote(run_id, execute=False):
    """Run promotion CLI and retrieve structured outputs."""

    with TemporaryDirectory() as directory:
        output_file = Path(directory) / "promotion-output"
        env = os.environ.copy()
        env["GITHUB_OUTPUT"] = str(output_file)

        command = [
            sys.executable,
            "-m",
            "ml.promote_candidate",
            "--run-id",
            run_id,
            "--model-name",
            MODEL_NAME,
        ]

        if execute:
            command.extend(["--execute", "--production-release"])

        subprocess.run(command, check=True, env=env)

        if not output_file.exists():
            raise RuntimeError("Promotion CLI did not report a decision")

        return dict(
            line.split("=", 1)
            for line in output_file.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )


def restore(previous_version, failed_version, url, sha, client):
    """Restore MLflow and redeploy the previous champion."""

    rollback_error = None

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ml.rollback",
                "--to-version",
                previous_version,
                "--expected-current",
                failed_version,
                "--execute",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        # MLflow alias may have been restored even if
        # subsequent audit logging failed.
        rollback_error = exc

    actual = client.get_model_version_by_alias(MODEL_NAME, "champion")

    if str(actual.version) != previous_version:
        raise RuntimeError(
            "CRITICAL: rollback did not restore the "
            "MLflow alias. Manual intervention required."
        ) from rollback_error

    # Restore the real application, not only MLflow.
    deploy_and_verify(url, previous_version, sha)

    if rollback_error:
        raise RuntimeError(
            "Model and API restored, but rollback audit failed."
        ) from rollback_error

    print("ROLLBACK VERIFIED: MLflow and Render restored.")


def release(run_id, url, execute=False):
    """Validate, promote, deploy, and recover if necessary."""

    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Invalid MLflow run ID")

    client = MlflowClient()
    run = client.get_run(run_id)

    # Strict provenance is required for production,
    # while the original promotion CLI remains unchanged.
    if run.data.tags.get("registry_model_name") != MODEL_NAME:
        raise RuntimeError("CT run does not identify the production model")

    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")

    if champion.status != "READY":
        raise RuntimeError("Current champion is not READY")

    previous_version = str(champion.version)

    # Do not promote if Render and MLflow disagree.
    verify(url, previous_version)

    # First evaluation is always read-only.
    check = promote(run_id, execute=False)

    if check.get("promotion_executed") != "false":
        raise RuntimeError("Preflight must not modify MLflow")

    if check.get("previous_version") != previous_version:
        raise RuntimeError("Champion changed during preflight")

    decision = check.get("promotion_decision")

    if decision == "REJECTED":
        print("REJECTED: current champion retained.")
        return

    if decision != "ELIGIBLE":
        raise RuntimeError("Unknown promotion decision")

    if not execute:
        print(
            "PREFLIGHT PASSED: candidate eligible. "
            "No production modification performed."
        )
        return

    # Authorization is mandatory for actual promotion.
    validate_production_release_context(True)

    sha = os.getenv("GITHUB_SHA", "")

    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("Missing or invalid production Git commit SHA")

    try:
        result = promote(run_id, execute=True)

        updated = client.get_model_version_by_alias(MODEL_NAME, "champion")

        new_version = str(updated.version)

        if (
            result.get("promotion_executed") != "true"
            or result.get("previous_version") != previous_version
            or result.get("new_version") != new_version
            or new_version == previous_version
            or updated.tags.get("ct_run_id") != run_id
            or updated.tags.get("previous_champion_version") != previous_version
        ):
            raise RuntimeError("Promoted model provenance is inconsistent")

        # Deploy the validated Git commit and new MLflow champion.
        deploy_and_verify(url, new_version, sha)

        final = client.get_model_version_by_alias(MODEL_NAME, "champion")

        if str(final.version) != new_version:
            raise RuntimeError("Champion changed during Render deployment")

        print(
            "CT RELEASE VERIFIED: model version",
            new_version,
        )

    except Exception as error:
        current = client.get_model_version_by_alias(MODEL_NAME, "champion")

        failed_version = str(current.version)

        if failed_version == previous_version:
            raise RuntimeError("Release failed; champion alias unchanged.") from error

        # Never overwrite an unexpected concurrent change.
        if (
            current.tags.get("ct_run_id") != run_id
            or current.tags.get("previous_champion_version") != previous_version
        ):
            raise RuntimeError(
                "CRITICAL: unexpected champion change. Manual recovery required."
            ) from error

        print("Deployment failed. Starting rollback.")

        try:
            restore(
                previous_version,
                failed_version,
                url,
                sha,
                client,
            )
        except Exception as rollback_error:
            raise RuntimeError(
                "CRITICAL: release failed and recovery was not fully verified."
            ) from rollback_error

        raise RuntimeError("CT release failed; rollback verified.") from error


def main():
    parser = argparse.ArgumentParser(description="PredictMaint CT production release")

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()

    release(
        run_id=args.run_id,
        url=args.url,
        execute=args.execute,
    )


if __name__ == "__main__":
    main()
