# PredictMaint AI — Deployment Documentation & Twelve-Factor App

This document describes the deployment architecture, the implementation of the **Twelve-Factor App** methodology, and the procedures for reproducing local and production deployments of **PredictMaint AI**.

---

## 1. Twelve-Factor App Methodology

The web application and inference service were designed according to the **Twelve-Factor App** principles to support portability, observability, and resilience in production.

| No. | Factor | Implementation in PredictMaint AI |
| --- | --- | --- |
| **I** | **Codebase** | A single centralized Git repository is shared by the team. Source code, tests, and workflow definitions are version-controlled using a structured branch flow (`dev` → `staging` → `main`). |
| **II** | **Dependencies** | Python dependencies are explicitly declared in `requirements.txt` and isolated within a Docker `python:3.11-slim` image. The application does not rely on undeclared host-system dependencies. |
| **III** | **Config** | Code and configuration are strictly separated. Sensitive and environment-specific settings (`MLFLOW_TRACKING_URI`, DagsHub/Azure credentials, `PORT`) are injected at runtime through environment variables. No secrets are stored in the Docker image or Git repository. |
| **IV** | **Backing Services** | External services (Azure Blob Storage for DVC, DagsHub for MLflow Tracking and Model Registry) are treated as attached resources, accessible through configurable URLs and credentials without changing application code. |
| **V** | **Build, Release, Run** | The three stages are separated: *Build* produces the Docker image associated with a Git commit SHA; *Release* combines the image with environment-specific configuration; *Run* starts the container on the cloud platform. |
| **VI** | **Processes** | The FastAPI inference service runs as a stateless process. No application state is retained on the container's local disk, allowing the process to be restarted or replaced. |
| **VII** | **Port Binding** | The application exposes an HTTP port directly through the Uvicorn ASGI server (`--host 0.0.0.0 --port ${PORT:-8000}`), without requiring a separately installed web server. |
| **VIII** | **Concurrency** | Horizontal scaling can be achieved by running additional lightweight containers behind the cloud platform's load balancer (Render, Azure Container Apps, or Cloud Run). |
| **IX** | **Disposability** | The application supports startup model loading through FastAPI lifespan and graceful shutdown handled by Uvicorn when it receives a SIGTERM signal. |
| **X** | **Dev/Prod Parity** | The same Dockerized application can run locally, in CI/CD, and in cloud production, minimizing environment-specific differences. |
| **XI** | **Logs** | Application logs are emitted to standard output (`stdout`/`stderr`) and can be collected by the hosting platform without writing local log files. |
| **XII** | **Admin Processes** | One-off administrative operations (continuous retraining through `ml/retrain.py`, audits, or rollback through `ml/rollback.py`) run as separate scripts or GitHub Actions jobs (`workflow_dispatch`), independently of the online inference process. |

### Factors with Limited Applicability

- **Factor VIII (Concurrency — OS-level clustering):** Since the application uses a lightweight inference pipeline, complex internal multiprocessing is less relevant than horizontal scaling through the container hosting platform (cloud autoscaling).

---

## 2. Branch Triggers & CI/CD Pipeline

The project follows a structured Git workflow:

1. **Pull requests targeting `dev`, `staging`, or `main`:**
   - Run linting and static analysis (`ruff`).
   - Run the full automated application test suite (`pytest`).
   - Build the Docker image (`docker build`).
   - Start the container and run a smoke test to verify startup and the `/ready` endpoint.
   - Require successful checks before a pull request may be merged.

2. **Merge into `staging`:**
   - Validate the release candidate in an isolated staging environment.

3. **Merge into `main`:**
   - Build and identify the Docker image using the Git commit SHA (`predictmaint-ai:sha-<commit>`).
   - Automatically deploy the application to the online production platform (Render / Azure / Cloud Run).
   - Wait for the readiness endpoint (`/ready`) to respond successfully.
   - Run an actual HTTP smoke test against the public production URL, checking a prediction and the loaded model version.

---

## 3. Reproducing the Deployment

### Local Deployment (Docker)

1. **Build the production Docker image:**

   ```bash
   docker build -t predictmaint-ai:local .
   ```

2. **Run the container in test mode (mock model enabled):**

   ```bash
   docker run -d -p 8000:8000 -e ENABLE_MOCK_MODEL=true --name predictmaint predictmaint-ai:local
   ```

3. **Check application readiness (local smoke test):**

   ```bash
   curl -f http://localhost:8000/ready
   curl -f http://localhost:8000/model-info
   ```

4. **Stop and remove the test container:**

   ```bash
   docker stop predictmaint && docker rm predictmaint
   ```

### Production Deployment (Cloud)

1. Configure secret environment variables on the target platform (Render / Cloud Run):
   - `MLFLOW_TRACKING_URI`: `https://dagshub.com/Abdel380/predictmaint-ai.mlflow`
   - `MLFLOW_TRACKING_USERNAME`: `<your_dagshub_username>`
   - `MLFLOW_TRACKING_PASSWORD`: `<your_dagshub_token>`
   - `MLFLOW_MODEL_NAME`: `predictmaint-ai`
   - `MLFLOW_MODEL_ALIAS`: `champion`
2. Deploy the image identified by the Git commit SHA.
3. Verify that the approved champion model has been loaded through the public URL:

   ```bash
   curl -f https://<your-app-url>/ready
   ```

---

## 4. MLflow Model Rollback

### 4.1. Objective

The rollback procedure restores a previously approved MLflow model version when a newly promoted champion must be replaced.

The procedure must:

1. Validate the current champion and the target version.
2. Restore the MLflow `champion` alias.
3. Restart or redeploy the inference application.
4. Verify the restored model version with a real HTTP prediction.

Changing the MLflow alias alone does not reload a model already running in FastAPI.

### 4.2. Safety Checks

The script `ml/rollback.py` enforces the following conditions:

- The current champion must match `--expected-current`.
- The target version must have status `READY`.
- The current champion must have the tag `approved_by=continuous_training_policy`.
- The target must match the champion's `previous_champion_version` tag.
- The target must pass a local inference check before the alias changes.
- The default execution is read-only. Only `--execute` changes the alias.
- The operation is recorded in MLflow.

The `--model-name` argument allows an isolated demonstration. By default, the script targets `predictmaint-ai`.

### 4.3. Controlled Rollback Demonstration

The demonstration uses a separate registered model:

`predictmaint-ai-ct-demo-henri-fixed`

This prevents modifications to the production model `predictmaint-ai`.

**Prerequisites:**

- Python dependencies installed.
- DagsHub / MLflow authentication configured.
- Validation dataset available locally.
- Champion version 4, with `previous_champion_version=1`, and target version 1 in `READY` status.

**Step 1 — Dry-run**

```bash
python -m ml.rollback --model-name predictmaint-ai-ct-demo-henri-fixed --expected-current 4 --to-version 1
```

Expected result:

```text
DRY-RUN PASSED
No champion alias modification.
```

**Step 2 — Execute the rollback**

Run only after validating the current registry state:

```bash
python -m ml.rollback --model-name predictmaint-ai-ct-demo-henri-fixed --expected-current 4 --to-version 1 --execute
```

Expected result:

```text
ROLLBACK SUCCESSFUL
Previous version: 4
New champion: 1
Rollback audit recorded in MLflow.
```

**Step 3 — Reload the application**

The model is resolved at FastAPI startup. Restart the application with these environment variables:

```text
MLFLOW_MODEL_NAME=predictmaint-ai-ct-demo-henri-fixed
MLFLOW_MODEL_ALIAS=champion
ENABLE_MOCK_MODEL=false
```

Use the shared DagsHub MLflow tracking server and valid credentials.

The application must load model version 1. A mock model is not acceptable for this verification.

**Step 4 — Verify the restored model**

With FastAPI running on localhost port 8001, run:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/model-info
```

Expected fields:

```text
model_name    : predictmaint-ai-ct-demo-henri-fixed
model_version : 1
alias         : champion
status        : loaded
```

Verify a real prediction:

```powershell
$body = @{
    type = "L"
    air_temperature = 298.1
    process_temperature = 308.6
    rotational_speed = 1500
    torque = 40
    tool_wear = 20
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8001/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

The response must contain a valid prediction, a risk score between 0 and 1, and `model_version=1`.

### 4.4. Verified Demonstration Evidence

A controlled rollback from version 4 to version 1 was performed on November 25, 2026.

Results:

- MLflow rollback: PASSED.
- FastAPI reload with restored v1: PASSED.
- HTTP `/model-info`: version 1 confirmed.
- HTTP `/predict`: prediction 0, risk score 0.0058, model version 1.

MLflow audit run:

https://dagshub.com/Abdel380/predictmaint-ai.mlflow/#/experiments/2/runs/62ee72d6e93a41de99705dbedd304dcd

This demonstration was performed using an isolated MLflow registered model and a local FastAPI instance. It was not a production Render rollback.

### 4.5. Production Release Recovery

The production CT release orchestrator `ml/ct_release.py` includes rollback logic.

If deployment verification fails after an approved model promotion, the orchestrator attempts to:

1. Restore the previous MLflow champion using `ml/rollback.py`.
2. Trigger a Render redeployment using `ml/render_redeploy.py`.
3. Verify `/ready`, `/model-info` and `/predict` against the restored version.

If the champion did not change, the release reports failure without attempting an unnecessary rollback.

If the model registry changes unexpectedly, the procedure stops and requires manual investigation.

### 4.6. Failure to Become Ready

If the application does not become ready after a model change:

1. Treat the release as failed; do not report successful deployment.
2. Inspect the Render deployment status and application logs.
3. Verify which version is assigned to the MLflow `champion` alias.
4. Check which version is actually loaded by the application, if reachable.
5. Attempt to restore the previously approved model and redeploy.
6. Verify readiness and a real prediction after recovery.

If recovery also fails, GitHub Actions must report a failed release and manual intervention is required.

**Important:** An MLflow alias restoration does not by itself prove that Render has recovered. The live HTTP checks are mandatory.

The automated production recovery scenario is implemented and covered by mocked tests, but a complete failure-and-recovery execution on Render has not yet been demonstrated.
