# PredictMaint AI — Deployment Documentation & Twelve-Factor App

This document describes the deployment architecture, the implementation of the **Twelve-Factor App** methodology, and the procedures for reproducing local and production deployments of **PredictMaint AI**.

---

## 1. Twelve-Factor App Methodology

The web application and inference service were designed according to the **Twelve-Factor App** principles to support portability, observability, and resilience in production.

| No. | Factor | Implementation in PredictMaint AI |
| --- | --- | --- |
| **I** | **Codebase** | A single centralized Git repository is shared by the team. Source code, tests, and workflow definitions are version-controlled using a structured branch flow (`dev` → `staging` → `main`). |
| **II** | **Dependencies** | The Dockerfile installs packages from `requirements.txt` in `python:3.11-slim` and installs `curl` for its health check. Python package versions are unpinned and the repository has no dependency lockfile. |
| **III** | **Config** | Render supplies MLflow settings and credentials as environment variables. Actions supplies Azure, DagsHub and Render credentials to the jobs that use them. The repository specifies defaults for the tracking URI, model name, alias and port. |
| **IV** | **Backing Services** | FastAPI retrieves the model from DagsHub MLflow at startup. Data and training jobs use the `myazure` DVC remote in Azure Blob Storage. The serving image includes the application and excludes the datasets and training scripts. |
| **V** | **Build, Release, Run** | Actions builds and smoke-tests a Docker image with an environment tag. Production deployment requests a Render deployment for the validated Git SHA and checks the commit on the live deploy. The immutable identity of the image Render actually serves remains unconfirmed. |
| **VI** | **Processes** | Inference needs no persistent local state. Each process holds its loaded model, Prometheus counters and drift window in memory. A restart reloads the champion and resets the process-local metrics. |
| **VII** | **Port Binding** | Uvicorn binds to `0.0.0.0` on `PORT`. The Dockerfile sets the default to `8000` and probes `/ready` on that port. |
| **VIII** | **Concurrency** | The Docker command starts one Uvicorn process per container. Each replica would load its own model and collect its own metrics. The repository defines no Render autoscaling policy. |
| **IX** | **Disposability** | FastAPI loads the model during startup and logs shutdown through its lifespan handler. A process that fails to load the real model returns HTTP 503 on `/ready` in production. |
| **X** | **Dev/Prod Parity** | Local, CI, staging and production use the project Dockerfile. CI container smoke tests explicitly enable the mock model. Staging and production checks require a real loaded version and a successful prediction. |
| **XI** | **Logs** | Application logs are emitted to standard output (`stdout`/`stderr`) and can be collected by the hosting platform without writing local log files. |
| **XII** | **Admin Processes** | One-off administrative operations (continuous retraining through `ml/retrain.py`, audits, or rollback through `ml/rollback.py`) run as separate scripts or GitHub Actions jobs (`workflow_dispatch`), independently of the online inference process. |

---

## 2. Branch Triggers & CI/CD Pipeline

The project follows a structured Git workflow:

1. **Pull requests targeting `dev`, `staging`, or `main`:**
   - Run linting and static analysis (`ruff`).
   - Run the full automated application test suite (`pytest`).
   - Build the Docker image (`docker build`).
   - Start the container and run a smoke test to verify startup and the `/ready` endpoint.
   - The required CI checks enforced by GitHub branch protection remain to be confirmed.

2. **Push to `staging` or manual dispatch on that branch**
   - Validate the code and Docker image, request a Render staging deployment for the workflow SHA, then check the public readiness, loaded version and prediction.

3. **Push to `main` or manual dispatch on that branch**
   - Validate the code and smoke-test `predictmaint-ai:production`. Call the Render deploy hook with the workflow SHA, poll the returned deploy ID and require status `live` with the same Git SHA.
   - Resolve the MLflow champion and verify it through `/ready`, `/model-info` and a real `POST /predict`. [Production workflow](.github/workflows/production.yml)

4. **Model release and monitoring**
   - Continuous Training logs a candidate and returns its promotion decision. An operator separately dispatches **CT Production Release** on `main` with the MLflow CT run ID. Preflight checks the current application and recomputes eligibility. In `release` mode, an eligible candidate proceeds to the `production-ct` environment for human approval, promotion, automatic Render redeployment and HTTP verification. [Protected release workflow](.github/workflows/ct-production-release.yml)
   - The production monitor compares MLflow with the live API after either production workflow completes, at minutes 17 and 47 each hour, or on manual dispatch. Grafana dashboards run in the local Compose stack. [Production monitor](.github/workflows/production-model-monitor.yml) · [Local monitoring](monitoring/README.md)

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

### Production Deployment (Render)

1. Configure the inference service on Render:
   - `MLFLOW_TRACKING_URI`: `https://dagshub.com/Abdel380/predictmaint-ai.mlflow`
   - `MLFLOW_TRACKING_USERNAME`: `<your_dagshub_username>`
   - `MLFLOW_TRACKING_PASSWORD`: `<your_dagshub_token>`
   - `MLFLOW_MODEL_NAME`: `predictmaint-ai`
   - `MLFLOW_MODEL_ALIAS`: `champion`
   - `ENABLE_MOCK_MODEL`: `false`
2. In GitHub Actions, configure the secrets `RENDER_PRODUCTION_DEPLOY_HOOK_URL`, `RENDER_API_KEY`, and `DAGSHUB_TOKEN`, and the variables `RENDER_PRODUCTION_SERVICE_ID`, `RENDER_PRODUCTION_URL`, and `DAGSHUB_USERNAME`.
3. Run **PredictMaint - Production Deployment** on `main`. The workflow validates the commit, waits for Render and verifies the champion as described above. To inspect the loaded version:

   ```bash
   curl -f https://predictmaint-ai-production.onrender.com/ready
   curl -f https://predictmaint-ai-production.onrender.com/model-info
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
- With `--execute`, the script rechecks the current champion before writing, verifies the restored alias and records a rollback audit in MLflow.

The `--model-name` argument allows an isolated demonstration. By default, the script targets `predictmaint-ai`.

### 4.3. Production rollback from v3 to v1

The team performed this procedure on 25 November 2026 after the approved v3 release and its synchronization check. It applies when `predictmaint-ai@champion` points to **v3** and v3 records **v1** as its previous champion. The verified presentation state is already **v1**, so the expected-current guard would stop a repeat at the first step. [Approved v3 release](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45712120942) · [Post-promotion monitor](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45712976568)

1. **Run the precheck.** From the repository root, use Python 3.11 with `requirements.txt` installed, DagsHub/MLflow credentials for the project and the DVC-tracked `data/splits/validation.csv`. If needed, retrieve the file with `dvc pull -r myazure data/splits/validation.csv.dvc` using Azure credentials. The command checks v3's approval and predecessor tags, loads v1 and tests its probabilities on three validation rows. Require `DRY-RUN PASSED` before proceeding.

   ```bash
   python -m ml.rollback --model-name predictmaint-ai --expected-current 3 --to-version 1
   ```

2. **Restore the alias.** Using credentials with registry write access, repeat the command with `--execute`. Confirm `New champion: 1` and the MLflow audit entry for `predictmaint-ai`, previous version `3`, restored version `1`. [Production rollback audit](https://dagshub.com/Abdel380/predictmaint-ai.mlflow/#/experiments/2/runs/87a607ce17f74f329995e2f4db0a085b)

   ```bash
   python -m ml.rollback --model-name predictmaint-ai --expected-current 3 --to-version 1 --execute
   ```

3. **Redeploy the serving process.** In GitHub Actions select **PredictMaint - Production Deployment → Run workflow**, choose `main` and run it once. Wait for **Deploy and verify Render production** to succeed. This reloads the champion at startup and checks the live Git SHA. The post-rollback run used commit `5578e0d`. [Post-rollback deployment and HTTP checks](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45716691906)

4. **Verify v1 over HTTP.** The following command checks HTTP 200 on all three endpoints. It requires `/ready` to report `loaded_version="1"`, `/model-info` to report `predictmaint-ai`, `champion` and `model_version="1"`, and `POST /predict` to return a binary prediction, a risk score in `[0, 1]` and version `"1"`. Require `Status: PASSED`. The subsequent `main` deployment at `b2044c0` and its synchronization monitor both passed. [Final deployment](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45718935803) · [Final monitor](https://github.com/eugeniettc/final-project-ml-in-prod/actions/runs/45719605125)

   ```bash
   python -m ml.verify_deployment \
     --url https://predictmaint-ai-production.onrender.com \
     --expected-version 1
   ```

If a precheck fails, stop before execution and inspect the reported version, provenance or inference error. If execution fails after the alias write, read the alias before retrying. If MLflow already points to v1, resume at step 3 and investigate any missing audit entry separately. A failed HTTP check leaves recovery unverified.

### 4.4. Historical isolated rollback

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

### 4.5. Isolated rollback evidence

A controlled rollback from version 4 to version 1 was performed on November 25, 2026.

Results:

- MLflow rollback: PASSED.
- FastAPI reload with restored v1: PASSED.
- HTTP `/model-info`: version 1 confirmed.
- HTTP `/predict`: prediction 0, risk score 0.0058, model version 1.

These results apply to the isolated registered model and local FastAPI instance. [Isolated rollback audit](https://dagshub.com/Abdel380/predictmaint-ai.mlflow/#/experiments/2/runs/62ee72d6e93a41de99705dbedd304dcd)

### 4.6. Automatic recovery in a protected production release

The production CT release orchestrator `ml/ct_release.py` includes rollback logic.

If deployment verification fails after an approved model promotion, the orchestrator attempts to:

1. Restore the previous MLflow champion using `ml/rollback.py`.
2. Trigger a Render redeployment using `ml/render_redeploy.py`.
3. Verify `/ready`, `/model-info` and `/predict` against the restored version.

If the champion did not change, the release reports failure without attempting an unnecessary rollback.

If the current version's `ct_run_id` or `previous_champion_version` identifies a different release, the orchestrator stops for manual investigation. A failed release retains its failed GitHub Actions status even when recovery succeeds.

### 4.7. Failure to Become Ready

If the application does not become ready after a model change:

1. Treat the release as failed; do not report successful deployment.
2. Inspect the Render deployment status and application logs.
3. Verify which version is assigned to the MLflow `champion` alias.
4. Check which version is actually loaded by the application, if reachable.
5. Attempt to restore the previously approved model and redeploy.
6. Verify readiness and a real prediction after recovery.

If recovery also fails, GitHub Actions must report a failed release and manual intervention is required.

**Important:** An MLflow alias restoration does not by itself prove that Render has recovered. The live HTTP checks are mandatory.

Controlled tests exercise a failed deployment, verified recovery, failed recovery and an unexpected alias change. The production v3 → v1 rehearsal was an operator-initiated rollback after a successful release. A deployment failure has yet to be induced on Render. [Release recovery tests](tests/ml/test_ct_release.py)
