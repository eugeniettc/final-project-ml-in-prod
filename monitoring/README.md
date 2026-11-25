# Monitoring & Alerting — PredictMaint AI

Observability stack for the FastAPI inference service using **Prometheus**, **Grafana**, **structlog**, and **Mailtrap** (SMTP alerting).

## 1. Setup & Startup

Add your SMTP credentials to `.env` at the repo root (ignored by Git):

```env
LOG_LEVEL=INFO
SMTP_HOST=sandbox.smtp.mailtrap.io:2525
SMTP_USER=<your_smtp_user>
SMTP_PASSWORD=<your_smtp_password>
SMTP_FROM=alerts@predictmaint.ai
```

Start the stack:

```bash
docker compose up --build -d
```

- **API (`/health`, `/metrics`, `/docs`):** `http://localhost:8000`
- **Prometheus:** `http://localhost:9090` (scrapes `api:8000/metrics` every 5s)
- **Grafana:** `http://localhost:3000` (`admin` / `admin`) — import `monitoring/dashboard.json`

## 2. Traffic & Incident Simulation

Run `monitoring/replay.py` to test the 6 dashboard panels (filtered on `path="/predict"`):

```bash
# Nominal traffic
python monitoring/replay.py --rate 25 --limit 500

# Error spike (drops a required feature -> HTTP 422)
python monitoring/replay.py --broken --rate 10 --limit 50

# Sensor drift (injects overheat ~340 K -> feature_drift >= 0.20)
python monitoring/replay.py --drift --rate 20 --limit 300
```

## 3. Logs & Alerting

- **Filter JSON error logs:**
  ```bash
  docker compose logs api --no-log-prefix | jq -R -c 'fromjson? | select(.level == "error")'
  ```
- **Alert rules (Grafana → Mailtrap, with `No data = Normal`):**
  - **API down:** `up{job="api"} < 1` for `1m`
  - **High error rate:** `> 5%` (`422` / `5xx` on `/predict`) for `2m`
  - **Data drift:** `feature_drift >= 0.20` for `2m` (investigate cause before retraining)
