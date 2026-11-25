import os
import time
import uuid
import logging
from collections import deque
from typing import Dict, Any

import numpy as np
import pandas as pd
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# =====================================================================
# 1. CONFIGURATION STRUCTLOG (Logs JSON stricts demandés en page 6-7)
# =====================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(),
    wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, LOG_LEVEL, logging.INFO)),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()

# =====================================================================
# 2. MÉTRIQUES PROMETHEUS (Exigences de la page 8 du sujet)
# =====================================================================

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Nombre total de requêtes HTTP",
    ["method", "path", "status"]
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Durée des requêtes HTTP en secondes",
    ["path"]
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Nombre total de prédictions réalisées",
    ["prediction", "model_version"]
)

MODEL_INFO = Gauge(
    "model_info",
    "Informations sur la version du modèle actuellement en service",
    ["model_version"]
)

FEATURE_DRIFT = Gauge(
    "feature_drift",
    "Score de dérive statistique par capteur (sur les 500 dernières prédictions)",
    ["feature"]
)

# =====================================================================
# 3. MOTEUR DE CALCUL DU DATA DRIFT (Sur les 6 capteurs de la machine)
# =====================================================================

# Moyennes et écarts-types de référence issus du jeu d'entraînement AI4I
BASELINE_STATS = {
    "Air temperature": {"mean": 300.0, "std": 2.0},
    "Process temperature": {"mean": 310.0, "std": 1.5},
    "Rotational speed": {"mean": 1538.0, "std": 179.0},
    "Torque": {"mean": 40.0, "std": 10.0},
    "Tool wear": {"mean": 108.0, "std": 63.0},
}


class DriftDetector:
    """
    Garde en mémoire les 500 dernières prédictions et recalcule la dérive
    statistique toutes les 100 prédictions (exigence page 8).
    """
    def __init__(self, window_size=500, update_every=100):
        self.window_size = window_size
        self.update_every = update_every
        self.buffer = deque(maxlen=window_size)
        self.counter = 0

    def add_prediction(self, features_dict: Dict[str, Any]):
        self.buffer.append(features_dict)
        self.counter += 1

        # Recalcul régulier de la dérive
        if self.counter % self.update_every == 0 or len(self.buffer) == 50:
            self._update_drift()

    def _update_drift(self):
        if not self.buffer:
            return

        df_window = pd.DataFrame(list(self.buffer))

        for col, stats in BASELINE_STATS.items():
            if col in df_window.columns:
                current_mean = df_window[col].mean()
                # Dérive normalisée : écart absolu par rapport à la moyenne de référence
                drift_score = min(1.0, abs(current_mean - stats["mean"]) / (3.0 * stats["std"]))
                FEATURE_DRIFT.labels(feature=col).set(round(float(drift_score), 4))


drift_detector = DriftDetector()

# =====================================================================
# 4. MIDDLEWARE FASTAPI (Traçabilité X-Request-ID et Logs)
# =====================================================================

class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Intercepte chaque requête HTTP pour :
    1. Attribuer un X-Request-ID unique.
    2. Mesurer la latence.
    3. Incrémenter les métriques Prometheus.
    4. Émettre un log JSON structuré.
    """
    async def dispatch(self, request: Request, call_next):
        # 1. Gestion du X-Request-ID
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start_time = time.time()
        path = request.url.path
        method = request.method

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            log.error(
                "request_failed",
                method=method,
                path=path,
                status=500,
                duration_ms=duration_ms,
                error=str(exc)
            )
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status="500").inc()
            raise exc

        # 2. Calcul du temps d'exécution
        duration_s = time.time() - start_time
        duration_ms = round(duration_s * 1000, 2)

        # 3. Mise à jour des métriques Prometheus (on filtre /metrics pour ne pas fausser les stats)
        if path != "/metrics":
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=str(status_code)).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(path=path).observe(duration_s)

        # 4. Journalisation JSON obligatoire
        response.headers["X-Request-ID"] = request_id

        if path != "/metrics":
            log.info(
                "http_request",
                method=method,
                path=path,
                status=status_code,
                duration_ms=duration_ms
            )

        return response