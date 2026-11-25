import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from app.schemas import MachineDataInput, PredictionOutput, ModelInfoOutput
from app.model_loader import model_service
from app.monitoring import (
    MonitoringMiddleware,
    PREDICTIONS_TOTAL,
    MODEL_INFO,
    drift_detector,
    generate_latest,
    CONTENT_TYPE_LATEST,
    log
)

# Chargement automatique des variables d'environnement (.env)
load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie de l'application :
    1. Tente de charger le modèle champion MLflow au démarrage.
    2. Initialise les métriques et les logs structurés.
    """
    model_service.load_model()

    if model_service.is_ready:
        version_str = str(model_service.loaded_version)
        MODEL_INFO.labels(model_version=version_str).set(1)
        log.info("application_startup", model_version=version_str, status="ready")
    else:
        log.warning("application_startup", status="not_ready", error=model_service.error_message)

    yield

    log.info("application_shutdown")


app = FastAPI(
    title="PredictMaint AI",
    description="API de maintenance prédictive industrielle avec monitoring Prometheus et structlog.",
    version="1.0.0",
    lifespan=lifespan
)

# 1. Activation du middleware de surveillance (chronomètre et logs JSON)
app.add_middleware(MonitoringMiddleware)

# 2. Montage des fichiers statiques pour l'interface web
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_home():
    """Sert l'interface utilisateur web (HTML)."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Bienvenue sur l'API PredictMaint AI. Rendez-vous sur /docs pour tester l'API Swagger."
    }


@app.get(
    "/metrics",
    summary="Exposition des métriques Prometheus",
    tags=["Monitoring"],
    include_in_schema=False
)
def get_metrics():
    """
    Endpoint collecté toutes les 5 secondes par Prometheus.
    Expose : requêtes, latence, dérive de capteurs et prédictions.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get(
    "/ready",
    summary="Vérification de la disponibilité du modèle (Readiness Probe)",
    tags=["Diagnostic"]
)
def check_readiness():
    """Ne répond 200 OK QUE si le modèle champion approuvé est chargé."""
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "reason": model_service.error_message or "Modèle champion non disponible."
            }
        )
    return {
        "status": "ready",
        "model_name": model_service.model_name,
        "loaded_version": model_service.loaded_version,
        "alias": model_service.model_alias
    }


@app.get(
    "/model-info",
    response_model=ModelInfoOutput,
    summary="Informations sur la version du modèle chargé",
    tags=["Diagnostic"]
)
def get_model_info():
    """Affiche la version concrète du modèle actuellement en mémoire."""
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aucun modèle n'est actuellement chargé."
        )
    return ModelInfoOutput(
        model_name=model_service.model_name,
        model_version=str(model_service.loaded_version),
        alias=model_service.model_alias,
        status="loaded"
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Prédire le risque de panne pour une machine",
    tags=["Inférence"]
)
def predict_machine_failure(data: MachineDataInput):
    """
    Reçoit les paramètres d'une machine, reconstruit les 6 colonnes,
    enregistre les métriques en temps réel et retourne le diagnostic.
    """
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le service ne peut pas répondre : le modèle n'est pas prêt."
        )

    # Chronomètre précis de l'inférence
    start_pred = time.time()

    # 1. Conversion des données vers le format exact du modèle
    features_df = data.to_model_dataframe()

    # 2. Inférence
    try:
        prediction, risk_score = model_service.predict(features_df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du calcul de la prédiction : {str(e)}"
        )

    pred_latency_ms = round((time.time() - start_pred) * 1000, 2)
    version_str = str(model_service.loaded_version)

    # 3. Mise à jour des métriques Prometheus
    PREDICTIONS_TOTAL.labels(prediction=str(prediction), model_version=version_str).inc()
    
    # 4. Alimentation du radar de dérive (Data Drift)
    drift_detector.add_prediction({
        "Air temperature": data.air_temperature,
        "Process temperature": data.process_temperature,
        "Rotational speed": data.rotational_speed,
        "Torque": data.torque,
        "Tool wear": data.tool_wear
    })

    # 5. Log JSON structuré exigé par le sujet (page 7)
    log.info(
        "prediction_made",
        model_version=version_str,
        prediction=prediction,
        risk_score=risk_score,
        latency_ms=pred_latency_ms,
        number_of_features=6,
        number_of_missing_values=0
    )

    # 6. Réponse
    return PredictionOutput(
        failure_prediction=prediction,
        risk_score=risk_score,
        model_name=model_service.model_name,
        model_version=version_str
    )