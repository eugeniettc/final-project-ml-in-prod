from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from app.schemas import MachineDataInput, PredictionOutput, ModelInfoOutput
from app.model_loader import model_service

# Chargement automatique des variables d'environnement (.env si présent)
load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie de l'application :
    Tente de charger le modèle champion MLflow au démarrage.
    """
    print("Démarrage de l'application... Tentative de chargement du modèle Champion...")
    model_service.load_model()
    yield
    print("Arrêt de l'application.")


app = FastAPI(
    title="PredictMaint AI",
    description="API de maintenance prédictive industrielle basée sur Scikit-learn et MLflow.",
    version="1.0.0",
    lifespan=lifespan
)

# Montage des fichiers statiques pour l'interface web
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_home():
    """
    Sert l'interface utilisateur web (HTML).
    """
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Bienvenue sur l'API PredictMaint AI. Rendez-vous sur /docs pour tester l'API Swagger."
    }


@app.get(
    "/ready",
    summary="Vérification de la disponibilité du modèle (Readiness Probe)",
    tags=["Diagnostic"]
)
def check_readiness():
    """
    Ne répond 200 OK QUE si le modèle champion approuvé est chargé
    et prêt à servir des prédictions. Sinon renvoie une erreur 503.
    """
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
    """
    Affiche la version concrète du modèle actuellement en mémoire.
    """
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
    Reçoit les paramètres d'une machine, reconstruit les 6 colonnes
    du modèle et retourne le diagnostic de défaillance.
    """
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le service ne peut pas répondre : le modèle n'est pas prêt."
        )

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

    # 3. Réponse conforme au contrat technique
    return PredictionOutput(
        failure_prediction=prediction,
        risk_score=risk_score,
        model_name=model_service.model_name,
        model_version=str(model_service.loaded_version)
    )