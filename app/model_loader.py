import os
import logging
from typing import Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("predictmaint.model_loader")

# Configuration MLflow / DagsHub
DEFAULT_TRACKING_URI = "https://dagshub.com/Abdel380/predictmaint-ai.mlflow"
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "predictmaint-ai")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


class MockModelPipeline:
    """
    Modèle de simulation STRICTEMENT réservé aux tests unitaires
    et au développement hors-ligne quand ENABLE_MOCK_MODEL=true.
    """
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Règle simple : panne si l'usure de l'outil est critique (> 200 min)
        return (X["Tool wear"] > 200).astype(int).values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Simule des probabilités [P(sain), P(panne)]
        p_failure = np.clip(X["Tool wear"] / 250.0, 0.05, 0.95)
        p_healthy = 1.0 - p_failure
        return np.column_stack([p_healthy, p_failure])


class ModelService:
    """
    Service gérant le cycle de vie du modèle Champion en production.
    """
    def __init__(self):
        self.model = None
        self.model_name: str = MODEL_NAME
        self.model_alias: str = MODEL_ALIAS
        self.loaded_version: Optional[str] = None
        self.is_ready: bool = False
        self.error_message: Optional[str] = None

    def load_model(self) -> bool:
        """
        Charge le modèle MLflow portant l'alias 'champion'.
        Résout d'abord la version concrète pour garantir la traçabilité.
        """
        # Vérifie si le mode MOCK est explicitement demandé (pour les tests locaux)
        if os.getenv("ENABLE_MOCK_MODEL", "false").lower() == "true":
            logger.warning("MODE MOCK ACTIVÉ : Utilisation d'un modèle simulé pour les tests.")
            self.model = MockModelPipeline()
            self.loaded_version = "mock-1.0"
            self.is_ready = True
            self.error_message = None
            return True

        # Mode Production : Connexion stricte à MLflow
        try:
            import mlflow
            from mlflow import MlflowClient
            import mlflow.sklearn

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
            mlflow.set_tracking_uri(tracking_uri)

            client = MlflowClient()
            logger.info(f"Résolution de l'alias '{self.model_alias}' pour '{self.model_name}'...")

            # 1. Résolution de l'alias vers la version concrète
            champion_version_info = client.get_model_version_by_alias(
                name=self.model_name,
                alias=self.model_alias
            )
            concrete_version = str(champion_version_info.version)

            # 2. Chargement de cette version précise
            model_uri = f"models:/{self.model_name}/{concrete_version}"
            logger.info(f"Chargement du modèle depuis : {model_uri}")
            
            loaded_model = mlflow.sklearn.load_model(model_uri)

            # 3. Validation de l'état
            self.model = loaded_model
            self.loaded_version = concrete_version
            self.is_ready = True
            self.error_message = None
            logger.info(f"Modèle {self.model_name} v{concrete_version} chargé avec succès.")
            return True

        except Exception as e:
            # RÈGLE D'OR : Aucun fallback silencieux en production !
            self.model = None
            self.loaded_version = None
            self.is_ready = False
            self.error_message = f"Échec du chargement du modèle : {str(e)}"
            logger.error(self.error_message)
            return False

    def predict(self, features_df: pd.DataFrame) -> Tuple[int, float]:
        """
        Exécute la prédiction sur le DataFrame formaté.
        Retourne (failure_prediction, risk_score).
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError("Le modèle n'est pas prêt à servir des prédictions.")

        # Calcul des prédictions
        prediction = int(self.model.predict(features_df)[0])

        # Calcul du score de risque
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features_df)
            risk_score = float(probabilities[0][1])  # Probabilité de la classe 1 (Panne)
        else:
            risk_score = float(prediction)

        return prediction, round(risk_score, 4)


# Instance unique (Singleton) partagée dans toute l'application
model_service = ModelService()