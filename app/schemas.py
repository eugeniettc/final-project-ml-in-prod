from typing import Literal
import pandas as pd
from pydantic import BaseModel, Field


class MachineDataInput(BaseModel):
    """
    Données d'entrée pour la prédiction de panne machine.
    Champs en snake_case pour une API REST propre.
    """
    type: Literal["L", "M", "H"] = Field(
        ..., 
        description="Type de produit : L (Low), M (Medium), H (High)"
    )
    air_temperature: float = Field(
        ..., 
        gt=0, 
        description="Température de l'air en Kelvin [K] (doit être > 0)"
    )
    process_temperature: float = Field(
        ..., 
        gt=0, 
        description="Température du processus en Kelvin [K] (doit être > 0)"
    )
    rotational_speed: float = Field(
        ..., 
        gt=0, 
        description="Vitesse de rotation en tours/minute [rpm] (doit être > 0)"
    )
    torque: float = Field(
        ..., 
        ge=0, 
        description="Couple en Newton-mètres [Nm] (doit être >= 0)"
    )
    tool_wear: float = Field(
        ..., 
        ge=0, 
        description="Usure de l'outil en minutes [min] (doit être >= 0)"
    )

    def to_model_dataframe(self) -> pd.DataFrame:
        """
        Reconstruit le DataFrame avec les 6 noms de colonnes EXACTS
        attendus par le modèle Scikit-learn de ton équipier.
        """
        return pd.DataFrame([{
            "Type": self.type,
            "Air temperature": self.air_temperature,
            "Process temperature": self.process_temperature,
            "Rotational speed": self.rotational_speed,
            "Torque": self.torque,
            "Tool wear": self.tool_wear,
        }])

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "L",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1450.0,
                "torque": 48.0,
                "tool_wear": 180.0
            }
        }
    }


class PredictionOutput(BaseModel):
    """
    Format de la réponse renvoyée par l'API /predict.
    """
    failure_prediction: int = Field(
        ..., 
        description="Prédiction binaire (0: Pas de panne, 1: Panne)"
    )
    risk_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Score de risque de défaillance (entre 0.0 et 1.0)"
    )
    model_name: str = Field(
        ..., 
        description="Nom du modèle enregistré dans MLflow"
    )
    model_version: str = Field(
        ..., 
        description="Version concrète du modèle actuellement chargée"
    )


class ModelInfoOutput(BaseModel):
    """
    Format de la réponse renvoyée par /model-info.
    """
    model_name: str
    model_version: str
    alias: str
    status: str